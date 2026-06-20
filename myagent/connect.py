"""`python -m myagent connect` — zero-config wiring for installed AI tools.

The "like cloud" onboarding: one command detects which AI tools are present on
this machine and writes the OpenBrain MCP server into each one's config, in that
tool's exact format and OS-correct location. Merges into existing config (never
clobbers other servers), backs up the original once, and is idempotent.

Design notes:
  * Data-driven registry — adding a tool is one entry below.
  * Cross-platform paths (macOS / Linux / Windows) via _config_bases().
  * Per-tool quirks: servers key (mcpServers/servers/context_servers), entry
    shape (url / httpUrl / serverUrl / {type:http}), and CLI tools (Claude Code,
    Codex) that own their config via their own `mcp add` command.
  * `--dry-run` shows the plan without writing. `--project DIR` also installs
    project-level hooks/rules via scripts/install-integrations.sh.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import load_config


def _config_bases() -> dict[str, Path]:
    """OS-correct base directories where apps keep their config."""
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        return {"home": home, "appsupport": home / "Library" / "Application Support",
                "xdg": home / ".config"}
    if system == "Windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return {"home": home, "appsupport": appdata, "xdg": appdata}
    return {"home": home, "appsupport": home / ".config", "xdg": home / ".config"}


@dataclass
class JsonTool:
    label: str
    detect: Path          # exists ⇒ tool is installed
    path: Path            # config file to write
    servers_key: str      # top-level key holding the servers map
    entry: dict           # the openbrain server entry for this tool


def _cu(base: str, client: str, token: str = "") -> str:
    """Append ?client=<tool> (provenance) and, when set, &token=<tok> (the local
    API token the daemon now requires). HTTP MCP is stateless, but we own the URL
    we write here, so baking the token in keeps wiring zero-friction."""
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}client={client}"
    if token:
        url += f"&token={token}"
    return url


def _json_tools(url: str, b: dict[str, Path], token: str = "") -> list[JsonTool]:
    """Every JSON-config tool we know how to wire, for the current platform.
    Each carries ?client=<key> (origin tool) and the API token."""
    def cu(client: str) -> str:
        return _cu(url, client, token)

    return [
        JsonTool("Cursor", b["home"] / ".cursor",
                 b["home"] / ".cursor" / "mcp.json",
                 "mcpServers", {"url": cu("cursor")}),
        JsonTool("Gemini CLI", b["home"] / ".gemini",
                 b["home"] / ".gemini" / "settings.json",
                 "mcpServers", {"httpUrl": cu("gemini")}),
        JsonTool("Windsurf", b["home"] / ".codeium" / "windsurf",
                 b["home"] / ".codeium" / "windsurf" / "mcp_config.json",
                 "mcpServers", {"serverUrl": cu("windsurf")}),
        # Google Antigravity: ~/.gemini/antigravity/mcp_config.json (its own
        # subdir under ~/.gemini, distinct from Gemini CLI's settings.json).
        # Supports streamable HTTP natively via `serverUrl` (like Windsurf), so
        # no mcp-remote bridge is needed.
        JsonTool("Antigravity", b["home"] / ".gemini" / "antigravity",
                 b["home"] / ".gemini" / "antigravity" / "mcp_config.json",
                 "mcpServers", {"serverUrl": cu("antigravity")}),
        JsonTool("Claude Desktop", b["appsupport"] / "Claude",
                 b["appsupport"] / "Claude" / "claude_desktop_config.json",
                 # Claude Desktop only accepts stdio MCP entries (command/args);
                 # an {"url": ...} entry is silently dropped with a popup. Bridge
                 # to our local HTTP server via npx mcp-remote.
                 "mcpServers", {"command": "npx",
                                "args": ["-y", "mcp-remote", cu("claude-desktop")]}),
        JsonTool("VS Code", b["appsupport"] / "Code" / "User",
                 b["appsupport"] / "Code" / "User" / "mcp.json",
                 "servers", {"type": "http", "url": cu("vscode")}),
        JsonTool("Zed", b["xdg"] / "zed",
                 b["xdg"] / "zed" / "settings.json",
                 "context_servers",
                 # Zed has no native HTTP transport yet — bridge via mcp-remote.
                 {"source": "custom", "command": "npx",
                  "args": ["-y", "mcp-remote", cu("zed")]}),
    ]


def _merge_server(path: Path, servers_key: str, entry: dict,
                  dry_run: bool = False) -> str:
    """Insert/refresh the 'openbrain' entry under servers_key, creating the file
    if needed. Non-destructive; backs up once. Returns a status string."""
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) or {}
        except (json.JSONDecodeError, OSError):
            return f"skipped (couldn't parse {path})"

    servers = existing.get(servers_key)
    if not isinstance(servers, dict):
        servers = {}
    if servers.get("openbrain") == entry:
        return "already wired"
    if dry_run:
        return f"would wire → {path}"

    if path.exists() and not path.with_suffix(path.suffix + ".bak").exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    servers["openbrain"] = entry
    existing[servers_key] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return f"wired → {path}"


def _cli_path(tool: str) -> str | None:
    """Find a CLI even when running under launchd's minimal PATH (the daemon's
    case) — search common install locations in addition to $PATH."""
    home = Path.home()
    common = os.pathsep.join(filter(None, [
        os.environ.get("PATH", ""),
        str(home / ".local/bin"), "/usr/local/bin", "/opt/homebrew/bin",
        str(home / ".bun/bin"), str(home / ".npm-global/bin"), "/usr/bin",
    ]))
    return shutil.which(tool, path=common)


def _connect_cli(tool: str, args: list[str], dry_run: bool) -> str | None:
    """Wire a tool that owns its config via its own CLI (Claude Code, Codex)."""
    exe = _cli_path(tool)
    if exe is None:
        return None
    tool = exe  # use the resolved absolute path
    if dry_run:
        return f"would run `{' '.join([tool, *args])}`"
    try:
        subprocess.run([tool, *args], check=False, capture_output=True, timeout=20)
        return f"wired via `{tool} mcp add`"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"skipped ({exc})"


def connect_tools(url: str, dry_run: bool = False, token: str = "") -> dict:
    """Detect installed tools and wire each. Returns a structured report so the
    UI 'Connect' button and the CLI share one implementation. `token` (the local
    API token) is baked into each tool's MCP URL so wiring stays zero-friction."""
    bases = _config_bases()
    connected: list[dict] = []
    not_installed: list[str] = []

    for label, tool, addargs in [
        ("Claude Code", "claude",
         ["mcp", "add", "-s", "user", "--transport", "http", "openbrain",
          _cu(url, "claude-code", token)]),
        ("Codex", "codex",
         ["mcp", "add", "openbrain", "--url", _cu(url, "codex", token)]),
    ]:
        status = _connect_cli(tool, addargs, dry_run)
        if status:
            connected.append({"label": label, "status": status})
        else:
            not_installed.append(label)

    for t in _json_tools(url, bases, token):
        if t.detect.exists():
            connected.append({"label": t.label,
                              "status": _merge_server(t.path, t.servers_key,
                                                      t.entry, dry_run)})
        else:
            not_installed.append(t.label)

    return {"connected": connected, "not_installed": not_installed}


def run_connect(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[2:]
    dry_run = "--dry-run" in argv
    project = None
    if "--project" in argv:
        i = argv.index("--project")
        project = argv[i + 1] if i + 1 < len(argv) else os.getcwd()

    cfg = load_config()
    url = f"http://127.0.0.1:{cfg.memory_port}/mcp"
    # Same token file the daemon uses (idempotent): whether the daemon or
    # `connect` runs first, both converge on one token, baked into each URL.
    from .auth import load_or_create_token
    token = load_or_create_token(cfg.db_path.parent)

    print("\n🔌  OpenBrain — connecting your AI tools"
          + ("  (dry run)" if dry_run else "") + "\n")

    report = connect_tools(url, dry_run, token)
    if report["connected"]:
        print("Connected:")
        for c in report["connected"]:
            print(f"  ✓ {c['label']}: {c['status']}")
    else:
        print("No supported tools detected on this machine yet.")
    if report["not_installed"]:
        print("\nNot detected (install the tool, then re-run `connect`):")
        print("  " + ", ".join(report["not_installed"]))

    # Optional project-level hooks/rules (Claude Code hook, Cursor rules, etc.).
    if project and not dry_run:
        installer = Path(__file__).resolve().parent.parent / "scripts" / "install-integrations.sh"
        if installer.exists():
            print(f"\nInstalling project-level files into {project} …")
            subprocess.run(["bash", str(installer), project], check=False)

    print(f"\nMemory dashboard: http://127.0.0.1:{cfg.memory_port}/")
    if not dry_run:
        print("Restart a tool after wiring for it to pick up the new server.")
    print()
