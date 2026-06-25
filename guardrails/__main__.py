"""Entry point / CLI for OpenBrain Guard Rails.

  openbrain-guardrails              run the daemon (default)
  openbrain-guardrails run         run the daemon
  openbrain-guardrails status      show event totals
  openbrain-guardrails dashboard   open the web UI (token in the URL)
  openbrain-guardrails connect     print the Claude Code capture-hook wiring
  openbrain-guardrails version     show version
"""

from __future__ import annotations

import json
import sys
import urllib.request

from . import __version__


def _url(path: str = "") -> str:
    from .config import load_config
    return f"http://127.0.0.1:{load_config().port}{path}"


def _token() -> str | None:
    from .auth import read_token
    from .config import load_config
    return read_token(load_config().db_path.parent)


def _get(path: str) -> dict:
    from .auth import HEADER
    tok = _token()
    headers = {HEADER: tok} if tok else {}
    req = urllib.request.Request(_url(path), headers=headers)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _run_daemon() -> None:
    import logging
    import uvicorn
    from .config import load_config
    from .server import create_app

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    # Loopback only, and access_log=False so the ?token= query param is never
    # written to a logfile in cleartext.
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=cfg.port,
                log_level="info", access_log=False)


def _connect() -> None:
    """Print the exact Claude Code hook wiring (we never silently edit a user's
    settings.json — they paste it). PostToolUse fires after a tool runs, which is
    where a complete approval event (action + result) is known."""
    snippet = {
        "hooks": {
            "PostToolUse": [{
                "matcher": "*",
                "hooks": [{"type": "command",
                           "command": "python -m guardrails.hooks.claude_code_capture"}],
            }],
        }
    }
    print("\n🛡  OpenBrain Guard Rails — Claude Code capture hook\n")
    print("Add this to ~/.claude/settings.json (or a project .claude/settings.json):\n")
    print(json.dumps(snippet, indent=2))
    print("\nThe hook reads the daemon token from $GUARDRAILS_TOKEN or "
          "~/.openbrain-guardrails/token, so no token needs to be pasted.")
    print(f"Make sure the daemon is running:  openbrain-guardrails run")
    print(f"Dashboard: {_url('/')}\n")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"

    if cmd in ("run", "daemon"):
        _run_daemon()
    elif cmd == "status":
        try:
            d = _get("/health")
            print(f"🛡  Guard Rails up on :{d['port']} — "
                  f"{d['total_events']} approval events logged")
        except Exception:
            print("Guard Rails unreachable — daemon not responding", file=sys.stderr)
            sys.exit(1)
    elif cmd == "connect":
        _connect()
    elif cmd in ("dashboard", "ui", "open"):
        import webbrowser
        tok = _token()
        webbrowser.open(_url("/") + (f"?token={tok}" if tok else ""))
    elif cmd in ("-v", "--version", "version"):
        print(f"openbrain-guardrails {__version__}")
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        print(f"unknown command: {cmd}\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
