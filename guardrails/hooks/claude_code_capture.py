"""Claude Code capture hook → OpenBrain Guard Rails.

Wire it as a PostToolUse hook (`openbrain-guardrails connect`). Claude Code runs
it with a JSON event on stdin after a tool executes; we translate that into a
structured approval event and POST it to the local daemon.

Honesty notes (these shape V1 and are recorded in each event's metadata):
  * Claude Code hooks fire for *every* tool use, not only when a permission
    dialog was shown, and the verbatim dialog text isn't passed to hooks. So we
    synthesise a faithful prompt ("Allow <tool>? …") and set
    metadata.prompt_source = "synthesized".
  * We record selected_option = "allow" because the tool ran — a denied tool
    never reaches PostToolUse. Capturing denials and the auto-approved-vs-prompted
    distinction needs the Notification hook + Pre/Post correlation (V1.1).

The hook ALWAYS exits 0 and never prints to stdout, so it can never block or
interfere with the agent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

_DEFAULT_DATA_DIR = "~/.openbrain-guardrails"


def _summarize_input(tool_name: str, tool_input: dict) -> str:
    """A one-line description of the agent's intended action, per tool shape."""
    if not isinstance(tool_input, dict):
        return tool_name
    for key in ("command", "file_path", "path", "url", "pattern", "query"):
        if tool_input.get(key):
            return f"{tool_name}: {tool_input[key]}"
    if tool_input:
        return f"{tool_name}: {json.dumps(tool_input)[:300]}"
    return tool_name


def _result_from_response(tool_response) -> tuple[str, str]:
    """(result, result_detail) from a PostToolUse tool_response. Best-effort:
    look for explicit error signals, else assume success since the tool ran."""
    detail = ""
    if isinstance(tool_response, dict):
        detail = json.dumps(tool_response)[:1000]
        err = (tool_response.get("error") or tool_response.get("stderr")
               or tool_response.get("is_error") or tool_response.get("isError"))
        if err:
            return "failure", detail
        return "success", detail
    if isinstance(tool_response, str):
        detail = tool_response[:1000]
        low = tool_response.lower()
        if "error" in low or "traceback" in low or "failed" in low:
            return "failure", detail
        return "success", detail
    return "success", detail


def build_event(payload: dict, *, repository: str | None = None,
                branch: str | None = None,
                user_request: str | None = None) -> dict:
    """Pure translation of a Claude Code hook payload into an approval event.
    Side-effect-free so it can be unit-tested without a daemon or filesystem."""
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    event_name = payload.get("hook_event_name") or ""
    is_post = event_name == "PostToolUse"

    action = _summarize_input(tool_name, tool_input)
    event: dict = {
        "ide": "claude-code",
        "agent": "claude-code",
        "session_id": payload.get("session_id"),
        "repository": repository,
        "branch": branch,
        "user_request": user_request,
        "agent_action": action,
        "prompt_text": f"Allow this action? {action}",
        "options": ["allow", "deny"],
        "tool_name": tool_name,
        "metadata": {
            "hook_event_name": event_name,
            "prompt_source": "synthesized",
            "cwd": payload.get("cwd"),
        },
    }
    if is_post:
        # The tool ran, so it was permitted; capture the outcome.
        event["selected_option"] = "allow"
        result, detail = _result_from_response(payload.get("tool_response"))
        event["result"] = result
        event["result_detail"] = detail
    return event


def last_user_request(transcript_path: str | None) -> str | None:
    """The most recent genuine user message from a Claude Code transcript JSONL
    (skipping tool-result turns). None if unavailable. Never raises."""
    if not transcript_path:
        return None
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user":
            continue
        message = entry.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:20000]
        if isinstance(content, list):
            # Skip turns that are purely tool results; keep real text parts.
            texts = [p.get("text", "") for p in content
                     if isinstance(p, dict) and p.get("type") == "text"]
            joined = " ".join(t for t in texts if t).strip()
            if joined:
                return joined[:20000]
    return None


def git_context(cwd: str | None) -> tuple[str | None, str | None]:
    """(repository, branch) for cwd, best-effort. Never raises."""
    if not cwd or not os.path.isdir(cwd):
        return None, None

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", cwd, *args],
                                 capture_output=True, text=True, timeout=3)
            return out.stdout.strip() or None if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    top = _git("rev-parse", "--show-toplevel")
    repo = os.path.basename(top) if top else None
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    return repo, branch


def _data_dir() -> Path:
    return Path(os.environ.get("GUARDRAILS_DATA_DIR", _DEFAULT_DATA_DIR)).expanduser()


def _token() -> str | None:
    tok = os.environ.get("GUARDRAILS_TOKEN")
    if tok:
        return tok
    try:
        return (_data_dir() / "token").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def post_event(event: dict) -> None:
    """POST the event to the local daemon. Swallows everything — capture is
    best-effort and must never affect the agent."""
    port = os.environ.get("GUARDRAILS_PORT", "3112")
    url = f"http://127.0.0.1:{port}/events"
    body = json.dumps({k: v for k, v in event.items() if v is not None}).encode()
    headers = {"Content-Type": "application/json"}
    tok = _token()
    if tok:
        headers["X-Guardrails-Token"] = tok
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        urllib.request.urlopen(req, timeout=3).read()
    except Exception:
        pass


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        sys.exit(0)
    if not isinstance(payload, dict):
        sys.exit(0)
    try:
        repo, branch = git_context(payload.get("cwd"))
        user_request = last_user_request(payload.get("transcript_path"))
        event = build_event(payload, repository=repo, branch=branch,
                            user_request=user_request)
        post_event(event)
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
