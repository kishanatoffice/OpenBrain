# OpenBrain Guard Rails (V2)

A standalone, local-first service that **captures and stores every approval /
permission prompt** AI agents raise inside your IDEs — Cursor, VS Code,
JetBrains, Claude Code, and friends.

This is **V1: collect & store only.** The point is a clean, structured event log.
Risk detection, policy enforcement, and auto-approval recommendations are
deliberately *not* here yet — they're future versions that will build on this
data.

It runs as its own loopback daemon — separate process, DB, port (`3112`), token,
and dashboard — fully decoupled from the OpenBrain memory daemon. The only shared
code is the memory daemon's hardened secret-redactor, reused so captured prompts
are scrubbed of API keys/PII with the same well-tested patterns.

## What each event records

| Field | Meaning |
|-------|---------|
| `user_request` | the user's original request |
| `agent_action` | the agent's intended action |
| `prompt_text` | the approval prompt, verbatim |
| `options` | all options offered |
| `selected_option` | the user's choice |
| `result` / `result_detail` | the execution outcome |
| `ide`, `agent`, `repository`, `branch`, `session_id`, `tool_name` | provenance |
| `created_at` / `decided_at` / `completed_at`, `status` | lifecycle |

`status` (`pending` → `decided` → `completed`) is derived from the fields, so it
can't drift: log a complete event in one POST, or log a `pending` one and `PATCH`
in the decision and result later.

Example (the canonical case):

```
User:   "Run the application"
Agent:  "Need to restart the server."
Options: Restart · Skip · Cancel
Choice:  Restart
Result:  Success
```

## Run it

```bash
openbrain-guardrails run         # start the daemon on 127.0.0.1:3112
openbrain-guardrails dashboard   # open the web UI (token baked into the URL)
openbrain-guardrails status      # event totals
openbrain-guardrails connect     # print the Claude Code capture-hook wiring
```

## Capture: how events get in

Two ingestion paths, both token-gated:

- **REST** — `POST /events` with the JSON body above (only `prompt_text` is
  required). `PATCH /events/{id}` fills in a decision/result later.
- **MCP** — `POST /mcp` exposes a single `log_approval` tool for MCP-native
  agents.

### Claude Code (reference capturer)

`openbrain-guardrails connect` prints a `PostToolUse` hook for
`~/.claude/settings.json`. After each tool runs, the hook
(`guardrails/hooks/claude_code_capture.py`) records a complete approval event —
deriving the user request from the transcript, the action from the tool input,
the result from the tool response, and repo/branch from git. It reads the daemon
token from `$GUARDRAILS_TOKEN` or `~/.openbrain-guardrails/token`, and always
exits 0 so it can never block the agent.

### Other IDEs

Cursor / VS Code / JetBrains don't expose their approval dialogs to third-party
code today, so they POST to the same `/events` API via adapters as those mature.
The API is the stable contract; capturers are pluggable.

## Honesty notes (V1 limits)

- Claude Code hooks fire for *every* tool use, not only when a permission dialog
  was actually shown, and the verbatim dialog text isn't passed to hooks — so the
  capturer **synthesises** the prompt (`metadata.prompt_source = "synthesized"`)
  and records `selected_option = "allow"` because the tool ran.
- **Denied** actions and the auto-approved-vs-prompted distinction aren't captured
  yet (no hook fires on denial); that needs the Notification hook + Pre/Post
  correlation, planned for V1.1.
- Secrets in captured text are redacted on ingest; events flagged
  `metadata._redacted = true` had something scrubbed.

## Config

Env (or `guardrails.toml`): `GUARDRAILS_PORT` (3112), `GUARDRAILS_DB_PATH`
(`~/.openbrain-guardrails/guardrails.db`), `GUARDRAILS_MAX_FIELD_CHARS` (20000).
