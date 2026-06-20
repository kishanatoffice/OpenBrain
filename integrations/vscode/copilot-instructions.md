# OpenBrain permanent memory

You have permanent local memory via the `openbrain` MCP server (tools:
`recall`, `remember`, `forget`). It persists across every session and every
AI tool. Use it automatically.

- **Before starting a task or answering a non-trivial question**, call `recall`
  first with a query derived from the user's request — pull past decisions,
  preferences, project state, people, and dates before writing code or planning.
- **After a task, or when a durable fact/decision/preference/project update is
  established**, call `remember` with a self-contained summary (include dates).

A SessionStart hook (`.github/hooks/`) loads your standing context once when the
session opens. Copilot's per-prompt hook cannot inject context, so for
task-specific recall you must call the `recall` tool yourself, per this rule.

**Opt-out:** if the user's message contains `--no-memory` (or `#nomem`), skip
`recall` for that turn and answer with the plain model; ignore the flag text.
