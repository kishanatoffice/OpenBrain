# OpenBrain permanent memory

You have permanent local memory via the `openbrain` MCP server (tools:
`recall`, `remember`, `forget`), shared across every session and every tool.

- **Before starting any task**, call `recall` first with a short query derived
  from the request to pull past decisions, preferences, project state, and
  dates — before writing code or proposing a plan.
- **After a task, or when a durable fact/decision/preference is established**,
  call `remember` with a self-contained summary (include dates).

Junie does not inject context via a pre-prompt hook, so recall is enforced by
this guideline. Honor it on every task.

**Opt-out:** if the user's message contains `--no-memory` (or `#nomem`), skip
`recall` for that turn and answer with the plain model; ignore the flag text.
