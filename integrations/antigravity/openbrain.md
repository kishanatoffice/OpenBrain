---
trigger: always_on
---

# OpenBrain permanent memory

You have permanent local memory via the `openbrain` MCP server (tools:
`recall`, `remember`, `forget`). It persists across every session and every AI
tool. Use it automatically — the user must never have to ask.

- **Before starting any task or answering a new question**, FIRST call `recall`
  with a query derived from the user's request. Pull past decisions,
  preferences, project state, people, and dates. Only after reviewing what
  comes back do you begin the normal build/planning process — so every feature
  is grounded in the user's history and persona, not just the current prompt.
- **After a task, or whenever a durable fact, decision, preference, or project
  update is established**, call `remember` with a self-contained summary
  (include dates and enough context to make sense standalone).

This is an `always_on` rule: it applies to every interaction. Recall first,
build second, remember last.

**Opt-out:** if the user's message contains `--no-memory` (or `#nomem`), do NOT
call `recall` for that turn — answer with the plain model — and ignore the flag
text itself.
