# OpenBrain IDE integrations

Wiring OpenBrain into every major AI coding tool. One local daemon (already
running on `http://127.0.0.1:3111`) is the shared brain; each tool connects to
it over MCP. Where a tool supports it, a **hook** forces memory to be injected
deterministically; where it doesn't, an **always-apply rule** instructs the
agent to call `recall` itself.

## The honest capability matrix

Storing memory is easy. The hard part is making *recall happen* before the
agent acts. Only one tool today can guarantee it per-prompt:

| Tool | MCP (recall/remember/forget) | Deterministic injection | How recall is enforced |
|------|------------------------------|--------------------------|------------------------|
| **Claude Code** | ✅ | ✅ **per-prompt** | `UserPromptSubmit` hook injects `/context` (stdout → context) |
| **VS Code Copilot** | ✅ | ⚠️ **session-start only** | `SessionStart` hook injects via `additionalContext`; per-prompt recall via rule + MCP |
| **Cursor** | ✅ | ❌ (hook is block/allow only) | always-apply rule + MCP tool |
| **JetBrains (Junie)** | ✅ | ❌ (hooks preview, no inject) | guideline + MCP tool |
| **Antigravity** | ✅ | ❌ (no injecting hook) | `always_on` rule + MCP tool |

**Takeaway:** MCP is the universal substrate — `recall`/`remember` work
everywhere. True per-prompt injection is a Claude Code feature today; Copilot
gets session-start injection; the rest are best-effort via rules. The only
fully tool-independent guarantee is the **Level 3 proxy** (see below).

## Level 3 — the memory-injecting proxy (tool-independent, built)

For tools whose hooks can't inject (Cursor, JetBrains, Antigravity), point the
tool's **OpenAI-compatible base URL** at the daemon instead of MCP/rules:

```
http://127.0.0.1:3111/v1
```

The daemon exposes `POST /v1/chat/completions`. On every request it recalls
memory for the latest user turn, folds it into the system message, and forwards
(streaming or not) to the configured upstream (`proxy_upstream_url`). The
client's `Authorization` header is passed straight through — keys are never
stored or logged. It **fails open**: any recall error forwards the request
unchanged. This is the only mechanism that guarantees injection regardless of a
tool's hook support. Configure upstream/floors in `config.toml` (`proxy_*`).

## Install — where each file goes

The daemon must be running (`launchctl kickstart -k gui/$(id -u)/com.myagent.memory`,
or `python -m myagent`). Then per tool:

### Claude Code  ✅ already wired in this repo
- `.claude/settings.json` → `UserPromptSubmit` → `scripts/recall-hook.sh`
- Nothing else to do. Fires per prompt.

### Cursor
- `integrations/cursor/mcp.json`     → `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global)
- `integrations/cursor/openbrain.mdc` → `.cursor/rules/openbrain.mdc`

### VS Code (GitHub Copilot)
- `integrations/vscode/mcp.json`               → `.vscode/mcp.json`
- `integrations/vscode/openbrain-hooks.json`   → `.github/hooks/openbrain.json`
- `integrations/vscode/copilot-instructions.md`→ `.github/copilot-instructions.md`

### JetBrains (Junie / AI Assistant)
- `integrations/jetbrains/mcp.json`      → `.junie/mcp/mcp.json`
- `integrations/jetbrains/guidelines.md` → `.junie/guidelines.md`

### Antigravity
- `integrations/antigravity/mcp_config.json` → `~/.gemini/config/mcp_config.json`
  (merge the `mcpServers` entry if the file already exists)
- `integrations/antigravity/openbrain.md`    → project `.antigravity/rules/openbrain.md`
  (or paste into your existing `.antigravityrules`)

## Opt-out — skip memory for one message

Include **`--no-memory`** (or `#nomem`) in a prompt to bypass OpenBrain for that
single turn and use the plain LLM. The Claude Code hook and the L3 proxy enforce
it directly (and strip the flag before the model sees it); the rule-based tools
(Cursor, JetBrains, Antigravity, Copilot) honor it via their rule files.

## Transport note

All configs use the **HTTP** transport (`http://127.0.0.1:3111/mcp`) so every
tool shares the one launchd-managed daemon — no per-tool process spawning. For
a tool that only speaks stdio, use instead:

```
/Users/kishankumar/Documents/open_brain/.venv/bin/python -m myagent.mcp
```

Run `scripts/install-integrations.sh` from a project root to drop the
project-level files into place automatically (global files are printed as
manual steps).
