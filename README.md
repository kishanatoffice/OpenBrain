# OpenBrain — your local memory, attachable to any AI

Every AI tool has the same weakness: a finite context window that forgets you
when the session ends. OpenBrain is a small daemon that runs on your computer
and acts as **permanent memory for all of them at once** — Claude Code, Claude
Desktop/Cowork, Gemini CLI, Cursor, Codex, anything. Set it up once; every
tool you connect shares one brain.

The bridge is **MCP (Model Context Protocol)** — the open JSON-RPC standard
those tools already speak. OpenBrain implements it from scratch (no SDK, no
extra dependencies) and exposes exactly three tools:

| Tool | What the AI does with it |
| --- | --- |
| `recall(query, max_tokens)` | get the most relevant memories, packed under a token budget |
| `remember(content)` | save something permanently, across all tools |
| `forget(memory_id)` | delete one memory |

Everything stays on `127.0.0.1`. Your memories live in SQLite plus a folder
of Markdown files you can open in Obsidian.

**Privacy & safety.** Secrets and PII (API keys, tokens, private keys, emails)
are **redacted before anything is stored** — on every write, the auto-capture
judge's output, and the dashboard. Deleting a memory **hard-purges** it (row +
search index + embeddings + the markdown file). Data is stored unencrypted at
rest, so rely on full-disk encryption (FileVault/BitLocker). The daemon never
phones home. Licensed Apache-2.0.

**Local API token.** The daemon binds to `127.0.0.1`, but loopback is shared by
every process and user on the machine. So every data/management endpoint
requires a random token (stored `0600` at `~/.myagent/token`, created on first
run). `connect` bakes it into each tool's MCP URL automatically and `openbrain
dashboard` opens the UI with it — so this is invisible in normal use. It blocks
*other users* and *malicious web pages* (CSRF/DNS-rebinding); it does **not**
claim to stop code running as you (which can read the token or DB directly —
only full-disk encryption addresses that). Liveness (`/health`) and the empty
dashboard shell (`/`) stay open; everything else is gated.

## Install & run

One command does everything — installs deps, starts the background service,
**auto-detects and wires every AI tool on your machine**, and offers to set up
your persona:

```bash
./install.sh
```

`python -m myagent connect` is the engine: it finds installed tools and writes
the OpenBrain MCP server into each, in that tool's own format and OS-correct
location — **Claude Code, Claude Desktop, Cursor, VS Code (Copilot), Gemini CLI,
Windsurf, Codex, Zed**. It merges (never clobbers other servers), backs up
originals, and is idempotent. Preview first with `--dry-run`; add project-level
hooks/rules with `--project <dir>`.

Then run **`openbrain dashboard`** (or `python -m myagent dashboard`) to open the
UI — it launches `http://127.0.0.1:3111` with your token attached, which the page
captures and then strips from the address bar. A clean dashboard with a
**Controls** panel (toggle, no config files, no restart):

- **Memory** — master on/off; when off, the controls below dim and tools use only their own context.
- **Ask before using** — when memory might not fit, the assistant asks first.
- **Auto-remember** — silently save important facts from your chats (secrets redacted first).
- **Sensitivity** — Broad / Balanced / Precise recall.

Plus live stats (memories / core / auto / used / 🔒 secrets protected), one-click
**Connect** for your tools, and a **memory workbench** built to scale to 100k+:
a smart-folder rail (Views · Sources · Tags with counts), keyword search, cursor
**pagination** ("Load more"), **source badges** showing which tool each memory
came from, and per-memory **⭐ favorite / 📦 archive** (archived memories are kept
but excluded from recall), pin, edit, and delete.

<details><summary>Manual install</summary>

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m myagent
```

Run at login (macOS): `scripts/install-launchd.sh` (remove with
`scripts/uninstall-launchd.sh`).
</details>

## Set up your persona (so a new chat already knows you)

```bash
.venv/bin/python -m myagent init
```

A handful of questions — role, how you like answers, your stack, what you're
working on — stored as **core** memories. Core memories are the *always-on
persona layer*: recall injects them into **every** request regardless of the
query, so the first message in any new chat already understands who you are and
how you think. Pin any existing memory to this layer from the web UI, or by
adding the `core` tag.

Requirements: Python 3.11+, [Ollama](https://ollama.com) with
`ollama pull nomic-embed-text` (embeddings) and any chat model for background
summaries. **If Ollama is down, nothing breaks** — writes and keyword recall
keep working; embeddings and summaries catch up automatically later.

## Selective auto-capture

Set `proxy_autocapture = true` (proxy path) and, after each turn, a local Ollama
judge decides whether your message held a **durable fact** (a decision,
preference, deadline, identity, project state). If so, it silently stores a
one-line summary tagged `auto` — reviewable and deletable in the UI. Coding
questions, tasks, and chit-chat are ignored, so the brain stays clean. Runs in
the background (no added latency) and fails closed (no judge → nothing stored).
Off by default.

## Ask-before-using (preflight gate)

Set `proxy_preflight = true` (proxy path) and, when memory's relevance to your
prompt is a **judgment call**, OpenBrain answers first with a tiny menu — *use
my memory* vs *skip to the plain LLM* — then proceeds with your pick. Clearly
relevant memory still injects silently; clearly irrelevant prompts forward
untouched. It only interrupts on the borderline, so it never becomes nagware.
The menu's opening line is **drafted on the fly** by your local model — it names
the specific saved context that might apply and asks a brief clarifying question
— while the numbered choices stay fixed for reliable parsing. If the model is
slow or offline it falls back to a static menu, so the gate always renders.
Off by default.

## Turn memory off (global switch)

Want a clean slate — a project or session that uses only the tool's own context,
with your shared brain staying out of it? Flip the global switch:

```bash
openbrain off     # memory off everywhere — every tool falls back to its LLM
openbrain on      # memory back on
```

Or click **Turn memory OFF / ON** in the dashboard header. The state persists
across restarts; while off, recall, injection, the gate, and auto-capture all
stand down (explicit saves still work). For one-off skips, use `--no-memory`
instead.

## Skip memory for one message

Add **`--no-memory`** (or `#nomem`) anywhere in a prompt to bypass OpenBrain for
that turn and answer with the plain LLM — no persona, no recall. The flag is
stripped before the model sees it. Honored by the Claude Code hook and the L3
proxy directly, and by the other editors' rules. Leave it out and memory is on
as usual.

## Connectors (the platform)

OpenBrain is a host for **connectors** — pluggable capabilities, each with its
own on/off switch in the dashboard **Connectors** card. **Memory** is the
built-in connector (always on; the global switch above is its mute). Add-on
connectors are off by default and toggle independently — switch one off and its
tools disappear from every connected AI tool (and are refused even if a client
cached the schema).

### Document & Image OCR (`ocr`)

Converts a **local** PDF, Office doc, or text-bearing screenshot to Markdown so
an AI tool reads it as cheap text instead of a costly binary/image attachment.
This shrinks the tokens you *send* the model (not the tokens it *generates*).

```bash
pip install 'openbrain-memory[ocr]'   # optional, heavy deps (markitdown[all])
```

Then enable **Document & Image OCR** in the dashboard and call the `digest` tool
with a path inside your configured ingest folder:

```
digest(path="~/.myagent/ingest/report.pdf", save_as_memory=false)
```

It is off by default and built to fail safe:

- **Path confinement** — only files inside `OCR_INGEST_DIRS` are read; the file is
  opened once with `O_NOFOLLOW` (symlinks are refused outright, closing the
  swap-after-check race), `..` escapes are resolved away, and there is **no URL
  fetching**.
- **Bounded work** — conversion runs in a worker thread with a wall-clock timeout
  so a pathological file can't freeze the daemon, and images are pixel-capped
  against decompression bombs.
- **Untrusted by default** — ingested text is fenced as data (never instructions)
  and, if saved, tagged `untrusted` so prompt-injection in a document can't be
  laundered into trusted memory.
- **Size + type caps** and a **zip-bomb guard** (Office/EPub files are zips —
  the *expanded* size, entry count, and compression ratio are checked before
  parsing).
- **Redaction on ingest** — secrets are stripped from the Markdown before it is
  returned or saved.
- **Honest accounting** — a token-savings estimate is reported **only for
  images** (where vision tokens are the real alternative); for documents the
  cost is reported with no savings claim. All figures are labeled estimates.

> Note: extracting *text from images* needs markitdown configured with an OCR
> backend; without one it reads only metadata. For purely visual images
> (diagrams, mockups, photos) prefer a vision model — `digest` will say so.

## Scope a turn to a namespace

Tag memories with `ns:<name>` (e.g. `ns:policy`, `ns:billing`) and a host can
scope a single recall to that bucket — useful when one assistant should never
see another's context (a billing agent shouldn't pull marketing notes).

```bash
# data endpoints require the local token (header or ?token=); $TOK below:
TOK=$(cat ~/.myagent/token)

# only ns:policy memories surface
curl -H "X-OpenBrain-Token: $TOK" 'http://127.0.0.1:3111/context?q=refund+window&ns=policy'

# ns:policy OR the always-on persona
curl -H "X-OpenBrain-Token: $TOK" 'http://127.0.0.1:3111/context?q=refund+window&ns=policy,core'

# discoverable list of namespaces and their counts
curl -H "X-OpenBrain-Token: $TOK" 'http://127.0.0.1:3111/namespaces'
```

Namespaces are just tags with an `ns:` prefix — no migration, no separate
store. Filtering to a non-`core` namespace skips the persona layer by default;
add `core` to the list to keep it.

## Connect your AI tools (one-time)

`openbrain connect` wires every detected tool **and bakes in your local token**
automatically — prefer it. Wiring by hand? HTTP transports need
`?token=<your token>` (from `~/.myagent/token`) on the URL; the stdio command
transport doesn't (it reads the DB directly as you).

**Claude Code** (covers every project, user-wide):

```bash
claude mcp add -s user --transport http openbrain "http://127.0.0.1:3111/mcp?client=claude-code&token=$(cat ~/.myagent/token)"
```

**Claude Desktop / Claude Cowork** — in `claude_desktop_config.json` under
`mcpServers` (stdio command; no token needed — use your absolute venv path):

```json
"openbrain": { "command": "/path/to/open_brain/.venv/bin/python", "args": ["-m", "myagent.mcp"] }
```

**Gemini CLI** — in `~/.gemini/settings.json` under `mcpServers` (HTTP — include
your token):

```json
"openbrain": { "httpUrl": "http://127.0.0.1:3111/mcp?client=gemini&token=YOUR_TOKEN" }
```

**Cursor / Windsurf / Antigravity / VS Code** — any MCP client config accepts
either the stdio command or the HTTP URL above. **OpenAI**: Codex CLI and the
Agents SDK support MCP servers the same way (ChatGPT's own connectors require
a public remote URL, so a tunnel would be needed there).

**No MCP at all?** Plain HTTP returns a ready-to-paste context block:

```bash
curl -H "X-OpenBrain-Token: $(cat ~/.myagent/token)" \
  'http://127.0.0.1:3111/context?q=database+decisions&max_tokens=2000'
```

### Make your agents actually use it

Tools call memory when prompted to. Add one line to your agent's instructions
(e.g. `CLAUDE.md`):

> You have permanent local memory via the `openbrain` MCP tools. At the start
> of a task, call `recall` with a short description of the task, then `expand`
> the ids you need from the index. Save durable facts and decisions with
> `remember`.

## How recall fits unbounded memory into a bounded window

A context window is a token budget **B**; selection is done in four steps
(see `myagent/search.py`):

1. **Fuse** — BM25 (SQLite FTS5) and cosine similarity over 768-d embeddings,
   combined with Reciprocal Rank Fusion: `RRF(d) = Σ 1/(60 + rank)`.
2. **Diversify** — Maximal Marginal Relevance,
   `argmax λ·rel(d) − (1−λ)·max cos(d, selected)`, λ=0.7, so the budget isn't
   spent on near-duplicates.
3. **Decay (optional)** — forgetting curve `w = 2^(−age/half_life)`; off by
   default (`RECALL_HALF_LIFE_DAYS = 0`) because old facts aren't less true.
4. **Pack** — greedy knapsack under B: full text if it fits, summary if not.

Writes are instant: an excerpt summary + embedding (~100 ms) so the calling
agent never waits; the daemon upgrades summaries with Ollama in the background
and rewrites the vault file.

### Layered recall (progressive disclosure)

The `recall` tool defaults to **`mode="index"`**: it returns a compact candidate
list (`#id · date · score · one-line summary`), and the agent then calls
**`expand(ids=[…])`** for only the memories it actually needs. `mode="full"`
keeps the old behavior (inline every body), and the always-on hook/proxy
injection path stays full — it is one-shot, with no agent to make a second call.

This is a **scale** optimization, measured not assumed:

- Small brain / short memories: roughly break-even (an index line ≈ the body).
- Mature brain (e.g. 40 long memories, 4k budget): the index lists **all 30**
  ranked candidates for ~1.3k tokens; expanding 2 adds ~0.5k — **~50% fewer
  tokens than inlining, and the model sees every relevant candidate** instead of
  only the ~15 that fit the budget. Better recall *and* cheaper.

The persona `core` block is always inlined in full regardless of mode.

## Configuration

Resolution order: **environment variables → `config.toml` → defaults**.

| Key | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `OLLAMA_MODEL` | `llama3` | model for background summaries |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `MEMORY_PORT` | `3111` | daemon port |
| `VAULT_PATH` | `~/.myagent/vault` | Markdown mirror (Obsidian vault) |
| `DB_PATH` | `~/.myagent/memories.db` | SQLite file |
| `RECALL_HALF_LIFE_DAYS` | `0` (off) | forgetting-curve half-life |
| `OCR_INGEST_DIRS` | `~/.myagent/ingest` | colon-separated folders the `ocr` connector may read |
| `OCR_MAX_FILE_MB` | `25` | source-file size cap for `digest` |
| `OCR_MAX_EXPANDED_MB` | `100` | zip-bomb guard: max decompressed size of an archive |
| `OCR_CONVERT_TIMEOUT_S` | `30` | wall-clock cap on one conversion (runs off the event loop) |
| `OCR_MAX_IMAGE_MP` | `40` | decoded-pixel cap (megapixels) — refuses image decompression bombs |

## REST API (besides MCP)

- `POST /memories` `{"content": "..."}` — store (201, instant)
- `GET /memories?q=&mode=hybrid|keyword|semantic&limit=&format=json|text`
- `GET /memories/{id}` / `DELETE /memories/{id}`
- `GET /context?q=&max_tokens=` — packed plain-text context block
- `GET /health` — status, counts, connection info

## Layout

```
myagent/
├── config.py          # TOML + env config loader
├── db.py              # SQLite: WAL, versioned migrations, FTS5, embeddings
├── search.py          # the math: RRF, cosine, MMR, decay, token packing
├── memory_service.py  # instant writes, budgeted recall, background enrichment
├── mcp.py             # MCP server (stdio entry: python -m myagent.mcp)
├── connectors.py      # connector registry (the platform); memory connector
├── ocr.py             # Document & Image OCR connector (markitdown, opt-in)
├── tokens.py          # token-savings estimator + session counter
├── redact.py          # secret redaction (applied on capture and on ingest)
├── vault.py           # Markdown mirror writer
├── ollama.py          # async client for /api/generate and /api/embed
├── server.py          # FastAPI: REST + /mcp HTTP transport + UI
├── ui/index.html      # traffic-light UI (no build step)
└── __main__.py        # python -m myagent (the daemon)
```

Core dependencies: `fastapi`, `uvicorn`, `httpx`. That's all — the MCP protocol,
vector math, and packing are implemented in-repo. The OCR connector adds
`markitdown[all]` only when installed via the optional `[ocr]` extra.

## Phase history

- **Phase 1** — daemon, config, SQLite + Markdown mirror, summarize-on-write
- **Phase 2** — FTS5 keyword search, embeddings, hybrid ranking (RRF),
  migrations, launchd agent
- **Phase 3** — (superseded) built-in chat agent + automations; removed in the
  Phase 4 pivot — the connected AI tools *are* the agents now
- **Phase 4** — the pivot: universal memory. MCP server (stdio + HTTP),
  budgeted recall (MMR + knapsack packing + optional decay), instant writes
  with background enrichment, traffic-light UI
- **Phase 5** — the connector platform: a registry of pluggable, individually
  switchable capabilities (Memory is the built-in). First add-on is the
  Document & Image OCR connector (`markitdown`, opt-in) with a hardened ingest
  envelope and honest token-savings accounting
- **Phase 6** — layered recall (progressive disclosure): `recall` returns a
  compact index by default, `expand` fetches full bodies by id — fewer tokens
  and wider candidate coverage as the brain grows
