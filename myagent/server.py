"""REST + MCP daemon — your local brain, attachable to any AI tool.

  GET    /              -> web UI (status + connection snippets + memories)
  POST   /mcp           -> MCP streamable-HTTP transport (stateless)
  GET    /context       -> packed context block (plain text, for non-MCP tools)
  POST   /memories      -> store (instant; AI summary enriched in background)
  GET    /memories      -> recent / keyword / semantic / hybrid search
  GET    /memories/{id} -> full record
  DELETE /memories/{id} -> remove from DB, index, and vault
  GET    /health        -> daemon + config + Ollama status
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .auth import extract_token, load_or_create_token, token_matches
from .config import Config, load_config
from .connectors import REGISTRY, is_enabled
from .db import MemoryStore
from .mcp import handle_message
from .memory_service import (
    Deps,
    create_memory,
    enrich_pending,
    recall_context,
)
from .ollama import OllamaClient, OllamaError
from .proxy import handle_chat_completions
from .redact import redact, redaction_count
from .tokens import tokens_saved
from .vault import Vault
from .vault_sync import sync_loop

log = logging.getLogger("myagent")

MAX_CONTENT_CHARS = 100_000
ENRICH_TICK_SECONDS = 20
UI_FILE = Path(__file__).parent / "ui" / "index.html"


# Runtime settings the user controls live from the dashboard (no restart, no
# config.toml edit). Persisted so they survive restarts. Defaults come from
# config.toml; the user's UI toggles override them.
_SETTING_KEYS = ("paused", "preflight", "autocapture", "min_relevance")


def _settings_path(cfg: Config) -> Path:
    return cfg.db_path.parent / "settings.json"


def _default_connector_state() -> dict[str, bool]:
    """Per-connector enable flags, seeded from each connector's default."""
    return {c.key: c.default_enabled for c in REGISTRY.values()}


def _enabled_connectors(settings: dict) -> frozenset[str]:
    """The active connector set, derived from live settings. Built-in
    (non-toggleable) connectors are always active regardless of stored flags,
    so the core can never be accidentally disabled via a hand-edited file."""
    state = settings.get("connectors") or {}
    return frozenset(
        c.key for c in REGISTRY.values()
        if not c.toggleable or state.get(c.key, c.default_enabled)
    )


def _load_settings(cfg: Config) -> dict:
    import json as _json
    settings = {
        "paused": False,
        "preflight": cfg.proxy_preflight,
        "autocapture": cfg.proxy_autocapture,
        "min_relevance": cfg.proxy_min_relevance,
        "connectors": _default_connector_state(),
    }
    try:
        saved = _json.loads(_settings_path(cfg).read_text())
        settings.update({k: v for k, v in saved.items() if k in _SETTING_KEYS})
        # Connectors are persisted as a sub-map; merge over defaults so a
        # newly-registered connector picks up its default rather than vanishing.
        if isinstance(saved.get("connectors"), dict):
            for k, v in saved["connectors"].items():
                if k in settings["connectors"]:
                    settings["connectors"][k] = bool(v)
    except Exception:
        # Migrate the old single-purpose switch.json (paused only), if present.
        try:
            settings["paused"] = bool(
                _json.loads((cfg.db_path.parent / "switch.json").read_text())["paused"])
        except Exception:
            pass
    return settings


def _save_settings(cfg: Config, settings: dict) -> None:
    import json as _json
    p = _settings_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: settings[k] for k in _SETTING_KEYS}
    payload["connectors"] = settings.get("connectors", _default_connector_state())
    p.write_text(_json.dumps(payload), encoding="utf-8")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


class MemoryIn(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    tags: list[str] | None = Field(default=None, max_length=20)


class MemoryPatch(BaseModel):
    content: str | None = Field(default=None, min_length=1,
                                max_length=MAX_CONTENT_CHARS)
    tags: list[str] | None = Field(default=None, max_length=20)
    favorite: bool | None = None
    archived: bool | None = None


class ImportIn(BaseModel):
    memories: list[MemoryIn] = Field(max_length=5000)


class SettingsPatch(BaseModel):
    paused: bool | None = None
    preflight: bool | None = None
    autocapture: bool | None = None
    min_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    # Partial map of connector_key -> enabled. Unknown keys and attempts to
    # toggle a built-in connector are ignored (validated against the registry).
    connectors: dict[str, bool] | None = None


async def _enrichment_loop(deps: Deps) -> None:
    """Continuously upgrade excerpt summaries / missing embeddings."""
    log.info("Enrichment loop started (tick %ss)", ENRICH_TICK_SECONDS)
    while True:
        try:
            while await enrich_pending(deps):
                pass  # keep going while there is work and Ollama answers
        except Exception:
            log.exception("Enrichment tick failed")
        await asyncio.sleep(ENRICH_TICK_SECONDS)


async def _supervised(make_coro, name: str, health: dict) -> None:
    """Run a background loop; if it crashes, record it in the health map and
    restart after a backoff so a single escape can't silently stop the daemon's
    enrichment/sync forever."""
    while True:
        try:
            await make_coro()
            return  # a clean return means the loop intentionally stopped
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("%s loop crashed; restarting in 5s", name)
            health[name] = f"crashed: {exc}".replace("\n", " ")[:200]
            await asyncio.sleep(5)


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = cfg
        # Lightweight in-process visibility: how often memory was actually
        # injected this session (recall blocks + proxy injections).
        app.state.stats = {"injections": 0, "last_query": None}
        # Degraded-state tracking for the unsupervised background loops.
        app.state.health = {"enrichment": None, "sync": None}
        deps = Deps(
            store=MemoryStore(cfg.db_path),
            vault=Vault(cfg.vault_path),
            ollama=OllamaClient(cfg.ollama_url, cfg.ollama_model,
                                cfg.ollama_embed_model),
            half_life_days=cfg.recall_half_life_days,
            min_similarity=cfg.recall_min_similarity,
            ocr_ingest_dirs=cfg.ocr_ingest_dirs,
            ocr_max_bytes=cfg.ocr_max_bytes,
            ocr_max_expanded_bytes=cfg.ocr_max_expanded_bytes,
            ocr_convert_timeout_s=cfg.ocr_convert_timeout_s,
            ocr_max_image_pixels=cfg.ocr_max_image_pixels,
        )
        # Local API token: gates every data/management endpoint so another local
        # user or a malicious web page can't reach the brain. Stored 0600 next to
        # the DB; created on first run.
        app.state.token = load_or_create_token(cfg.db_path.parent)
        app.state.settings = _load_settings(cfg)
        deps.paused = app.state.settings["paused"]
        deps.enabled_connectors = _enabled_connectors(app.state.settings)
        app.state.deps = deps
        log.info(
            "openbrain up — db=%s vault=%s ollama=%s model=%s embed=%s",
            cfg.db_path, cfg.vault_path, cfg.ollama_url,
            cfg.ollama_model, cfg.ollama_embed_model,
        )
        background = [
            asyncio.create_task(_supervised(lambda: _enrichment_loop(deps),
                                            "enrichment", app.state.health)),
            asyncio.create_task(_supervised(lambda: sync_loop(deps),
                                            "sync", app.state.health)),
        ]
        try:
            yield
        finally:
            for task in background:
                task.cancel()
            await deps.ollama.aclose()

    app = FastAPI(title="openbrain memory daemon", version=__version__,
                  lifespan=lifespan)

    # Open endpoints (no token): liveness and the dashboard shell. The shell
    # carries no data — it bootstraps the token from its `?token=` URL and sends
    # it on every API call, so every data/management route below stays gated.
    _OPEN_PATHS = {"/health", "/"}

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        """Gate every endpoint on the local API token, except the open shell +
        liveness. CORS preflights pass (they carry no token by spec)."""
        if request.method == "OPTIONS" or request.url.path in _OPEN_PATHS:
            return await call_next(request)
        token = getattr(request.app.state, "token", None)
        provided = extract_token(request.headers, request.query_params)
        if not token_matches(provided, token):
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid OpenBrain token — open "
                                   "the dashboard via `openbrain dashboard`"},
            )
        return await call_next(request)

    # ---- UI + health ---------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def ui_index():
        return FileResponse(UI_FILE, media_type="text/html")

    @app.get("/health")
    async def health(request: Request):
        c: Config = request.app.state.config
        deps: Deps = request.app.state.deps
        return {
            "status": "ok",
            "version": app.version,
            "ollama_url": c.ollama_url,
            "ollama_model": c.ollama_model,
            "ollama_embed_model": c.ollama_embed_model,
            "ollama_reachable": await deps.ollama.is_reachable(),
            "vault_path": str(c.vault_path),
            "db_path": str(c.db_path),
            "memory_count": deps.store.count(),
            "mcp_http_url": f"http://127.0.0.1:{c.memory_port}/mcp",
            "mcp_stdio_command": [sys.executable, "-m", "myagent.mcp"],
            "injections": request.app.state.stats["injections"],
            # last_query is intentionally NOT exposed here: /health is the one
            # unauthenticated endpoint, and recent query text — even redacted —
            # is private. It stays on the token-gated /stats.
            "paused": request.app.state.deps.paused,
            "degraded": any(request.app.state.health.values()),
            "background": request.app.state.health,
        }

    @app.get("/stats")
    async def stats(request: Request):
        deps: Deps = request.app.state.deps
        s = request.app.state.stats
        cfg_settings = request.app.state.settings
        return {
            "total": deps.store.count(),
            "core": deps.store.count_by_tag("core"),
            "auto": deps.store.count_by_tag("auto"),
            "injections": s["injections"],
            "last_query": s["last_query"],
            "redactions": redaction_count(),
            "ocr_tokens_saved": tokens_saved(),
            # Live, user-controlled settings — the dashboard renders controls from these.
            "paused": deps.paused,
            "preflight": cfg_settings["preflight"],
            "autocapture": cfg_settings["autocapture"],
            "min_relevance": cfg_settings["min_relevance"],
            # The connector platform: one card per registered capability.
            "connectors": [
                {
                    "key": c.key,
                    "label": c.label,
                    "description": c.description,
                    "enabled": is_enabled(c.key, deps.enabled_connectors),
                    "toggleable": c.toggleable,
                    "tools": sorted(c.tool_names),
                }
                for c in REGISTRY.values()
            ],
            "degraded": any(request.app.state.health.values()),
            "ollama_reachable": await deps.ollama.is_reachable(),
        }

    @app.get("/settings")
    async def get_settings(request: Request):
        return request.app.state.settings

    @app.put("/settings")
    async def put_settings(body: SettingsPatch, request: Request):
        """Update live controls (no restart). Returns the full settings."""
        s = request.app.state.settings
        for k in _SETTING_KEYS:
            v = getattr(body, k)
            if v is not None:
                s[k] = v
        if body.connectors is not None:
            state = s.setdefault("connectors", _default_connector_state())
            for key, enabled in body.connectors.items():
                c = REGISTRY.get(key)
                # Only real, toggleable connectors can be switched; built-ins
                # and unknown keys are silently ignored (no footgun, no error).
                if c is not None and c.toggleable:
                    state[key] = bool(enabled)
        request.app.state.deps.paused = s["paused"]
        request.app.state.deps.enabled_connectors = _enabled_connectors(s)
        _save_settings(request.app.state.config, s)
        log.info("settings updated: %s", s)
        return s

    @app.post("/connect")
    async def connect_endpoint(request: Request):
        """One-click wiring of every detected AI tool (the dashboard button)."""
        from .connect import connect_tools
        c: Config = request.app.state.config
        return connect_tools(f"http://127.0.0.1:{c.memory_port}/mcp",
                             token=request.app.state.token)

    @app.post("/pause")
    @app.post("/resume")
    async def set_switch(request: Request):
        """Global memory ON/OFF (also exposed as a setting). Kept for the CLI."""
        paused = request.url.path.endswith("/pause")
        request.app.state.settings["paused"] = paused
        request.app.state.deps.paused = paused
        _save_settings(request.app.state.config, request.app.state.settings)
        log.info("memory switch -> %s", "OFF (paused)" if paused else "ON")
        return {"paused": paused}

    @app.get("/export")
    async def export_memories(request: Request):
        """Full JSON backup of every memory (re-importable via POST /import)."""
        rows = request.app.state.deps.store.all_for_export()
        return {"version": app.version, "count": len(rows), "memories": rows}

    @app.post("/import")
    async def import_memories(body: ImportIn, request: Request):
        """Bulk-create memories from an export bundle. Dedup applies; explicit
        import is never gated by the write-quality guard."""
        deps: Deps = request.app.state.deps
        imported = duplicates = 0
        for m in body.memories:
            result = await create_memory(deps, m.content, tags=m.tags,
                                         source="import")
            if result.get("duplicate"):
                duplicates += 1
            else:
                imported += 1
        return {"imported": imported, "duplicates": duplicates,
                "total": deps.store.count()}

    # ---- MCP over streamable HTTP (stateless) ----------------------------------

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        # DNS-rebinding guard: browsers attach Origin; CLIs normally don't.
        origin = request.headers.get("origin")
        if origin and (urlparse(origin).hostname or "") not in _LOCAL_HOSTS:
            raise HTTPException(status_code=403, detail="forbidden origin")
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": None,
                         "error": {"code": -32700, "message": "parse error"}},
            )

        # Per-request copy: HTTP is stateless, so writes get a fixed source.
        # `connect` bakes ?client=<tool> into each tool's URL, so we recover the
        # originating tool (cursor/claude-desktop/…) for provenance.
        client = (request.query_params.get("client") or "mcp-http")[:60]
        deps = dataclasses.replace(request.app.state.deps, source=client)
        if isinstance(payload, list):  # batch (older protocol revisions)
            responses = [r for r in
                         [await handle_message(deps, m) for m in payload]
                         if r is not None]
            if not responses:
                return Response(status_code=202)
            return JSONResponse(responses)

        response = await handle_message(deps, payload)
        if response is None:  # notification — accepted, no body
            return Response(status_code=202)
        return JSONResponse(response)

    @app.get("/mcp")
    @app.delete("/mcp")
    async def mcp_not_allowed():
        # Stateless transport: no server-initiated streams, no sessions.
        raise HTTPException(status_code=405, detail="POST JSON-RPC messages here")

    # ---- Level 3: memory-injecting LLM proxy (OpenAI-compatible) ----------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        c: Config = request.app.state.config
        s = request.app.state.settings  # live, user-controlled
        return await handle_chat_completions(
            request.app.state.deps, request,
            upstream_url=c.proxy_upstream_url,
            min_relevance=s["min_relevance"],
            max_tokens=c.proxy_recall_tokens,
            timeout=c.proxy_timeout_seconds,
            preflight=s["preflight"],
            gate_high=c.proxy_gate_high,
            autocapture=s["autocapture"],
            stats=request.app.state.stats,
        )

    # ---- plain-text context for non-MCP consumers -------------------------------

    @app.get("/context")
    async def context(
        request: Request,
        q: str = Query(default="", description="what to look for"),
        max_tokens: int = Query(default=2000, ge=100, le=20000),
        tag: str | None = Query(default=None),
        ns: str | None = Query(default=None, description="namespace scope: one "
                               "name or comma-separated (e.g. 'policy' or "
                               "'policy,core'). Mapped to tags 'ns:<name>'."),
        min_relevance: float | None = Query(default=None, ge=0.0, le=1.0,
                                     description="drop hits below this score; "
                                                 "defaults to the live Sensitivity "
                                                 "setting when omitted"),
    ):
        # Fall back to the user's live Sensitivity setting when the caller
        # doesn't specify (a hook may still override via its own env).
        floor = (min_relevance if min_relevance is not None
                 else request.app.state.settings["min_relevance"])
        # `ns=policy,core` means "scope to ns:policy OR core (the persona)".
        # Bare 'core' is passed through as-is so hosts can ask for persona only.
        tag_filter: str | list[str] | None = tag
        if ns:
            names = [n.strip().lower() for n in ns.split(",") if n.strip()]
            tag_filter = [n if n == "core" else f"ns:{n}" for n in names]
            if tag:  # if caller also passed a raw tag, union with the namespaces
                tag_filter.append(tag)
        block = await recall_context(request.app.state.deps, q, max_tokens,
                                     tag_filter, floor)
        if block and "nothing found" not in block and "no memories match" not in block:
            stats = request.app.state.stats
            stats["injections"] += 1
            stats["last_query"] = redact(q)[0] or "(recent)"
        return PlainTextResponse(block)

    # ---- memories ----------------------------------------------------------------

    @app.post("/memories", status_code=201)
    async def add_memory(body: MemoryIn, request: Request):
        source = request.headers.get("x-source", "rest")[:60]
        return await create_memory(request.app.state.deps, body.content,
                                   tags=body.tags, source=source)

    @app.get("/memories")
    async def get_memories(
        request: Request,
        q: str | None = Query(default=None, description="keyword search"),
        tag: str | None = Query(default=None),
        source: str | None = Query(default=None, description="filter by origin tool"),
        kind: str | None = Query(default=None, pattern="^(core|auto)$"),
        favorite: bool = Query(default=False),
        archived: bool = Query(default=False),
        after: str | None = Query(default=None, description="pagination cursor"),
        limit: int = Query(default=50, ge=1, le=200),
        format: str = Query(default="json", pattern="^(json|text)$"),
    ):
        """Filtered, paginated browse. Scales via keyset pagination (browse) /
        FTS (search) — never loads the whole store. Returns a `next` cursor."""
        store = request.app.state.deps.store
        result = store.browse(q=q, source=source, tag=tag, kind=kind,
                              favorite=favorite, archived=archived,
                              after=after, limit=limit)
        rows = result["rows"]
        if format == "text":
            blocks = [f"[{r['created_at']} | memory #{r['id']}]\n{r['summary']}"
                      for r in rows]
            return PlainTextResponse("\n\n".join(blocks) or "No memories found.")
        return {"count": len(rows), "memories": rows, "next": result["next"]}

    @app.get("/facets")
    async def facets(request: Request):
        """Filter-rail counts: sources, top tags, types, totals."""
        store = request.app.state.deps.store
        f = store.facet_counts()
        f["core"] = store.count_by_tag("core")
        f["auto"] = store.count_by_tag("auto")
        return f

    @app.get("/namespaces")
    async def namespaces(request: Request):
        """Discoverable list of namespaces (tags prefixed `ns:`) with counts.
        A host scoping a turn to a namespace passes the short name to
        /context?ns=<name>."""
        store = request.app.state.deps.store
        rows = [t for t in store.facet_counts()["tags"]
                if t["tag"].startswith("ns:")]
        return {"namespaces": [{"name": t["tag"][3:], "count": t["count"]}
                                for t in rows]}

    @app.patch("/memories/{memory_id}")
    async def edit_memory(memory_id: int, body: MemoryPatch, request: Request):
        """Edit content and/or replace tags. Pinning a memory to the always-on
        persona layer is just adding the 'core' tag here."""
        deps: Deps = request.app.state.deps
        existing = deps.store.get(memory_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="memory not found")

        if body.content is not None and body.content.strip() != existing["content"]:
            # Invalidates summary + embeddings; the enricher regenerates both.
            deps.store.update_content(memory_id, body.content.strip())
            updated = deps.store.get(memory_id)
            md_path = updated.get("md_path")
            # Keep the markdown mirror in step, but never touch user-authored files.
            if md_path and updated.get("source") != "vault":
                deps.vault.write_memory(updated, path=md_path)

        if body.tags is not None:
            deps.store.set_tags(memory_id, body.tags)
        if body.favorite is not None or body.archived is not None:
            deps.store.set_flags(memory_id, favorite=body.favorite,
                                 archived=body.archived)

        return deps.store.get(memory_id)

    @app.get("/memories/{memory_id}")
    async def get_memory(memory_id: int, request: Request):
        memory = request.app.state.deps.store.get(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return memory

    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: int, request: Request):
        deps: Deps = request.app.state.deps
        deleted = deps.store.delete(memory_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="memory not found")

        vault_path: Path = deps.vault.path
        md_file = Path(deleted["md_path"]) if deleted.get("md_path") else None
        if md_file is None:
            # Rows written before md_path tracking: locate by the id in the filename.
            matches = list(vault_path.glob(f"????-??-??-??????-{memory_id:04d}-*.md"))
            md_file = matches[0] if len(matches) == 1 else None

        md_removed = False
        # Refuse to unlink anything that has wandered outside the vault.
        if md_file and md_file.is_relative_to(vault_path) and md_file.is_file():
            md_file.unlink()
            md_removed = True
        return {"deleted": memory_id, "markdown_removed": md_removed}

    @app.exception_handler(OllamaError)
    async def ollama_error_handler(_, exc: OllamaError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    return app


app = create_app()
