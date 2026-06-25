"""Guard Rails daemon — REST + MCP ingestion for AI approval events.

  GET    /             -> dashboard (open shell; bootstraps token from ?token=)
  GET    /health       -> liveness + totals (open)
  POST   /events       -> log an approval event (full or partial)
  GET    /events       -> filtered, paginated list
  GET    /events/{id}  -> full record
  PATCH  /events/{id}  -> fill in decision / result on a pending event
  DELETE /events/{id}  -> remove an event
  GET    /stats        -> facet counts
  GET    /export       -> JSON dump (bounded)
  POST   /mcp          -> MCP streamable-HTTP transport (log_approval tool)

Everything except /health and the dashboard shell is gated by the local token.
Captured text is run through the memory daemon's hardened secret redactor before
storage, so an approval prompt that quotes a command containing an API key
(`deploy --token=…`) never lands in the log verbatim.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

# Sole shared dependency on the memory package: the hardened, well-tested secret
# redactor. Reusing it (rather than reimplementing) keeps the two services'
# privacy boundary identical and avoids security-relevant drift.
from myagent.redact import redact

from . import __version__
from .auth import extract_token, load_or_create_token, token_matches
from .config import Config, load_config
from .db import ApprovalStore, TEXT_FIELDS
from .mcp import handle_mcp
from .models import ApprovalEventIn, ApprovalEventPatch

log = logging.getLogger("guardrails")

UI_FILE = Path(__file__).parent / "ui" / "index.html"

# The text fields that may carry secrets and so get redacted on the way in.
_REDACT_FIELDS = ("user_request", "agent_action", "prompt_text",
                  "result_detail", "selected_option", "result")


def _sanitize(event: dict, max_chars: int) -> dict:
    """Redact secrets from free-text fields and length-cap every text field.
    Records `_redacted: true` in metadata when anything was scrubbed, as a trust
    signal for the dashboard."""
    redacted_any = False
    for field in _REDACT_FIELDS:
        val = event.get(field)
        if isinstance(val, str) and val:
            cleaned, found = redact(val)
            if found:
                redacted_any = True
            event[field] = cleaned
    # Cap every text field after redaction.
    for field in TEXT_FIELDS:
        val = event.get(field)
        if isinstance(val, str) and len(val) > max_chars:
            event[field] = val[:max_chars]
    # Redact secrets in the options too (they can echo a command).
    if isinstance(event.get("options"), list):
        cleaned_opts = []
        for opt in event["options"]:
            if isinstance(opt, str):
                c, found = redact(opt)
                redacted_any = redacted_any or bool(found)
                cleaned_opts.append(c[:max_chars])
            else:
                cleaned_opts.append(opt)
        event["options"] = cleaned_opts
    if redacted_any:
        meta = event.get("metadata") or {}
        if isinstance(meta, dict):
            meta["_redacted"] = True
            event["metadata"] = meta
    return event


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or load_config()
    store = ApprovalStore(cfg.db_path)
    token = load_or_create_token(cfg.db_path.parent)

    app = FastAPI(title="OpenBrain Guard Rails", version=__version__)
    app.state.store = store
    app.state.config = cfg
    app.state.token = token

    _OPEN_PATHS = {"/health", "/"}

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in _OPEN_PATHS:
            return await call_next(request)
        provided = extract_token(request.headers, request.query_params)
        if not token_matches(provided, request.app.state.token):
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid Guard Rails token — open "
                                   "the dashboard via `openbrain-guardrails dashboard`"})
        return await call_next(request)

    # ---- UI + health ---------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def home():
        if UI_FILE.is_file():
            return FileResponse(UI_FILE)
        return JSONResponse({"service": "openbrain-guardrails", "version": __version__})

    @app.get("/health")
    async def health(request: Request):
        s = request.app.state
        return {"status": "ok", "service": "openbrain-guardrails",
                "version": __version__, "port": s.config.port,
                "total_events": s.store.count()}

    # ---- ingestion -----------------------------------------------------------

    def _ingest(raw: dict) -> dict:
        """Validate, sanitize, and store one event. Shared by REST and MCP."""
        data = ApprovalEventIn(**raw).model_dump()
        data = _sanitize(data, cfg.max_field_chars)
        return store.create(data)

    @app.post("/events", status_code=201)
    async def create_event(body: ApprovalEventIn, request: Request):
        return _ingest(body.model_dump())

    @app.get("/events")
    async def list_events(
        request: Request,
        session_id: str | None = Query(default=None),
        ide: str | None = Query(default=None),
        agent: str | None = Query(default=None),
        repository: str | None = Query(default=None),
        status: str | None = Query(default=None),
        selected_option: str | None = Query(default=None),
        after: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
    ):
        result = store.list(session_id=session_id, ide=ide, agent=agent,
                            repository=repository, status=status,
                            selected_option=selected_option, after=after,
                            limit=limit)
        return {"count": len(result["rows"]), "events": result["rows"],
                "next": result["next"]}

    @app.get("/events/{event_id}")
    async def get_event(event_id: int):
        row = store.get(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        return row

    @app.patch("/events/{event_id}")
    async def patch_event(event_id: int, body: ApprovalEventPatch):
        patch = body.model_dump(exclude_unset=True)
        if not patch:
            raise HTTPException(status_code=400, detail="no fields to update")
        patch = _sanitize(patch, cfg.max_field_chars)
        row = store.update(event_id, patch)
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        return row

    @app.delete("/events/{event_id}")
    async def delete_event(event_id: int):
        row = store.delete(event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="event not found")
        return {"deleted": event_id}

    @app.get("/stats")
    async def stats():
        return store.stats()

    @app.get("/export")
    async def export(limit: int = Query(default=10000, ge=1, le=100000)):
        """Bounded JSON dump, newest first — for backup / offline analysis."""
        rows: list[dict] = []
        cursor = None
        while len(rows) < limit:
            page = store.list(after=cursor, limit=min(500, limit - len(rows)))
            rows.extend(page["rows"])
            cursor = page["next"]
            if not cursor:
                break
        return {"version": __version__, "count": len(rows), "events": rows}

    # ---- MCP -----------------------------------------------------------------

    @app.post("/mcp")
    async def mcp(request: Request):
        try:
            message = await request.json()
        except Exception:
            return JSONResponse(status_code=400,
                                content={"detail": "invalid JSON"})
        # Batch (JSON-RPC array) support: respond only to non-notifications.
        if isinstance(message, list):
            responses = [r for r in (handle_mcp(m, _ingest) for m in message) if r]
            return JSONResponse(content=responses) if responses else JSONResponse(
                status_code=202, content=None)
        response = handle_mcp(message, _ingest)
        if response is None:
            return JSONResponse(status_code=202, content=None)
        return JSONResponse(content=response)

    return app
