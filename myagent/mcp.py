"""Model Context Protocol server — the universal bridge to any AI tool.

MCP is an open JSON-RPC 2.0 standard spoken by Claude Code, Claude
Desktop/Cowork, Gemini CLI, Cursor, Codex CLI, and others. This module
implements the server side by hand (tools capability only — ~no other
surface is needed), so the brain attaches to every one of those tools with
zero extra dependencies.

Two transports share the same dispatcher:
  stdio            — `python -m myagent.mcp` (spawned by desktop apps/CLIs)
  streamable HTTP  — POST /mcp on the daemon (stateless; for URL-based config)

stdio rule: stdout carries protocol messages ONLY; all logging goes to stderr.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from . import __version__
from .connectors import connector_for_tool, is_enabled, tool_specs_for
from .memory_service import Deps

log = logging.getLogger("myagent.mcp")

SUPPORTED_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}
LATEST_PROTOCOL_VERSION = "2025-06-18"

SERVER_INFO = {"name": "openbrain", "version": __version__}


async def call_tool(deps: Deps, name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call to its owning connector.

    Enforces the connector switch at the protocol boundary: a tool whose
    connector is disabled is rejected even if the client cached its schema,
    so a switched-off connector can never act."""
    if not isinstance(args, dict):
        return "Error: tool arguments must be an object."
    connector = connector_for_tool(name)
    if connector is None:
        raise ValueError(f"unknown tool {name!r}")
    if not is_enabled(connector.key, deps.enabled_connectors):
        return (f"Error: the '{connector.label}' connector is switched off. "
                f"Enable it in the OpenBrain dashboard to use this tool.")
    return await connector.handler(deps, name, args)


async def handle_message(deps: Deps, msg: Any) -> dict | None:
    """Dispatch one JSON-RPC message. Returns the response dict, or None for
    notifications (which must not be answered)."""
    if not isinstance(msg, dict):
        return _error(None, -32600, "invalid request")
    msg_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    is_notification = "id" not in msg

    if not method:
        return None  # a response or malformed message — nothing to do

    try:
        if method == "initialize":
            client_version = params.get("protocolVersion", "")
            client_name = (params.get("clientInfo") or {}).get("name", "")
            if client_name:
                # stdio servers are per-client processes, so this sticks for
                # the client's lifetime; the HTTP transport passes a
                # per-request copy of deps, so mutation is harmless there.
                deps.source = client_name[:60]
            result = {
                "protocolVersion": (client_version
                                    if client_version in SUPPORTED_PROTOCOL_VERSIONS
                                    else LATEST_PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            }
        elif method in ("notifications/initialized", "notifications/cancelled"):
            return None
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": tool_specs_for(deps.enabled_connectors)}
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            try:
                text = await call_tool(deps, name, args)
                result = {"content": [{"type": "text", "text": text}],
                          "isError": False}
            except Exception as exc:
                log.exception("tool %s failed", name)
                result = {"content": [{"type": "text", "text": f"Error: {exc}"}],
                          "isError": True}
        elif method in ("resources/list", "resources/templates/list"):
            result = {"resources": []}  # some clients probe without checking caps
        elif method == "prompts/list":
            result = {"prompts": []}
        else:
            if is_notification:
                return None
            return _error(msg_id, -32601, f"method not found: {method}")
    except Exception as exc:
        log.exception("error handling %s", method)
        if is_notification:
            return None
        return _error(msg_id, -32603, f"internal error: {exc}")

    if is_notification:
        return None
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


# ---- stdio transport ------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .config import load_config
    from .db import MemoryStore
    from .ollama import OllamaClient
    from .vault import Vault

    cfg = load_config()
    loop = asyncio.new_event_loop()
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
        source="mcp-stdio",  # replaced by clientInfo.name on initialize
    )
    log.info("openbrain MCP server on stdio (db=%s)", cfg.db_path)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                response = _error(None, -32700, "parse error")
            else:
                response = loop.run_until_complete(handle_message(deps, msg))
            if response is not None:
                sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                sys.stdout.flush()
    finally:
        loop.run_until_complete(deps.ollama.aclose())
        loop.close()


if __name__ == "__main__":
    main()
