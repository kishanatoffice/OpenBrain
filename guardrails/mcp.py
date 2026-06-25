"""Minimal MCP (JSON-RPC 2.0) surface for Guard Rails.

Exposes one tool, `log_approval`, so an agent that *can* call tools is able to
self-report an approval event without a separate HTTP client. The REST API is
the primary ingestion path (that's what the hooks use); this is the equivalent
door for MCP-native callers. Stateless streamable-HTTP transport, served at
POST /mcp by server.py.

Deliberately tiny: only the three methods a client needs to discover and call
the tool. Anything else returns a proper JSON-RPC "method not found".
"""

from __future__ import annotations

from typing import Any, Callable

PROTOCOL_VERSION = "2025-06-18"
_SUPPORTED = {"2024-11-05", "2025-03-26", "2025-06-18"}

_LOG_APPROVAL_TOOL = {
    "name": "log_approval",
    "description": (
        "Record an approval/permission prompt and its outcome in OpenBrain "
        "Guard Rails. Call this whenever a permission decision is made. Provide "
        "the verbatim prompt and as much surrounding context as is known."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["prompt_text"],
        "properties": {
            "prompt_text": {"type": "string", "description": "The approval prompt, verbatim."},
            "user_request": {"type": "string", "description": "The user's original request."},
            "agent_action": {"type": "string", "description": "The agent's intended action."},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": "All options offered to the user."},
            "selected_option": {"type": "string", "description": "The option the user chose."},
            "result": {"type": "string", "description": "Execution result (e.g. success, failure)."},
            "result_detail": {"type": "string"},
            "session_id": {"type": "string"},
            "ide": {"type": "string"},
            "agent": {"type": "string"},
            "repository": {"type": "string"},
            "branch": {"type": "string"},
            "tool_name": {"type": "string"},
        },
    },
}


def _result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_mcp(message: dict, ingest: Callable[[dict], dict]) -> dict | None:
    """Handle one JSON-RPC message. `ingest` stores a validated event dict and
    returns the stored row. Returns a response dict, or None for notifications
    (which by spec get no reply)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(message.get("id") if isinstance(message, dict) else None,
                      -32600, "Invalid Request")

    method = message.get("method")
    msg_id = message.get("id")
    # Notifications carry no id and expect no response.
    if "id" not in message:
        return None

    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        version = requested if requested in _SUPPORTED else PROTOCOL_VERSION
        return _result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "openbrain-guardrails", "version": "0.1.0"},
        })

    if method == "tools/list":
        return _result(msg_id, {"tools": [_LOG_APPROVAL_TOOL]})

    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != "log_approval":
            return _error(msg_id, -32602, f"unknown tool: {params.get('name')}")
        args = params.get("arguments") or {}
        if not (args.get("prompt_text") or "").strip():
            return _result(msg_id, {
                "content": [{"type": "text", "text": "prompt_text is required."}],
                "isError": True,
            })
        try:
            row = ingest(args)
        except Exception as exc:  # never crash the transport on a bad write
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"failed to log: {exc}"}],
                "isError": True,
            })
        return _result(msg_id, {
            "content": [{"type": "text",
                         "text": f"Logged approval event #{row['id']} (status: {row['status']})."}],
            "isError": False,
        })

    return _error(msg_id, -32601, f"method not found: {method}")
