"""Connector registry — the platform layer.

OpenBrain is not a single feature; it is a host for *connectors*. Each
connector is a self-contained capability that contributes MCP tools and owns
its handler. Memory (recall/remember/forget) is the first connector; OCR,
and others follow the same contract.

Design rules that keep this foolproof as the platform grows:

  * One uniform `Connector` shape — adding a capability is registering a value,
    never threading a new special case through the MCP dispatcher or the UI.
  * Tools are exposed ONLY for connectors that are enabled. Disabling a
    connector makes its tools vanish from `tools/list` AND rejects any direct
    `tools/call` — defense in depth at the protocol boundary, so a disabled
    connector cannot act even if a client cached its schema.
  * Built-in connectors (`toggleable=False`, e.g. Memory — the core promise)
    cannot be switched off; only add-on connectors carry a user toggle. This
    prevents the footgun of a user uninstalling the product's own reason to
    exist. Memory's live "mute" is the separate `paused` master switch.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memory_service import (
    DEFAULT_RECALL_TOKENS,
    EXPAND_LIMIT,
    MAX_RECALL_TOKENS,
    MIN_RECALL_TOKENS,
    Deps,
    create_memory,
    expand_memories,
    recall_context,
    recall_index,
)

ToolHandler = Callable[[Deps, str, dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class Connector:
    """One pluggable capability.

    key          stable identifier, used in settings + URLs (never shown raw)
    label        human name for the dashboard card
    description  one-line, plain-English (the UI convention)
    tool_specs   MCP tool schemas this connector contributes
    handler      async (deps, tool_name, args) -> result text
    default_enabled  whether it is on out of the box
    toggleable   False for built-ins that must never be switched off (Memory)
    """

    key: str
    label: str
    description: str
    tool_specs: list[dict[str, Any]]
    handler: ToolHandler
    default_enabled: bool = True
    toggleable: bool = True
    tool_names: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Derive the tool-name set once so dispatch is an O(1) lookup, and the
        # name<->connector mapping can never drift from tool_specs.
        object.__setattr__(
            self, "tool_names",
            frozenset(spec["name"] for spec in self.tool_specs),
        )


# ---- helpers shared by tool handlers ----------------------------------------


def _safe_int(value: Any, default: int) -> int:
    """Coerce an MCP argument to int without trusting the client's type."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _remove_vault_file(deps: Deps, deleted: dict) -> None:
    if not deleted.get("md_path"):
        return
    md_file = Path(deleted["md_path"])
    if md_file.is_relative_to(deps.vault.path) and md_file.is_file():
        md_file.unlink()


# ---- memory connector --------------------------------------------------------

MEMORY_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "recall",
        "description": (
            "Search the user's permanent local memory (their 'second brain'). "
            "Call this at the START of a task with a short description of the "
            "task, and whenever the user mentions something you don't have in "
            "context — past decisions, preferences, people, projects, dates. "
            "By default returns a COMPACT INDEX (id + one-line summary + score) "
            "— cheap to read; then call `expand` with the ids you actually need "
            "to read their full text. Use mode='full' only when you want every "
            "match inlined at once. An empty query returns the most recent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, e.g. 'database migration decisions'",
                },
                "mode": {
                    "type": "string",
                    "enum": ["index", "full"],
                    "description": "'index' (default) lists compact candidates to "
                                   "expand; 'full' inlines every match's body.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": f"Token budget for the result "
                                   f"({MIN_RECALL_TOKENS}-{MAX_RECALL_TOKENS}, "
                                   f"default {DEFAULT_RECALL_TOKENS})",
                },
                "tag": {
                    "type": "string",
                    "description": "Optional: only return memories carrying this tag",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "expand",
        "description": (
            "Fetch the FULL text of specific memories by id — step two after a "
            "`recall` index. Pass only the ids you actually need; fetching less "
            "is the whole point (it keeps context small). Ids are the numbers "
            "shown as '#N' in the recall index."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": f"Memory ids to expand (up to {EXPAND_LIMIT})",
                },
            },
            "required": ["ids"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Save important information to the user's permanent local memory. "
            "It persists forever, across every session and every AI tool the "
            "user works with. Use it for durable facts, decisions, "
            "preferences, project state, and anything the user asks you to "
            "remember. Write the content so it makes sense standalone, "
            "without this conversation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The self-contained text to remember",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels, e.g. ['work', 'health']",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "forget",
        "description": (
            "Permanently delete one memory by its id (the number shown by "
            "recall as 'memory #N'). Only use when the user asks to forget "
            "or correct something."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer", "description": "id from recall"},
            },
            "required": ["memory_id"],
        },
    },
]


async def memory_handler(deps: Deps, name: str, args: dict[str, Any]) -> str:
    if name == "recall":
        # Default to the compact index (cheap); 'full' inlines every body.
        recall = (recall_context if str(args.get("mode") or "").lower() == "full"
                  else recall_index)
        return await recall(
            deps,
            str(args.get("query", "")),
            _safe_int(args.get("max_tokens"), DEFAULT_RECALL_TOKENS),
            tag=str(args.get("tag") or "") or None,
        )
    if name == "expand":
        ids = args.get("ids")
        if not isinstance(ids, list):
            return "Error: 'ids' must be a list of memory ids, e.g. [12, 34]."
        return expand_memories(deps, ids)
    if name == "remember":
        content = str(args.get("content", "")).strip()
        if not content:
            return "Error: content must not be empty."
        raw_tags = args.get("tags") or []
        # A client may send a string ("a,b") or a single tag instead of a list;
        # coerce so we never iterate a string into single-character tags.
        if isinstance(raw_tags, str):
            raw_tags = [t for t in raw_tags.split(",")] if "," in raw_tags else [raw_tags]
        tags = [str(t) for t in raw_tags]
        # Agent-initiated saves are gated: meta/test chatter is skipped so it
        # never pollutes future recall. The user's own writes (web UI / REST)
        # are never gated.
        memory = await create_memory(deps, content, tags=tags, force=False)
        if memory.get("skipped"):
            return (f"Not saved — {memory['reason']}. (Only durable facts, "
                    f"decisions, and preferences are worth remembering.)")
        if memory.get("duplicate"):
            return (f"Already known — this matches memory #{memory['id']} "
                    f"(similarity {memory['similarity']}). Nothing new saved.")
        return (f"Saved as memory #{memory['id']}. It is now permanent and "
                f"available to every connected AI tool.")
    if name == "forget":
        memory_id = _safe_int(args.get("memory_id"), -1)
        deleted = deps.store.delete(memory_id)
        if deleted is None:
            return f"Error: no memory #{memory_id}."
        _remove_vault_file(deps, deleted)
        return f"Memory #{memory_id} permanently deleted."
    raise ValueError(f"unknown tool {name!r}")


MEMORY_CONNECTOR = Connector(
    key="memory",
    label="Memory",
    description="Remembers your context, preferences, and decisions across every AI tool.",
    tool_specs=MEMORY_TOOL_SPECS,
    handler=memory_handler,
    default_enabled=True,
    toggleable=False,  # the core promise — muted via the master switch, never removed
)


# ---- registry ----------------------------------------------------------------

# Order here is the order connectors render in the dashboard.
REGISTRY: dict[str, Connector] = {c.key: c for c in (MEMORY_CONNECTOR,)}


def default_enabled_keys() -> frozenset[str]:
    return frozenset(c.key for c in REGISTRY.values() if c.default_enabled)


def _resolve_enabled(enabled: Iterable[str] | None) -> frozenset[str]:
    """None means 'use defaults' — so a Deps built without explicit connector
    state (e.g. the stdio entrypoint, tests) still exposes the built-ins."""
    return default_enabled_keys() if enabled is None else frozenset(enabled)


def is_enabled(key: str, enabled: Iterable[str] | None) -> bool:
    return key in _resolve_enabled(enabled)


def tool_specs_for(enabled: Iterable[str] | None) -> list[dict[str, Any]]:
    """The MCP tool list for the currently-enabled connectors."""
    active = _resolve_enabled(enabled)
    specs: list[dict[str, Any]] = []
    for c in REGISTRY.values():
        if c.key in active:
            specs.extend(c.tool_specs)
    return specs


def connector_for_tool(name: str) -> Connector | None:
    for c in REGISTRY.values():
        if name in c.tool_names:
            return c
    return None


# ---- add-on connectors -------------------------------------------------------
# Importing the module triggers it to register itself into REGISTRY. We import
# the MODULE (`from . import ocr`), never a symbol from it: a bare module import
# tolerates a partially-initialized module, so this is safe regardless of which
# module Python imports first (no circular-import failure). Each add-on registers
# itself; built-in Memory is already first, preserving dashboard order.
from . import ocr  # noqa: E402,F401
