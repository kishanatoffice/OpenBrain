"""Approval-event store (SQLite).

One table, `approval_events`, is the entire V1 product: a faithful, structured
log of every approval/permission prompt an AI agent raised, what the user chose,
and how it turned out. Schema is migration-versioned from day one so later
versions (risk scores, policy verdicts) can extend it without a rewrite.

An event has a small lifecycle, which `status` tracks:
  pending   — the prompt was captured, no choice recorded yet
  decided   — the user picked an option (selected_option set)
  completed — the action ran and a result is known (result set)

Clients may create an event at any stage: a Pre-action hook logs a `pending`
event and patches it later, while a Post-action hook can log a fully `completed`
event in one shot. `status` is always derived from the fields so it can't drift.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_MIGRATION_V1 = """
CREATE TABLE approval_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    ide             TEXT,
    agent           TEXT,
    repository      TEXT,
    branch          TEXT,
    user_request    TEXT,            -- the user's original request
    agent_action    TEXT,            -- the agent's intended action
    prompt_text     TEXT NOT NULL,   -- the approval prompt, verbatim
    options         TEXT NOT NULL DEFAULT '[]',   -- JSON array of offered options
    selected_option TEXT,            -- the user's choice
    result          TEXT,            -- execution result (success / failure / …)
    result_detail   TEXT,            -- optional free-text detail
    status          TEXT NOT NULL DEFAULT 'pending',
    tool_name       TEXT,            -- e.g. Bash, Edit (when known)
    metadata        TEXT NOT NULL DEFAULT '{}',    -- JSON catch-all provenance
    created_at      TEXT NOT NULL,
    decided_at      TEXT,
    completed_at    TEXT
);
CREATE INDEX idx_events_created ON approval_events (created_at DESC, id DESC);
CREATE INDEX idx_events_session ON approval_events (session_id);
CREATE INDEX idx_events_ide     ON approval_events (ide);
CREATE INDEX idx_events_status  ON approval_events (status);
"""

_MIGRATIONS = [_MIGRATION_V1]

# Columns selected for a full record, in a stable order.
_COLUMNS = ("id, session_id, ide, agent, repository, branch, user_request, "
            "agent_action, prompt_text, options, selected_option, result, "
            "result_detail, status, tool_name, metadata, created_at, "
            "decided_at, completed_at")

# Fields stored as JSON text and rehydrated on read.
_JSON_FIELDS = ("options", "metadata")

# Free-text fields that should be length-capped on the way in.
TEXT_FIELDS = ("session_id", "ide", "agent", "repository", "branch",
               "user_request", "agent_action", "prompt_text",
               "selected_option", "result", "result_detail", "tool_name")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for field in _JSON_FIELDS:
        if field in d:
            try:
                d[field] = json.loads(d[field]) if d[field] else ([] if field == "options" else {})
            except (TypeError, json.JSONDecodeError):
                d[field] = [] if field == "options" else {}
    return d


def _derive_status(selected_option: Any, result: Any) -> str:
    """status is a pure function of the two lifecycle fields, so it never drifts
    from reality regardless of which order a client fills them in."""
    if result not in (None, ""):
        return "completed"
    if selected_option not in (None, ""):
        return "decided"
    return "pending"


class ApprovalStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for v in range(version, len(_MIGRATIONS)):
                conn.executescript(_MIGRATIONS[v])
                conn.execute(f"PRAGMA user_version = {v + 1}")

    # ---- writes ----------------------------------------------------------

    def create(self, event: dict[str, Any]) -> dict[str, Any]:
        """Insert one approval event. Missing fields default to NULL/empty;
        only prompt_text is required (validated at the API layer). status,
        created_at, and the decided/completed timestamps are derived here."""
        created_at = _now()
        selected = event.get("selected_option")
        result = event.get("result")
        status = _derive_status(selected, result)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO approval_events "
                "(session_id, ide, agent, repository, branch, user_request, "
                " agent_action, prompt_text, options, selected_option, result, "
                " result_detail, status, tool_name, metadata, created_at, "
                " decided_at, completed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event.get("session_id"), event.get("ide"), event.get("agent"),
                    event.get("repository"), event.get("branch"),
                    event.get("user_request"), event.get("agent_action"),
                    event.get("prompt_text") or "",
                    json.dumps(event.get("options") or []),
                    selected, result, event.get("result_detail"), status,
                    event.get("tool_name"),
                    json.dumps(event.get("metadata") or {}),
                    created_at,
                    created_at if status in ("decided", "completed") else None,
                    created_at if status == "completed" else None,
                ),
            )
            row_id = cur.lastrowid
        return self.get(row_id)

    def update(self, event_id: int, patch: dict[str, Any]) -> dict[str, Any] | None:
        """Patch an event's lifecycle fields (selected_option, result,
        result_detail, options, metadata). Recomputes status and stamps the
        decided/completed timestamps the first time each milestone is reached.
        Returns the updated row, or None if the id is unknown."""
        current = self.get(event_id)
        if current is None:
            return None

        selected = patch.get("selected_option", current["selected_option"])
        result = patch.get("result", current["result"])
        status = _derive_status(selected, result)

        sets: list[str] = []
        params: list[Any] = []
        for field in ("selected_option", "result", "result_detail"):
            if field in patch:
                sets.append(f"{field} = ?")
                params.append(patch[field])
        if "options" in patch:
            sets.append("options = ?")
            params.append(json.dumps(patch["options"] or []))
        if "metadata" in patch:
            sets.append("metadata = ?")
            params.append(json.dumps(patch["metadata"] or {}))

        sets.append("status = ?")
        params.append(status)
        # Stamp the first time we reach each milestone; don't overwrite on later patches.
        if status in ("decided", "completed") and not current["decided_at"]:
            sets.append("decided_at = ?")
            params.append(_now())
        if status == "completed" and not current["completed_at"]:
            sets.append("completed_at = ?")
            params.append(_now())

        params.append(event_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE approval_events SET {', '.join(sets)} WHERE id = ?", params)
        return self.get(event_id)

    def delete(self, event_id: int) -> dict[str, Any] | None:
        row = self.get(event_id)
        if row is None:
            return None
        with self._connect() as conn:
            conn.execute("DELETE FROM approval_events WHERE id = ?", (event_id,))
        return row

    # ---- reads -----------------------------------------------------------

    def get(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM approval_events WHERE id = ?",
                (event_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list(self, *, session_id: str | None = None, ide: str | None = None,
             agent: str | None = None, repository: str | None = None,
             status: str | None = None, selected_option: str | None = None,
             after: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Filtered, keyset-paginated event list (newest first). `after` is an
        opaque cursor 'ks:<created_at>|<id>'; a malformed cursor is treated as
        the first page rather than raising. Returns {rows, next}."""
        limit = max(1, min(int(limit), 500))
        where: list[str] = []
        params: list[Any] = []
        for col, val in (("session_id", session_id), ("ide", ide),
                         ("agent", agent), ("repository", repository),
                         ("status", status), ("selected_option", selected_option)):
            if val:
                where.append(f"{col} = ?")
                params.append(val)
        if after and after.startswith("ks:"):
            try:
                cur_at, cur_id = after[3:].rsplit("|", 1)
                where.append("(created_at < ? OR (created_at = ? AND id < ?))")
                params += [cur_at, cur_at, int(cur_id)]
            except ValueError:
                pass  # malformed cursor → first page
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM approval_events{clause} "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (*params, limit + 1)).fetchall()
        items = [_row_to_dict(r) for r in rows[:limit]]
        nxt = None
        if len(rows) > limit and items:
            last = items[-1]
            nxt = f"ks:{last['created_at']}|{last['id']}"
        return {"rows": items, "next": nxt}

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM approval_events").fetchone()[0]

    def stats(self) -> dict[str, Any]:
        """Facet counts for the dashboard rail and at-a-glance totals."""
        def group(col: str) -> list[dict[str, Any]]:
            with self._connect() as conn:
                return [dict(r) for r in conn.execute(
                    f"SELECT {col} AS value, count(*) AS count FROM approval_events "
                    f"WHERE {col} IS NOT NULL AND {col} != '' "
                    f"GROUP BY {col} ORDER BY count DESC").fetchall()]
        with self._connect() as conn:
            total = conn.execute("SELECT count(*) FROM approval_events").fetchone()[0]
            by_status = {r["value"]: r["count"] for r in group("status")}
        return {
            "total": total,
            "by_status": by_status,
            "by_ide": group("ide"),
            "by_agent": group("agent"),
            "by_selected_option": group("selected_option"),
            "by_result": group("result"),
        }
