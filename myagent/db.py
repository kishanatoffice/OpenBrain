"""SQLite storage engine for memories.

Schema is versioned via PRAGMA user_version and migrated in order on startup,
so a Phase 1 database upgrades in place. Connections are opened per operation:
SQLite in WAL mode handles this cheaply and it keeps the module safe under
FastAPI's threadpool without shared state.
"""

from __future__ import annotations

import json
import re
import sqlite3
from array import array
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# v1 — Phase 1 base schema.
_MIGRATION_V1 = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at DESC);
"""

# v2 — Phase 2: vault-file tracking, FTS5 index, embeddings.
_MIGRATION_V2 = """
ALTER TABLE memories ADD COLUMN md_path TEXT;

CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, summary,
    content='memories', content_rowid='id',
    tokenize='porter unicode61'
);
INSERT INTO memories_fts(rowid, content, summary)
    SELECT id, content, summary FROM memories;

CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, summary)
        VALUES (new.id, new.content, new.summary);
END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
        VALUES ('delete', old.id, old.content, old.summary);
END;
CREATE TRIGGER memories_au AFTER UPDATE OF content, summary ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, summary)
        VALUES ('delete', old.id, old.content, old.summary);
    INSERT INTO memories_fts(rowid, content, summary)
        VALUES (new.id, new.content, new.summary);
END;

CREATE TABLE embeddings (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    model     TEXT    NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB    NOT NULL
);
"""

# v3 — Phase 3: chat history and scheduled automations (dropped again in v4).
_MIGRATION_V3 = """
CREATE TABLE chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE TABLE automations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL,
    prompt           TEXT    NOT NULL,
    kind             TEXT    NOT NULL CHECK (kind IN ('daily', 'interval')),
    at_time          TEXT,
    interval_minutes INTEGER,
    enabled          INTEGER NOT NULL DEFAULT 1,
    last_run_at      TEXT,
    last_result      TEXT,
    created_at       TEXT    NOT NULL
);
"""

# v4 — Phase 4 pivot: pure memory engine. The product is the memory, not an
# agent; writes become instant (excerpt summary) and a background pass
# upgrades them to AI summaries (`summarized` flag).
_MIGRATION_V4 = """
DROP TABLE IF EXISTS chat_messages;
DROP TABLE IF EXISTS automations;
ALTER TABLE memories ADD COLUMN summarized INTEGER NOT NULL DEFAULT 0;
UPDATE memories SET summarized = 1 WHERE tokens_used > 0;
"""

# v5 — recall quality + provenance: chunked embeddings (one vector per ~1500
# chars, max-pooled at query time), tags, source attribution, and the vault
# mtime needed for two-way sync. Existing whole-memory vectors carry over as
# chunk 0; the new task-prefixed embed key makes the enricher re-embed anyway.
_MIGRATION_V5 = """
ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';
ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT '';
ALTER TABLE memories ADD COLUMN md_mtime REAL NOT NULL DEFAULT 0;

CREATE TABLE chunk_embeddings (
    memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model       TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB    NOT NULL,
    PRIMARY KEY (memory_id, model, chunk_index)
);
INSERT INTO chunk_embeddings (memory_id, model, chunk_index, dim, vector)
    SELECT memory_id, model, 0, dim, vector FROM embeddings;
DROP TABLE embeddings;
"""

# v6 — memory management at scale: favorite/archive flags, an AI-assignable
# category (folder-like), and indexes for keyset pagination + faceting.
_MIGRATION_V6 = """
ALTER TABLE memories ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN category TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories (source);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category);
CREATE INDEX IF NOT EXISTS idx_memories_keyset ON memories (archived, created_at DESC, id DESC);
"""

_MIGRATIONS = [_MIGRATION_V1, _MIGRATION_V2, _MIGRATION_V3, _MIGRATION_V4,
               _MIGRATION_V5, _MIGRATION_V6]

_ROW_COLUMNS = ("id, content, summary, tokens_used, created_at, md_path, "
                "summarized, tags, source, favorite, archived, category")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if "tags" in d:
        try:
            d["tags"] = json.loads(d["tags"])
        except (TypeError, json.JSONDecodeError):
            d["tags"] = []
    return d


def _normalize_tags(tags: list[str] | str | None) -> str:
    # Guard against a caller passing tags as a string (e.g. an MCP client that
    # sends "research" instead of ["research"]) — iterating a str would explode
    # it into single-character tags. Also drop 1-char tags as low-signal noise.
    if isinstance(tags, str):
        tags = [tags]
    cleaned = sorted({t.strip().lower() for t in (tags or [])
                      if isinstance(t, str) and len(t.strip()) > 1})
    return json.dumps(cleaned)


def _pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> array:
    vec = array("f")
    vec.frombytes(blob)
    return vec


_STOPWORDS = frozenset(
    "a an and are as at be but by do for from has have how i in is it me my "
    "of on or our she so that the their them they this to was we what when "
    "where which who will with you your".split()
)


def _fts_quote(query: str) -> str:
    """Free text -> safe FTS5 query: content words quoted, AND-joined.

    Keyword search is the precision channel of hybrid recall (semantic search
    is the recall channel), so stopwords are dropped — 'when is my physics
    exam' must match on (physics AND exam), not on 'is' and 'my'."""
    terms = [t for t in re.findall(r"\w+", query.lower())
             if t not in _STOPWORDS]
    return " ".join(f'"{t}"' for t in terms)


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
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
            # A Phase 1 DB predates versioning: tables exist but user_version is 0.
            if version == 0:
                has_memories = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
                ).fetchone()
                if has_memories:
                    version = 1
            for v in range(version, len(_MIGRATIONS)):
                conn.executescript(_MIGRATIONS[v])
                conn.execute(f"PRAGMA user_version = {v + 1}")

    # ---- writes ----------------------------------------------------------

    def add(self, content: str, summary: str, tokens_used: int = 0,
            summarized: bool = False, tags: list[str] | None = None,
            source: str = "") -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tags_json = _normalize_tags(tags)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO memories "
                "(content, summary, tokens_used, created_at, summarized, tags, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (content, summary, tokens_used, created_at, int(summarized),
                 tags_json, source),
            )
            row_id = cur.lastrowid
        return {
            "id": row_id,
            "content": content,
            "summary": summary,
            "tokens_used": tokens_used,
            "created_at": created_at,
            "md_path": None,
            "summarized": int(summarized),
            "tags": json.loads(tags_json),
            "source": source,
            "favorite": 0,
            "archived": 0,
            "category": "",
        }

    def update_content(self, memory_id: int, content: str) -> None:
        """Replace content (vault edit): stale summary and embeddings are
        invalidated so the enricher regenerates both."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET content = ?, summarized = 0 WHERE id = ?",
                (content, memory_id),
            )
            conn.execute(
                "DELETE FROM chunk_embeddings WHERE memory_id = ?", (memory_id,)
            )

    def unsummarized(self, limit: int = 5) -> list[dict[str, Any]]:
        """Memories still carrying an excerpt summary, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM memories WHERE summarized = 0 "
                "ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def mark_summarized(self, memory_id: int, summary: str, tokens_used: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET summary = ?, tokens_used = ?, summarized = 1 "
                "WHERE id = ?",
                (summary, tokens_used, memory_id),
            )

    def set_tags(self, memory_id: int, tags: list[str] | None) -> bool:
        """Replace a memory's tags (normalized). Returns False if absent."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET tags = ? WHERE id = ?",
                (_normalize_tags(tags), memory_id),
            )
            return cur.rowcount > 0

    def set_md_path(self, memory_id: int, md_path: Path | str,
                    md_mtime: float = 0.0) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET md_path = ?, md_mtime = ? WHERE id = ?",
                (str(md_path), md_mtime, memory_id),
            )

    def clear_md_path(self, memory_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET md_path = NULL, md_mtime = 0 WHERE id = ?",
                (memory_id,),
            )

    def delete(self, memory_id: int) -> dict[str, Any] | None:
        """Hard-purge a memory: row + FTS entry (via trigger) + all chunk
        embeddings. Returns the deleted row, or None if absent. Explicitly
        deletes embeddings rather than relying on the FK cascade, so a secret
        can be verifiably and fully erased even if PRAGMA foreign_keys is off."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM chunk_embeddings WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return _row_to_dict(row)

    # ---- reads -----------------------------------------------------------

    def get(self, memory_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        # Archived memories are kept but never surface in recall.
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM memories WHERE archived = 0 "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def by_tag(self, tag: str, limit: int = 20) -> list[dict[str, Any]]:
        """Memories carrying `tag`, newest first (archived excluded). Tags are
        stored as a JSON array of lowercased strings."""
        needle = f'"{tag.strip().lower()}"'
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM memories "
                "WHERE archived = 0 AND tags LIKE ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (f"%{needle}%", limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def search_keyword(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """BM25-ranked FTS5 search (archived excluded). Best (lowest bm25) first."""
        fts_query = _fts_quote(query)
        if not fts_query:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {', '.join('m.' + c for c in _ROW_COLUMNS.split(', '))} "
                "FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                "WHERE memories_fts MATCH ? AND m.archived = 0 "
                "ORDER BY bm25(memories_fts) LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def browse(self, *, q: str | None = None, source: str | None = None,
               tag: str | None = None, kind: str | None = None,
               favorite: bool = False, archived: bool = False,
               after: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Filtered, paginated memory list for the dashboard. Scales to large
        stores: browse (no q) uses indexed keyset pagination on (created_at,id);
        keyword search (q) uses FTS5 + offset. `after` is a single opaque cursor
        ('ks:<created_at>|<id>' for browse, 'off:<n>' for search); a malformed or
        mismatched cursor is ignored (treated as the first page) rather than
        raising. Returns {rows, next}."""
        limit = max(1, min(int(limit), 500))
        where = ["m.archived = 1" if archived else "m.archived = 0"]
        params: list[Any] = []
        if source:
            where.append("m.source = ?"); params.append(source)
        if favorite:
            where.append("m.favorite = 1")
        if kind in ("core", "auto"):
            where.append("m.tags LIKE ?"); params.append(f'%"{kind}"%')
        if tag:
            where.append("m.tags LIKE ?"); params.append(f'%"{tag.strip().lower()}"%')

        cols = ", ".join("m." + c for c in _ROW_COLUMNS.split(", "))
        with self._connect() as conn:
            if q and _fts_quote(q):
                offset = 0
                if after and after.startswith("off:"):
                    try:
                        offset = max(0, int(after[4:]))
                    except ValueError:
                        offset = 0  # ignore a malformed cursor
                rows = conn.execute(
                    f"SELECT {cols} FROM memories_fts f JOIN memories m ON m.id = f.rowid "
                    f"WHERE memories_fts MATCH ? AND {' AND '.join(where)} "
                    "ORDER BY bm25(memories_fts) LIMIT ? OFFSET ?",
                    (_fts_quote(q), *params, limit + 1, offset),
                ).fetchall()
                items = [_row_to_dict(r) for r in rows[:limit]]
                nxt = f"off:{offset + limit}" if len(rows) > limit else None
                return {"rows": items, "next": nxt}

            # Browse: keyset pagination (no slow OFFSET at scale).
            if after and after.startswith("ks:"):
                try:
                    cur_at, cur_id = after[3:].rsplit("|", 1)
                    where.append("(m.created_at < ? OR (m.created_at = ? AND m.id < ?))")
                    params += [cur_at, cur_at, int(cur_id)]
                except ValueError:
                    pass  # malformed cursor → start from the first page
            clause = " AND ".join(where)
            rows = conn.execute(
                f"SELECT {cols} FROM memories m WHERE {clause} "
                "ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
                (*params, limit + 1),
            ).fetchall()
        items = [_row_to_dict(r) for r in rows[:limit]]
        nxt = None
        if len(rows) > limit and items:
            last = items[-1]
            nxt = f"ks:{last['created_at']}|{last['id']}"
        return {"rows": items, "next": nxt}

    def facet_counts(self) -> dict[str, Any]:
        """Counts for the dashboard's filter rail — sources, top tags, types,
        totals. Excludes archived. Uses json_each for tag counts (scales)."""
        with self._connect() as conn:
            sources = [dict(r) for r in conn.execute(
                "SELECT source, count(*) AS count FROM memories "
                "WHERE archived = 0 AND source != '' GROUP BY source "
                "ORDER BY count DESC").fetchall()]
            tags = [dict(r) for r in conn.execute(
                "SELECT je.value AS tag, count(*) AS count "
                "FROM memories, json_each(memories.tags) je "
                "WHERE memories.archived = 0 GROUP BY je.value "
                "ORDER BY count DESC LIMIT 50").fetchall()]
            total = conn.execute("SELECT count(*) FROM memories WHERE archived = 0").fetchone()[0]
            archived = conn.execute("SELECT count(*) FROM memories WHERE archived = 1").fetchone()[0]
            favorite = conn.execute("SELECT count(*) FROM memories WHERE favorite = 1 AND archived = 0").fetchone()[0]
        return {"sources": sources, "tags": tags, "total": total,
                "archived": archived, "favorite": favorite}

    def set_flags(self, memory_id: int, *, favorite: bool | None = None,
                  archived: bool | None = None) -> bool:
        sets, params = [], []
        if favorite is not None:
            sets.append("favorite = ?"); params.append(int(favorite))
        if archived is not None:
            sets.append("archived = ?"); params.append(int(archived))
        if not sets:
            return False
        params.append(memory_id)
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params)
            return cur.rowcount > 0

    def get_many(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_ROW_COLUMNS} FROM memories WHERE id IN ({marks})", ids
            ).fetchall()
        return {r["id"]: _row_to_dict(r) for r in rows}

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM memories").fetchone()[0]

    def count_by_tag(self, tag: str) -> int:
        needle = f'"{tag.strip().lower()}"'
        with self._connect() as conn:
            return conn.execute(
                "SELECT count(*) FROM memories WHERE tags LIKE ?",
                (f"%{needle}%",),
            ).fetchone()[0]

    def all_for_export(self) -> list[dict[str, Any]]:
        """Every memory's portable fields, newest first — for backup/export."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content, summary, tags, source, created_at FROM memories "
                "ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def vault_rows(self) -> list[dict[str, Any]]:
        """id, content, md_path, md_mtime, source — what the vault sync needs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content, md_path, md_mtime, source FROM memories"
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- chunk embeddings -------------------------------------------------

    def replace_chunk_embeddings(self, memory_id: int, model: str,
                                 vectors: list[list[float]]) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunk_embeddings WHERE memory_id = ? AND model = ?",
                (memory_id, model),
            )
            conn.executemany(
                "INSERT INTO chunk_embeddings "
                "(memory_id, model, chunk_index, dim, vector) VALUES (?, ?, ?, ?, ?)",
                [(memory_id, model, i, len(v), _pack(v))
                 for i, v in enumerate(vectors)],
            )

    def missing_embeddings(self, model: str) -> list[dict[str, Any]]:
        """Memories with no chunk embeddings for the given model."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT m.id, m.content FROM memories m WHERE NOT EXISTS "
                "(SELECT 1 FROM chunk_embeddings e "
                " WHERE e.memory_id = m.id AND e.model = ?)",
                (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def all_chunk_embeddings(self, model: str) -> list[tuple[int, int, array]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT memory_id, chunk_index, vector FROM chunk_embeddings "
                "WHERE model = ?",
                (model,),
            ).fetchall()
        return [(r["memory_id"], r["chunk_index"], _unpack(r["vector"]))
                for r in rows]
