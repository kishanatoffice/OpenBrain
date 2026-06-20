"""Storage engine tests: migrations, FTS sync, chunk embeddings."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from myagent.db import MemoryStore


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_database_migrates_to_latest(self):
        conn = sqlite3.connect(self.store.db_path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 6)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("memories", tables)
        self.assertIn("chunk_embeddings", tables)
        self.assertNotIn("embeddings", tables)
        self.assertNotIn("automations", tables)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
        self.assertTrue({"favorite", "archived", "category"} <= cols)  # v6
        conn.close()

    def test_tags_are_normalized_and_roundtripped(self):
        memory = self.store.add("content", "summary",
                                tags=[" Work ", "health", "work"])
        self.assertEqual(memory["tags"], ["health", "work"])
        self.assertEqual(self.store.get(memory["id"])["tags"], ["health", "work"])

    def test_by_tag_and_counts_and_set_tags(self):
        a = self.store.add("alpha fact", "s", tags=["core"])
        self.store.add("beta fact", "s", tags=["auto"])
        self.store.add("gamma fact", "s", tags=["core", "auto"])
        self.assertEqual(self.store.count_by_tag("core"), 2)
        self.assertEqual(self.store.count_by_tag("auto"), 2)
        self.assertEqual({r["id"] for r in self.store.by_tag("core")},
                         {a["id"], a["id"] + 2})
        # set_tags replaces tags wholesale
        self.assertTrue(self.store.set_tags(a["id"], ["auto"]))
        self.assertEqual(self.store.count_by_tag("core"), 1)
        self.assertFalse(self.store.set_tags(9999, ["x"]))

    def test_browse_filters_pagination_and_flags(self):
        for i in range(5):
            self.store.add(f"memory number {i}", "s", tags=["work"], source="cursor")
        self.store.add("a gemini note", "s", tags=["personal"], source="gemini")
        # keyset pagination: page of 3 then the rest
        p1 = self.store.browse(limit=3)
        self.assertEqual(len(p1["rows"]), 3)
        self.assertIsNotNone(p1["next"])
        p2 = self.store.browse(limit=3, after=p1["next"])
        self.assertEqual(len(p2["rows"]), 3)  # remaining 3
        ids = {r["id"] for r in p1["rows"]} | {r["id"] for r in p2["rows"]}
        self.assertEqual(len(ids), 6)  # no overlap, full coverage
        # source filter
        self.assertEqual(len(self.store.browse(source="gemini")["rows"]), 1)
        # tag filter
        self.assertEqual(len(self.store.browse(tag="work")["rows"]), 5)

    def test_search_pagination_honors_offset_cursor(self):
        for i in range(5):
            self.store.add(f"widget report number {i}", "s")
        p1 = self.store.browse(q="widget", limit=2)
        self.assertEqual(len(p1["rows"]), 2)
        self.assertEqual(p1["next"], "off:2")
        p2 = self.store.browse(q="widget", limit=2, after=p1["next"])
        self.assertEqual(len(p2["rows"]), 2)
        # pages must not overlap (the bug was: off cursor ignored → page 1 forever)
        self.assertFalse({r["id"] for r in p1["rows"]} & {r["id"] for r in p2["rows"]})

    def test_browse_tolerates_malformed_cursor(self):
        self.store.add("anything here", "s")
        self.assertEqual(len(self.store.browse(after="garbage")["rows"]), 1)
        self.assertEqual(len(self.store.browse(after="ks:bad")["rows"]), 1)
        self.assertEqual(len(self.store.browse(q="anything", after="off:x")["rows"]), 1)

    def test_archive_excludes_from_recall_paths(self):
        m = self.store.add("archive me", "s", tags=["x"])
        self.store.set_flags(m["id"], archived=True)
        self.assertEqual(self.store.recent(10), [])              # recall: gone
        self.assertEqual(self.store.search_keyword("archive"), [])  # recall: gone
        self.assertEqual(len(self.store.browse(archived=True)["rows"]), 1)  # browse: visible
        self.assertEqual(len(self.store.browse()["rows"]), 0)    # default browse: hidden

    def test_facet_counts(self):
        self.store.add("one", "s", tags=["work", "core"], source="cursor")
        self.store.add("two", "s", tags=["work"], source="gemini")
        f = self.store.facet_counts()
        self.assertEqual(f["total"], 2)
        srcs = {s["source"]: s["count"] for s in f["sources"]}
        self.assertEqual(srcs, {"cursor": 1, "gemini": 1})
        tags = {t["tag"]: t["count"] for t in f["tags"]}
        self.assertEqual(tags["work"], 2)

    def test_delete_hard_purges_embeddings_and_fts(self):
        def counts(mid: int) -> tuple[int, int]:
            with sqlite3.connect(self.store.db_path) as c:
                rows = c.execute("SELECT count(*) FROM memories WHERE id=?",
                                 (mid,)).fetchone()[0]
                emb = c.execute("SELECT count(*) FROM chunk_embeddings WHERE memory_id=?",
                                (mid,)).fetchone()[0]
            return rows, emb

        m = self.store.add("secret leaked token here", "summary")
        self.store.replace_chunk_embeddings(m["id"], "fake-embed",
                                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(len(self.store.search_keyword("token")), 1)
        self.assertEqual(counts(m["id"]), (1, 2))

        self.store.delete(m["id"])
        self.assertEqual(counts(m["id"]), (0, 0))           # fully gone
        self.assertEqual(self.store.search_keyword("token"), [])  # FTS cleared

    def test_all_for_export_shape(self):
        self.store.add("exportable", "summary", tags=["t"], source="unit")
        rows = self.store.all_for_export()
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]),
                         {"content", "summary", "tags", "source", "created_at"})

    def test_tags_string_not_exploded_and_short_dropped(self):
        # Regression: a string tag must NOT become single-char tags, and 1-char
        # tags are dropped as noise.
        m = self.store.add("x", "s", tags="research")
        self.assertEqual(self.store.get(m["id"])["tags"], ["research"])
        m2 = self.store.add("y", "s", tags=["work", "a", "b", "ok"])
        self.assertEqual(self.store.get(m2["id"])["tags"], ["ok", "work"])

    def test_fts_follows_content_updates(self):
        memory = self.store.add("the quick brown fox", "summary")
        self.assertEqual(len(self.store.search_keyword("fox")), 1)
        self.store.update_content(memory["id"], "a lazy elephant instead")
        self.assertEqual(self.store.search_keyword("fox"), [])
        self.assertEqual(len(self.store.search_keyword("elephant")), 1)

    def test_update_content_invalidates_summary_and_embeddings(self):
        memory = self.store.add("original", "summary", summarized=True)
        self.store.replace_chunk_embeddings(memory["id"], "m", [[1.0, 0.0]])
        self.store.update_content(memory["id"], "changed")
        self.assertEqual(self.store.get(memory["id"])["summarized"], 0)
        self.assertEqual(self.store.all_chunk_embeddings("m"), [])
        self.assertEqual([r["id"] for r in self.store.missing_embeddings("m")],
                         [memory["id"]])

    def test_chunk_embeddings_replace_and_cascade(self):
        memory = self.store.add("content", "summary")
        self.store.replace_chunk_embeddings(memory["id"], "m",
                                            [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(len(self.store.all_chunk_embeddings("m")), 2)
        self.store.replace_chunk_embeddings(memory["id"], "m", [[0.5, 0.5]])
        self.assertEqual(len(self.store.all_chunk_embeddings("m")), 1)
        self.store.delete(memory["id"])
        self.assertEqual(self.store.all_chunk_embeddings("m"), [])

    def test_unsummarized_and_mark(self):
        memory = self.store.add("content", "excerpt")
        self.assertEqual([r["id"] for r in self.store.unsummarized()],
                         [memory["id"]])
        self.store.mark_summarized(memory["id"], "ai summary", 99)
        self.assertEqual(self.store.unsummarized(), [])
        fresh = self.store.get(memory["id"])
        self.assertEqual((fresh["summary"], fresh["tokens_used"]),
                         ("ai summary", 99))

    def test_md_path_set_and_clear(self):
        memory = self.store.add("content", "summary")
        self.store.set_md_path(memory["id"], "/tmp/x.md", 123.0)
        row = self.store.vault_rows()[0]
        self.assertEqual((row["md_path"], row["md_mtime"]), ("/tmp/x.md", 123.0))
        self.store.clear_md_path(memory["id"])
        self.assertIsNone(self.store.vault_rows()[0]["md_path"])


if __name__ == "__main__":
    unittest.main()
