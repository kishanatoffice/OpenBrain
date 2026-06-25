"""Approval-event store: schema, lifecycle, filters, pagination, stats."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from guardrails.db import ApprovalStore


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ApprovalStore(Path(self.tmp.name) / "g.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_migrates_to_v1(self):
        conn = sqlite3.connect(self.store.db_path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("approval_events", tables)
        conn.close()

    def test_create_minimal_is_pending(self):
        ev = self.store.create({"prompt_text": "Allow restart?"})
        self.assertEqual(ev["status"], "pending")
        self.assertEqual(ev["prompt_text"], "Allow restart?")
        self.assertEqual(ev["options"], [])
        self.assertIsNone(ev["selected_option"])
        self.assertIsNone(ev["decided_at"])
        self.assertIsNone(ev["completed_at"])

    def test_create_complete_event_in_one_shot(self):
        ev = self.store.create({
            "prompt_text": "Need to restart the server.",
            "options": ["Restart", "Skip", "Cancel"],
            "selected_option": "Restart",
            "result": "success",
            "ide": "claude-code", "agent": "claude-code",
        })
        self.assertEqual(ev["status"], "completed")
        self.assertEqual(ev["options"], ["Restart", "Skip", "Cancel"])
        self.assertIsNotNone(ev["decided_at"])
        self.assertIsNotNone(ev["completed_at"])

    def test_lifecycle_via_updates(self):
        ev = self.store.create({"prompt_text": "Allow?", "options": ["Yes", "No"]})
        self.assertEqual(ev["status"], "pending")

        decided = self.store.update(ev["id"], {"selected_option": "Yes"})
        self.assertEqual(decided["status"], "decided")
        self.assertIsNotNone(decided["decided_at"])
        self.assertIsNone(decided["completed_at"])

        done = self.store.update(ev["id"], {"result": "success"})
        self.assertEqual(done["status"], "completed")
        self.assertIsNotNone(done["completed_at"])
        # decided_at stamp is not overwritten on the later patch
        self.assertEqual(done["decided_at"], decided["decided_at"])

    def test_update_unknown_returns_none(self):
        self.assertIsNone(self.store.update(999, {"result": "success"}))

    def test_delete(self):
        ev = self.store.create({"prompt_text": "x"})
        self.assertIsNotNone(self.store.delete(ev["id"]))
        self.assertIsNone(self.store.get(ev["id"]))
        self.assertIsNone(self.store.delete(ev["id"]))

    def test_list_filters_and_pagination(self):
        for i in range(5):
            self.store.create({"prompt_text": f"p{i}", "ide": "cursor"})
        self.store.create({"prompt_text": "other", "ide": "vscode"})
        # filter by ide
        self.assertEqual(self.store.list(ide="cursor")["rows"].__len__(), 5)
        self.assertEqual(len(self.store.list(ide="vscode")["rows"]), 1)
        # pagination: no skips/dupes across pages
        seen, cursor, pages = [], None, 0
        while True:
            page = self.store.list(limit=2, after=cursor)
            seen += [r["id"] for r in page["rows"]]
            cursor = page["next"]; pages += 1
            if not cursor or pages > 10:
                break
        self.assertEqual(len(seen), len(set(seen)))   # unique
        self.assertEqual(len(seen), 6)                # complete

    def test_malformed_cursor_is_first_page(self):
        self.store.create({"prompt_text": "a"})
        page = self.store.list(after="ks:garbage")
        self.assertEqual(len(page["rows"]), 1)

    def test_stats(self):
        self.store.create({"prompt_text": "a", "ide": "cursor",
                           "selected_option": "Restart", "result": "success"})
        self.store.create({"prompt_text": "b", "ide": "cursor",
                           "selected_option": "Cancel", "result": "failure"})
        self.store.create({"prompt_text": "c"})  # pending
        s = self.store.stats()
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["by_status"]["completed"], 2)
        self.assertEqual(s["by_status"]["pending"], 1)
        ides = {r["value"]: r["count"] for r in s["by_ide"]}
        self.assertEqual(ides["cursor"], 2)
        results = {r["value"]: r["count"] for r in s["by_result"]}
        self.assertEqual(results, {"success": 1, "failure": 1})


if __name__ == "__main__":
    unittest.main()
