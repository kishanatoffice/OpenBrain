"""Two-way vault sync tests: import, edit, detach."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest

from myagent.memory_service import create_memory
from myagent.vault_sync import extract_content, sync_once

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


def age(path, seconds=10):
    """Backdate a file so the settle-time guard doesn't skip it."""
    t = time.time() - seconds
    os.utime(path, (t, t))
    return t


class TestExtractContent(unittest.TestCase):
    def test_daemon_template_takes_content_section(self):
        text = ("---\nid: 1\n---\n\n# Title\n\n## Summary\n\nsumm\n\n"
                "## Content\n\nthe real body\n")
        self.assertEqual(extract_content(text), "the real body")

    def test_plain_note_taken_whole_minus_frontmatter(self):
        text = "---\ntags: [x]\n---\n# My note\n\nhello world\n"
        self.assertEqual(extract_content(text), "# My note\n\nhello world")

    def test_note_without_frontmatter(self):
        self.assertEqual(extract_content("just text"), "just text")


class TestSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.deps = make_deps(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_file_is_imported(self):
        note = self.deps.vault.path / "idea.md"
        note.write_text("# Idea\n\nBuild a treehouse this summer.",
                        encoding="utf-8")
        age(note)
        counts = run(sync_once(self.deps))
        self.assertEqual(counts["imported"], 1)
        rows = self.deps.store.recent(5)
        self.assertEqual(rows[0]["source"], "vault")
        self.assertIn("treehouse", rows[0]["content"])

    def test_fresh_file_waits_for_settle(self):
        note = self.deps.vault.path / "hot.md"
        note.write_text("still being written", encoding="utf-8")
        counts = run(sync_once(self.deps))
        self.assertEqual(counts["imported"], 0)

    def test_user_edit_updates_memory(self):
        memory = run(create_memory(self.deps, "original body", source="rest"))
        path = memory["md_path"]
        text = open(path, encoding="utf-8").read()
        open(path, "w", encoding="utf-8").write(
            text.replace("original body", "edited body"))
        mtime = age(path, seconds=5)
        # pretend our recorded mtime is older than the disk file
        self.deps.store.set_md_path(memory["id"], path, mtime - 60)

        counts = run(sync_once(self.deps))
        self.assertEqual(counts["updated"], 1)
        fresh = self.deps.store.get(memory["id"])
        self.assertEqual(fresh["content"], "edited body")
        self.assertEqual(fresh["summarized"], 0)  # re-summarize queued
        self.assertEqual(  # re-embed queued
            [r["id"] for r in self.deps.store.missing_embeddings("fake-embed")],
            [memory["id"]])

    def test_daemon_write_is_not_treated_as_edit(self):
        run(create_memory(self.deps, "untouched", source="rest"))
        counts = run(sync_once(self.deps))
        self.assertEqual(counts["updated"], 0)

    def test_unlinked_daemon_file_is_adopted_not_duplicated(self):
        memory = run(create_memory(self.deps, "legacy row", source="rest"))
        path = memory["md_path"]
        age(path)
        # simulate a pre-md_path-tracking row: file on disk, link lost
        self.deps.store.clear_md_path(memory["id"])

        counts = run(sync_once(self.deps))
        self.assertEqual(counts.get("adopted"), 1)
        self.assertEqual(counts["imported"], 0)
        self.assertEqual(self.deps.store.count(), 1)  # no duplicate row
        self.assertEqual(self.deps.store.vault_rows()[0]["md_path"], path)

    def test_deleted_file_detaches_but_keeps_memory(self):
        memory = run(create_memory(self.deps, "keep me", source="rest"))
        os.unlink(memory["md_path"])
        counts = run(sync_once(self.deps))
        self.assertEqual(counts["detached"], 1)
        fresh = self.deps.store.get(memory["id"])
        self.assertIsNotNone(fresh)
        self.assertIsNone(fresh["md_path"])

    def test_renamed_file_relinks_instead_of_duplicating(self):
        # Regression: a rename is one delete + one new file. The new file must
        # re-link to the detached row by content, not import a duplicate memory.
        memory = run(create_memory(self.deps, "unique renamable body", source="rest"))
        old = memory["md_path"]
        new = self.deps.vault.path / "user-renamed-note.md"
        os.rename(old, new)
        age(new)
        counts = run(sync_once(self.deps))
        self.assertEqual(self.deps.store.count(), 1)        # no duplicate
        self.assertEqual(counts["imported"], 0)
        self.assertEqual(counts.get("adopted"), 1)
        self.assertEqual(self.deps.store.get(memory["id"])["md_path"], str(new))


if __name__ == "__main__":
    unittest.main()
