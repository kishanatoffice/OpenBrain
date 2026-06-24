"""Memory engine tests: chunking, dedupe, threshold, packing, enrichment."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from myagent.memory_service import (
    MIN_RECALL_TOKENS,
    _pack,
    apply_supersession,
    chunk_text,
    create_memory,
    enrich_pending,
    excerpt_summary,
    is_low_value,
    normalize_source,
    recall_context,
)
from myagent.search import estimate_tokens

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


class TestChunking(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("hello"), ["hello"])
        self.assertEqual(chunk_text("   "), [])

    def test_windows_overlap(self):
        text = "x" * 4000
        chunks = chunk_text(text, size=1500, overlap=200)
        self.assertTrue(all(len(c) <= 1500 for c in chunks))
        # consecutive windows share `overlap` chars: chunk i+1 starts at
        # i*(size-overlap), i.e. 200 before chunk i ends
        # windows at 0/1300/2600; the 3900 window is fully inside the
        # previous chunk and gets dropped
        step = 1500 - 200
        self.assertEqual(len(chunks), 3)
        joined = "".join(c[: step] for c in chunks[:-1]) + chunks[-1]
        self.assertEqual(joined, text)  # full coverage, nothing lost


class TestWriteAndDedupe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_writes_db_vault_and_chunks(self):
        deps = make_deps(self.tmp.name)
        memory = run(create_memory(deps, "A fact worth keeping.",
                                   tags=["Test"], source="unit"))
        self.assertFalse(memory.get("duplicate"))
        self.assertEqual(memory["tags"], ["test"])
        self.assertEqual(memory["source"], "unit")
        self.assertTrue(Path(memory["md_path"]).is_file())
        self.assertEqual(len(deps.store.all_chunk_embeddings("fake-embed")), 1)

    def test_near_duplicate_is_not_stored_twice(self):
        deps = make_deps(self.tmp.name)  # all texts embed to the same vector
        first = run(create_memory(deps, "Remember the cake recipe."))
        second = run(create_memory(deps, "Remember the cake recipe please."))
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(deps.store.count(), 1)

    def test_distinct_vectors_are_both_stored(self):
        deps = make_deps(self.tmp.name, vectors={
            "first fact": [1.0, 0.0, 0.0],
            "second fact": [0.0, 1.0, 0.0],
        })
        run(create_memory(deps, "first fact"))
        second = run(create_memory(deps, "second fact"))
        self.assertFalse(second.get("duplicate"))
        self.assertEqual(deps.store.count(), 2)

    def test_write_survives_ollama_down(self):
        deps = make_deps(self.tmp.name, down=True)
        memory = run(create_memory(deps, "Stored even when the model is off."))
        self.assertIn("warning", memory)
        self.assertEqual(deps.store.count(), 1)

    def test_secrets_redacted_before_storage(self):
        deps = make_deps(self.tmp.name)
        m = run(create_memory(deps, "deploy key is sk-proj-ABCDEFGHIJKLMNOP1234567890 keep it"))
        stored = deps.store.get(m["id"])
        self.assertNotIn("sk-proj-ABCDEFGHIJKLMNOP", stored["content"])
        self.assertIn("[REDACTED:openai-key]", stored["content"])

    def test_dedup_matches_any_chunk_not_just_first(self):
        # Two long docs share an identical TAIL chunk but differ in the head.
        # Old code dedup'd on chunk 0 only and missed this; now it must catch it.
        tail = "Z" * 310
        a = "A" * 1300 + tail
        b = "B" * 1300 + tail
        ca, cb = chunk_text(a), chunk_text(b)
        self.assertEqual(len(ca), 2)  # sanity: two chunks each
        deps = make_deps(self.tmp.name, vectors={
            ca[0]: [1.0, 0.0, 0.0], ca[1]: [0.0, 1.0, 0.0],   # shared tail vector
            cb[0]: [0.0, 0.0, 1.0], cb[1]: [0.0, 1.0, 0.0],   # cb[1]==ca[1]
        })
        run(create_memory(deps, a))
        second = run(create_memory(deps, b))
        self.assertTrue(second.get("duplicate"))  # caught via the shared tail
        self.assertEqual(deps.store.count(), 1)


class TestRecall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_similarity_floor_keeps_junk_out(self):
        deps = make_deps(self.tmp.name, vectors={
            "dentist appointment in july": [1.0, 0.0, 0.0],
            "jenkins pipeline migration": [0.2, 0.98, 0.0],
            "the dentist": [1.0, 0.0, 0.0],   # query embeds like the dentist doc
        })
        run(create_memory(deps, "dentist appointment in july"))
        run(create_memory(deps, "jenkins pipeline migration"))
        block = run(recall_context(deps, "the dentist", 2000))
        self.assertIn("dentist appointment", block)
        self.assertNotIn("jenkins", block)  # cos ≈ 0.2 < 0.50 floor

    def test_relative_cutoff_drops_also_rans(self):
        deps = make_deps(self.tmp.name, vectors={
            "strong match": [0.95, 0.312, 0.0],   # cos 0.95 with query
            "weak match": [0.80, 0.60, 0.0],      # cos 0.80 — above floor,
            "q": [1.0, 0.0, 0.0],                 # below 0.9 x 0.95 = 0.855
        })
        run(create_memory(deps, "strong match"))
        run(create_memory(deps, "weak match"))
        block = run(recall_context(deps, "q", 2000))
        self.assertIn("strong match", block)
        self.assertNotIn("weak match", block)

    def test_tag_filter(self):
        deps = make_deps(self.tmp.name)
        run(create_memory(deps, "tagged fact", tags=["work"]))
        block = run(recall_context(deps, "", 2000, tag="health"))
        self.assertIn("nothing found", block)
        block = run(recall_context(deps, "", 2000, tag="work"))
        self.assertIn("tagged fact", block)

    def test_empty_brain(self):
        deps = make_deps(self.tmp.name)
        self.assertIn("nothing found", run(recall_context(deps, "", 500)))


class TestWriteQualityGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_flags_junk_and_keeps_facts(self):
        self.assertTrue(is_low_value("ok")[0])                       # too short
        self.assertTrue(is_low_value("just testing")[0])            # too few words
        self.assertTrue(is_low_value(
            "so this is the prompt i am writing, lets see what you "
            "have stored about me")[0])                             # meta chatter
        self.assertFalse(is_low_value(
            "Dentist appointment booked for July 3rd at 4pm.")[0])  # real fact

    def test_auto_save_skips_junk_but_stores_facts(self):
        deps = make_deps(self.tmp.name, vectors={
            "lets see what you have stored about me": [0.0, 1.0, 0.0],
            "Migrated CI from Jenkins to GitHub Actions in July.": [1.0, 0.0, 0.0],
        })
        skipped = run(create_memory(
            deps, "lets see what you have stored about me", force=False))
        self.assertTrue(skipped.get("skipped"))
        self.assertEqual(deps.store.count(), 0)

        kept = run(create_memory(
            deps, "Migrated CI from Jenkins to GitHub Actions in July.",
            force=False))
        self.assertFalse(kept.get("skipped"))
        self.assertEqual(deps.store.count(), 1)

    def test_explicit_write_is_never_gated(self):
        deps = make_deps(self.tmp.name)
        kept = run(create_memory(deps, "ok", force=True))  # short, but explicit
        self.assertFalse(kept.get("skipped"))
        self.assertEqual(deps.store.count(), 1)


class TestCorePersonaLayer(unittest.TestCase):
    """The always-on layer: core-tagged memories inject regardless of query."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_core_injected_for_unrelated_query(self):
        # Core fact and the query embed to orthogonal vectors: relevance alone
        # would never surface the core memory, but it must appear anyway.
        deps = make_deps(self.tmp.name, vectors={
            "About me: I prefer concise answers": [0.0, 1.0, 0.0],
            "how do I sort a list in python": [1.0, 0.0, 0.0],
        })
        run(create_memory(deps, "About me: I prefer concise answers",
                          tags=["core", "persona"]))
        block = run(recall_context(deps, "how do I sort a list in python", 2000))
        self.assertIn("persona & preferences", block)
        self.assertIn("I prefer concise answers", block)

    def test_core_injected_on_empty_query(self):
        deps = make_deps(self.tmp.name)
        run(create_memory(deps, "About me: I am a product manager",
                          tags=["core"]))
        block = run(recall_context(deps, "", 2000))
        self.assertIn("product manager", block)

    def test_total_output_respects_small_budget_with_core(self):
        # Regression: the persona block and query block were each independently
        # floored back to MIN_RECALL_TOKENS, letting total output reach ~2x the
        # requested cap. Total must stay within ~budget.
        deps = make_deps(self.tmp.name)
        for i in range(6):
            run(create_memory(
                deps, ("Persona fact %d: a fairly long sentence of context "
                       "about the user. " % i) * 4, tags=["core"]))
        block = run(recall_context(deps, "", MIN_RECALL_TOKENS))
        self.assertLessEqual(estimate_tokens(block), int(MIN_RECALL_TOKENS * 1.35))

    def test_no_duplicate_when_core_also_matches_query(self):
        deps = make_deps(self.tmp.name, vectors={
            "About me: I love rust": [1.0, 0.0, 0.0],
            "tell me about rust": [1.0, 0.0, 0.0],
        })
        run(create_memory(deps, "About me: I love rust", tags=["core"]))
        block = run(recall_context(deps, "tell me about rust", 2000))
        # Appears in the persona section, but not echoed again in results.
        self.assertEqual(block.count("I love rust"), 1)

    def test_specific_tag_filter_skips_persona(self):
        deps = make_deps(self.tmp.name, vectors={
            "About me: persona fact": [1.0, 0.0, 0.0],
            "a work note": [0.0, 1.0, 0.0],  # distinct so it isn't deduped
        })
        run(create_memory(deps, "About me: persona fact", tags=["core"]))
        run(create_memory(deps, "a work note", tags=["work"]))
        block = run(recall_context(deps, "", 2000, tag="work"))
        self.assertIn("work note", block)
        self.assertNotIn("persona fact", block)

    def test_namespace_list_isolates_recall(self):
        # ns:policy memory only surfaces when its namespace is in scope.
        deps = make_deps(self.tmp.name, vectors={
            "refund window is 30 days": [1.0, 0.0, 0.0],
            "marketing tone is upbeat": [0.0, 1.0, 0.0],
        })
        run(create_memory(deps, "refund window is 30 days", tags=["ns:policy"]))
        run(create_memory(deps, "marketing tone is upbeat", tags=["ns:marketing"]))
        scoped = run(recall_context(deps, "", 2000, tag=["ns:policy"]))
        self.assertIn("refund window", scoped)
        self.assertNotIn("marketing tone", scoped)

    def test_namespace_list_with_core_includes_persona(self):
        # Multi-tag filter that includes 'core' brings the persona back in.
        deps = make_deps(self.tmp.name, vectors={
            "About me: persona fact": [1.0, 0.0, 0.0],
            "refund window is 30 days": [0.0, 1.0, 0.0],
        })
        run(create_memory(deps, "About me: persona fact", tags=["core"]))
        run(create_memory(deps, "refund window is 30 days", tags=["ns:policy"]))
        block = run(recall_context(deps, "", 2000, tag=["ns:policy", "core"]))
        self.assertIn("refund window", block)
        self.assertIn("persona fact", block)

    def test_empty_tag_list_means_no_filter(self):
        # An empty/whitespace list normalizes to None and recalls everything.
        deps = make_deps(self.tmp.name, vectors={
            "a work note": [1.0, 0.0, 0.0],
        })
        run(create_memory(deps, "a work note", tags=["work"]))
        block = run(recall_context(deps, "", 2000, tag=[" ", ""]))
        self.assertIn("work note", block)


class TestGlobalSwitch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_paused_recall_returns_empty(self):
        deps = make_deps(self.tmp.name)
        run(create_memory(deps, "About me: I prefer tabs", tags=["core"]))
        # ON: returns the persona; OFF: returns nothing at all.
        self.assertIn("prefer tabs", run(recall_context(deps, "anything", 2000)))
        deps.paused = True
        self.assertEqual(run(recall_context(deps, "anything", 2000)), "")
        self.assertEqual(run(recall_context(deps, "", 2000)), "")  # empty query too


class TestPacking(unittest.TestCase):
    def _row(self, i, content):
        return {"id": i, "created_at": "2026-06-12T00:00:00+00:00",
                "content": content, "summary": "short summary", "tags": []}

    def test_budget_is_respected(self):
        scored = [(self._row(i, "word " * 300), 1.0) for i in range(10)]
        block = _pack(scored, budget=500, header="test")
        self.assertLessEqual(estimate_tokens(block), 600)  # header slack only

    def test_falls_back_to_summary_when_full_text_too_big(self):
        scored = [(self._row(1, "huge " * 2000), 1.0)]
        block = _pack(scored, budget=200, header="test")
        self.assertIn("short summary", block)
        self.assertNotIn("huge huge", block)


class TestEnrichment(unittest.TestCase):
    def test_enrich_upgrades_summary_and_skips_vault_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp)
            memory = run(create_memory(deps, "Some note. " * 20, source="rest"))
            vault_note = deps.store.add("vault authored", excerpt_summary("x"),
                                        source="vault")
            note_path = deps.vault.path / "user-note.md"
            note_path.write_text("vault authored", encoding="utf-8")
            deps.store.set_md_path(vault_note["id"], note_path, 1.0)

            touched = run(enrich_pending(deps, batch=10))
            self.assertGreaterEqual(touched, 2)
            self.assertEqual(
                deps.store.get(memory["id"])["summary"], "This is the AI summary.")
            # daemon-authored file was rewritten with the AI summary...
            self.assertIn("This is the AI summary.",
                          Path(memory["md_path"]).read_text(encoding="utf-8"))
            # ...but the user's vault-authored file was left untouched
            self.assertEqual(note_path.read_text(encoding="utf-8"),
                             "vault authored")


class TestSupersession(unittest.TestCase):
    """A correction invalidates the stale memory it contradicts — but only when
    the contradiction judge confirms it (trust: never drop a still-true fact)."""

    def _seed_pair(self, deps):
        # Stored directly (bypassing create_memory dedup) with similar vectors so
        # the new fact surfaces the old one as a supersede candidate.
        old = deps.store.add("production database is MySQL", "s")
        new = deps.store.add("production database is Postgres, not MySQL", "s")
        for m in (old, new):
            deps.store.replace_chunk_embeddings(m["id"], deps.ollama.embed_key,
                                                [[1.0, 0.0, 0.0]])
        return old, new

    def test_invalidates_contradicting_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp, default=[1.0, 0.0, 0.0], contradiction=True)
            old, new = self._seed_pair(deps)
            self.assertEqual(run(apply_supersession(deps, new)), old["id"])
            self.assertIsNotNone(deps.store.get(old["id"])["invalidated_at"])
            self.assertEqual(deps.store.get(old["id"])["invalidated_by"], new["id"])

    def test_keeps_memory_when_judge_says_no_contradiction(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp, default=[1.0, 0.0, 0.0], contradiction=False)
            old, new = self._seed_pair(deps)
            self.assertIsNone(run(apply_supersession(deps, new)))
            self.assertIsNone(deps.store.get(old["id"])["invalidated_at"])

    def test_does_not_supersede_core_persona_memory(self):
        # The always-on persona layer is user-curated; a background correction
        # must never silently invalidate it, even on a confirmed contradiction.
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp, default=[1.0, 0.0, 0.0], contradiction=True)
            old = deps.store.add("the user prefers MySQL", "s", tags=["core"])
            new = deps.store.add("the user prefers Postgres now", "s")
            for m in (old, new):
                deps.store.replace_chunk_embeddings(m["id"], deps.ollama.embed_key,
                                                    [[1.0, 0.0, 0.0]])
            self.assertIsNone(run(apply_supersession(deps, new)))
            self.assertIsNone(deps.store.get(old["id"])["invalidated_at"])

    def test_no_supersede_when_nothing_similar(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp, contradiction=True)
            # Orthogonal vectors → similarity below the candidate floor.
            old = deps.store.add("I enjoy hiking on weekends", "s")
            new = deps.store.add("the API base url is v2", "s")
            deps.store.replace_chunk_embeddings(old["id"], deps.ollama.embed_key,
                                                [[1.0, 0.0, 0.0]])
            deps.store.replace_chunk_embeddings(new["id"], deps.ollama.embed_key,
                                                [[0.0, 1.0, 0.0]])
            # apply_supersession embeds new's content; default vector is [1,0,0],
            # but the stored new vector is orthogonal to old — the judge is never
            # even consulted for a below-floor candidate.
            deps.ollama.vectors = {new["content"]: [0.0, 1.0, 0.0]}
            self.assertIsNone(run(apply_supersession(deps, new)))


class TestProvenance(unittest.TestCase):
    """source attribution: canonicalized on write, surfaced on recall."""

    def test_normalize_known_tools(self):
        cases = {
            "Claude Code": "claude-code",
            "claude_code": "claude-code",
            "Claude Desktop": "claude-desktop",
            "Cowork (Claude)": "claude-desktop",
            "Cursor": "cursor",
            "gemini-cli": "gemini",
            "Windsurf": "windsurf",
        }
        for raw, want in cases.items():
            self.assertEqual(normalize_source(raw), want, raw)

    def test_normalize_unknown_is_slugified_not_dropped(self):
        self.assertEqual(normalize_source("My Tool 2.0!"), "my-tool-2-0")
        # already-canonical internal slugs survive unchanged
        self.assertEqual(normalize_source("proxy-autocapture"), "proxy-autocapture")
        self.assertEqual(normalize_source(""), "")
        self.assertEqual(normalize_source(None), "")

    def test_create_memory_stores_normalized_deps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp)
            deps.source = "Claude Code"  # as a stdio clientInfo.name would arrive
            memory = run(create_memory(deps, "The user ships on Fridays only."))
            self.assertEqual(deps.store.get(memory["id"])["source"], "claude-code")

    def test_explicit_source_overrides_and_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp)
            deps.source = "cursor"
            memory = run(create_memory(deps, "A durable fact about the project.",
                                       source="Claude Desktop"))
            self.assertEqual(deps.store.get(memory["id"])["source"], "claude-desktop")

    def test_pack_surfaces_source(self):
        row = {"id": 7, "created_at": "2026-06-20T00:00:00+00:00",
               "content": "fact body", "summary": "s", "tags": [],
               "source": "claude-code"}
        block = _pack([(row, 0.9)], budget=2000, header="test")
        self.assertIn("via claude-code", block)
        # absent source must not render a dangling separator
        row_no_src = {**row, "source": ""}
        self.assertNotIn("via ", _pack([(row_no_src, 0.9)], budget=2000, header="t"))


if __name__ == "__main__":
    unittest.main()
