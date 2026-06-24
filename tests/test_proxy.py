"""Memory-injecting proxy tests: extraction, injection, and fail-open."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from myagent.memory_service import create_memory
from myagent.proxy import (
    _GATE_MARKER,
    _already_injected,
    _capture,
    _compose_menu,
    _draft_gate_menu,
    _find_gate,
    _has_opt_out,
    _inject,
    _latest_user_text,
    _maybe_inject,
    _parse_gate_choice,
    _prior_assistant_text,
    _resolve_gate_answer,
    _strip_opt_out,
)

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


class TestExtraction(unittest.TestCase):
    def test_latest_user_text_string(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"}]
        self.assertEqual(_latest_user_text(msgs), "second")

    def test_latest_user_text_parts(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "world"}]}]
        self.assertEqual(_latest_user_text(msgs), "hello world")

    def test_no_user_message(self):
        self.assertEqual(_latest_user_text([{"role": "system", "content": "x"}]), "")

    def test_prior_assistant_text_returns_last_real_reply(self):
        msgs = [{"role": "user", "content": "q1"},
                {"role": "assistant", "content": "claim A"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "claim B"},
                {"role": "user", "content": "no, that's wrong"}]
        self.assertEqual(_prior_assistant_text(msgs), "claim B")

    def test_prior_assistant_text_skips_gate_menu(self):
        msgs = [{"role": "assistant", "content": "real claim"},
                {"role": "assistant", "content": f"{_GATE_MARKER} menu"},
                {"role": "user", "content": "1"}]
        self.assertEqual(_prior_assistant_text(msgs), "real claim")

    def test_prior_assistant_text_empty_on_first_turn(self):
        self.assertEqual(_prior_assistant_text([{"role": "user", "content": "hi"}]), "")


class TestInjection(unittest.TestCase):
    def test_augments_existing_system_message(self):
        msgs = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}]
        out = _inject(msgs, "MEMORY BLOCK")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("You are helpful.", out[0]["content"])
        self.assertIn("MEMORY BLOCK", out[0]["content"])
        self.assertIn("<openbrain-memory>", out[0]["content"])
        # original list untouched (new list returned)
        self.assertEqual(msgs[0]["content"], "You are helpful.")

    def test_inserts_system_when_absent(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _inject(msgs, "MEMORY BLOCK")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("MEMORY BLOCK", out[0]["content"])

    def test_already_injected_detects_marker(self):
        self.assertTrue(_already_injected(
            [{"role": "system", "content": "x <openbrain-memory> y"}]))
        self.assertFalse(_already_injected(
            [{"role": "user", "content": "plain"}]))


class TestMaybeInject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _deps_with_memory(self):
        # Query and stored doc embed to the same vector -> cosine 1.0, clears
        # both the 0.5 similarity floor and the 0.6 proxy relevance floor.
        deps = make_deps(self.tmp.name, vectors={
            "user prefers tabs over spaces": [1.0, 0.0, 0.0],
            "what indentation should I use": [1.0, 0.0, 0.0],
        })
        run(create_memory(deps, "user prefers tabs over spaces"))
        return deps

    def test_injects_relevant_memory(self):
        deps = self._deps_with_memory()
        body = {"model": "gpt", "messages": [
            {"role": "user", "content": "what indentation should I use"}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        self.assertEqual(out["messages"][0]["role"], "system")
        self.assertIn("tabs over spaces", out["messages"][0]["content"])

    def test_skips_when_already_injected(self):
        deps = self._deps_with_memory()
        body = {"messages": [
            {"role": "system", "content": "<openbrain-memory>old</openbrain-memory>"},
            {"role": "user", "content": "what indentation should I use"}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        self.assertEqual(out, body)  # untouched

    def test_skips_when_no_user_text(self):
        deps = self._deps_with_memory()
        body = {"messages": [{"role": "system", "content": "sys only"}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        self.assertEqual(out, body)

    def test_fails_open_on_irrelevant_query(self):
        deps = make_deps(self.tmp.name, vectors={
            "unrelated stored fact": [0.0, 1.0, 0.0],
            "completely different question": [1.0, 0.0, 0.0],
        })
        run(create_memory(deps, "unrelated stored fact"))
        body = {"messages": [
            {"role": "user", "content": "completely different question"}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        self.assertEqual(out, body)  # cos 0 < floor -> nothing injected

    def test_opt_out_skips_injection_and_strips_flag(self):
        deps = self._deps_with_memory()
        body = {"messages": [
            {"role": "user",
             "content": "what indentation should I use --no-memory"}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        joined = str(out["messages"])
        self.assertNotIn("<openbrain-memory>", joined)   # not injected
        self.assertNotIn("--no-memory", joined)          # flag stripped
        self.assertIn("what indentation should I use", out["messages"][-1]["content"])

    def test_opt_out_in_parts_content(self):
        deps = self._deps_with_memory()
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what indentation should I use #nomem"}]}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        joined = str(out["messages"])
        self.assertNotIn("<openbrain-memory>", joined)
        self.assertNotIn("#nomem", joined)

    def test_opt_out_not_triggered_by_substring(self):
        # Regression: the bare #nomem/​/nomem tokens need word boundaries, or
        # ordinary paths/words silently disable memory.
        self.assertFalse(_has_opt_out("see /var/nomemory/config for details"))
        self.assertFalse(_has_opt_out("the xnomem channel"))
        self.assertTrue(_has_opt_out("--no-memory"))
        self.assertTrue(_has_opt_out("please #nomem thanks"))
        self.assertTrue(_has_opt_out("answer #nomemory"))

    def test_strip_opt_out_tolerates_part_without_text_key(self):
        # Regression: a text part missing the "text" field must not raise
        # KeyError (which would be swallowed, leaking the flag upstream).
        msgs = [{"role": "user", "content": [
            {"type": "text"},  # no "text" key
            {"type": "text", "text": "hello --no-memory"}]}]
        out = _strip_opt_out(msgs)
        joined = str(out)
        self.assertNotIn("--no-memory", joined)
        self.assertIn("hello", joined)

    def test_fails_open_on_bad_body(self):
        deps = self._deps_with_memory()
        body = {"no_messages_key": True}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        self.assertEqual(out, body)

    def test_fails_open_when_ollama_down(self):
        deps = make_deps(self.tmp.name, down=True)
        run(create_memory(deps, "some fact"))  # stored without embedding
        body = {"messages": [{"role": "user", "content": "any query"}]}
        out = run(_maybe_inject(deps, body, 0.6, 1000))
        # recall degrades to keyword-only; "any query" matches nothing -> no inject
        self.assertNotIn("<openbrain-memory>",
                         str(out["messages"][0].get("content", "")))


class TestPreflightGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _deps(self):
        deps = make_deps(self.tmp.name, vectors={
            "About me: I prefer tabs": [1.0, 0.0, 0.0],
            "what indentation should I use": [1.0, 0.0, 0.0],
        })
        run(create_memory(deps, "About me: I prefer tabs"))
        return deps

    def test_find_gate_and_parse_choice(self):
        msgs = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": f"{_GATE_MARKER} menu"},
                {"role": "user", "content": "1"}]
        self.assertEqual(_find_gate(msgs), 1)
        self.assertEqual(_find_gate([{"role": "user", "content": "hi"}]), -1)
        self.assertEqual(_parse_gate_choice("skip"), "skip")
        self.assertEqual(_parse_gate_choice("2"), "skip")
        self.assertEqual(_parse_gate_choice("1"), "keep")
        self.assertEqual(_parse_gate_choice("use my memory"), "keep")
        self.assertEqual(_parse_gate_choice("banana"), "other")

    def test_resolve_keep_injects_for_original_question(self):
        deps = self._deps()
        msgs = [{"role": "user", "content": "what indentation should I use"},
                {"role": "assistant", "content": f"{_GATE_MARKER} menu"},
                {"role": "user", "content": "1"}]
        frag, injected = run(_resolve_gate_answer(deps, msgs, 1, 0.6, 1000))
        self.assertTrue(injected)
        # menu + choice dropped; system memory prepended to the original question
        self.assertEqual(frag["messages"][0]["role"], "system")
        self.assertIn("I prefer tabs", frag["messages"][0]["content"])
        self.assertEqual(frag["messages"][-1]["content"],
                         "what indentation should I use")
        self.assertNotIn(_GATE_MARKER, str(frag["messages"]))

    def test_compose_menu_keeps_parseable_options(self):
        menu = _compose_menu("Tailored: you use Postgres — want me to apply that?")
        self.assertIn(_GATE_MARKER, menu)
        self.assertIn("Tailored: you use Postgres", menu)        # dynamic intro
        self.assertIn("Use my OpenBrain memory", menu)            # fixed options intact
        self.assertEqual(_parse_gate_choice("2"), "skip")         # parsing unaffected

    def test_draft_gate_menu_uses_dynamic_intro(self):
        deps = make_deps(self.tmp.name,
                         gate_intro="You prefer tabs; should I apply that here?")
        run(create_memory(deps, "About me: I prefer tabs"))
        menu = run(_draft_gate_menu(deps, "what indentation?", 1000, 0.0))
        self.assertIn("You prefer tabs; should I apply that here?", menu)
        self.assertIn(_GATE_MARKER, menu)

    def test_draft_gate_menu_falls_back_when_no_draft(self):
        deps = make_deps(self.tmp.name, gate_intro=None)  # draft unavailable
        run(create_memory(deps, "About me: I prefer tabs"))
        menu = run(_draft_gate_menu(deps, "what indentation?", 1000, 0.0))
        self.assertIn("might* apply here", menu)  # static fallback intro
        self.assertIn(_GATE_MARKER, menu)

    def test_resolve_skip_does_not_inject(self):
        deps = self._deps()
        msgs = [{"role": "user", "content": "what indentation should I use"},
                {"role": "assistant", "content": f"{_GATE_MARKER} menu"},
                {"role": "user", "content": "skip"}]
        frag, injected = run(_resolve_gate_answer(deps, msgs, 1, 0.6, 1000))
        self.assertFalse(injected)
        self.assertNotIn("<openbrain-memory>", str(frag["messages"]))
        self.assertEqual(frag["messages"], [msgs[0]])  # only the original question


class TestAutoCapture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_stores_when_judge_finds_durable_fact(self):
        deps = make_deps(self.tmp.name,
                         judge_summary="The user prefers tabs over spaces.")
        run(_capture(deps, "btw I always use tabs, never spaces"))
        self.assertEqual(deps.store.count(), 1)
        self.assertIn("auto", deps.store.recent(1)[0]["tags"])

    def test_skips_when_judge_says_no(self):
        deps = make_deps(self.tmp.name, judge_summary=None)  # judge returns NO
        run(_capture(deps, "how do I reverse a string in python"))
        self.assertEqual(deps.store.count(), 0)

    def test_fails_closed_when_ollama_down(self):
        deps = make_deps(self.tmp.name, down=True,
                         judge_summary="would-be summary")
        run(_capture(deps, "anything"))
        self.assertEqual(deps.store.count(), 0)

    def test_low_value_summary_still_guarded(self):
        # Even if the judge returns something, the write guard rejects junk.
        deps = make_deps(self.tmp.name, judge_summary="ok")
        run(_capture(deps, "trigger"))
        self.assertEqual(deps.store.count(), 0)

    def test_correction_is_captured_against_prior_assistant(self):
        deps = make_deps(
            self.tmp.name,
            judge_summary=None,  # standalone durable judge would say NO
            correction_summary="The project uses Postgres, not MySQL.")
        run(_capture(deps, "no, we use Postgres not MySQL",
                     prior_assistant="Your stack uses MySQL for storage."))
        self.assertEqual(deps.store.count(), 1)
        stored = deps.store.recent(1)[0]
        self.assertIn("correction", stored["tags"])
        self.assertEqual(stored["source"], "proxy-correction")

    def test_no_prior_assistant_falls_back_to_durable(self):
        # First turn (no assistant claim yet): correction path can't fire, so the
        # standalone durable judge decides.
        deps = make_deps(self.tmp.name,
                         judge_summary="The user prefers Postgres.",
                         correction_summary="should not be used")
        run(_capture(deps, "I prefer Postgres", prior_assistant=""))
        self.assertEqual(deps.store.count(), 1)
        self.assertNotIn("correction", deps.store.recent(1)[0]["tags"])

    def test_non_correction_reply_falls_back_to_durable(self):
        # Prior assistant exists but the reply isn't a correction → durable path.
        deps = make_deps(self.tmp.name,
                         judge_summary="The user's launch is on Friday.",
                         correction_summary=None)
        run(_capture(deps, "my launch is on Friday",
                     prior_assistant="Here is how to deploy."))
        self.assertEqual(deps.store.count(), 1)
        self.assertNotIn("correction", deps.store.recent(1)[0]["tags"])

    def test_correction_supersedes_the_stale_memory(self):
        # End-to-end: an old fact is stored, then the user corrects it; the
        # correction is captured AND the stale memory is invalidated (not piled
        # next to the new one). Vectors are similar-but-distinct so dedup doesn't
        # fire yet the old fact still surfaces as a supersede candidate.
        corrected = "The project uses Postgres, not MySQL."
        vecs = {"production database is MySQL": [1.0, 0.0, 0.0],
                corrected: [0.8, 0.6, 0.0]}  # cosine 0.8: not a dup, but related
        deps = make_deps(self.tmp.name, vectors=vecs, default=[0.0, 0.0, 1.0],
                         correction_summary=corrected, contradiction=True)
        old = run(create_memory(deps, "production database is MySQL",
                                source="cursor"))
        run(_capture(deps, "no, we use Postgres not MySQL",
                     prior_assistant="Your production database is MySQL."))
        # New correction stored, old fact invalidated and gone from recall.
        self.assertIsNotNone(deps.store.get(old["id"])["invalidated_at"])
        live_ids = [r["id"] for r in deps.store.recent(10)]
        self.assertNotIn(old["id"], live_ids)
        self.assertTrue(any("correction" in deps.store.get(i)["tags"]
                            for i in live_ids))


if __name__ == "__main__":
    unittest.main()
