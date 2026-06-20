"""Holdout/lift eval logic — grading wiring and lift math, with a fake model.

The full eval needs a real Ollama (it generates and grades answers); these tests
cover the deterministic parts: which queries are gradeable, that the treatment
arm is fed recalled context while the control arm is not, and that lift
aggregates correctly."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from evals.holdout import _gradeable_queries, grade_query, summarize_results
from myagent.memory_service import create_memory

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


class TestGradeableFilter(unittest.TestCase):
    def test_only_queries_with_answer_are_gradeable(self):
        rows = [
            {"type": "seed", "alias": "a", "content": "x"},
            {"type": "query", "q": "no answer key", "expect": ["a"]},
            {"type": "query", "q": "has key", "expect": ["a"], "answer": "yes"},
        ]
        got = _gradeable_queries(rows)
        self.assertEqual([r["q"] for r in got], ["has key"])


class TestLiftMath(unittest.TestCase):
    def test_summarize_counts_helped_hurt_and_lift(self):
        results = [
            {"control_ok": False, "treatment_ok": True},   # helped
            {"control_ok": False, "treatment_ok": True},   # helped
            {"control_ok": True, "treatment_ok": True},    # neutral
            {"control_ok": True, "treatment_ok": False},   # hurt (regression)
        ]
        s = summarize_results(results)
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["control"], 0.5)       # 2/4
        self.assertEqual(s["treatment"], 0.75)    # 3/4
        self.assertEqual(s["lift"], 0.25)
        self.assertEqual(s["helped"], 2)
        self.assertEqual(s["hurt"], 1)

    def test_empty_results_are_safe(self):
        self.assertEqual(summarize_results([])["lift"], 0.0)


class TestGradeQueryWiring(unittest.TestCase):
    def test_treatment_gets_context_control_does_not(self):
        # FakeOllama: same vector for every text → the seeded memory is recalled,
        # answer() tags the arm 'ctx'/'noctx', and judge_answer() passes only the
        # context-grounded arm — so a recalled memory must produce lift.
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp)
            run(create_memory(deps, "Production uses Postgres 16 on RDS.",
                              source="eval"))
            q_row = {"type": "query", "q": "which database in production?",
                     "answer": "Postgres 16"}
            r = run(grade_query(deps, q_row, budget=2000))
            self.assertTrue(r["treatment_ok"])    # grounded answer graded correct
            self.assertFalse(r["control_ok"])      # bare answer graded wrong
            self.assertIn("grounded", r["treatment"])
            self.assertIn("bare", r["control"])

    def test_no_recall_collapses_treatment_to_control(self):
        # Empty brain → nothing recalled → both arms answer without context, so
        # there is no lift to claim (honest: memory can't help what it lacks).
        with tempfile.TemporaryDirectory() as tmp:
            deps = make_deps(tmp)
            q_row = {"type": "query", "q": "unknowable fact", "answer": "42"}
            r = run(grade_query(deps, q_row, budget=2000))
            self.assertFalse(r["treatment_ok"])
            self.assertFalse(r["control_ok"])


if __name__ == "__main__":
    unittest.main()
