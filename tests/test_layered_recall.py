"""Layered recall: compact index (step 1) + expand by id (step 2).

The token win is the reason this exists, so the headline test asserts the index
is materially smaller than the full pack over the same memories.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from myagent import memory_service as ms
from myagent.mcp import call_tool
from myagent.memory_service import (
    estimate_tokens,
    expand_memories,
    recall_context,
    recall_index,
)

from .helpers import make_deps


def run(coro):
    return asyncio.run(coro)


_BODY = ("This is a substantial memory body number {n}. It records a durable "
         "decision with enough prose that its full text costs many more tokens "
         "than a one-line summary would. Lorem ipsum dolor sit amet, consectetur "
         "adipiscing elit, sed do eiusmod tempor incididunt ut labore.")


class LayeredRecallCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # down=True → no embeddings, so near-identical bodies aren't deduped into
        # one. Empty-query recall uses the recent path (no embeddings needed).
        self.deps = make_deps(self.tmp.name, down=True)
        self.ids = []
        for n in range(8):
            m = run(ms.create_memory(self.deps, _BODY.format(n=n), tags=["work"]))
            self.ids.append(m["id"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_is_smaller_than_full(self):
        full = run(recall_context(self.deps, "", 4000))
        index = run(recall_index(self.deps, "", 4000))
        self.assertLess(estimate_tokens(index), estimate_tokens(full))
        # The index must still name the memories and tell the model how to expand.
        self.assertIn("(index)", index)
        self.assertIn("expand(ids=", index)

    def test_expand_returns_full_bodies_in_order(self):
        out = expand_memories(self.deps, [self.ids[2], self.ids[0]])
        self.assertIn("Expanded memories", out)
        self.assertIn(f"memory #{self.ids[2]}", out)
        self.assertIn("substantial memory body number 2", out)
        # Requested order is preserved (#2 before #0).
        self.assertLess(out.index(f"#{self.ids[2]}"), out.index(f"#{self.ids[0]}"))

    def test_expand_skips_unknown_and_dedupes(self):
        out = expand_memories(self.deps, [self.ids[0], self.ids[0], 9999])
        self.assertEqual(out.count(f"memory #{self.ids[0]}"), 1)  # deduped
        self.assertNotIn("#9999", out)

    def test_expand_empty_is_friendly(self):
        self.assertIn("No memory ids", expand_memories(self.deps, []))
        self.assertIn("No memory ids", expand_memories(self.deps, ["nope"]))

    def test_expand_respects_limit(self):
        out = expand_memories(self.deps, list(range(1, 100)))  # > EXPAND_LIMIT
        # Never more than EXPAND_LIMIT bodies, and it doesn't error.
        self.assertLessEqual(out.count("· relevance"), ms.EXPAND_LIMIT)


class LayeredRecallDispatchCase(unittest.TestCase):
    """The MCP recall/expand wiring: recall defaults to index, mode=full inlines."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.deps = make_deps(self.tmp.name)
        self.deps.enabled_connectors = frozenset({"memory"})
        self.mid = run(ms.create_memory(
            self.deps, "The deploy key rotation happens every 90 days per policy.",
            tags=["ops"]))["id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_recall_defaults_to_index(self):
        out = run(call_tool(self.deps, "recall", {"query": ""}))
        self.assertIn("(index)", out)
        self.assertIn(f"#{self.mid}", out)

    def test_recall_full_mode_inlines(self):
        out = run(call_tool(self.deps, "recall", {"query": "", "mode": "full"}))
        self.assertNotIn("(index)", out)
        self.assertIn("90 days", out)  # full body present

    def test_expand_tool_roundtrip(self):
        out = run(call_tool(self.deps, "expand", {"ids": [self.mid]}))
        self.assertIn("90 days", out)

    def test_expand_tool_rejects_non_list(self):
        out = run(call_tool(self.deps, "expand", {"ids": "12"}))
        self.assertIn("must be a list", out)


if __name__ == "__main__":
    unittest.main()
