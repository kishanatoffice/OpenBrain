"""Unit tests for the retrieval math."""

from __future__ import annotations

import unittest
from array import array
from datetime import datetime, timedelta, timezone

from myagent.search import (
    cosine_similarity,
    decay_weight,
    estimate_tokens,
    mmr_order,
    pool_chunk_similarities,
    reciprocal_rank_fusion_scored,
)


def vec(*values: float) -> array:
    return array("f", values)


class TestCosine(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(cosine_similarity(vec(1, 2, 3), vec(1, 2, 3)), 1.0,
                               places=5)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(cosine_similarity(vec(1, 0), vec(0, 1)), 0.0)

    def test_zero_vector_is_safe(self):
        self.assertEqual(cosine_similarity(vec(0, 0), vec(1, 1)), 0.0)


class TestTokens(unittest.TestCase):
    def test_estimate_is_chars_over_four(self):
        self.assertEqual(estimate_tokens("x" * 400), 101)
        self.assertEqual(estimate_tokens(""), 1)


class TestRRF(unittest.TestCase):
    def test_best_item_normalizes_to_one(self):
        fused = reciprocal_rank_fusion_scored([[1, 2], [1, 3]], 10)
        self.assertEqual(fused[0][0], 1)  # appears top of both lists
        self.assertAlmostEqual(fused[0][1], 1.0)
        self.assertTrue(all(score <= 1.0 for _, score in fused))

    def test_empty_rankings(self):
        self.assertEqual(reciprocal_rank_fusion_scored([[], []], 5), [])


class TestMMR(unittest.TestCase):
    def test_duplicate_vector_is_demoted(self):
        # a and b are identical vectors; c is orthogonal but slightly less
        # relevant. MMR must pick c second because b adds nothing new.
        vectors = {1: vec(1, 0), 2: vec(1, 0), 3: vec(0, 1)}
        candidates = [(1, 1.0), (2, 0.95), (3, 0.90)]
        ordered = [m for m, _ in mmr_order(candidates, vectors, lam=0.7)]
        self.assertEqual(ordered[0], 1)
        self.assertEqual(ordered[1], 3)
        self.assertEqual(ordered[2], 2)

    def test_missing_vectors_are_treated_as_novel(self):
        ordered = mmr_order([(1, 1.0), (2, 0.5)], vectors={})
        self.assertEqual([m for m, _ in ordered], [1, 2])


class TestDecay(unittest.TestCase):
    def test_half_life(self):
        created = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self.assertAlmostEqual(decay_weight(created, 30), 0.5, places=2)

    def test_disabled(self):
        created = (datetime.now(timezone.utc) - timedelta(days=3650)).isoformat()
        self.assertEqual(decay_weight(created, 0), 1.0)

    def test_naive_timestamp_does_not_crash(self):
        # Regression: a tz-naive created_at (e.g. an imported row) must be
        # treated as UTC, not raise "can't subtract offset-naive and aware".
        w = decay_weight("2020-01-01T00:00:00", 30.0)
        self.assertGreater(w, 0.0)
        self.assertLessEqual(w, 1.0)


class TestChunkPooling(unittest.TestCase):
    def test_max_pool_picks_best_chunk_per_memory(self):
        chunks = [
            (1, 0, vec(1, 0)),   # memory 1 chunk 0: cos 1.0 with query
            (1, 1, vec(0, 1)),   # memory 1 chunk 1: cos 0.0
            (2, 0, vec(0.6, 0.8)),
        ]
        pooled = pool_chunk_similarities([1.0, 0.0], chunks, 10)
        self.assertEqual(pooled[0][0], 1)
        self.assertAlmostEqual(pooled[0][1], 1.0, places=5)
        self.assertAlmostEqual(pooled[1][1], 0.6, places=5)

if __name__ == "__main__":
    unittest.main()
