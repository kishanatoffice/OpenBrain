"""The retrieval mathematics: how unbounded memory fits a bounded window.

A context window is a token budget B. Selection works in four steps:

  1. Fuse   — lexical (BM25) and semantic (cosine) rankings combined with
              Reciprocal Rank Fusion: RRF(d) = Σ 1/(k + rank_i(d)), k = 60.
  2. Diversify — Maximal Marginal Relevance re-ordering so the budget is not
              spent on near-duplicates:
              pick argmax_d [ λ·rel(d) − (1−λ)·max_{s∈selected} cos(d, s) ].
  3. Decay  — optional forgetting curve w(Δt) = 2^(−Δt / half_life); off by
              default because old facts are not less true.
  4. Pack   — greedy knapsack: maximize Σ rel subject to Σ tokens ≤ B, trying
              each memory's full text first, then its summary, else skipping.

Pure Python on array('f'): at this scale (thousands of memories, 768-dim
vectors) a full scan is milliseconds, so no vector index or numpy yet.
"""

from __future__ import annotations

import math
from array import array
from datetime import datetime, timezone

MMR_LAMBDA = 0.7
RRF_K = 60


def estimate_tokens(text: str) -> int:
    """tokens ≈ ⌈chars / 4⌉ — the standard English heuristic; close enough
    for budgeting and dependency-free."""
    return len(text) // 4 + 1


def cosine_similarity(a: array | list[float], b: array | list[float]) -> float:
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def pool_chunk_similarities(
    query_vector: list[float],
    chunk_embeddings: list[tuple[int, int, array]],
    limit: int,
) -> list[tuple[int, float, array]]:
    """Max-pool chunk cosines per memory: sim(q, M) = max over chunks of M.

    Returns [(memory_id, best_cosine, best_chunk_vector)] sorted best first;
    the winning chunk's vector is what MMR uses for redundancy comparisons.
    """
    best: dict[int, tuple[float, array]] = {}
    for memory_id, _index, vec in chunk_embeddings:
        if len(vec) != len(query_vector):
            continue
        cos = cosine_similarity(query_vector, vec)
        if memory_id not in best or cos > best[memory_id][0]:
            best[memory_id] = (cos, vec)
    ranked = sorted(
        ((mid, cos, vec) for mid, (cos, vec) in best.items()),
        key=lambda t: -t[1],
    )
    return ranked[:limit]


def reciprocal_rank_fusion_scored(
    rankings: list[list[int]], limit: int, k: int = RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked id lists; returns [(id, score)] with scores normalized so
    the best item is 1.0. Rank-based, so BM25 and cosine need no calibration."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, memory_id in enumerate(ranking):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank + 1)
    if not scores:
        return []
    top = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    best = top[0][1]
    return [(memory_id, score / best) for memory_id, score in top]


def mmr_order(
    candidates: list[tuple[int, float]],
    vectors: dict[int, array],
    lam: float = MMR_LAMBDA,
) -> list[tuple[int, float]]:
    """Re-order (id, relevance) pairs by Maximal Marginal Relevance.

    Candidates without a vector contribute no redundancy signal (treated as
    novel). O(n²·dim) — n ≤ ~30 here, so well under a millisecond per query.
    """
    remaining = dict(candidates)
    selected: list[tuple[int, float]] = []
    while remaining:
        best_id, best_score = None, -math.inf
        for memory_id, rel in remaining.items():
            vec = vectors.get(memory_id)
            redundancy = 0.0
            if vec is not None:
                redundancy = max(
                    (cosine_similarity(vec, vectors[s])
                     for s, _ in selected if s in vectors),
                    default=0.0,
                )
            score = lam * rel - (1.0 - lam) * redundancy
            if score > best_score:
                best_id, best_score = memory_id, score
        selected.append((best_id, remaining.pop(best_id)))
    return selected


def decay_weight(created_at_iso: str, half_life_days: float) -> float:
    """Forgetting curve w(Δt) = 2^(−Δt/h). half_life_days <= 0 disables it."""
    if half_life_days <= 0:
        return 1.0
    created = datetime.fromisoformat(created_at_iso)
    age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86_400
    return 2.0 ** (-max(age_days, 0.0) / half_life_days)
