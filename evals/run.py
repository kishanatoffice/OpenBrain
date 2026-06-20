"""Tiny retrieval eval — seeds memories, runs queries, scores hit@k + MRR.

Runs offline against a deterministic bag-of-words embedder so the result is
repeatable in CI and doesn't need Ollama. The goal is regression detection
on the retrieval pipeline (ranking, fusion, namespace scoping, budget) — not
absolute semantic quality.

Usage:  .venv/bin/python -m evals.run [path/to/golden.jsonl]
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from myagent.db import MemoryStore
from myagent.memory_service import Deps, create_memory, recall_context
from myagent.ollama import OllamaError
from myagent.vault import Vault

EMBED_DIM = 64
ID_PATTERN = re.compile(r"\[memory #(\d+)")
K_VALUES = (1, 3, 5)
WORD_RE = re.compile(r"[a-z0-9]+")


class HashBoWOllama:
    """Deterministic bag-of-words embedder. Words hash into EMBED_DIM buckets;
    shared vocabulary → high cosine. Good enough for retrieval-pipeline
    regression tests; not a substitute for a real model.
    """
    embed_model = "hash-bow"
    embed_key = "hash-bow"
    model = "hash-bow"

    async def embed(self, text: str, kind: str = "query") -> list[float]:
        return (await self.embed_many([text], kind))[0]

    async def embed_many(self, texts: list[str],
                         kind: str = "document") -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for word in WORD_RE.findall(text.lower()):
            h = int(hashlib.blake2b(word.encode(), digest_size=4).hexdigest(), 16)
            vec[h % EMBED_DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm > 0 else vec

    async def judge_durable(self, text: str) -> str | None:
        return None  # skip the durability gate during eval

    async def summarize(self, text: str) -> tuple[str, int]:
        return (text[:120], 1)

    async def is_reachable(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


def _hit_at(ranked: list[int], expected: set[int], k: int) -> int:
    return 1 if any(r in expected for r in ranked[:k]) else 0


def _mrr(ranked: list[int], expected: set[int]) -> float:
    for i, mid in enumerate(ranked, 1):
        if mid in expected:
            return 1.0 / i
    return 0.0


async def _seed(deps: Deps, rows: list[dict]) -> dict[str, int]:
    """Insert all seed rows, return alias → memory_id map."""
    aliases: dict[str, int] = {}
    for r in rows:
        result = await create_memory(deps, r["content"], tags=r.get("tags") or [],
                                     source="eval")
        if result.get("id") is None:
            raise RuntimeError(f"seed insert failed for alias={r['alias']!r}: {result}")
        aliases[r["alias"]] = result["id"]
    return aliases


async def _run(golden_path: Path) -> int:
    rows = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    seeds = [r for r in rows if r["type"] == "seed"]
    queries = [r for r in rows if r["type"] == "query"]
    if not queries:
        print("no queries in golden file", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        deps = Deps(
            store=MemoryStore(Path(tmp) / "memories.db"),
            vault=Vault(Path(tmp) / "vault"),
            ollama=HashBoWOllama(),
            min_similarity=0.05,  # bag-of-words cosines are lower than real embeds
        )
        aliases = await _seed(deps, seeds)

        per_k = {k: 0 for k in K_VALUES}
        mrr_sum = 0.0
        failures: list[tuple[str, list[str], list[int]]] = []

        for q in queries:
            expected_ids = {aliases[a] for a in q["expect"]}
            tag_filter: str | list[str] | None = None
            if q.get("ns"):
                tag_filter = [f"ns:{q['ns']}"]
            try:
                block = await recall_context(deps, q["q"], 2000, tag_filter, 0.0)
            except OllamaError as e:
                print(f"recall_context errored on {q['q']!r}: {e}", file=sys.stderr)
                return 2
            ranked = [int(m) for m in ID_PATTERN.findall(block)]
            for k in K_VALUES:
                per_k[k] += _hit_at(ranked, expected_ids, k)
            mrr_sum += _mrr(ranked, expected_ids)
            if not _hit_at(ranked, expected_ids, max(K_VALUES)):
                failures.append((q["q"], q["expect"], ranked))

    n = len(queries)
    print(f"\nOpenBrain retrieval eval — {n} queries against {len(seeds)} seed memories\n")
    print(f"  hit@1  {per_k[1]:>3}/{n}  ({per_k[1] / n:.0%})")
    print(f"  hit@3  {per_k[3]:>3}/{n}  ({per_k[3] / n:.0%})")
    print(f"  hit@5  {per_k[5]:>3}/{n}  ({per_k[5] / n:.0%})")
    print(f"  MRR    {mrr_sum / n:.3f}")

    if failures:
        print(f"\n{len(failures)} miss(es) at k={max(K_VALUES)}:")
        for q, expect, ranked in failures:
            print(f"  {q!r}  expected={expect}  ranked={ranked or '[]'}")

    # v1: print only, always exit 0. Wire a baseline comparison once the
    # steady-state numbers stabilize (and once a real-embedding mode lands).
    return 0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent / "golden.jsonl"
    return asyncio.run(_run(path))


if __name__ == "__main__":
    sys.exit(main())
