"""Holdout / lift eval — does injecting memory actually improve answers?

`evals.run` measures whether the RIGHT memories rank highly (retrieval). This
measures the thing the user actually cares about: whether grounding the model in
those memories produces BETTER answers. For each gradeable query we answer twice
— TREATMENT (memory injected) and CONTROL (no memory — the holdout) — then an
LLM judge grades each answer against the query's expected fact. The headline
number is LIFT = treatment accuracy − control accuracy.

This is the paired (within-query) form of Headroom's random-holdout idea: rather
than leaving X% of traffic untreated and comparing populations, we run both arms
on every query, which gives a clean lift from a small golden set. An online
random-holdout in the proxy (skip injection on X% of live turns, grade later
from a feedback signal) is the natural follow-on.

Unlike `evals.run` this needs a real generative model, so it is OPT-IN and not a
CI gate. Run it when you change recall, the persona layer, or the injection
prompt and want a data-backed "memory helps" number rather than a vibe.

Usage:  .venv/bin/python -m evals.holdout [path/to/golden.jsonl]
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from myagent.config import load_config
from myagent.db import MemoryStore
from myagent.memory_service import DEFAULT_RECALL_TOKENS, Deps, recall_context
from myagent.ollama import OllamaClient, OllamaError
from myagent.vault import Vault

from .run import _seed  # reuse the seeder so both evals stay in lockstep


async def grade_query(deps: Deps, q_row: dict, budget: int) -> dict:
    """Answer one query in both arms and grade each against its expected fact.

    Returns {q, expected, control_ok, treatment_ok, control, treatment}. Kept
    free of I/O beyond `deps` so it can be unit-tested with a fake client."""
    q = q_row["q"]
    expected = q_row["answer"]
    tag_filter = [f"ns:{q_row['ns']}"] if q_row.get("ns") else None

    block = await recall_context(deps, q, budget, tag_filter, 0.0)
    if "nothing found" in block.lower() or "no memories match" in block.lower():
        block = ""  # nothing recalled → treatment collapses to control

    treatment = await deps.ollama.answer(q, context=block)
    control = await deps.ollama.answer(q, context="")
    return {
        "q": q,
        "expected": expected,
        "treatment_ok": await deps.ollama.judge_answer(q, expected, treatment),
        "control_ok": await deps.ollama.judge_answer(q, expected, control),
        "treatment": treatment,
        "control": control,
    }


def summarize_results(results: list[dict]) -> dict:
    """Aggregate per-query grades into control/treatment accuracy and lift."""
    n = len(results)
    if n == 0:
        return {"n": 0, "control": 0.0, "treatment": 0.0, "lift": 0.0,
                "helped": 0, "hurt": 0}
    control = sum(r["control_ok"] for r in results) / n
    treatment = sum(r["treatment_ok"] for r in results) / n
    helped = sum(1 for r in results if r["treatment_ok"] and not r["control_ok"])
    hurt = sum(1 for r in results if r["control_ok"] and not r["treatment_ok"])
    return {"n": n, "control": control, "treatment": treatment,
            "lift": treatment - control, "helped": helped, "hurt": hurt}


def _gradeable_queries(rows: list[dict]) -> list[dict]:
    """Query rows that carry an expected-answer fact (others are retrieval-only)."""
    return [r for r in rows if r.get("type") == "query" and r.get("answer")]


async def _run(golden_path: Path) -> int:
    rows = [json.loads(line) for line in golden_path.read_text().splitlines()
            if line.strip()]
    seeds = [r for r in rows if r["type"] == "seed"]
    queries = _gradeable_queries(rows)
    if not queries:
        print("no gradeable queries (add an \"answer\" field to query rows)",
              file=sys.stderr)
        return 2

    cfg = load_config()
    ollama = OllamaClient(cfg.ollama_url, cfg.ollama_model, cfg.ollama_embed_model)
    if not await ollama.is_reachable():
        print(f"Ollama not reachable at {cfg.ollama_url} — the holdout eval needs "
              f"a real model. Start Ollama (model {cfg.ollama_model!r}) and retry.",
              file=sys.stderr)
        await ollama.aclose()
        return 2

    try:
        with tempfile.TemporaryDirectory() as tmp:
            deps = Deps(
                store=MemoryStore(Path(tmp) / "memories.db"),
                vault=Vault(Path(tmp) / "vault"),
                ollama=ollama,
                half_life_days=cfg.recall_half_life_days,
                min_similarity=cfg.recall_min_similarity,
            )
            await _seed(deps, seeds)
            try:
                results = [await grade_query(deps, q, DEFAULT_RECALL_TOKENS)
                           for q in queries]
            except OllamaError as e:
                print(f"holdout eval aborted: {e}", file=sys.stderr)
                return 2
    finally:
        await ollama.aclose()

    s = summarize_results(results)
    print(f"\nOpenBrain holdout eval — {s['n']} gradeable queries "
          f"(real Ollama: {cfg.ollama_model})\n")
    print(f"  control   (no memory)   {s['control']:.0%}")
    print(f"  treatment (memory)      {s['treatment']:.0%}")
    print(f"  lift                    {s['lift']:+.0%}  "
          f"(helped {s['helped']}, hurt {s['hurt']})\n")
    for r in results:
        mark = {(True, True): "✓→✓", (False, True): "·→✓",
                (True, False): "✓→· REGRESSION", (False, False): "·→·"}[
            (r["control_ok"], r["treatment_ok"])]
        print(f"  [{mark}] {r['q']}")
    return 0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent / "golden.jsonl"
    return asyncio.run(_run(path))


if __name__ == "__main__":
    sys.exit(main())
