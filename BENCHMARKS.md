# OpenBrain benchmarks

Two questions, two harnesses — both seeded from [`evals/golden.jsonl`](evals/golden.jsonl)
and fully reproducible on your own machine. Numbers below are a **small smoke
benchmark** (10 seed memories, 10 queries), not a public leaderboard score; they
exist to validate the mechanism and catch regressions. See *Caveats* before
quoting them.

## 1. Retrieval — do the right memories rank first?

Seeds a fresh store, runs each query through the hybrid recall pipeline (FTS +
semantic, RRF-fused, MMR-diversified), and scores hit@k + MRR.

```bash
python -m evals.run            # deterministic hash-BoW embedder (CI baseline)
python -m evals.run --ollama   # real embeddings (nomic-embed-text)
```

| Embedder | hit@1 | hit@3 | hit@5 | MRR |
|----------|:-----:|:-----:|:-----:|:---:|
| hash-BoW (deterministic, CI) | 50% | 60% | 60% | 0.550 |
| **real `nomic-embed-text`**  | **90%** | **100%** | **100%** | **0.950** |

The hash-BoW row is an intentional **floor** — a bag-of-words stand-in that keeps
CI repeatable without Ollama. It only tests pipeline wiring (fusion, namespace
scoping, budgeting). Real embeddings are the number that reflects recall quality.

## 2. Lift — does injecting memory actually improve answers?

For each query, answer twice with a real model — **treatment** (memory injected)
vs **control** (no memory, the holdout) — then an LLM judge grades each answer
against the expected fact. Headline = **lift = treatment − control**.

```bash
python -m evals.holdout        # needs Ollama running
```

| Arm | Correct |
|-----|:-------:|
| control (no memory) | 0% |
| **treatment (memory)** | **100%** |
| **lift** | **+100%** (helped 10, hurt 0) |

A 0% control is expected and is the point: the golden set is *private* facts
(your refund policy, your DB choice, a person's birthday) that a base model
cannot know. Memory takes the model from "can't answer" to "answers correctly,"
with zero regressions (`hurt 0`).

## Environment (for the numbers above)

- OpenBrain 0.6.0, run 2026-06-20
- Ollama 0.30.8 · embeddings `nomic-embed-text` (137M) · chat/judge `qwen3.5:4b`
- Dataset: `evals/golden.jsonl` — 10 seed memories, 10 queries across policy /
  dev / personal namespaces, incl. one multi-hop ("refund and return together")
  and a noise memory that should never surface

## Caveats (read before quoting)

- **Small N (10).** This is a smoke benchmark, not a statistical result. It
  catches regressions and demonstrates the mechanism; it is not a LoCoMo /
  LongMemEval leaderboard score.
- **The set favors memory by construction** — private facts a base model can't
  know, so control is near-0 and lift is near-maximal. That is the honest,
  intended demonstration of *what memory is for*, not a claim that OpenBrain adds
  +100% on arbitrary tasks.
- **Generation isn't deterministic**, so the lift harness is opt-in, not a CI
  gate; the retrieval harness (hash-BoW) is the deterministic CI signal.

## Roadmap

- A proper public benchmark on a **LoCoMo / LongMemEval** subset (multi-session,
  temporal, multi-hop, adversarial no-answer cases) with real embeddings.
- A committed baseline + non-zero exit on regression for `evals.run`.
- An online random-holdout in the proxy graded from a real user-feedback signal.
