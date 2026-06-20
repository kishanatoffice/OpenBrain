# Evals

Two complementary harnesses, both seeded from [golden.jsonl](golden.jsonl):

| Eval | Question it answers | Needs Ollama? | CI gate? |
|------|---------------------|---------------|----------|
| `evals.run` | Do the **right memories rank highly**? (retrieval) | No | Yes (fast, deterministic) |
| `evals.holdout` | Does injecting memory **improve the answer**? (lift) | Yes | No (opt-in) |

# Retrieval eval

Tiny offline harness that seeds a fresh OpenBrain store, runs a set of
queries through `recall_context`, and scores hit@k + MRR.

## Run

```bash
.venv/bin/python -m evals.run            # deterministic hash-BoW (CI baseline)
.venv/bin/python -m evals.run --ollama   # real embeddings (nomic-embed-text)
```

Default is ~1 second and needs no Ollama. `--ollama` uses your configured
embedder for true semantic numbers. Published results: [../BENCHMARKS.md](../BENCHMARKS.md).

## What the numbers mean

Today's baseline against a **deterministic hash bag-of-words embedder**
(intentional — keeps the eval repeatable in CI without Ollama):

```
hit@1  5/10  (50%)   hit@3  6/10  (60%)   hit@5  6/10  (60%)   MRR  0.550
```

This is a floor, not a ceiling. The hash-BoW embedder is a stand-in to test
**retrieval-pipeline behavior** (fusion, namespace scoping, budget,
relative-cutoff). With real `nomic-embed-text` vectors these scores will be
substantially higher.

What this eval is good at catching:
- a code change that breaks namespace filtering → hit@k drops to 0 on the
  scoped queries
- a change to RRF / MMR / decay that hurts ranking → MRR moves
- a change that breaks token budgeting → empty / truncated blocks

What it is **not** good at:
- judging semantic quality with real embeddings (use a manual sample for that)
- detecting prompt-quality regressions

## Extending

Add rows to [golden.jsonl](golden.jsonl). Two types:

```json
{"type": "seed",  "alias": "policy_refund", "content": "...", "tags": ["ns:policy"]}
{"type": "query", "q": "how long for a refund?", "expect": ["policy_refund"], "ns": "policy",
 "answer": "Refunds are issued within 30 days; after that, store credit only."}
```

- `alias` is a stable handle; `expect` references aliases so memory ids
  don't matter
- `ns` is optional; when set, the query is scoped via the namespace tag
- Multiple expected aliases (`"expect": ["a", "b"]`) score as a hit if *any*
  is in the top-k
- `answer` is optional and only used by the holdout eval — the expected fact a
  correct answer must convey. Query rows without it are retrieval-only.

# Holdout eval — does memory actually help?

`evals.run` proves the right memories *rank*; this proves they *change the
answer*. For each gradeable query (one with an `answer` field) it produces two
answers with a real model — **treatment** (memory injected) and **control** (no
memory, the holdout) — then an LLM judge grades each against the expected fact.

```bash
.venv/bin/python -m evals.holdout
```

The headline number is **lift = treatment accuracy − control accuracy**, plus
how many queries memory *helped* vs *hurt* (a `✓→· REGRESSION` means the model
got it right without memory but wrong with it — exactly what you want to catch
before shipping a recall/persona/injection change). Needs Ollama running; it is
opt-in, not a CI gate, because answer generation isn't deterministic.

This is the paired (within-query) form of a holdout: both arms run on every
query. The online version — randomly skip injection on X% of live proxy turns
and grade later from a user-feedback signal — is the natural follow-on.

## Next

- baseline.json + non-zero exit on regression (retrieval)
- online random-holdout in the proxy + a feedback signal to grade live traffic
- expand to a LoCoMo / LongMemEval subset (~30+ multi-session/temporal/no-answer)
