"""The memory engine: instant writes, budgeted recall, background enrichment.

Phase 2 quality rules:
- Long content is embedded in chunks (~1500 chars, 200 overlap); a memory's
  similarity to a query is the max over its chunks.
- Writes are deduplicated: if the best stored cosine ≥ 0.95, nothing new is
  saved and the existing memory is reported instead.
- Recall applies an absolute cosine floor (config RECALL_MIN_SIMILARITY) so
  rank-based scores can't smuggle irrelevant memories into the context.
- Writes stay instant: excerpt summary + embeddings (~100 ms); the daemon
  upgrades summaries with Ollama in the background. Vault-authored files are
  never rewritten by the daemon — that folder belongs to the user.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import MemoryStore
from .ollama import OllamaClient, OllamaError
from .redact import redact
from .search import (
    decay_weight,
    estimate_tokens,
    mmr_order,
    pool_chunk_similarities,
    reciprocal_rank_fusion_scored,
)
from .vault import Vault

log = logging.getLogger("myagent")

CHUNK_CHARS = 1_500
CHUNK_OVERLAP = 200
MAX_CHUNKS = 64
DUPLICATE_COSINE = 0.95
# Semantic gate: keep a memory only if cos >= max(floor, 0.90 * best_cos).
# Measured on nomic-embed-text+prefixes: true matches 0.56-0.76, junk
# 0.34-0.52, nonsense queries peak ~0.48 — no single absolute floor separates
# them, but floor + relative dominance does.
RELATIVE_CUTOFF = 0.90
CANDIDATE_POOL = 30
DEFAULT_RECALL_TOKENS = 2_000
MIN_RECALL_TOKENS = 100
MAX_RECALL_TOKENS = 20_000
EXPAND_LIMIT = 25  # max ids one expand() call will fetch full bodies for

# The always-on persona layer. Memories tagged CORE_TAG describe who the user
# is and how they like to work; they are injected on EVERY recall regardless of
# how the query scores, so a brand-new chat in any tool already understands the
# user. Relevance-based recall alone never surfaces these — a coding question
# doesn't semantically match "I prefer concise answers".
CORE_TAG = "core"
CORE_LIMIT = 12            # max core memories to inject
CORE_BUDGET_FRACTION = 0.4  # share of the token budget reserved for core


@dataclass
class Deps:
    store: MemoryStore
    vault: Vault
    ollama: OllamaClient
    half_life_days: float = 0.0
    min_similarity: float = 0.45
    source: str = ""
    # Global ON/OFF switch. When paused, all recall/injection returns nothing,
    # so every tool falls back to its own LLM context. Explicit writes still
    # work. Toggled at runtime via /pause and /resume; persisted across restarts.
    paused: bool = False
    # Which connectors are active for this process/request. None means "use the
    # registry defaults" — so a Deps built without explicit state still exposes
    # the built-in tools. The HTTP layer populates this from live settings.
    enabled_connectors: frozenset[str] | None = None
    # OCR connector security envelope (populated from Config). Empty allowlist
    # means the connector can read nothing — safe by default.
    ocr_ingest_dirs: tuple[Path, ...] = ()
    ocr_max_bytes: int = 25 * 1024 * 1024
    ocr_max_expanded_bytes: int = 100 * 1024 * 1024
    ocr_convert_timeout_s: float = 30.0
    ocr_max_image_pixels: int = 40 * 1_000_000


def excerpt_summary(text: str) -> str:
    """First three sentences — the instant stand-in until enrichment runs."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3])[:500]


def chunk_text(text: str, size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding windows with overlap so facts straddling a boundary still
    land whole in at least one chunk."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = size - overlap
    chunks = [text[start:start + size] for start in range(0, len(text), step)]
    # Drop a trailing window fully contained in the previous one.
    if len(chunks) > 1 and len(chunks[-1]) <= overlap:
        chunks.pop()
    return chunks[:MAX_CHUNKS]


# ---- write path ---------------------------------------------------------------

# Meta/test chatter an agent might reflexively try to "remember". Once stored,
# it gets injected into every future prompt, so it's worse than noise. These
# patterns are specific to talking-about-the-memory rather than real facts.
_LOW_VALUE_PATTERNS = re.compile(
    r"(let'?s see what you|let me see (if|whether|what)|is this (being )?stored"
    r"|are you storing|just testing|test(ing)? (the )?memory"
    r"|checking (if|whether|what)|see what you have stored)",
    re.IGNORECASE,
)


def is_low_value(content: str) -> tuple[bool, str]:
    """Heuristic gate for auto-saves (never applied to explicit user writes).
    Returns (low_value, reason)."""
    text = content.strip()
    if len(text) < 15:
        return True, "too short to be a durable fact"
    if len(text.split()) < 4:
        return True, "too few words to be a durable fact"
    if _LOW_VALUE_PATTERNS.search(text):
        return True, "looks like meta/test chatter, not a fact worth keeping"
    return False, ""


# Provenance — the origin tool of a memory. The HTTP transport already receives
# clean slugs (connect.py bakes ?client=<slug> into each tool's URL); this mainly
# tames the free-form clientInfo.name reported over stdio, so "Claude Code",
# "claude-code", and "claude_code" all facet as one source. Substring match on
# the first hit wins, so order specifics (claude+code) before generics.
_SOURCE_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("claude", "code"), "claude-code"),
    (("claude", "desktop"), "claude-desktop"),
    (("claude", "cowork"), "claude-desktop"),
    (("cursor",), "cursor"),
    (("antigravity",), "antigravity"),
    (("gemini",), "gemini"),
    (("windsurf",), "windsurf"),
    (("codex",), "codex"),
    (("cline",), "cline"),
    (("copilot",), "copilot"),
    (("vscode",), "vscode"),
    (("zed",), "zed"),
)


def normalize_source(raw: str | None) -> str:
    """Canonicalize a provenance label so the same tool, reported in different
    shapes across transports, facets as one origin. Known tools map to a fixed
    slug; unknown sources (and already-canonical internal ones like
    'proxy-autocapture') are slugified, never dropped."""
    s = (raw or "").strip().lower()
    if not s:
        return ""
    for needles, label in _SOURCE_ALIASES:
        if all(n in s for n in needles):
            return label
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:40]


async def create_memory(deps: Deps, content: str,
                        tags: list[str] | None = None,
                        source: str | None = None,
                        force: bool = True) -> dict[str, Any]:
    """Store with excerpt summary + chunk embeddings; returns immediately.
    Near-duplicates (cosine ≥ 0.95) are not stored twice. When force=False
    (automatic agent saves), low-value chatter is skipped rather than stored."""
    if not force:
        low, reason = is_low_value(content)
        if low:
            log.info("create_memory: skipped low-value content (%s)", reason)
            return {"skipped": True, "reason": reason}

    # Privacy boundary: scrub secrets/PII before anything is embedded or stored,
    # on every write (auto and explicit). Applies to imports and judge output too.
    content, redacted = redact(content)
    if redacted:
        log.info("create_memory: redacted %s before storing", sorted(set(redacted)))

    vectors: list[list[float]] | None = None
    warning = None
    try:
        vectors = await deps.ollama.embed_many(chunk_text(content), kind="document")
    except OllamaError as exc:
        warning = f"stored without embedding (will backfill): {exc}"
        log.warning("Embedding failed (will backfill): %s", exc)

    if vectors:
        stored = deps.store.all_chunk_embeddings(deps.ollama.embed_key)
        # Dedup against the BEST match of ANY new chunk (not just the first):
        # two long docs that differ in their opening but match later are dupes.
        best_id, best_cos = None, 0.0
        for vec in vectors:
            pooled = pool_chunk_similarities(vec, stored, 1)
            if pooled and pooled[0][1] > best_cos:
                best_id, best_cos = pooled[0][0], pooled[0][1]
        if best_id is not None and best_cos >= DUPLICATE_COSINE:
            existing = deps.store.get(best_id)
            if existing is not None:
                existing["duplicate"] = True
                existing["similarity"] = round(best_cos, 3)
                return existing

    memory = deps.store.add(content, excerpt_summary(content),
                            tags=tags,
                            source=normalize_source(source or deps.source))
    md_path = deps.vault.write_memory(memory)
    deps.store.set_md_path(memory["id"], md_path, md_path.stat().st_mtime)
    memory["md_path"] = str(md_path)

    if vectors:
        deps.store.replace_chunk_embeddings(memory["id"],
                                            deps.ollama.embed_key, vectors)
    elif warning:
        memory["warning"] = warning
    return memory


# ---- recall path --------------------------------------------------------------


async def top_relevance(deps: Deps, query: str) -> float:
    """Best query-to-memory cosine across all stored chunks, 0.0 if none or if
    embeddings are unavailable. Used by the preflight gate to decide whether
    memory relevance is an obvious match, borderline, or clearly irrelevant."""
    query = (query or "").strip()
    if deps.paused or not query:
        return 0.0
    try:
        query_vector = await deps.ollama.embed(query, kind="query")
    except OllamaError:
        return 0.0
    pooled = pool_chunk_similarities(
        query_vector, deps.store.all_chunk_embeddings(deps.ollama.embed_key), 1)
    return pooled[0][1] if pooled else 0.0


async def semantic_candidates(deps: Deps, query: str,
                              limit: int = 20) -> list[tuple[int, float, Any]]:
    """[(memory_id, cosine, best_chunk_vector)] above the semantic gate."""
    query_vector = await deps.ollama.embed(query, kind="query")
    chunks = deps.store.all_chunk_embeddings(deps.ollama.embed_key)
    pooled = pool_chunk_similarities(query_vector, chunks, limit)
    if not pooled:
        return []
    cutoff = max(deps.min_similarity, pooled[0][1] * RELATIVE_CUTOFF)
    return [(m, cos, vec) for m, cos, vec in pooled if cos >= cutoff]


def _normalize_tag_filter(tag: str | list[str] | None) -> set[str] | None:
    """Coerce a tag/list/None filter into a set of lowercased tags. Empty result
    becomes None (no filter), which keeps the unfiltered fast path."""
    if tag is None:
        return None
    items = [tag] if isinstance(tag, str) else list(tag)
    cleaned = {t.strip().lower() for t in items if t and t.strip()}
    return cleaned or None


async def recall_context(deps: Deps, query: str,
                         max_tokens: int = DEFAULT_RECALL_TOKENS,
                         tag: str | list[str] | None = None,
                         min_relevance: float = 0.0,
                         render=None) -> str:
    """Return a plain-text context block packed under the token budget.

    `render` selects how the query results are formatted: the default `_pack`
    inlines full bodies (used by the always-on hook/proxy path, which is
    one-shot and must inject something complete); `_pack_index` emits compact
    candidate lines for the agent-driven, two-step recall. The persona `core`
    block is always rendered in full regardless — it is the standing promise.

    When the global switch is OFF (deps.paused), returns "" so no memory is
    injected anywhere — every tool falls back to its own context.

    `tag` may be a single tag or a list (OR-matched) — a memory passes the
    filter if any of its tags is in the set. Pass `["ns:policy"]` to scope a
    turn to one namespace; `["ns:policy", "core"]` to include the persona too.

    min_relevance drops hits whose final score (cosine × recency decay) falls
    below the floor. Hybrid recall almost always surfaces *some* candidate, so
    an automatic caller (e.g. a pre-prompt hook firing on every prompt) sets a
    floor to stay silent on prompts with no genuinely relevant memory, rather
    than injecting low-relevance noise."""
    if deps.paused:
        return ""
    render = render or _pack
    budget = max(MIN_RECALL_TOKENS, min(int(max_tokens), MAX_RECALL_TOKENS))
    query = (query or "").strip()
    tag_set = _normalize_tag_filter(tag)

    # Always-on persona layer first (skipped when the caller is filtering and
    # CORE_TAG is not in the filter set — then they want exactly those tags,
    # not the persona).
    core_block, core_ids = "", set()
    if tag_set is None or CORE_TAG in tag_set:
        core_block, core_ids = _core_section(deps, int(budget * CORE_BUDGET_FRACTION))
        budget -= estimate_tokens(core_block) if core_block else 0
        budget = max(MIN_RECALL_TOKENS, budget)

    def _matches(row_tags: list[str]) -> bool:
        return tag_set is None or bool(tag_set.intersection(row_tags))

    def _combine(query_block: str) -> str:
        parts = [p for p in (core_block, query_block) if p]
        return "\n\n".join(parts) if parts else query_block

    if not query:
        rows = [r for r in deps.store.recent(CANDIDATE_POOL)
                if _matches(r["tags"]) and r["id"] not in core_ids]
        scored = [(row, 1.0) for row in rows]
        header = f"Local memory — most recent (of {deps.store.count()} total)"
        # When the brain is empty but core exists, still return the persona.
        if not rows and core_block:
            return core_block
        return _combine(render(scored, budget, header))

    keyword_ids = [r["id"] for r in deps.store.search_keyword(query, 20)]
    vectors: dict[int, Any] = {}
    cosines: dict[int, float] = {}
    semantic_ids: list[int] = []
    try:
        for memory_id, cos, vec in await semantic_candidates(deps, query):
            semantic_ids.append(memory_id)
            vectors[memory_id] = vec
            cosines[memory_id] = cos
    except OllamaError as exc:
        log.warning("Semantic recall unavailable, keyword only: %s", exc)

    fused = reciprocal_rank_fusion_scored([keyword_ids, semantic_ids],
                                          CANDIDATE_POOL)
    if not fused:
        return core_block or f"Local memory — no memories match {query!r}."

    ordered = mmr_order(fused, vectors)
    by_id = deps.store.get_many([m for m, _ in ordered])
    # Shown relevance is the honest cosine where one exists; rank score only
    # for keyword-only hits. Core memories are already in the persona section,
    # so drop them here to avoid showing the same memory twice.
    scored = [
        (by_id[m], cosines.get(m, rel) * decay_weight(by_id[m]["created_at"],
                                                      deps.half_life_days))
        for m, rel in ordered
        if m in by_id and m not in core_ids and not by_id[m].get("archived")
        and _matches(by_id[m]["tags"])
    ]
    if min_relevance > 0:
        scored = [(row, rel) for row, rel in scored if rel >= min_relevance]
    if not scored:
        return core_block or f"Local memory — no memories match {query!r}."
    header = f"Local memory — results for {query!r}"
    if tag_set:
        header += f" (tags: {', '.join(sorted(tag_set))})"
    return _combine(render(scored, budget, header))


def _core_section(deps: Deps, budget: int) -> tuple[str, set[int]]:
    """The always-on persona block and the set of memory ids it used (so the
    relevance pass can skip them). Empty string + empty set when no core
    memories exist."""
    rows = deps.store.by_tag(CORE_TAG, CORE_LIMIT)
    if not rows:
        return "", set()
    scored = [(row, 1.0) for row in rows]
    block = _pack(scored, max(MIN_RECALL_TOKENS, budget),
                  "Who you are — persona & preferences (always loaded)")
    used = {row["id"] for row, _ in scored}
    return block, used


def _pack(scored: list[tuple[dict, float]], budget: int, header: str) -> str:
    """Greedy knapsack: full text if it fits, summary if not, else skip."""
    if not scored:
        return f"{header}: nothing found."
    blocks: list[str] = []
    used = 0
    shown = 0
    for row, rel in scored:
        tags = f" · tags: {', '.join(row['tags'])}" if row.get("tags") else ""
        src = f" · via {row['source']}" if row.get("source") else ""
        tag_line = (f"[memory #{row['id']} · {row['created_at'][:10]} · "
                    f"relevance {rel:.2f}{src}{tags}]")
        for body in (row["content"].strip(), row["summary"].strip()):
            block = f"{tag_line}\n{body}"
            tokens = estimate_tokens(block)
            if used + tokens <= budget:
                blocks.append(block)
                used += tokens
                shown += 1
                break
        if budget - used < 40:  # not even a summary fits anymore
            break
    if not blocks:
        return f"{header}: results exist but none fit a {budget}-token budget."
    title = f"=== {header} · {shown} shown · ≈{used}/{budget} tokens ==="
    return "\n\n".join([title, *blocks])


_INDEX_SUMMARY_CHARS = 140


def _pack_index(scored: list[tuple[dict, float]], budget: int, header: str) -> str:
    """Compact candidate list: one line per memory (id + date + score + a
    one-line summary) instead of full bodies. The agent reads this cheaply, then
    calls `expand` for only the ids it actually needs — the progressive-
    disclosure path that keeps recall from paying for every body up front."""
    if not scored:
        return f"{header}: nothing found."
    lines: list[str] = []
    used = 0
    for row, rel in scored:
        summary = " ".join(row["summary"].split())  # collapse to a single line
        if len(summary) > _INDEX_SUMMARY_CHARS:
            summary = summary[:_INDEX_SUMMARY_CHARS - 1] + "…"
        line = (f"#{row['id']} · {row['created_at'][:10]} · "
                f"rel {rel:.2f} · {summary}")
        tokens = estimate_tokens(line)
        if used + tokens > budget:
            break
        lines.append(line)
        used += tokens
    if not lines:
        return f"{header}: results exist but none fit a {budget}-token budget."
    title = f"=== {header} · {len(lines)} shown (index) · ≈{used} tokens ==="
    hint = ("→ Call expand(ids=[…]) with the ids you need to read their full "
            "text. Fetch only what's relevant — that's the token saving.")
    return "\n".join([title, *lines, "", hint])


async def recall_index(deps: Deps, query: str,
                       max_tokens: int = DEFAULT_RECALL_TOKENS,
                       tag: str | list[str] | None = None,
                       min_relevance: float = 0.0) -> str:
    """Two-step recall, step one: rank exactly like `recall_context` but render
    the query results as a compact index. The persona `core` block is still
    inlined in full (the standing promise)."""
    return await recall_context(deps, query, max_tokens, tag, min_relevance,
                                render=_pack_index)


def expand_memories(deps: Deps, ids: list[int],
                    max_tokens: int = MAX_RECALL_TOKENS) -> str:
    """Two-step recall, step two: return the full text of the requested ids
    (already redacted at store time), in the order asked, bounded by a token
    budget. Unknown ids are silently skipped."""
    seen: set[int] = set()
    clean: list[int] = []
    for raw in list(ids)[:EXPAND_LIMIT]:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if i not in seen:
            seen.add(i)
            clean.append(i)
    if not clean:
        return "No memory ids given. Pass ids from a recall index, e.g. expand(ids=[12, 34])."
    by_id = deps.store.get_many(clean)
    scored = [(by_id[i], 1.0) for i in clean if i in by_id]
    if not scored:
        return f"No memories found for ids {clean}."
    return _pack(scored, max(MIN_RECALL_TOKENS, min(int(max_tokens), MAX_RECALL_TOKENS)),
                 "Expanded memories")


# ---- background enrichment -----------------------------------------------------


async def enrich_pending(deps: Deps, batch: int = 5) -> int:
    """Upgrade excerpt summaries to AI summaries and backfill embeddings.
    Returns the number of rows touched; 0 means nothing left to do."""
    touched = 0

    for row in deps.store.unsummarized(batch):
        try:
            summary, tokens = await deps.ollama.summarize(row["content"])
        except OllamaError:
            return touched  # Ollama is down; retry on a later tick
        deps.store.mark_summarized(row["id"], summary, tokens)
        # Vault-authored notes belong to the user — never rewrite those files.
        if row.get("md_path") and row.get("source") != "vault":
            fresh = deps.store.get(row["id"])
            try:
                path = deps.vault.write_memory(fresh, path=fresh["md_path"])
                deps.store.set_md_path(row["id"], path, path.stat().st_mtime)
            except OSError as exc:
                log.warning("Vault rewrite failed for memory %s: %s",
                            row["id"], exc)
        touched += 1
        log.info("Enriched memory #%s with AI summary (%d tokens)",
                 row["id"], tokens)

    for row in deps.store.missing_embeddings(deps.ollama.embed_key)[:batch]:
        try:
            vectors = await deps.ollama.embed_many(chunk_text(row["content"]),
                                                   kind="document")
        except OllamaError:
            return touched
        deps.store.replace_chunk_embeddings(row["id"], deps.ollama.embed_key,
                                            vectors)
        touched += 1
        log.info("Embedded memory #%s (%d chunks)", row["id"], len(vectors))

    return touched
