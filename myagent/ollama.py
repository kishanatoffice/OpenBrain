"""Minimal async HTTP client for Ollama: summaries and embeddings only."""

from __future__ import annotations

import re

import httpx

_SYSTEM_PROMPT = (
    "You are a note summarizer. Summarize the user's text in exactly three "
    "short sentences. Reply with the summary only — no preamble, no headings, "
    "no quotes."
)

_JUDGE_SYSTEM = (
    "You decide whether a message a user sent to an AI contains information "
    "worth remembering PERMANENTLY about the user: a durable fact, decision, "
    "preference, plan, deadline, or identity detail. Coding questions, requests "
    "to perform a task, debugging, brainstorming, and one-off queries are NOT "
    "worth remembering. Be strict — when in doubt, answer NO.\n"
    "If it IS worth remembering, reply with ONE concise third-person sentence "
    "stating the fact (e.g. 'The user prefers tabs over spaces.'). "
    "If it is NOT, reply with exactly: NO"
)

_CORRECTION_SYSTEM = (
    "You detect when a user is CORRECTING a factual claim the assistant just "
    "made. You are given the assistant's previous message and the user's reply. "
    "If the user corrects or contradicts a concrete fact — e.g. 'no, we use "
    "Postgres, not MySQL', 'actually the deadline is Friday', 'that's wrong, the "
    "API is v2' — reply with ONE concise third-person sentence stating the "
    "CORRECTED fact (what is now true), e.g. 'The project uses Postgres, not "
    "MySQL.' If the user is NOT correcting a fact (a follow-up question, an "
    "opinion, a new unrelated request, or mere disagreement without a fact), "
    "reply with exactly: NO. Be strict — when in doubt, answer NO."
)

_CONTRADICTION_SYSTEM = (
    "You decide whether a NEW fact directly CONTRADICTS an OLD stored fact about "
    "the user — i.e. they cannot both be true now, so the new one supersedes the "
    "old (e.g. OLD 'uses MySQL' vs NEW 'uses Postgres, not MySQL'; OLD 'lives in "
    "Paris' vs NEW 'moved to Berlin'). Facts that are merely related, additive, "
    "or about different things do NOT contradict (OLD 'likes Python' vs NEW "
    "'likes Go' is NOT a contradiction — both can be true). Reply with exactly "
    "YES if the new fact supersedes the old one, otherwise exactly NO. When in "
    "doubt, answer NO — never invalidate a still-true memory."
)

_GATE_SYSTEM = (
    "A memory assistant is about to ask the user whether to use their saved "
    "context before answering. Given the user's request and the context the "
    "assistant holds about them, write a SHORT preface (1-2 sentences): name "
    "the specific saved thing(s) that could shape the answer, and optionally "
    "one brief clarifying question. Be concrete. Do NOT answer the request, do "
    "NOT list options or say 'reply 1/2' — that is added separately. Plain "
    "text only, no headings."
)

# Answer + grade prompts for the holdout/lift eval (evals.holdout).
_ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the user's question concisely. When a "
    "Context section is provided and relevant, ground your answer in it; if the "
    "context does not cover the question, answer from general knowledge or say "
    "you don't know. Do not invent specifics."
)
_ANSWER_JUDGE_SYSTEM = (
    "You grade whether an ANSWER correctly conveys an EXPECTED FACT for a "
    "question. Reply with exactly YES if the answer states or clearly implies "
    "the expected fact, otherwise exactly NO. Judge only on the expected fact, "
    "ignoring style, verbosity, or extra detail."
)

# Reasoning models (qwen3, deepseek-r1, ...) may emit <think>...</think> blocks.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaError(Exception):
    """Raised when Ollama is unreachable or returns an error."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        embed_model: str = "nomic-embed-text",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.embed_model = embed_model
        # nomic models are trained with asymmetric task prefixes; using them
        # measurably improves retrieval. The distinct storage key makes the
        # enricher transparently re-embed anything stored without prefixes.
        self._prefixed = "nomic" in embed_model
        self.embed_key = (f"{embed_model}+task-prefix" if self._prefixed
                          else embed_model)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def is_reachable(self) -> bool:
        try:
            resp = await self._client.get("/api/version", timeout=3)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def summarize(self, text: str) -> tuple[str, int]:
        """Return (three_sentence_summary, total_tokens_used)."""
        payload = {
            "model": self.model,
            "system": _SYSTEM_PROMPT,
            "prompt": f"Summarize the following text in three sentences:\n\n{text}",
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OllamaError(
                f"Ollama returned {exc.response.status_code} for model "
                f"{self.model!r}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

        data = resp.json()
        summary = _THINK_RE.sub("", data.get("response", "")).strip()
        if not summary:
            raise OllamaError("Ollama returned an empty summary")

        tokens = int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
        return summary, tokens

    async def judge_durable(self, text: str) -> str | None:
        """If `text` holds a durable fact worth remembering, return a one-line
        third-person summary; otherwise None. Returns None (never raises) when
        Ollama is unavailable — auto-capture fails open to 'store nothing'."""
        payload = {
            "model": self.model,
            "system": _JUDGE_SYSTEM,
            "prompt": text,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        verdict = _THINK_RE.sub("", resp.json().get("response", "")).strip()
        if not verdict or verdict.upper().startswith("NO"):
            return None
        return verdict[:500]

    async def judge_correction(self, prior_assistant: str,
                               user_msg: str) -> str | None:
        """If the user is correcting a fact the assistant just stated, return a
        one-line third-person summary of the corrected fact; otherwise None.
        Returns None (never raises) when Ollama is unavailable — like
        judge_durable, correction-capture fails open to 'store nothing'."""
        if not prior_assistant.strip() or not user_msg.strip():
            return None
        payload = {
            "model": self.model,
            "system": _CORRECTION_SYSTEM,
            "prompt": (f"Assistant's previous message:\n{prior_assistant}\n\n"
                       f"User's reply:\n{user_msg}"),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        verdict = _THINK_RE.sub("", resp.json().get("response", "")).strip()
        if not verdict or verdict.upper().startswith("NO"):
            return None
        return verdict[:500]

    async def judge_contradiction(self, old_fact: str, new_fact: str) -> bool:
        """True if `new_fact` supersedes `old_fact` (they can't both be true).
        Returns False (never raises) on error — fail closed, so an unreachable
        Ollama never invalidates a memory."""
        if not old_fact.strip() or not new_fact.strip():
            return False
        payload = {
            "model": self.model,
            "system": _CONTRADICTION_SYSTEM,
            "prompt": f"OLD fact:\n{old_fact}\n\nNEW fact:\n{new_fact}",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError:
            return False
        verdict = _THINK_RE.sub("", resp.json().get("response", "")).strip()
        return verdict.upper().startswith("YES")

    async def draft_gate(self, query: str, context: str,
                         timeout: float = 8.0) -> str | None:
        """Draft a short, tailored preface for the preflight gate naming the
        relevant saved context. Returns None (never raises) on error/timeout so
        the gate can fall back to its static intro."""
        payload = {
            "model": self.model,
            "system": _GATE_SYSTEM,
            "prompt": f"User request:\n{query}\n\nContext I have about the user:\n{context}",
            "stream": False,
            # think=false skips the slow <think> phase on reasoning models (qwen3,
            # deepseek-r1, ...); num_predict caps output — the gate is interactive
            # so it must be fast. Unknown options are ignored by Ollama.
            "think": False,
            "options": {"temperature": 0.3, "num_predict": 100},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload,
                                           timeout=timeout)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        text = _THINK_RE.sub("", resp.json().get("response", "")).strip()
        return text[:600] or None

    async def answer(self, question: str, context: str = "") -> str:
        """Answer a question, optionally grounded in a context block. Used by the
        holdout eval to produce the treatment (context) and control (no context)
        arms. Raises OllamaError on failure so the eval can stop and report
        rather than silently scoring zeros."""
        prompt = (f"Context:\n{context}\n\nQuestion: {question}"
                  if context.strip() else f"Question: {question}")
        payload = {
            "model": self.model,
            "system": _ANSWER_SYSTEM,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"answer generation failed: {exc}") from exc
        return _THINK_RE.sub("", resp.json().get("response", "")).strip()

    async def judge_answer(self, question: str, expected: str,
                           answer: str) -> bool:
        """Grade whether `answer` conveys the `expected` fact for `question`.
        Returns True/False; raises OllamaError on failure."""
        payload = {
            "model": self.model,
            "system": _ANSWER_JUDGE_SYSTEM,
            "prompt": (f"Question: {question}\nExpected fact: {expected}\n"
                       f"Answer: {answer}\n\nDoes the answer convey the expected "
                       f"fact? Reply YES or NO."),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = await self._client.post("/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"answer judging failed: {exc}") from exc
        verdict = _THINK_RE.sub("", resp.json().get("response", "")).strip()
        return verdict.upper().startswith("YES")

    async def embed(self, text: str, kind: str = "query") -> list[float]:
        """Embed one text. kind is 'query' or 'document' (asymmetric models)."""
        return (await self.embed_many([text], kind))[0]

    async def embed_many(self, texts: list[str],
                         kind: str = "document") -> list[list[float]]:
        """Embed a batch in one request; preserves input order."""
        if self._prefixed:
            prefix = "search_query: " if kind == "query" else "search_document: "
            texts = [prefix + t for t in texts]
        payload = {"model": self.embed_model, "input": texts}
        try:
            resp = await self._client.post("/api/embed", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OllamaError(
                f"Ollama returned {exc.response.status_code} for embed model "
                f"{self.embed_model!r}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

        vectors = resp.json().get("embeddings") or []
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise OllamaError(
                f"Embed model {self.embed_model!r} returned "
                f"{len(vectors)} embeddings for {len(texts)} inputs"
            )
        return vectors
