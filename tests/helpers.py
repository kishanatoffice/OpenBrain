"""Shared test fixtures: deterministic fake Ollama + temp-dir deps."""

from __future__ import annotations

from pathlib import Path

from myagent.db import MemoryStore
from myagent.memory_service import Deps
from myagent.ollama import OllamaError
from myagent.vault import Vault


class FakeOllama:
    """Deterministic stand-in: embeddings come from a text->vector mapping
    (with a constant default), summaries are canned, and `down=True`
    simulates an unreachable Ollama."""

    def __init__(self, vectors: dict[str, list[float]] | None = None,
                 default: list[float] | None = None, down: bool = False,
                 judge_summary: str | None = None,
                 correction_summary: str | None = None,
                 contradiction: bool = False,
                 gate_intro: str | None = None):
        self.embed_model = "fake-embed"
        self.embed_key = "fake-embed"
        self.model = "fake-chat"
        self.vectors = vectors or {}
        self.default = default or [1.0, 0.0, 0.0]
        self.down = down
        # What judge_durable returns: a summary (store) or None (skip).
        self.judge_summary = judge_summary
        # What judge_correction returns: the corrected fact, or None (no correction).
        self.correction_summary = correction_summary
        # What judge_contradiction returns (does NEW supersede OLD).
        self.contradiction = contradiction
        # What draft_gate returns: a tailored intro, or None (static fallback).
        self.gate_intro = gate_intro

    async def judge_durable(self, text: str) -> str | None:
        if self.down:
            return None  # fails closed
        return self.judge_summary

    async def judge_correction(self, prior_assistant: str,
                               user_msg: str) -> str | None:
        if self.down or not prior_assistant.strip() or not user_msg.strip():
            return None  # fails closed; mirrors the real client's guard
        return self.correction_summary

    async def judge_contradiction(self, old_fact: str, new_fact: str) -> bool:
        if self.down or not old_fact.strip() or not new_fact.strip():
            return False  # fail closed: never invalidate when judge is unsure
        return self.contradiction

    async def draft_gate(self, query: str, context: str,
                         timeout: float = 8.0) -> str | None:
        if self.down:
            return None
        return self.gate_intro

    async def embed(self, text: str, kind: str = "query") -> list[float]:
        return (await self.embed_many([text], kind))[0]

    async def embed_many(self, texts: list[str],
                         kind: str = "document") -> list[list[float]]:
        if self.down:
            raise OllamaError("fake ollama is down")
        return [list(self.vectors.get(t, self.default)) for t in texts]

    async def answer(self, question: str, context: str = "") -> str:
        if self.down:
            raise OllamaError("fake ollama is down")
        # Tag the arm so judge_answer can simulate "memory helps": the
        # context-grounded (treatment) answer grades correct, the bare control
        # answer does not. Markers must not substring-collide.
        return f"ANSWER[{'grounded' if context.strip() else 'bare'}]: {question}"

    async def judge_answer(self, question: str, expected: str,
                           answer: str) -> bool:
        if self.down:
            raise OllamaError("fake ollama is down")
        return "grounded" in answer

    async def summarize(self, text: str) -> tuple[str, int]:
        if self.down:
            raise OllamaError("fake ollama is down")
        return ("This is the AI summary.", 42)

    async def is_reachable(self) -> bool:
        return not self.down

    async def aclose(self) -> None:
        pass


def make_deps(tmpdir: str | Path, **ollama_kwargs) -> Deps:
    tmp = Path(tmpdir)
    return Deps(
        store=MemoryStore(tmp / "memories.db"),
        vault=Vault(tmp / "vault"),
        ollama=FakeOllama(**ollama_kwargs),
        min_similarity=0.50,
    )
