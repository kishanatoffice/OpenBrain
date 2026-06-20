"""`python -m myagent init` — one-time persona seed.

Cold-start fix: a brand-new brain is empty, so the first chat in any tool has
nothing to ground on. This asks a handful of questions and stores the answers
as CORE memories (tagged `core`, `persona`) — the always-on layer that
recall injects into every request. Run it once; edit answers later in the web
UI. Writes go straight to the same DB the daemon uses (SQLite WAL is safe for
this), so it works whether or not the daemon is running.
"""

from __future__ import annotations

import asyncio

from .config import load_config
from .db import MemoryStore
from .memory_service import CORE_TAG, Deps, create_memory
from .ollama import OllamaClient
from .vault import Vault

# (prompt shown, label that prefixes the stored memory so it reads standalone)
QUESTIONS = [
    ("Your name and what you do (role, field)",
     "About me"),
    ("How do you like an AI to respond? (tone, length, format — e.g. "
     "'concise, code-first, no preamble')",
     "How I like AI to respond"),
    ("Your main tech stack / tools (languages, frameworks, services)",
     "My tech stack and tools"),
    ("What are you working on right now? (projects, goals)",
     "What I'm currently working on"),
    ("Anything else an AI should ALWAYS know about you? (constraints, "
     "preferences, context)",
     "Standing context about me"),
]


def _build_deps() -> Deps:
    cfg = load_config()
    return Deps(
        store=MemoryStore(cfg.db_path),
        vault=Vault(cfg.vault_path),
        ollama=OllamaClient(cfg.ollama_url, cfg.ollama_model,
                            cfg.ollama_embed_model),
        half_life_days=cfg.recall_half_life_days,
        min_similarity=cfg.recall_min_similarity,
        source="init",
    )


async def _seed(deps: Deps, answers: list[tuple[str, str]]) -> int:
    saved = 0
    for label, text in answers:
        content = f"{label}: {text.strip()}"
        result = await create_memory(deps, content,
                                     tags=[CORE_TAG, "persona"], source="init")
        if not result.get("duplicate") and not result.get("skipped"):
            saved += 1
    await deps.ollama.aclose()
    return saved


def run_init() -> None:
    print("\n🧠  OpenBrain — set up your always-on persona")
    print("    Answer what you like; press Enter to skip any question.")
    print("    These become 'core' memories, loaded into every AI chat.\n")

    answers: list[tuple[str, str]] = []
    for prompt, label in QUESTIONS:
        try:
            text = input(f"• {prompt}\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled — nothing saved.")
            return
        if text:
            answers.append((label, text))

    if not answers:
        print("\nNothing entered — no core memories created.")
        return

    cfg = load_config()
    deps = _build_deps()
    saved = asyncio.run(_seed(deps, answers))
    print(f"\n✅  Saved {saved} core memory(ies). Every new AI chat will now "
          "open knowing this about you.")
    print(f"    Review or edit them anytime at http://127.0.0.1:"
          f"{cfg.memory_port}/\n")
