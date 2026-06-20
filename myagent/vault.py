"""Markdown mirror — writes each memory as an Obsidian-friendly .md file."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SLUG_MAX_WORDS = 6
_SLUG_MAX_CHARS = 48


def _slugify(text: str) -> str:
    words = re.sub(r"[^\w\s-]", "", text.lower()).split()[:_SLUG_MAX_WORDS]
    slug = "-".join(words)[:_SLUG_MAX_CHARS].strip("-")
    return slug or "memory"


def _escape_frontmatter(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Vault:
    def __init__(self, vault_path: Path) -> None:
        self.path = vault_path
        self.path.mkdir(parents=True, exist_ok=True)

    def write_memory(self, memory: dict[str, Any],
                     path: str | Path | None = None) -> Path:
        """Mirror a memory to Markdown. Pass `path` to rewrite an existing
        file in place (used when enrichment upgrades the summary)."""
        if path is None:
            created = datetime.fromisoformat(memory["created_at"]).astimezone(
                timezone.utc)
            stamp = created.strftime("%Y-%m-%d-%H%M%S")
            slug = _slugify(memory["summary"] or memory["content"])
            file_path = self.path / f"{stamp}-{memory['id']:04d}-{slug}.md"
        else:
            file_path = Path(path)

        title = (memory["summary"].split(".")[0] or "Memory").strip()
        body = (
            "---\n"
            f"id: {memory['id']}\n"
            f"created: {memory['created_at']}\n"
            f"tokens_used: {memory['tokens_used']}\n"
            "tags: [memory]\n"
            f"title: {_escape_frontmatter(title)}\n"
            "---\n"
            "\n"
            f"# {title}\n"
            "\n"
            "## Summary\n"
            "\n"
            f"{memory['summary'].strip()}\n"
            "\n"
            "## Content\n"
            "\n"
            f"{memory['content'].strip()}\n"
        )
        file_path.write_text(body, encoding="utf-8")
        return file_path
