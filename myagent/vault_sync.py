"""Two-way vault sync: the Markdown folder is a real interface, not just a
mirror.

Every tick the daemon scans VAULT_PATH for *.md files:
  new file      -> imported as a memory (source="vault"); summary and
                   embeddings arrive via the normal enrichment loop
  edited file   -> memory content updated, embeddings + summary invalidated
  deleted file  -> memory kept but detached (md_path cleared) — sync never
                   silently destroys data

Feedback-loop guard: the DB stores the mtime of every file the daemon itself
wrote; only files whose disk mtime is newer count as user edits. Files
modified in the last 2 seconds are skipped (possibly still being written).
Vault-authored files are never rewritten by the daemon (see enrich_pending).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from .memory_service import Deps, excerpt_summary

log = logging.getLogger("myagent")

TICK_SECONDS = 15
MTIME_TOLERANCE = 1.0   # filesystem timestamp slack
SETTLE_SECONDS = 2.0    # ignore files modified this recently
MAX_FILE_BYTES = 1_000_000

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_CONTENT_SECTION_RE = re.compile(r"^## Content\s*$", re.MULTILINE)
# Daemon-written filenames embed the memory id: 2026-06-12-160926-0001-slug.md
# The id is zero-padded to *at least* 4 digits (vault.py uses :04d), so match a
# variable-width run — hard-coding {4} silently fails for ids >= 10000 and would
# re-import that file as a duplicate memory instead of re-linking it.
_DAEMON_FILE_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}-\d{6}-(\d+)-")


def extract_content(text: str) -> str:
    """Pull the memory content out of a Markdown file.

    Daemon-written files keep the content under '## Content'; arbitrary
    user notes are taken whole (minus YAML frontmatter)."""
    body = _FRONTMATTER_RE.sub("", text)
    match = _CONTENT_SECTION_RE.search(body)
    if match:
        return body[match.end():].strip()
    return body.strip()


async def sync_once(deps: Deps) -> dict[str, int]:
    """One scan pass. Returns counts: imported / updated / detached."""
    counts = {"imported": 0, "updated": 0, "detached": 0}
    now = time.time()
    rows = deps.store.vault_rows()
    by_path = {r["md_path"]: r for r in rows if r["md_path"]}
    disk_files = {str(p) for p in deps.vault.path.glob("*.md")}
    unlinked = {r["id"]: r for r in rows if not r["md_path"]}

    # Known files first -> detect user edits / deletions. Detaching deleted files
    # *before* the import pass is what lets a rename (which the daemon sees as one
    # deletion + one new file) re-link to the now-unlinked row below instead of
    # forking one note into two memories.
    for path_str, row in by_path.items():
        path = Path(path_str)
        if path_str not in disk_files:
            deps.store.clear_md_path(row["id"])
            unlinked[row["id"]] = row  # eligible for re-link in the import pass
            counts["detached"] += 1
            log.warning("Vault sync: %s deleted on disk; memory #%s kept "
                        "(detached)", path.name, row["id"])
            continue
        try:
            stat = path.stat()
            if (stat.st_mtime <= row["md_mtime"] + MTIME_TOLERANCE
                    or now - stat.st_mtime < SETTLE_SECONDS
                    or stat.st_size > MAX_FILE_BYTES):
                continue
            content = extract_content(path.read_text(encoding="utf-8"))
        except OSError as exc:
            log.warning("Vault sync: cannot read %s: %s", path.name, exc)
            continue
        if content and content != row["content"]:
            deps.store.update_content(row["id"], content)
            counts["updated"] += 1
            log.info("Vault sync: memory #%s updated from %s edit",
                     row["id"], path.name)
        # Absorb the new mtime either way so we don't re-read every tick.
        deps.store.set_md_path(row["id"], path, stat.st_mtime)

    # New files -> adopt (our own unlinked file), re-link a renamed file, or
    # import a genuinely new user note.
    for path_str in sorted(disk_files - set(by_path)):
        path = Path(path_str)
        try:
            stat = path.stat()
            if now - stat.st_mtime < SETTLE_SECONDS or stat.st_size > MAX_FILE_BYTES:
                continue
            content = extract_content(path.read_text(encoding="utf-8"))
        except OSError as exc:
            log.warning("Vault sync: cannot read %s: %s", path.name, exc)
            continue
        if not content:
            continue

        # A daemon-written file whose memory exists but lost its link (e.g.
        # rows from before md_path tracking): re-link instead of duplicating.
        match = _DAEMON_FILE_RE.match(path.name)
        if match and int(match.group(1)) in unlinked:
            row = unlinked.pop(int(match.group(1)))
            if content != row["content"]:
                deps.store.update_content(row["id"], content)
                counts["updated"] += 1
            deps.store.set_md_path(row["id"], path, stat.st_mtime)
            counts["adopted"] = counts.get("adopted", 0) + 1
            log.info("Vault sync: re-linked %s to memory #%s",
                     path.name, row["id"])
            continue

        # A renamed file: its content matches an unlinked row (the file it was
        # renamed from was detached above). Re-link rather than import a twin.
        twin = next((mid for mid, r in unlinked.items()
                     if r["content"] == content), None)
        if twin is not None:
            row = unlinked.pop(twin)
            deps.store.set_md_path(row["id"], path, stat.st_mtime)
            counts["adopted"] = counts.get("adopted", 0) + 1
            log.info("Vault sync: re-linked renamed %s to memory #%s",
                     path.name, row["id"])
            continue

        memory = deps.store.add(content, excerpt_summary(content),
                                source="vault")
        deps.store.set_md_path(memory["id"], path, stat.st_mtime)
        counts["imported"] += 1
        log.info("Vault sync: imported %s as memory #%s", path.name, memory["id"])

    return counts


async def sync_loop(deps: Deps, tick: float = TICK_SECONDS) -> None:
    log.info("Vault sync started (tick %ss, watching %s)", tick, deps.vault.path)
    while True:
        try:
            await sync_once(deps)
        except Exception:
            log.exception("Vault sync tick failed")
        await asyncio.sleep(tick)
