"""Local API token — the privacy boundary for the Guard Rails loopback daemon.

Same model and rationale as the memory daemon's token, vendored here so Guard
Rails stays self-contained (its own token file, its own data dir): a random
256-bit token (file-stored, 0600) gates every data endpoint so another local
user or a browser page can't read or write the approval log. It does NOT defend
against code running as you — only OS-level controls can.

The token is supplied as the `X-Guardrails-Token` header or a `?token=` query
param.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Mapping

TOKEN_FILENAME = "token"
HEADER = "x-guardrails-token"
QUERY_PARAM = "token"
_TOKEN_BYTES = 32  # 256 bits, url-safe base64


def token_path(data_dir: Path) -> Path:
    return data_dir / TOKEN_FILENAME


def load_or_create_token(data_dir: Path) -> str:
    """Return the daemon's token, generating and persisting one (0600) on first
    run. Idempotent: an existing non-empty token is reused (perms re-hardened)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    p = token_path(data_dir)
    existing = read_token(data_dir)
    if existing:
        _harden(p)
        return existing
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    # O_CREAT with mode 0o600 so the secret is never briefly world-readable.
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    return token


def read_token(data_dir: Path) -> str | None:
    """The stored token, or None if absent/empty. Never creates the file."""
    try:
        token = token_path(data_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _harden(p: Path) -> None:
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def token_matches(provided: str | None, actual: str | None) -> bool:
    """Constant-time comparison. False if either side is missing — fail closed."""
    if not provided or not actual:
        return False
    return secrets.compare_digest(provided, actual)


def extract_token(headers: Mapping[str, str],
                  query_params: Mapping[str, str]) -> str | None:
    return headers.get(HEADER) or query_params.get(QUERY_PARAM)
