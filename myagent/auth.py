"""Local API token — the privacy boundary for the loopback daemon.

OpenBrain binds to 127.0.0.1, but loopback is shared by every process and every
user on the machine, and a malicious web page can try to reach it via CSRF /
DNS-rebinding. A random token (file-stored, 0600) gates the data and management
endpoints so that:

  * another user on a shared machine cannot read or write your brain, and
  * a web page in your browser cannot either — it can't read the token
    (cross-origin reads are blocked) nor set our custom header on a simple
    cross-site request without a CORS preflight we reject.

What it deliberately does NOT defend against: code running AS you. Such code can
read the token file or the SQLite database directly — no app-layer check can
stop that (only OS-level encryption/sandboxing could). The boundary is
*other users* and *the browser*, not your own processes. We state this plainly
rather than overclaim.

The token is passed as the `X-OpenBrain-Token` header or a `?token=` query
param — never via `Authorization`, which the memory-injecting proxy reserves for
the user's upstream model key.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Mapping

TOKEN_FILENAME = "token"
HEADER = "x-openbrain-token"
QUERY_PARAM = "token"
_TOKEN_BYTES = 32  # 256 bits of entropy, url-safe base64


def token_path(data_dir: Path) -> Path:
    return data_dir / TOKEN_FILENAME


def load_or_create_token(data_dir: Path) -> str:
    """Return the daemon's token, generating and persisting one (0600) on first
    run. Idempotent: an existing non-empty token is reused (and its perms are
    re-hardened in case the file was created loosely)."""
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
    """The stored token, or None if absent/empty. Never creates the file —
    callers that only consume (CLI) must not race the daemon into existence."""
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
    """Pull the caller's token from the custom header or the query param.
    `Authorization` is intentionally ignored — the proxy needs it for the
    upstream model key."""
    return headers.get(HEADER) or query_params.get(QUERY_PARAM)
