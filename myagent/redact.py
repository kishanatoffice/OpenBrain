"""Secret/PII redaction — the privacy boundary on the write path.

OpenBrain auto-captures and injects memory into every prompt across every tool.
Without this, a user message containing an API key, token, or email could be
stored permanently and replayed into every future request. This module scrubs
known-secret shapes (and emails) before anything is persisted or surfaced.

Deliberately a curated regex set, not entropy-only: entropy detection produces
false positives on hashes/IDs/base64 and would silently redact legitimate facts.
We err toward precision — high-confidence patterns only — and log what was hit
so it's never silent. Run it at the persistence boundary (create_memory), on the
auto-capture judge's output, and on anything surfaced in the dashboard.
"""

from __future__ import annotations

import re

# (label, compiled pattern). High-confidence shapes only.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private-key-block",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
                re.DOTALL)),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai-key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("github-pat", re.compile(r"github_pat_[A-Za-z0-9_]{40,}")),
    ("slack-token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe-key", re.compile(r"\b[rsp]k_(?:live|test)_[A-Za-z0-9]{10,}\b")),
    ("sendgrid-key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),
    ("gitlab-pat", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack-webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+")),
    ("azure-account-key", re.compile(r"(?i)AccountKey=[A-Za-z0-9+/=]{20,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer", re.compile(r"[Bb]earer\s+[A-Za-z0-9._-]{20,}")),
    # Credentials embedded in a URI: scheme://user:password@host  — redact the
    # whole `user:pass@` segment (extremely common in screenshots of .env/logs).
    ("uri-credentials",
     re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]+:[^\s/:@]+@")),
    # `api_key = "..."`, `aws_secret_access_key: ...`, `DB_PASSWORD=...` — match a
    # key NAME containing a secret-ish word (group 1), then redact its VALUE
    # (group 2). The value runs to whitespace/quote, so passwords with special
    # characters ($ @ ! % …) are fully scrubbed, not partially. The 8-char floor
    # keeps it off ordinary prose ("the secret: be kind" won't trip it).
    ("assigned-secret",
     re.compile(r"(?i)\b([\w.\-]*(?:api[_-]?key|secret|access[_-]?key|password|passwd|token)[\w.\-]*)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

_PLACEHOLDER = "[REDACTED:{}]"

# Running total of secrets/PII scrubbed this session — surfaced in the dashboard
# as a trust signal ("🔒 N protected"). Single-process daemon, so a module
# counter is fine.
_REDACTION_COUNT = 0


def redaction_count() -> int:
    return _REDACTION_COUNT


def redact(text: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, labels_found). Empty list ⇒ nothing redacted."""
    if not text:
        return text, []
    found: list[str] = []

    def _sub(label: str):
        def repl(m: re.Match) -> str:
            found.append(label)
            # For assigned-secret, keep the key name (group 1), redact only the
            # value (group 2).
            if label == "assigned-secret":
                return m.group(0).replace(m.group(2), _PLACEHOLDER.format(label))
            return _PLACEHOLDER.format(label)
        return repl

    for label, pat in _PATTERNS:
        text = pat.sub(_sub(label), text)
    if found:
        global _REDACTION_COUNT
        _REDACTION_COUNT += len(found)
    return text, found


def has_secret(text: str) -> bool:
    return bool(redact(text)[1])
