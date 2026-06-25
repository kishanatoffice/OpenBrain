"""Configuration for the Guard Rails daemon.

Resolution order (highest wins):
  1. Environment variables (GUARDRAILS_PORT, GUARDRAILS_DB_PATH, …)
  2. guardrails.toml in the working directory (or $GUARDRAILS_CONFIG)
  3. Built-in defaults

Kept intentionally separate from the memory daemon's config: Guard Rails is its
own service with its own port and data directory, so a user can run one, both,
or neither without the two interfering.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Own port (memory daemon owns 3111) and own data dir so the two services never
# share a DB, token, or socket.
DEFAULTS = {
    "GUARDRAILS_PORT": "3112",
    "GUARDRAILS_DB_PATH": "~/.openbrain-guardrails/guardrails.db",
    # Hard ceiling on any single stored text field (request / action / prompt).
    # Approval prompts are short; this just stops a pathological payload from
    # bloating the DB.
    "GUARDRAILS_MAX_FIELD_CHARS": "20000",
}


@dataclass(frozen=True)
class Config:
    port: int
    db_path: Path
    max_field_chars: int


def _read_toml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    flat = data.get("guardrails", data)
    return {str(k).upper(): str(v) for k, v in flat.items() if not isinstance(v, dict)}


def load_config() -> Config:
    toml_path = Path(os.environ.get("GUARDRAILS_CONFIG", "guardrails.toml"))
    file_values = _read_toml(toml_path)

    def get(key: str) -> str:
        return os.environ.get(key) or file_values.get(key) or DEFAULTS[key]

    def get_int(key: str) -> int:
        raw = get(key)
        try:
            return int(raw)
        except ValueError:
            raise SystemExit(f"{key} must be an integer, got {raw!r}")

    return Config(
        port=get_int("GUARDRAILS_PORT"),
        db_path=Path(get("GUARDRAILS_DB_PATH")).expanduser().resolve(),
        max_field_chars=get_int("GUARDRAILS_MAX_FIELD_CHARS"),
    )
