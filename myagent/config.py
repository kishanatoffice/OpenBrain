"""Configuration loader.

Resolution order (highest wins):
  1. Environment variables  (OLLAMA_URL, OLLAMA_MODEL, MEMORY_PORT, VAULT_PATH, DB_PATH)
  2. config.toml in the current working directory (or $MYAGENT_CONFIG)
  3. Built-in defaults
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULTS = {
    "OLLAMA_URL": "http://localhost:11434",
    "OLLAMA_MODEL": "llama3",
    "OLLAMA_EMBED_MODEL": "nomic-embed-text",
    "MEMORY_PORT": "3111",
    "VAULT_PATH": "~/.myagent/vault",
    "DB_PATH": "~/.myagent/memories.db",
    # Forgetting curve w(t) = 2^(-age/half_life) applied to recall relevance.
    # 0 disables it: old facts are not less true by default.
    "RECALL_HALF_LIFE_DAYS": "0",
    # Absolute cosine floor for semantic recall: memories below this never
    # enter the context, no matter how they rank relatively. (A relative
    # cutoff of 0.90 x best-match is applied on top — see memory_service.)
    "RECALL_MIN_SIMILARITY": "0.50",
    # Level 3 memory-injecting LLM proxy (POST /v1/chat/completions). Point a
    # tool's OpenAI-compatible base URL at http://127.0.0.1:<port>/v1 and every
    # request gets relevant memory injected before reaching the real model.
    "PROXY_UPSTREAM_URL": "https://api.openai.com/v1",
    "PROXY_MIN_RELEVANCE": "0.60",  # drop hits below this before injecting
    "PROXY_RECALL_TOKENS": "1000",  # token budget for the injected block
    "PROXY_TIMEOUT_SECONDS": "600",  # upstream timeout (long for streaming)
    # Preflight gate: when ON, if the query-relevant memory is in the BORDERLINE
    # band [PROXY_MIN_RELEVANCE, PROXY_GATE_HIGH) — relevant enough to matter but
    # not an obvious match — the proxy answers first with a small menu asking
    # whether to use OpenBrain or skip to the plain LLM, instead of guessing.
    # Clearly-relevant (>= high) injects silently; clearly-irrelevant forwards
    # plain. Off by default so the silent path stays the default experience.
    "PROXY_PREFLIGHT": "false",
    "PROXY_GATE_HIGH": "0.78",
    # Selective auto-capture: after a turn through the proxy, an Ollama judge
    # decides whether the user's message held a durable fact and, if so, stores
    # a one-line summary (tagged 'auto'). Off by default — auto-writing to
    # permanent memory is a deliberate opt-in. Fails closed (no judge = nothing
    # stored) and reuses dedup + the write-quality guard.
    "PROXY_AUTOCAPTURE": "false",
    # OCR / document-ingest connector (off by default; see connectors.py).
    # Allowlisted roots the `digest` tool may read from — colon-separated.
    # Anything outside these (after resolving symlinks and ..) is refused, so an
    # agent can never coax the tool into reading ~/.ssh, secrets, or system
    # files. Default is a single dedicated ingest dir the user drops files into.
    "OCR_INGEST_DIRS": "~/.myagent/ingest",
    "OCR_MAX_FILE_MB": "25",  # hard size cap; larger files are refused, not read
    # Office/EPub files are ZIP containers, so a tiny upload can expand to
    # gigabytes (a "zip bomb"). Before parsing, the connector sums the archive's
    # declared uncompressed size and refuses anything over this cap — defending
    # memory even though the source file passed the byte cap above.
    "OCR_MAX_EXPANDED_MB": "100",
    # Wall-clock ceiling on a single markitdown conversion. The conversion runs
    # in a worker thread; if it overruns, the request fails instead of freezing
    # the daemon's event loop forever on a parser-pathological file.
    "OCR_CONVERT_TIMEOUT_S": "30",
    # Decoded-pixel ceiling for images (megapixels). A small compressed image can
    # declare a gigapixel canvas (a decompression bomb); we refuse before any
    # decoder allocates the raster.
    "OCR_MAX_IMAGE_MP": "40",
}


@dataclass(frozen=True)
class Config:
    ollama_url: str
    ollama_model: str
    ollama_embed_model: str
    memory_port: int
    vault_path: Path
    db_path: Path
    recall_half_life_days: float
    recall_min_similarity: float
    proxy_upstream_url: str
    proxy_min_relevance: float
    proxy_recall_tokens: int
    proxy_timeout_seconds: float
    proxy_preflight: bool
    proxy_gate_high: float
    proxy_autocapture: bool
    ocr_ingest_dirs: tuple[Path, ...]
    ocr_max_bytes: int
    ocr_max_expanded_bytes: int
    ocr_convert_timeout_s: float
    ocr_max_image_pixels: int


def _read_toml(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    # Accept both flat keys and a [myagent] table; normalize keys to upper snake.
    flat = data.get("myagent", data)
    return {str(k).upper(): str(v) for k, v in flat.items() if not isinstance(v, dict)}


def load_config() -> Config:
    toml_path = Path(os.environ.get("MYAGENT_CONFIG", "config.toml"))
    file_values = _read_toml(toml_path)

    def get(key: str) -> str:
        return os.environ.get(key) or file_values.get(key) or DEFAULTS[key]

    port_raw = get("MEMORY_PORT")
    try:
        port = int(port_raw)
    except ValueError:
        raise SystemExit(f"MEMORY_PORT must be an integer, got {port_raw!r}")

    def get_float(key: str) -> float:
        raw = get(key)
        try:
            return float(raw)
        except ValueError:
            raise SystemExit(f"{key} must be a number, got {raw!r}")

    half_life = get_float("RECALL_HALF_LIFE_DAYS")
    min_similarity = get_float("RECALL_MIN_SIMILARITY")

    def get_int(key: str) -> int:
        raw = get(key)
        try:
            return int(raw)
        except ValueError:
            raise SystemExit(f"{key} must be an integer, got {raw!r}")

    return Config(
        ollama_url=get("OLLAMA_URL").rstrip("/"),
        ollama_model=get("OLLAMA_MODEL"),
        ollama_embed_model=get("OLLAMA_EMBED_MODEL"),
        memory_port=port,
        vault_path=Path(get("VAULT_PATH")).expanduser().resolve(),
        db_path=Path(get("DB_PATH")).expanduser().resolve(),
        recall_half_life_days=half_life,
        recall_min_similarity=min_similarity,
        proxy_upstream_url=get("PROXY_UPSTREAM_URL").rstrip("/"),
        proxy_min_relevance=get_float("PROXY_MIN_RELEVANCE"),
        proxy_recall_tokens=get_int("PROXY_RECALL_TOKENS"),
        proxy_timeout_seconds=get_float("PROXY_TIMEOUT_SECONDS"),
        proxy_preflight=get("PROXY_PREFLIGHT").strip().lower() in ("1", "true", "yes", "on"),
        proxy_gate_high=get_float("PROXY_GATE_HIGH"),
        proxy_autocapture=get("PROXY_AUTOCAPTURE").strip().lower() in ("1", "true", "yes", "on"),
        ocr_ingest_dirs=tuple(
            Path(p).expanduser().resolve()
            for p in get("OCR_INGEST_DIRS").split(":") if p.strip()
        ),
        ocr_max_bytes=get_int("OCR_MAX_FILE_MB") * 1024 * 1024,
        ocr_max_expanded_bytes=get_int("OCR_MAX_EXPANDED_MB") * 1024 * 1024,
        ocr_convert_timeout_s=get_float("OCR_CONVERT_TIMEOUT_S"),
        ocr_max_image_pixels=get_int("OCR_MAX_IMAGE_MP") * 1_000_000,
    )
