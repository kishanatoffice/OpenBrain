"""Token estimation — deliberately honest about being an *estimate*.

We do not vendor a model-specific tokenizer (it would add a heavy dependency
and still be wrong for whatever model the user's tool actually calls). Instead
we use two documented, conservative heuristics and label every number as an
estimate. The point is a defensible *order-of-magnitude* signal for "was this
worth it", not an exact billing figure.

A module-level counter accumulates estimated savings across the session, mirror-
ing redact.redaction_count(), so the dashboard can show a running total without
threading state through every call.
"""

from __future__ import annotations

# ~4 characters per token is the long-standing rule of thumb for English text
# across GPT/Claude BPE tokenizers. Good enough for an order-of-magnitude.
_CHARS_PER_TOKEN = 4

# Vision cost is tile-based on the major APIs: an image is split into ~512px
# tiles and each tile costs a roughly fixed number of tokens. ~750 px² per
# token is a conservative blend of the published Claude/GPT-4o figures. This
# is an APPROXIMATION — it is only used to estimate savings, never billed.
_PIXELS_PER_VISION_TOKEN = 750
_MIN_VISION_TOKENS = 85  # even a tiny image carries a base cost

_TOKENS_SAVED = 0


def estimate_text_tokens(text: str) -> int:
    """Estimated tokens for a block of text (~chars/4)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_image_vision_tokens(width: int, height: int) -> int:
    """Estimated tokens a vision model would spend to 'see' an image of these
    pixel dimensions. Conservative; an estimate, not a billed figure."""
    if width <= 0 or height <= 0:
        return _MIN_VISION_TOKENS
    return max(_MIN_VISION_TOKENS, (width * height) // _PIXELS_PER_VISION_TOKEN)


def record_savings(saved: int) -> None:
    """Add to the running estimated-savings counter (clamped at >= 0)."""
    global _TOKENS_SAVED
    if saved > 0:
        _TOKENS_SAVED += saved


def tokens_saved() -> int:
    return _TOKENS_SAVED
