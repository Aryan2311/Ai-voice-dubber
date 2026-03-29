"""
Rewrite stage removed (no LLM). Kept as identity so callers can still import `rewrite` if needed.
"""
from typing import Optional


def rewrite(
    text: str,
    language: str = "hi",
    style_hint: Optional[str] = None,
    target_syllables: Optional[int] = None,
) -> str:
    """Return text unchanged (translation-only dubbing pipeline)."""
    return text
