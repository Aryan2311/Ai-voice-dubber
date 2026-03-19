"""
Casual/natural language rewrite with Phi-3-mini. CPU. Mandatory; no identity fallback.
Supports target_syllables for syllable-aware dubbing (match original segment rhythm).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def rewrite(
    text: str,
    language: str = "hi",
    style_hint: Optional[str] = None,
    target_syllables: Optional[int] = None,
) -> str:
    """Rewrite text to sound natural and conversational. Optional target_syllables for alignment."""
    from worker.models.llm_loader import rewrite_with_llm
    return rewrite_with_llm(
        text, language=language, style_hint=style_hint, target_syllables=target_syllables
    )
