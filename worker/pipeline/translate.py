"""
Translation: NLLB-200 (or MarianMT fallback). Runs on CPU.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def translate_segments(
    segments: List[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
) -> List[Dict[str, Any]]:
    """
    Translate each segment's text. Preserves start/end. Returns same structure with translated text.
    """
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    if src == tgt:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]
    from worker.models.nllb_loader import translate_text
    result = []
    for s in segments:
        text = translate_text(s.get("text", ""), source_lang=src, target_lang=tgt)
        result.append({
            "start": s["start"],
            "end": s["end"],
            "text": text,
            "original_text": s.get("text", ""),
        })
    return result
