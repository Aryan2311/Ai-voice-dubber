"""
Translation: NLLB-200 only. Runs on CPU.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from worker.utils.job_logging import log_preview

logger = logging.getLogger(__name__)


def translate_segments(
    segments: List[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    *,
    log_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Translate each segment's text. Preserves start/end. Returns same structure with translated text.
    log_context: optional prefix (e.g. media_id=...) for INFO logs.
    """
    ctx = ("[%s] " % log_context) if log_context else ""
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    n = len(segments)
    if src == tgt:
        logger.info(
            "%s[translate] skip NLLB (source==target): %s segments lang=%s",
            ctx,
            n,
            src,
        )
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]
    from worker.models.nllb_loader import translate_text

    t_batch = time.monotonic()
    logger.info(
        "%s[translate] NLLB begin: %s segments %s→%s",
        ctx,
        n,
        src,
        tgt,
    )
    result = []
    for i, s in enumerate(segments):
        raw = s.get("text", "") or ""
        t0 = time.monotonic()
        text = translate_text(raw, source_lang=src, target_lang=tgt)
        seg_elapsed = time.monotonic() - t0
        logger.info(
            "%s[translate] segment %d/%d in_chars=%d out_chars=%d elapsed_sec=%.2f in_preview=%r out_preview=%r",
            ctx,
            i + 1,
            n,
            len(raw),
            len(text),
            seg_elapsed,
            log_preview(raw, 100),
            log_preview(text, 100),
        )
        result.append({
            "start": s["start"],
            "end": s["end"],
            "text": text,
            "original_text": s.get("text", ""),
        })
    logger.info(
        "%s[translate] NLLB done: %s segments total_elapsed_sec=%.1f",
        ctx,
        n,
        time.monotonic() - t_batch,
    )
    return result
