"""
ASR: Whisper. Runs on GPU; use with GPU_LOCK. Returns segments [{start, end, text}, ...].
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def transcribe(audio_path: str, language: str = None) -> Dict[str, Any]:
    """
    Transcribe audio. Returns {"segments": [...], "full_text": str, "language": str}.
    Segments: [{"start": float, "end": float, "text": str}, ...].
    """
    from worker.models.whisper_loader import get_whisper_model
    model = get_whisper_model()
    result = model.transcribe(audio_path, language=language, word_timestamps=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": s["start"],
            "end": s["end"],
            "text": (s.get("text") or "").strip(),
        })
    return {
        "segments": segments,
        "full_text": (result.get("text") or "").strip(),
        "language": result.get("language", "en"),
    }
