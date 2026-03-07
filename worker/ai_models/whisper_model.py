"""
OpenAI Whisper: load once at worker startup, transcribe audio to segments.
"""
import os
from typing import List, Dict, Any

# Lazy load to avoid import errors if not installed
_whisper_model = None


def load_whisper(model_size: str = None):
    """Load Whisper model once. Keep in memory."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        size = model_size or os.getenv("WHISPER_MODEL_SIZE", "base")
        _whisper_model = whisper.load_model(size)
        return _whisper_model
    except ImportError:
        raise RuntimeError("openai-whisper not installed. pip install openai-whisper")


def transcribe_audio(audio_path: str, language: str = None) -> List[Dict[str, Any]]:
    """
    Transcribe audio file. Returns list of segments: [{ start, end, text }, ...]
    """
    model = load_whisper()
    result = model.transcribe(audio_path, language=language, word_timestamps=False)
    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
        })
    full_text = result.get("text", "").strip()
    return {
        "segments": segments,
        "full_text": full_text,
        "language": result.get("language", "en"),
    }
