"""
XTTS v2: load once at worker startup, generate speech from text (optionally with voice clone).
"""
import os
from pathlib import Path

_xtts_model = None
_xtts_speaker_wav = None


def load_xtts():
    """Load XTTS model once. Keep in memory."""
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    try:
        from TTS.api import TTS
        # XTTS v2 model
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        return _xtts_model
    except ImportError:
        raise RuntimeError("TTS (Coqui) not installed. pip install TTS")


def generate_speech(
    text: str,
    output_path: str,
    language: str = "en",
    speaker_wav_path: str = None,
) -> None:
    """
    Generate speech from text. If speaker_wav_path is set, clone that voice.
    """
    model = load_xtts()
    if speaker_wav_path and os.path.isfile(speaker_wav_path):
        model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=speaker_wav_path,
            language=language,
        )
    else:
        # Default speaker
        model.tts_to_file(
            text=text,
            file_path=output_path,
            language=language,
        )
