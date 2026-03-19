"""
XTTS loader (current TTS). GPU. Used when StyleTTS2 is not available.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

from worker.ai_models import xtts_model


def generate_speech_xtts(
    text: str,
    output_path: str,
    language: str = "en",
    speaker_wav_path: Optional[str] = None,
) -> None:
    xtts_model.generate_speech(
        text, output_path,
        language=language,
        speaker_wav_path=speaker_wav_path,
    )
