"""
TTS: StyleTTS2 only. GPU. Uses style_audio_path for prosody, speaker_wav_path for voice clone.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_speech(
    text: str,
    output_path: str,
    language: str = "en",
    speaker_wav_path: Optional[str] = None,
    prosody: Optional[Dict[str, Any]] = None,
    style_audio_path: Optional[str] = None,
) -> None:
    """
    Generate speech with StyleTTS2. style_audio_path = prosody ref (e.g. original segment).
    speaker_wav_path = voice clone reference. At least one can be set for best quality.
    """
    from worker.models.styletts_loader import generate_speech_styletts
    generate_speech_styletts(
        text, output_path,
        language=language,
        speaker_wav_path=speaker_wav_path,
        style_audio_path=style_audio_path,
        prosody=prosody,
    )
