"""
TTS: StyleTTS2 only. GPU. Requires speaker_wav_path and style_audio_path (merged ref_s).

Dubbing passes segment audio as style_audio_path; standalone TTS passes the same path for both.
"""
import logging

logger = logging.getLogger(__name__)


def generate_speech(
    text: str,
    output_path: str,
    *,
    language: str = "en",
    speaker_wav_path: str,
    style_audio_path: str,
) -> None:
    from worker.models.styletts_loader import generate_speech_styletts

    generate_speech_styletts(
        text,
        output_path,
        language=language,
        speaker_wav_path=speaker_wav_path,
        style_audio_path=style_audio_path,
    )
