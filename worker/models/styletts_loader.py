"""
StyleTTS2 TTS. GPU. Mandatory; no XTTS fallback.

Dual reference: prosody vs voice identity.
- Prosody (rhythm, emotion) → from original segment audio (style_audio_path). Use in dubbing.
- Voice identity → from RVC after TTS, not from StyleTTS2 when dubbing.
So when style_audio_path is set we use only that (prosody from original); speaker_wav is for RVC.
"""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_styletts_model = None


def get_styletts_model():
    global _styletts_model
    if _styletts_model is not None:
        return _styletts_model
    from styletts2 import tts
    checkpoint = os.getenv("STYLETTS2_CHECKPOINT")
    config = os.getenv("STYLETTS2_CONFIG")
    if checkpoint and config:
        _styletts_model = tts.StyleTTS2(
            model_checkpoint_path=checkpoint,
            config_path=config,
        )
    else:
        _styletts_model = tts.StyleTTS2()
    logger.info("StyleTTS2 loaded.")
    return _styletts_model


def generate_speech_styletts(
    text: str,
    output_path: str,
    language: str = "en",
    speaker_wav_path: Optional[str] = None,
    style_audio_path: Optional[str] = None,
    prosody: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate speech. Dual reference:
    - style_audio_path: prosody from original segment (use in dubbing; do not pass speaker here).
    - speaker_wav_path: voice reference only when no style_audio_path (e.g. standalone TTS).
    Voice identity in dubbing is applied by RVC after this step.
    """
    model = get_styletts_model()
    # Prefer prosody from original segment; only use speaker_wav when no segment ref (e.g. TTS job).
    ref_path = style_audio_path if style_audio_path and os.path.isfile(style_audio_path) else (speaker_wav_path if speaker_wav_path and os.path.isfile(speaker_wav_path) else None)
    kwargs = {}
    if ref_path:
        kwargs["target_voice_path"] = ref_path
    model.inference(
        text,
        output_wav_file=output_path,
        output_sample_rate=24000,
        **kwargs,
    )
