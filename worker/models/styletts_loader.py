"""
StyleTTS2 TTS. GPU. Mandatory; no XTTS fallback.

Dual reference in TTS context:
- Prosody (rhythm, emotion) → from original segment audio (style_audio_path). Use in dubbing.
- Voice guidance (standalone TTS) → optional speaker_wav_path when no style_audio_path.
So when style_audio_path is set we use only that reference.
"""
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_styletts_model = None


def _ensure_nltk_for_styletts():
    """
    NLTK 3.8+: word_tokenize/sent_tokenize need punkt_tab; StyleTTS2 only pulled punkt during its init.
    nltk.download is a no-op when data is already cached (e.g. Docker image bake step).
    """
    import nltk

    logger.info("[styletts] Ensuring NLTK punkt + punkt_tab …")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def get_styletts_model():
    global _styletts_model
    if _styletts_model is not None:
        return _styletts_model
    _ensure_nltk_for_styletts()
    from styletts2 import tts
    checkpoint = os.getenv("STYLETTS2_CHECKPOINT")
    config = os.getenv("STYLETTS2_CONFIG")
    if checkpoint and config:
        logger.info(
            "[styletts] Loading StyleTTS2 from checkpoint=%r config=%r …",
            checkpoint,
            config,
        )
        _styletts_model = tts.StyleTTS2(
            model_checkpoint_path=checkpoint,
            config_path=config,
        )
    else:
        logger.info("[styletts] Loading StyleTTS2 with default checkpoints (GPU) …")
        _styletts_model = tts.StyleTTS2()
    logger.info("[styletts] StyleTTS2 ready.")
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
    - style_audio_path: prosody from original segment (use in dubbing).
    - speaker_wav_path: voice reference only when no style_audio_path (e.g. standalone TTS).
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
