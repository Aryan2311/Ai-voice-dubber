"""
StyleTTS2 TTS. GPU. Mandatory; no XTTS fallback.

Dual reference only: timbre from speaker_wav_path, prosody half from style_audio_path.
StyleTTS2 ref_s = concat(style_encoder, predictor_encoder); inference blends [:half] as timbre
and [half:] as prosody — we merge voice[:half] + segment[half:].

No single-reference fallback: both local WAV paths must exist.
"""
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Public default clip (LibriVox / StyleTTS2 demo); used for startup warmup only.
STYLETTS_DEFAULT_REF_URL = "https://styletts2.github.io/wavs/LJSpeech/OOD/GT/00001.wav"

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
    import styletts2.tts as tts
    from worker.models.styletts_pred_dur_fix import apply_styletts_pred_dur_fix

    apply_styletts_pred_dur_fix(tts)
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


def _merged_ref_s_voice_and_segment(model: Any, speaker_wav_path: str, style_audio_path: str) -> torch.Tensor:
    """Timbre from clone reference, prosody-related half from style segment (StyleTTS2 layout)."""
    ref_voice = model.compute_style(speaker_wav_path)
    ref_segment = model.compute_style(style_audio_path)
    dim = ref_voice.shape[-1]
    half = dim // 2
    if ref_segment.shape[-1] != dim:
        raise ValueError("Style reference embedding size mismatch")
    return torch.cat([ref_voice[:, :half], ref_segment[:, half:]], dim=-1)


def generate_speech_styletts(
    text: str,
    output_path: str,
    *,
    language: str = "en",
    speaker_wav_path: str,
    style_audio_path: str,
) -> None:
    """
    StyleTTS2 with merged ref_s only. Both WAV paths must exist (same file allowed, e.g. warmup).
    """
    _ = language  # reserved for future multilingual TTS; StyleTTS2 path is English phonemizer today
    if not speaker_wav_path or not os.path.isfile(speaker_wav_path):
        raise FileNotFoundError("speaker_wav_path must be a readable WAV: %r" % (speaker_wav_path,))
    if not style_audio_path or not os.path.isfile(style_audio_path):
        raise FileNotFoundError("style_audio_path must be a readable WAV: %r" % (style_audio_path,))
    model = get_styletts_model()
    ref_s = _merged_ref_s_voice_and_segment(model, speaker_wav_path, style_audio_path)
    model.inference(
        text,
        output_wav_file=output_path,
        output_sample_rate=24000,
        ref_s=ref_s,
    )


def warmup_generate_short(output_wav: str, language: str = "en") -> None:
    """Prime StyleTTS2 with the public default reference used for both halves (valid merged ref_s)."""
    from cached_path import cached_path

    ref = cached_path(STYLETTS_DEFAULT_REF_URL)
    generate_speech_styletts(
        "hello",
        output_wav,
        language=language,
        speaker_wav_path=ref,
        style_audio_path=ref,
    )
