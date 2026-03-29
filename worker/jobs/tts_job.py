"""
Text-to-speech job: text + voice_sample -> StyleTTS2 -> speech.wav

Uses the same preprocessed voice clip for both StyleTTS2 reference halves (merged ref_s equals
a full single-file style embedding). Dubbing jobs use voice_sample + per-segment source audio instead.
"""
import logging
import os
import tempfile
import time

from worker.utils import s3_utils
from worker.pipeline import tts
from worker.utils.job_logging import brief_job

logger = logging.getLogger(__name__)


def run_tts_job(job: dict) -> None:
    """
    job: {
      job_type: "TEXT_TO_SPEECH",
      request_id: str,
      text: str,
      language: str,
      voice_sample: str,  # S3 key — sole reference (timbre + style halves from this clip)
    }
    Output: tts/{request_id}/speech.wav
    """
    request_id = job["request_id"]
    text = job["text"]
    language = job.get("language", "en")
    voice_sample_s3 = job.get("voice_sample")
    if not voice_sample_s3 or not str(voice_sample_s3).strip():
        raise ValueError("TEXT_TO_SPEECH requires non-empty voice_sample (S3 key)")
    voice_sample_s3 = str(voice_sample_s3).strip()
    t0 = time.monotonic()
    logger.info(
        "TEXT_TO_SPEECH begin request_id=%s payload=%s",
        request_id,
        brief_job(job),
    )

    with tempfile.TemporaryDirectory() as tmp:
        raw_voice = s3_utils.download_to_temp(voice_sample_s3, suffix=".wav")
        from worker.utils.reference_audio import ensure_preprocessed_reference

        speaker_wav = ensure_preprocessed_reference(raw_voice, tmp, isolate_voice=True)
        if not os.path.isfile(speaker_wav) or os.path.getsize(speaker_wav) == 0:
            raise RuntimeError("voice_sample produced no usable WAV after preprocessing")

        out_wav = os.path.join(tmp, "speech.wav")
        # Same path for both args → merged ref_s recovers a single-file embedding.
        tts.generate_speech(
            text,
            out_wav,
            language=language,
            speaker_wav_path=speaker_wav,
            style_audio_path=speaker_wav,
        )

        s3_key = f"tts/{request_id}/speech.wav"
        s3_utils.upload_file(out_wav, s3_key, content_type="audio/wav")
        wav_size = os.path.getsize(out_wav) if os.path.isfile(out_wav) else 0
        logger.info(
            "TEXT_TO_SPEECH done request_id=%s uploaded_key=%s wav_bytes=%d elapsed_sec=%.1f",
            request_id,
            s3_key,
            wav_size,
            time.monotonic() - t0,
        )
