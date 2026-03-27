"""
DUB_MEDIA: full pipeline (ASR → translate → rewrite → TTS → align → merge).
Output: video -> dubbed/{media_id}/{lang}.mp4; audio-only -> audio/{media_id}/{lang}.wav
Requires voice_sample on every job (clone timbre); segment prosody comes from source audio.
"""
import logging

from worker.pipeline.dub import run_dub

logger = logging.getLogger(__name__)


def run_dub_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job.get("language", "es")
    voice_sample_s3 = job.get("voice_sample")
    if not voice_sample_s3 or not str(voice_sample_s3).strip():
        raise ValueError("DUB_MEDIA requires non-empty voice_sample (S3 key to clone WAV)")
    logger.info("DUB_MEDIA media_id=%s language=%s", media_id, language)

    run_dub(
        media_id=media_id,
        language=language,
        voice_sample_s3=str(voice_sample_s3).strip(),
    )
