"""
DUB_MEDIA: full pipeline (ASR → translate → rewrite → TTS → align → merge).
Output: video -> dubbed/{media_id}/{lang}.mp4; audio-only -> audio/{media_id}/{lang}.wav
Same S3 contract as before; backend APIs unchanged.
"""
import logging

from worker.pipeline.dub import run_dub

logger = logging.getLogger(__name__)


def run_dub_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job.get("language", "es")
    voice_sample_s3 = job.get("voice_sample")
    # Default to prosody-enabled dubbing when backend does not pass this field.
    use_prosody = job.get("use_prosody", True)
    logger.info("DUB_MEDIA media_id=%s language=%s use_prosody=%s", media_id, language, use_prosody)

    run_dub(
        media_id=media_id,
        language=language,
        voice_sample_s3=voice_sample_s3,
        use_prosody=use_prosody,
    )
