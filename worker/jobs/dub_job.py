"""
DUB_MEDIA: full pipeline (ASR → translate → rewrite → TTS → align → merge).
Output: video -> dubbed/{media_id}/{lang}.mp4; audio-only -> audio/{media_id}/{lang}.wav
Requires voice_sample on every job (clone timbre); segment prosody comes from source audio.
Writes job_completions/{job_id}.json on success so the backend manifest can mark the job completed.
"""
import logging

from worker.pipeline.dub import run_dub
from worker.utils import s3_utils

logger = logging.getLogger(__name__)


def run_dub_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job.get("language", "es")
    voice_sample_s3 = job.get("voice_sample")
    job_id = job.get("job_id")
    if not job_id or not str(job_id).strip():
        raise ValueError("DUB_MEDIA job must include job_id (backend must send unique id per request)")
    if not voice_sample_s3 or not str(voice_sample_s3).strip():
        raise ValueError("DUB_MEDIA requires non-empty voice_sample (S3 key to clone WAV)")
    logger.info("DUB_MEDIA job_id=%s media_id=%s language=%s", job_id, media_id, language)

    skip = bool(job.get("skip_if_exists", False))
    out_key = run_dub(
        media_id=media_id,
        language=language,
        voice_sample_s3=str(voice_sample_s3).strip(),
        skip_if_output_exists=skip,
    )
    if out_key:
        s3_utils.upload_json(
            f"job_completions/{job_id}.json",
            {"status": "completed", "result_s3_key": out_key},
        )
