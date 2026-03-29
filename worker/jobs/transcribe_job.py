"""
TRANSCRIBE: ensure source audio exists (preprocess), run ASR (Whisper), upload transcripts/{media_id}/original.json
"""
import logging
import os
import tempfile
import time

from worker.utils import s3_utils, media_preprocess
from worker.pipeline import asr
from worker.utils.job_logging import brief_job

logger = logging.getLogger(__name__)


def run_transcribe_job(job: dict) -> None:
    media_id = job["media_id"]
    t0 = time.monotonic()
    logger.info(
        "TRANSCRIBE begin media_id=%s payload=%s",
        media_id,
        brief_job(job),
    )

    media_preprocess.ensure_source_audio(media_id)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "source.wav")
        s3_utils.download_file(f"audio/{media_id}/source.wav", audio_path)
        t_asr = time.monotonic()
        logger.info("TRANSCRIBE media_id=%s running Whisper", media_id)

        result = asr.transcribe(audio_path)
        logger.info(
            "TRANSCRIBE media_id=%s Whisper done elapsed_sec=%.1f",
            media_id,
            time.monotonic() - t_asr,
        )
        transcript = {
            "media_id": media_id,
            "segments": result["segments"],
            "full_text": result["full_text"],
            "language": result.get("language", "en"),
        }

        key = f"transcripts/{media_id}/original.json"
        s3_utils.upload_json(key, transcript)
        nseg = len(result.get("segments", []))
        logger.info(
            "TRANSCRIBE done media_id=%s uploaded_key=%s segments=%d detected_lang=%s elapsed_sec=%.1f",
            media_id,
            key,
            nseg,
            transcript.get("language", "en"),
            time.monotonic() - t0,
        )
