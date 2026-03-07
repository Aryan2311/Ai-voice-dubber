"""
TRANSCRIBE: ensure source audio exists (preprocess), run Whisper, upload transcripts/{media_id}/original.json
"""
import logging
import os
import tempfile

from worker.utils import s3_utils, media_preprocess
from worker.ai_models import whisper_model

logger = logging.getLogger(__name__)


def run_transcribe_job(job: dict) -> None:
    media_id = job["media_id"]
    logger.info("TRANSCRIBE media_id=%s ensuring source audio", media_id)

    media_preprocess.ensure_source_audio(media_id)

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "source.wav")
        s3_utils.download_file(f"audio/{media_id}/source.wav", audio_path)
        logger.info("TRANSCRIBE media_id=%s running Whisper", media_id)

        result = whisper_model.transcribe_audio(audio_path)
        transcript = {
            "media_id": media_id,
            "segments": result["segments"],
            "full_text": result["full_text"],
            "language": result.get("language", "en"),
        }

        key = f"transcripts/{media_id}/original.json"
        s3_utils.upload_json(key, transcript)
        logger.info("TRANSCRIBE media_id=%s uploaded key=%s segments=%d", media_id, key, len(result.get("segments", [])))
