"""
TRANSLATE_TRANSCRIPT: read transcripts/{media_id}/original.json, translate to language, upload transcripts/{media_id}/{lang}.json
"""
import json
import logging
import os
import tempfile
import time

from worker.utils import s3_utils
from worker.pipeline import translate as translate_pipeline
from worker.utils.job_logging import brief_job

logger = logging.getLogger(__name__)


def run_translate_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job["language"]
    t0 = time.monotonic()
    ctx = "media_id=%s language=%s" % (media_id, language)
    logger.info(
        "TRANSLATE_TRANSCRIPT begin %s payload=%s",
        ctx,
        brief_job(job),
    )

    original_key = f"transcripts/{media_id}/original.json"
    if not s3_utils.object_exists(original_key):
        raise FileNotFoundError("Original transcript missing. Run TRANSCRIBE first: %s" % original_key)

    with tempfile.TemporaryDirectory() as tmp:
        local_orig = os.path.join(tmp, "original.json")
        s3_utils.download_file(original_key, local_orig)
        with open(local_orig, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_lang = data.get("language", "en")
        segments = data.get("segments", [])
        logger.info(
            "TRANSLATE_TRANSCRIPT %s source_lang=%s segments=%s",
            ctx,
            source_lang,
            len(segments),
        )
        translated = translate_pipeline.translate_segments(
            segments, source_lang, language, log_context=ctx
        )
        full_text = " ".join(s["text"] for s in translated)

        out = {
            "media_id": media_id,
            "language": language,
            "segments": translated,
            "full_text": full_text,
        }
        key = f"transcripts/{media_id}/{language}.json"
        s3_utils.upload_json(key, out)
        logger.info(
            "TRANSLATE_TRANSCRIPT done %s uploaded_key=%s full_text_chars=%d elapsed_sec=%.1f",
            ctx,
            key,
            len(full_text),
            time.monotonic() - t0,
        )
