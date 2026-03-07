"""
TRANSLATE_TRANSCRIPT: read transcripts/{media_id}/original.json, translate to language, upload transcripts/{media_id}/{lang}.json
"""
import json
import logging
import tempfile
import os

from worker.utils import s3_utils
from worker.ai_models import translator

logger = logging.getLogger(__name__)


def run_translate_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job["language"]
    logger.info("TRANSLATE_TRANSCRIPT media_id=%s language=%s", media_id, language)

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
        translated = translator.translate_segments(segments, source_lang, language)
        full_text = " ".join(s["text"] for s in translated)

        out = {
            "media_id": media_id,
            "language": language,
            "segments": translated,
            "full_text": full_text,
        }
        key = f"transcripts/{media_id}/{language}.json"
        s3_utils.upload_json(key, out)
        logger.info("TRANSLATE_TRANSCRIPT media_id=%s language=%s uploaded key=%s", media_id, language, key)
