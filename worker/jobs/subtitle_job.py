"""
GENERATE_SUBTITLE: ensure translated transcript exists, generate SRT or VTT, upload to subtitles/{media_id}/{lang}.srt|.vtt
"""
import json
import logging
import os
import tempfile

from worker.utils import s3_utils
from worker.jobs import translate_job

logger = logging.getLogger(__name__)


def _segments_to_srt(segments: list) -> str:
    def _ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return "%02d:%02d:%02d,%03d" % (h, m, s, ms)

    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip().replace("\n", " ")
        lines.append("%d\n%s --> %s\n%s\n" % (i, _ts(start), _ts(end), text))
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    def _ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec % 1) * 1000)
        return "%02d:%02d:%02d.%03d" % (h, m, s, ms)

    lines = ["WEBVTT", ""]
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip().replace("\n", " ")
        lines.append("%s --> %s" % (_ts(start), _ts(end)))
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def run_subtitle_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job["language"]
    fmt = (job.get("format") or "srt").lower()
    if fmt not in ("srt", "vtt"):
        fmt = "srt"
    logger.info("GENERATE_SUBTITLE media_id=%s language=%s format=%s", media_id, language, fmt)

    transcript_key = f"transcripts/{media_id}/{language}.json"
    if not s3_utils.object_exists(transcript_key):
        logger.info("GENERATE_SUBTITLE media_id=%s running TRANSLATE first", media_id)
        translate_job.run_translate_job({"job_type": "TRANSLATE_TRANSCRIPT", "media_id": media_id, "language": language})

    with tempfile.TemporaryDirectory() as tmp:
        local_json = os.path.join(tmp, "transcript.json")
        s3_utils.download_file(transcript_key, local_json)
        with open(local_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])

        if fmt == "srt":
            content = _segments_to_srt(segments)
            content_type = "application/x-subrip"
        else:
            content = _segments_to_vtt(segments)
            content_type = "text/vtt"

        key = f"subtitles/{media_id}/{language}.{fmt}"
        s3_utils.upload_bytes(content.encode("utf-8"), key, content_type=content_type)
        logger.info("GENERATE_SUBTITLE media_id=%s uploaded key=%s segments=%d", media_id, key, len(segments))
