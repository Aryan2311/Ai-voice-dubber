"""
Compact job summaries for worker logs (no large payloads).
"""
from __future__ import annotations

from typing import Any, Dict


def _trunc(s: str, max_len: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def brief_job(job: Dict[str, Any]) -> str:
    """Single-line context: ids, language, flags, lengths — safe for INFO logs."""
    parts: list[str] = []
    jt = job.get("job_type")
    if jt:
        parts.append("job_type=%s" % jt)
    for key in ("job_id", "request_id"):
        v = job.get(key)
        if v is not None and str(v).strip():
            parts.append("%s=%s" % (key, str(v).strip()))
            break
    mid = job.get("media_id")
    if mid:
        parts.append("media_id=%s" % mid)
    lang = job.get("language")
    if lang is not None and str(lang).strip():
        parts.append("language=%s" % lang)
    fmt = job.get("format")
    if fmt:
        parts.append("format=%s" % fmt)
    vs = job.get("voice_sample")
    if vs is not None and str(vs).strip():
        vs_s = str(vs).strip()
        parts.append("voice_sample=%s" % (vs_s[-72:] if len(vs_s) > 72 else vs_s))
    txt = job.get("text")
    if isinstance(txt, str) and txt:
        parts.append("text_len=%d" % len(txt))
        parts.append("text_preview=%r" % _trunc(txt, 72))
    if "skip_if_exists" in job:
        parts.append("skip_if_exists=%s" % bool(job.get("skip_if_exists")))
    return " ".join(parts) if parts else "(empty job)"


def log_preview(text: str, max_len: int = 120) -> str:
    """Short quoted preview for translation/rewrite logs."""
    return _trunc(text, max_len)
