"""
Worker job status helpers backed by S3.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from worker.utils import s3_utils


def job_status_key(job_id: str) -> str:
    return f"job_status/{job_id}.json"


def read_job_status(job_id: str) -> Optional[dict[str, Any]]:
    if not job_id:
        return None
    key = job_status_key(job_id)
    try:
        resp = s3_utils.get_s3_client().get_object(Bucket=s3_utils.get_bucket(), Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception:
        return None


def write_job_status(job_id: str, status: str, **payload: Any) -> None:
    if not job_id:
        return
    body = {
        "job_id": job_id,
        "status": status,
        "updated_at": time.time(),
        **payload,
    }
    s3_utils.upload_json(job_status_key(job_id), body)
