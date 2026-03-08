"""
Worker S3 helpers: download from S3, upload to S3. Uses same bucket and key layout as backend.
"""
import os
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config


def get_s3_client(region: str = None, access_key: str = None, secret_key: str = None):
    return boto3.client(
        "s3",
        region_name=region or os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


def get_bucket():
    return os.getenv("S3_BUCKET", "ai-dubbing-platform")


def download_file(s3_key: str, local_path: str) -> None:
    client = get_s3_client()
    client.download_file(get_bucket(), s3_key, local_path)


def download_to_temp(s3_key: str, suffix: str = "") -> str:
    """Download S3 object to a temp file. Returns path. Caller must clean up."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    download_file(s3_key, path)
    return path


def upload_file(local_path: str, s3_key: str, content_type: str = None) -> None:
    client = get_s3_client()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    client.upload_file(local_path, get_bucket(), s3_key, ExtraArgs=extra or None)


def upload_bytes(body: bytes, s3_key: str, content_type: str = "application/octet-stream") -> None:
    client = get_s3_client()
    client.put_object(Bucket=get_bucket(), Key=s3_key, Body=body, ContentType=content_type)


def upload_json(s3_key: str, data: dict) -> None:
    import json
    upload_bytes(json.dumps(data).encode("utf-8"), s3_key, content_type="application/json")


def list_keys(prefix: str) -> list:
    """List object keys under prefix."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=get_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def object_exists(s3_key: str) -> bool:
    try:
        get_s3_client().head_object(Bucket=get_bucket(), Key=s3_key)
        return True
    except Exception:
        return False


def object_exists_and_non_empty(s3_key: str, min_size: int = 1024) -> bool:
    """True only if object exists and has at least min_size bytes (avoids treating empty/corrupt as done)."""
    try:
        resp = get_s3_client().head_object(Bucket=get_bucket(), Key=s3_key)
        return (resp.get("ContentLength") or 0) >= min_size
    except Exception:
        return False
