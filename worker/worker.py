"""
EC2 GPU Worker: SQS → local queue → GPU scheduler → one job at a time.
- Listener thread: receive SQS messages, enqueue (receipt_handle, job).
- Processor thread: get job from queue; GPU jobs run inside gpu_session(), then delete SQS message.
Job types: TRANSCRIBE, TRANSLATE_TRANSCRIPT, GENERATE_SUBTITLE (CPU), TEXT_TO_SPEECH, DUB_MEDIA.
"""
import json
import os
import time
import logging
import threading

import boto3
from botocore.config import Config

from worker.jobs import transcribe_job, translate_job, subtitle_job, tts_job, dub_job
from worker.gpu import gpu_session
from worker.scheduler import add_job, get_job
from worker.utils import s3_utils

# Clear format for docker logs: UTC timestamp, level, name, message
logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

JOB_TRANSCRIBE = "TRANSCRIBE"
JOB_TRANSLATE_TRANSCRIPT = "TRANSLATE_TRANSCRIPT"
JOB_GENERATE_SUBTITLE = "GENERATE_SUBTITLE"
JOB_TEXT_TO_SPEECH = "TEXT_TO_SPEECH"
JOB_DUB_MEDIA = "DUB_MEDIA"

# Only GENERATE_SUBTITLE is CPU-only (format SRT/VTT from JSON; translate dependency runs in same thread)
GPU_JOB_TYPES = {JOB_TRANSCRIBE, JOB_TRANSLATE_TRANSCRIPT, JOB_TEXT_TO_SPEECH, JOB_DUB_MEDIA}


def requires_gpu(job: dict) -> bool:
    return job.get("job_type") in GPU_JOB_TYPES


def get_sqs_client():
    return boto3.client(
        "sqs",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(region_name=os.getenv("AWS_REGION", "us-east-1")),
    )


def get_queue_url():
    url = os.getenv("SQS_QUEUE_URL")
    if not url:
        raise ValueError("SQS_QUEUE_URL is not set")
    return url


def handle_job(job: dict) -> None:
    job_type = job.get("job_type")
    if job_type == JOB_TRANSCRIBE:
        transcribe_job.run_transcribe_job(job)
    elif job_type == JOB_TRANSLATE_TRANSCRIPT:
        translate_job.run_translate_job(job)
    elif job_type == JOB_GENERATE_SUBTITLE:
        subtitle_job.run_subtitle_job(job)
    elif job_type == JOB_TEXT_TO_SPEECH:
        tts_job.run_tts_job(job)
    elif job_type == JOB_DUB_MEDIA:
        dub_job.run_dub_job(job)
    else:
        raise ValueError("Unknown job_type: %s" % job_type)


def load_models_once():
    logger.info("Loading AI models (once) at startup...")
    try:
        from worker.models import whisper_loader, nllb_loader, styletts_loader
        whisper_loader.load_whisper()
        logger.info("Whisper loaded.")
        nllb_loader.load_nllb()
        logger.info("NLLB loaded.")
        styletts_loader.get_styletts_model()
        logger.info("StyleTTS2 loaded.")
        from worker.models import llm_loader
        llm_loader.get_llm()
        logger.info("Rewrite LLM loaded.")
        _warmup_models(whisper_loader, styletts_loader)
    except Exception as e:
        logger.warning("Some models failed to load: %s", e)
    logger.info("Model loading done.")


def _warmup_models(whisper_loader, styletts_loader):
    """Optional warmup to avoid first-inference delay. Skip if it fails."""
    import tempfile
    try:
        with gpu_session(clear_cache_on_exit=True):
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                styletts_loader.generate_speech_styletts("hello", path, language="en")
            finally:
                if os.path.exists(path):
                    os.remove(path)
        logger.info("Model warmup done.")
    except Exception as e:
        logger.debug("Warmup skipped: %s", e)


def sqs_listener_loop(sqs, queue_url, wait_time_seconds=20):
    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=900,
            )
            messages = resp.get("Messages", [])
            for msg in messages:
                receipt_handle = msg["ReceiptHandle"]
                try:
                    job = json.loads(msg["Body"])
                    add_job(receipt_handle, job)
                    job_type = job.get("job_type", "?")
                    job_id = job.get("job_id") or job.get("media_id") or job.get("request_id") or "—"
                    logger.info("Enqueued job_type=%s job_id=%s", job_type, job_id)
                except Exception as e:
                    logger.exception("Failed to enqueue message: %s", e)
        except Exception as e:
            logger.exception("SQS receive error: %s", e)
            time.sleep(5)


def processor_loop(sqs, queue_url):
    while True:
        try:
            item = get_job(block=True, timeout=30)
            if item is None:
                continue
            receipt_handle, job = item
            job_type = job.get("job_type", "?")
            job_id = job.get("job_id") or job.get("media_id") or job.get("request_id") or "—"
            started = time.monotonic()
            try:
                logger.info("Starting job_type=%s job_id=%s", job_type, job_id)
                if requires_gpu(job):
                    with gpu_session():
                        handle_job(job)
                else:
                    handle_job(job)
                elapsed = time.monotonic() - started
                logger.info("Completed job_type=%s job_id=%s elapsed_sec=%.1f", job_type, job_id, elapsed)
            except Exception as e:
                elapsed = time.monotonic() - started
                logger.exception("Failed job_type=%s job_id=%s elapsed_sec=%.1f error=%s", job_type, job_id, elapsed, e)
                # Record failure in S3 so backend/UI can show failed status
                sid = job.get("job_id")
                if sid:
                    try:
                        key = f"job_failures/{sid}.json"
                        s3_utils.upload_bytes(
                            json.dumps({"status": "failed", "error": str(e)}).encode("utf-8"),
                            key,
                            content_type="application/json",
                        )
                        logger.info("Written job failure to s3 key=%s", key)
                    except Exception as s3_err:
                        logger.warning("Could not write job failure to S3: %s", s3_err)
            finally:
                # Always remove from queue once processed (success or failure) so the same job is never redelivered
                try:
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                except Exception as del_err:
                    logger.warning("Failed to delete SQS message job_id=%s: %s", job_id, del_err)
        except Exception as e:
            logger.exception("Processor error: %s", e)
            time.sleep(5)


def run_worker():
    load_models_once()
    sqs = get_sqs_client()
    queue_url = get_queue_url()

    logger.info("Worker started. SQS: %s. Listener + processor threads.", queue_url)

    listener = threading.Thread(target=sqs_listener_loop, args=(sqs, queue_url), daemon=False)
    processor = threading.Thread(target=processor_loop, args=(sqs, queue_url), daemon=False)
    listener.start()
    processor.start()
    listener.join()
    processor.join()


if __name__ == "__main__":
    run_worker()
