"""
EC2 GPU Worker: SQS → local queue → GPU scheduler → one job at a time.
- Listener thread: receive SQS messages, enqueue (receipt_handle, job).
- Processor thread: get job from queue; GPU jobs run inside gpu_session(), then delete SQS message.
Job types: TRANSCRIBE, TRANSLATE_TRANSCRIPT, GENERATE_SUBTITLE (CPU), TEXT_TO_SPEECH, DUB_MEDIA.
TEXT_TO_SPEECH requires voice_sample only; DUB_MEDIA requires voice_sample (plus source audio on S3 for segment prosody).
"""
import contextlib
import gc
import json
import os
import time
import logging
import threading

import boto3
from botocore.config import Config

from worker.jobs import transcribe_job, translate_job, subtitle_job, tts_job, dub_job
from worker.gpu import gpu_session
from worker.scheduler import add_job, get_job, queue_size
from worker.utils import s3_utils
from worker.utils.job_logging import brief_job

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


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _stagger_after_load():
    """Optional pause + GC between eager loads to reduce peak RAM/IO (see MODEL_LOAD_STAGGER_SEC)."""
    sec = float(os.getenv("MODEL_LOAD_STAGGER_SEC", "0") or "0")
    if sec > 0:
        logger.info(
            "[startup] Stagger: sleeping %.1f s (MODEL_LOAD_STAGGER_SEC) before next phase…",
            sec,
        )
        time.sleep(sec)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _configure_sequential_hub_downloads():
    """
    One model init at a time in this process; also discourage parallel HF shard downloads.
    (Transformers loads are still sequential — this mainly limits hub thread pool / transfer.)
    """
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_MAX_WORKERS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info(
        "[startup] Hugging Face / tokenizer env: HF_HUB_ENABLE_HF_TRANSFER=%r "
        "HF_HUB_DOWNLOAD_MAX_WORKERS=%r TOKENIZERS_PARALLELISM=%r",
        os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"),
        os.environ.get("HF_HUB_DOWNLOAD_MAX_WORKERS"),
        os.environ.get("TOKENIZERS_PARALLELISM"),
    )


def _process_rss_mib_linux():
    """Best-effort RSS on Linux (worker runs on EC2 Linux)."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            parts = f.read().split()
        rss_pages = int(parts[1])
        page = 4096
        try:
            page = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            pass
        return (rss_pages * page) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None


def _log_startup_resources(tag: str):
    """Memory snapshot after each startup phase."""
    rss = _process_rss_mib_linux()
    if rss is not None:
        logger.info("[startup] %s — process RSS ~%.0f MiB", tag, rss)
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                "[startup] %s — GPU allocated=%.2f GiB reserved=%.2f GiB",
                tag,
                alloc,
                reserved,
            )
        else:
            logger.info("[startup] %s — CUDA not available", tag)
    except Exception as ex:
        logger.info("[startup] %s — GPU stats unavailable: %s", tag, ex)


@contextlib.contextmanager
def _startup_phase(phase_id: str, title: str, detail: str = ""):
    """Sequential phase: log begin, resources, elapsed, resources again."""
    extra = f" — {detail}" if detail else ""
    logger.info("[startup] %s BEGIN %s%s", phase_id, title, extra)
    _log_startup_resources(f"{phase_id} (before {title})")
    t0 = time.monotonic()
    try:
        yield
    except Exception:
        logger.exception("[startup] %s FAILED: %s", phase_id, title)
        raise
    elapsed = time.monotonic() - t0
    logger.info("[startup] %s END %s — finished in %.1f s", phase_id, title, elapsed)
    _log_startup_resources(f"{phase_id} (after {title})")
    _stagger_after_load()


def load_models_once():
    # Lighter startup: models load on first real job (first request is slower).
    if _env_truthy("WORKER_LAZY_LOAD"):
        logger.info(
            "WORKER_LAZY_LOAD=1: skipping startup model load; models load on first use."
        )
        return

    _configure_sequential_hub_downloads()
    logger.info(
        "[startup] ========== Sequential model load (one component at a time) =========="
    )
    t_all = time.monotonic()
    try:
        from worker.models import whisper_loader, nllb_loader, styletts_loader

        w_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        with _startup_phase(
            "1/5",
            "Whisper (ASR)",
            f"model_size={w_size!r} (env WHISPER_MODEL_SIZE)",
        ):
            whisper_loader.load_whisper()
            logger.info(
                "[startup] Whisper weights ready (OpenAI whisper.load_model)."
            )

        with _startup_phase(
            "2/5",
            "NLLB-200 translation",
            "facebook/nllb-200-distilled-600M — tokenizer then weights (CPU)",
        ):
            nllb_loader.load_nllb()
            logger.info("[startup] NLLB tokenizer + seq2seq model ready.")

        ck = os.getenv("STYLETTS2_CHECKPOINT")
        cf = os.getenv("STYLETTS2_CONFIG")
        stts_detail = (
            f"custom checkpoint={ck!r} config={cf!r}"
            if ck and cf
            else "default checkpoints (StyleTTS2())"
        )
        with _startup_phase("3/5", "StyleTTS2 (GPU TTS)", stts_detail):
            styletts_loader.get_styletts_model()
            logger.info("[startup] StyleTTS2 inference object ready.")

        if _env_truthy("WORKER_SKIP_LLM_PRELOAD"):
            logger.info(
                "[startup] 4/5 SKIP: WORKER_SKIP_LLM_PRELOAD=1 — "
                "rewrite LLM (Phi-3) will load on first dub/rewrite job."
            )
        else:
            from worker.models import llm_loader

            llm_name = os.getenv("REWRITE_LLM_MODEL", llm_loader.DEFAULT_MODEL)
            with _startup_phase(
                "4/5",
                "Rewrite LLM (Phi-3, CPU)",
                f"model={llm_name!r} — tokenizer download/load then weights (can take many minutes)",
            ):
                llm_loader.get_llm()
                logger.info("[startup] Rewrite LLM + tokenizer ready.")

        if _env_truthy("WORKER_SKIP_WARMUP"):
            logger.info(
                "[startup] 5/5 SKIP: WORKER_SKIP_WARMUP=1 — no StyleTTS GPU warmup."
            )
        else:
            with _startup_phase(
                "5/5",
                "StyleTTS2 GPU warmup",
                "short inference 'hello' inside gpu_session()",
            ):
                _warmup_models(styletts_loader)

    except Exception as e:
        logger.warning("Startup model load aborted or partial failure: %s", e)
    total = time.monotonic() - t_all
    logger.info(
        "[startup] ========== Sequential model load finished (total wall %.1f s) ==========",
        total,
    )


def _warmup_models(styletts_loader):
    """One GPU inference to prime StyleTTS2; must run inside a startup phase."""
    import tempfile

    logger.info("[startup] Warmup: generating short WAV via StyleTTS2…")
    with gpu_session(clear_cache_on_exit=True):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            styletts_loader.warmup_generate_short(path, language="en")
        finally:
            if os.path.exists(path):
                os.remove(path)
    logger.info("[startup] Warmup: StyleTTS2 completed OK.")


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
                    logger.info(
                        "SQS→local_queue job_type=%s job_id=%s queue_depth_after=%d payload=%s",
                        job_type,
                        job_id,
                        queue_size(),
                        brief_job(job),
                    )
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
                logger.info(
                    "Job BEGIN job_type=%s job_id=%s requires_gpu=%s payload=%s",
                    job_type,
                    job_id,
                    requires_gpu(job),
                    brief_job(job),
                )
                if requires_gpu(job):
                    with gpu_session():
                        handle_job(job)
                else:
                    handle_job(job)
                elapsed = time.monotonic() - started
                logger.info(
                    "Job OK job_type=%s job_id=%s elapsed_sec=%.1f",
                    job_type,
                    job_id,
                    elapsed,
                )
            except Exception as e:
                elapsed = time.monotonic() - started
                logger.exception(
                    "Job FAIL job_type=%s job_id=%s elapsed_sec=%.1f payload=%s",
                    job_type,
                    job_id,
                    elapsed,
                    brief_job(job),
                )
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
