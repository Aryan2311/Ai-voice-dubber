"""
Local job queue. SQS messages are enqueued here; a single processor thread
pulls jobs and runs them (with GPU lock for GPU jobs).
"""
import queue
from typing import Optional, Tuple

# (receipt_handle, job_body)
QueuedItem = Tuple[str, dict]

JOB_QUEUE: "queue.Queue[QueuedItem]" = queue.Queue()


def add_job(receipt_handle: str, job: dict) -> None:
    JOB_QUEUE.put((receipt_handle, job))


def get_job(block: bool = True, timeout: Optional[float] = None) -> Optional[QueuedItem]:
    try:
        return JOB_QUEUE.get(block=block, timeout=timeout)
    except queue.Empty:
        return None


def queue_size() -> int:
    return JOB_QUEUE.qsize()
