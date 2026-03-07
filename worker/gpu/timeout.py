"""
Optional timeout for GPU jobs (e.g. prevent hung inference).
On Unix: signal.alarm; on Windows: no-op or use threading.Timer (more involved).
"""
import logging
import os
import signal
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Job exceeded timeout")


@contextmanager
def job_timeout(seconds: Optional[int] = None):
    """
    Raise TimeoutError after seconds (Unix only; Windows is no-op if seconds set).
    Use with care: signal.alarm can interfere with other code. Prefer process-level timeouts.
    """
    if seconds is None or seconds <= 0:
        yield
        return
    if os.name != "posix":
        logger.debug("Timeout not supported on this platform")
        yield
        return
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
