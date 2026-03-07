"""
GPU session context manager. Acquire global lock for GPU work, clear cache on exit.
"""
import logging
from contextlib import contextmanager

from worker.gpu.gpu_locks import GPU_LOCK
from worker.gpu.vram_monitor import log_vram

logger = logging.getLogger(__name__)


@contextmanager
def gpu_session(clear_cache_on_exit=True):
    """
    Acquire GPU lock, run block, release lock.
    Optionally clear CUDA cache on exit to free VRAM for next job.
    """
    logger.info("Acquiring GPU lock")
    GPU_LOCK.acquire()
    try:
        log_vram()
        yield
    finally:
        if clear_cache_on_exit:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log_vram()
            except ImportError:
                pass
        GPU_LOCK.release()
        logger.info("Released GPU lock")


def finish_job_clear_cache():
    """Call after GPU work to free VRAM (e.g. after run in gpu_session)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_vram()
    except ImportError:
        pass
