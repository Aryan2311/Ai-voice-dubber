"""
GPU VRAM monitoring. Use for logging and optional pre-job checks.
"""
import logging

logger = logging.getLogger(__name__)


def get_vram_usage():
    """Return allocated and reserved VRAM in bytes. Returns zeros if CUDA not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
    except ImportError:
        return {"allocated": 0, "reserved": 0}


def get_vram_usage_mb():
    """Return allocated/reserved in MB."""
    u = get_vram_usage()
    return {
        "allocated_mb": round(u["allocated"] / (1024 ** 2), 2),
        "reserved_mb": round(u["reserved"] / (1024 ** 2), 2),
    }


def has_enough_vram(max_allocated_bytes=None):
    """
    Optional: check before starting a GPU job.
    max_allocated_bytes: e.g. 20e9 for 20GB; if current allocated >= this, return False.
    """
    if max_allocated_bytes is None:
        return True
    u = get_vram_usage()
    return u["allocated"] < max_allocated_bytes


def log_vram():
    """Log current VRAM usage."""
    mb = get_vram_usage_mb()
    logger.info("VRAM allocated=%s MB reserved=%s MB", mb["allocated_mb"], mb["reserved_mb"])
