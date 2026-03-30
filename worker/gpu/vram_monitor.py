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


def get_vram_free_total_mb():
    """Return free/total VRAM in MB. Returns zeros if CUDA is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"free_mb": 0, "total_mb": 0}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_mb": round(free_bytes / (1024 ** 2), 2),
            "total_mb": round(total_bytes / (1024 ** 2), 2),
        }
    except ImportError:
        return {"free_mb": 0, "total_mb": 0}


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
    free_total = get_vram_free_total_mb()
    logger.info(
        "VRAM allocated=%s MB reserved=%s MB free=%s MB total=%s MB",
        mb["allocated_mb"],
        mb["reserved_mb"],
        free_total["free_mb"],
        free_total["total_mb"],
    )
