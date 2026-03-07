"""
Global GPU mutex. Ensures only one GPU job runs at a time (single EC2 GPU worker).
Never allow parallel GPU jobs unless VRAM is explicitly managed for multi-job.
"""
import threading

GPU_LOCK = threading.Lock()

# Max concurrent GPU jobs (1 = safe for ~24GB; increase only with VRAM budget)
MAX_GPU_JOBS = 1
