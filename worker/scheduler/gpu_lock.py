"""
GPU lock for sequential GPU use. Only one GPU model (ASR, TTS) runs at a time.
Use: with GPU_LOCK: run_whisper() / run_tts()
"""
from worker.gpu.gpu_locks import GPU_LOCK

__all__ = ["GPU_LOCK"]
