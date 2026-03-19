"""
Whisper model loader. Load once; unload not required if running sequentially with other GPU models
(via GPU_LOCK + empty_cache between stages). Kept for consistent pipeline interface.
"""
import os

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    import whisper
    size = os.getenv("WHISPER_MODEL_SIZE", "base")
    _whisper_model = whisper.load_model(size)
    return _whisper_model


def load_whisper(model_size: str = None):
    return get_whisper_model()


def unload_whisper():
    """Release Whisper from GPU so other models can use VRAM."""
    global _whisper_model
    _whisper_model = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
