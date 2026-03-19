"""
RVC (Retrieval-based Voice Conversion). GPU. No passthrough: when conversion is requested,
RVC_MODEL_PATH must be set to a trained .pth; otherwise raises.
"""
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)

_rvc_inference: Optional[Any] = None


def load_rvc() -> Optional[Any]:
    """
    Load RVC model from RVC_MODEL_PATH (.pth). Returns inference object or None if path not set.
    """
    global _rvc_inference
    if _rvc_inference is not None:
        return _rvc_inference
    path = os.getenv("RVC_MODEL_PATH", "").strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        from rvc_python.infer import RVCInference
        device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        _rvc_inference = RVCInference(device=device)
        _rvc_inference.load_model(path)
        logger.info("RVC loaded: %s", path)
        return _rvc_inference
    except ImportError as e:
        raise RuntimeError("RVC requires rvc-python. pip install rvc-python") from e


def convert_voice(input_wav_path: str, output_wav_path: str, speaker_wav_path: str = None) -> bool:
    """
    Run RVC conversion. Requires RVC_MODEL_PATH to be set; raises if conversion requested but RVC not loaded.
    No passthrough: does not copy input to output.
    """
    rvc = load_rvc()
    if rvc is None:
        raise RuntimeError(
            "RVC conversion requested but RVC_MODEL_PATH is not set or file not found. "
            "Set RVC_MODEL_PATH to a trained .pth model file."
        )
    rvc.infer_file(input_wav_path, output_wav_path)
    return True
