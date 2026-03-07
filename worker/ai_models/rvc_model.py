"""
Retrieval-based Voice Conversion: load once, convert speech to cloned voice.
Optional step after XTTS for closer voice match.
"""
import os
from pathlib import Path

_rvc_model = None


def load_rvc():
    """Load RVC model once. Keep in memory if applicable."""
    global _rvc_model
    if _rvc_model is not None:
        return _rvc_model
    # RVC has multiple backends (rvc-python, etc.). Stub: optional.
    try:
        # Example: from rvc import load_rvc_model
        # _rvc_model = load_rvc_model(...)
        _rvc_model = "rvc_placeholder"  # Replace with actual load when integrating
        return _rvc_model
    except Exception:
        _rvc_model = None
        return None


def convert_voice(input_wav_path: str, output_wav_path: str, speaker_wav_path: str = None) -> bool:
    """
    If RVC is available, convert input_wav to sound like speaker_wav. Otherwise copy input to output.
    Returns True if conversion was done, False if skipped (passthrough).
    """
    model = load_rvc()
    if model is None or model == "rvc_placeholder":
        import shutil
        shutil.copy(input_wav_path, output_wav_path)
        return False
    # Actual RVC inference here
    return True
