"""
Reference audio preprocessing for StyleTTS2 and RVC.

Best practice:
- Short clean segments: 3–10 s (we use 3–8 s)
- Neutral speech for voice ref; emotion comes from prosody transfer
- Normalize volume for stable embeddings
- Optional: voice isolation (Demucs/UVR) for noisy input
- Consistent format: 22050 Hz mono

Pipeline: [optional isolate] → normalize → resample 22050 mono → trim to max_duration
"""
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# StyleTTS2 / RVC reference format (user recommendation)
REFERENCE_TARGET_SR = 22050
REFERENCE_MAX_DURATION_SEC = 8
REFERENCE_MIN_DURATION_SEC = 3


def preprocess_reference_audio(
    input_path: str,
    output_path: str,
    max_duration_sec: float = REFERENCE_MAX_DURATION_SEC,
    target_sr: int = REFERENCE_TARGET_SR,
    normalize: bool = True,
    isolate_voice: bool = False,
) -> None:
    """
    Preprocess reference audio for voice cloning / StyleTTS2.
    Steps: [optional voice isolation] → load → normalize → resample mono → trim → save.
    """
    import numpy as np
    import librosa
    import soundfile as sf

    if not os.path.isfile(input_path):
        raise FileNotFoundError("Reference audio not found: %s" % input_path)

    if isolate_voice:
        input_path = _isolate_voice_track(input_path)

    y, sr = librosa.load(input_path, sr=None, mono=True)
    if len(y) == 0:
        raise ValueError("Reference audio is empty: %s" % input_path)

    if normalize:
        y = librosa.util.normalize(y.astype(np.float32))

    # Resample to target_sr and ensure mono
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Trim to max_duration_sec (take first N seconds of clean speech)
    max_samples = int(max_duration_sec * target_sr)
    if len(y) > max_samples:
        y = y[:max_samples]
        logger.debug("Trimmed reference to %.1f s", max_duration_sec)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, y, target_sr, subtype="PCM_16")
    logger.debug("Preprocessed reference -> %s (%d Hz, %.2f s)", output_path, target_sr, len(y) / target_sr)


def preprocess_prosody_segment(
    input_path: str,
    output_path: str,
    target_sr: int = REFERENCE_TARGET_SR,
    normalize: bool = True,
) -> None:
    """
    Preprocess a segment used as prosody reference (original segment slice).
    Same sample rate and normalization as voice refs for stable StyleTTS2 conditioning.
    """
    import numpy as np
    import librosa
    import soundfile as sf

    if not os.path.isfile(input_path):
        return
    y, sr = librosa.load(input_path, sr=None, mono=True)
    if len(y) == 0:
        return
    if normalize:
        y = librosa.util.normalize(y.astype(np.float32))
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, y, target_sr, subtype="PCM_16")


def _isolate_voice_track(input_path: str) -> str:
    """Run Demucs CLI to extract vocals. Returns path to isolated vocal WAV. Requires: pip install demucs."""
    import subprocess
    out_dir = tempfile.mkdtemp(prefix="demucs_")
    try:
        subprocess.run(
            ["python", "-m", "demucs", "-n", "htdemucs", input_path, "-o", out_dir],
            check=True,
            capture_output=True,
            timeout=120,
        )
        name = os.path.splitext(os.path.basename(input_path))[0]
        vocals_path = os.path.join(out_dir, "htdemucs", name, "vocals.wav")
        if os.path.isfile(vocals_path):
            return vocals_path
    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        logger.warning("Voice isolation (Demucs) skipped: %s", e)
    return input_path


def ensure_preprocessed_reference(
    raw_path: str,
    tmp_dir: str,
    max_duration_sec: float = REFERENCE_MAX_DURATION_SEC,
    isolate_voice: bool = False,
) -> str:
    """
    If raw_path exists, preprocess to a new file in tmp_dir and return its path.
    Otherwise return raw_path. Caller can use the returned path for StyleTTS2/RVC.
    """
    if not raw_path or not os.path.isfile(raw_path):
        return raw_path
    out_path = os.path.join(tmp_dir, "reference_preprocessed.wav")
    preprocess_reference_audio(
        raw_path,
        out_path,
        max_duration_sec=max_duration_sec,
        target_sr=REFERENCE_TARGET_SR,
        normalize=True,
        isolate_voice=isolate_voice,
    )
    return out_path
