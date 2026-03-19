"""
Extract prosody (pitch, energy) from segment audio for TTS conditioning. CPU only.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_prosody(audio_path: str) -> Dict[str, Any]:
    """
    Extract prosody features from audio. Returns dict with pitch_mean, energy_mean, etc.
    Used by TTS (e.g. StyleTTS2) for style embedding when available.
    """
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            return {"pitch_mean": 0.0, "energy_mean": 0.0}
        # Simple pitch (YIN) and energy (RMS)
        fmin, fmax = librosa.note_to_hz("C2"), librosa.note_to_hz("C7")
        pitch = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
        pitch_mean = float(np.nanmean(pitch)) if pitch.size else 0.0
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms)) if rms.size else 0.0
        return {"pitch_mean": pitch_mean, "energy_mean": energy_mean}
    except Exception as e:
        logger.debug("Prosody extraction failed: %s", e)
        return {"pitch_mean": 0.0, "energy_mean": 0.0}
