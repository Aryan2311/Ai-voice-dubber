"""
Detect pauses in segment audio. Used to insert pauses in translated text for natural rhythm. CPU.
"""
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def detect_pauses(audio_path: str, top_db: int = 30) -> List[Tuple[float, float]]:
    """
    Return list of (start_sec, end_sec) of non-silent intervals. Gaps are pauses.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        intervals = librosa.effects.split(y, top_db=top_db)
        return [(s / float(sr), e / float(sr)) for s, e in intervals]
    except Exception as e:
        logger.debug("Pause detection failed: %s", e)
        return []


def estimate_speech_rate(audio_path: str, num_words: int) -> float:
    """Words per second. num_words from segment text."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        dur = len(y) / float(sr)
        if dur <= 0:
            return 2.0
        return num_words / dur
    except Exception:
        return 2.0
