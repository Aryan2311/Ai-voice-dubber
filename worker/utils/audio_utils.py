"""
Audio utilities: load WAV, save WAV, simple concatenation or segment handling.
Worker can use this for XTTS output handling.
Timestamp-aligned dubbing: place each TTS clip at segment start and time-stretch to fit (end - start).
"""
import logging
import wave
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Default sample rate for timeline output (XTTS typically 24000)
TIMELINE_SAMPLE_RATE = 24000


def read_wav_frames(path: str) -> Tuple[bytes, int, int, int]:
    """Returns (frames_bytes, nchannels, sampwidth, framerate)."""
    with wave.open(path, "rb") as w:
        return (w.readframes(w.getnframes()), w.getnchannels(), w.getsampwidth(), w.getframerate())


def write_wav(path: str, frames: bytes, nchannels: int = 1, sampwidth: int = 2, framerate: int = 22050) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(frames)


def concat_wav_files(paths: List[str], output_path: str) -> None:
    """Concatenate WAV files (same format assumed)."""
    if not paths:
        raise ValueError("No input paths")
    frames_list = []
    nchannels, sampwidth, framerate = None, None, None
    for p in paths:
        with wave.open(p, "rb") as w:
            if nchannels is None:
                nchannels, sampwidth, framerate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames_list.append(w.readframes(w.getnframes()))
    with wave.open(output_path, "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(b"".join(frames_list))


def get_wav_duration_sec(path: str) -> float:
    """Return duration in seconds from WAV file."""
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        r = w.getframerate()
        return n / r if r else 0.0


def _write_float_wav(path: str, y: "np.ndarray", sr: int) -> None:
    """Write float32 array (-1..1) to WAV."""
    import numpy as np
    frames = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    write_wav(path, frames, nchannels=1, sampwidth=2, framerate=sr)


def time_stretch_to_duration(input_path: str, output_path: str, target_duration_sec: float) -> None:
    """
    Time-stretch audio so its duration equals target_duration_sec (preserves pitch).
    Uses librosa.effects.time_stretch. Overwrites output_path.
    """
    import numpy as np
    import librosa
    y, sr = librosa.load(input_path, sr=None, mono=True)
    current_dur = len(y) / float(sr)
    if current_dur <= 0 or target_duration_sec <= 0:
        n = int(round(target_duration_sec * sr))
        _write_float_wav(output_path, np.zeros(max(0, n)), sr)
        return
    rate = current_dur / target_duration_sec
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    _write_float_wav(output_path, y_stretch, sr)


def build_timeline_wav(
    segment_list: List[Tuple[float, float, str]],
    total_duration_sec: float,
    output_path: str,
    sample_rate: int = TIMELINE_SAMPLE_RATE,
) -> None:
    """
    Build a single WAV where each segment's audio is placed at its start time and
    time-stretched to fit (end - start). segment_list = [(start_sec, end_sec, wav_path), ...].
    Caller should pass gapless, non-overlapping (start, end) so there is no silence and no overlap.
    """
    import numpy as np
    import librosa
    import tempfile
    import os

    n_total = int(round(total_duration_sec * sample_rate))
    timeline = np.zeros(n_total, dtype=np.float32)

    for start_sec, end_sec, wav_path in segment_list:
        if not os.path.isfile(wav_path):
            continue
        target_dur = end_sec - start_sec
        if target_dur <= 0:
            continue
        start_sample = int(round(start_sec * sample_rate))
        if start_sample >= n_total:
            continue
        # Time-stretch this segment to fit the slot
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            time_stretch_to_duration(wav_path, tmp, target_dur)
            y, _ = librosa.load(tmp, sr=sample_rate, mono=True)
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        seg_len = len(y)
        end_sample = min(start_sample + seg_len, n_total)
        actual_len = end_sample - start_sample
        timeline[start_sample:end_sample] = y[:actual_len]

    _write_float_wav(output_path, timeline, sample_rate)
