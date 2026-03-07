"""
Audio utilities: load WAV, save WAV, simple concatenation or segment handling.
Worker can use this for RVC/XTTS output handling.
"""
import wave
import struct
from pathlib import Path
from typing import List, Tuple


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
