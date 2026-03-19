"""
Duration alignment: time-stretch segment audio to target duration (segment end - start).
Uses librosa time_stretch; optional ffmpeg atempo for simple scaling.
"""
import logging
import os
from typing import List, Tuple

from worker.utils import audio_utils

logger = logging.getLogger(__name__)


def align_segment_to_duration(
    input_wav_path: str,
    output_wav_path: str,
    target_duration_sec: float,
) -> None:
    """
    Time-stretch input audio to match target_duration_sec. Preserves pitch.
    """
    if target_duration_sec <= 0 or not os.path.isfile(input_wav_path):
        if os.path.isfile(input_wav_path):
            import shutil
            shutil.copy(input_wav_path, output_wav_path)
        return
    audio_utils.time_stretch_to_duration(input_wav_path, output_wav_path, target_duration_sec)


def build_timeline_wav(
    segment_list: List[Tuple[float, float, str]],
    total_duration_sec: float,
    output_path: str,
) -> None:
    """
    Place each segment's wav at (start_sec, end_sec) and time-stretch to fit. Merge to one WAV.
    segment_list: [(start_sec, end_sec, wav_path), ...]. Gapless, non-overlapping.
    """
    audio_utils.build_timeline_wav(segment_list, total_duration_sec, output_path)
