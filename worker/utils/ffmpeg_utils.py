"""
FFmpeg utilities: extract audio, merge audio with video.
"""
import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000, mono: bool = True) -> None:
    """
    Extract audio from video to WAV. Default 16kHz mono for Whisper.
    ffmpeg -i video.mp4 -ac 1 -ar 16000 audio.wav
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def merge_audio_with_video(video_path: str, audio_path: str, output_path: str) -> None:
    """
    Replace video track's audio with new audio. Copy video stream.
    ffmpeg -i video.mp4 -i new_audio.wav -c:v copy -map 0:v:0 -map 1:a:0 output.mp4
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def convert_audio_to_wav(input_path: str, output_path: str, sample_rate: int = 16000, mono: bool = True) -> None:
    """
    Convert any audio (mp3, m4a, etc.) to mono 16kHz WAV for ASR.
    ffmpeg -i input.mp3 -ac 1 -ar 16000 output.wav
    """
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def get_audio_duration_seconds(audio_path: str) -> float:
    """Probe duration with ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())
