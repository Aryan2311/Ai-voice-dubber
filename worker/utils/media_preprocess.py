"""
Preprocess uploaded media: download, extract or convert to 16kHz mono WAV, upload to audio/{media_id}/source.wav.
"""
import os
import tempfile

from worker.utils import s3_utils, ffmpeg_utils

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")


def _get_media_s3_key(media_id: str) -> str:
    keys = s3_utils.list_keys(f"uploads/media/{media_id}/")
    if not keys:
        raise FileNotFoundError("No media found for media_id=%s" % media_id)
    return keys[0]


def _is_video(key: str) -> bool:
    return key.lower().endswith(VIDEO_EXTENSIONS)


def ensure_source_audio(media_id: str) -> str:
    """
    If audio/{media_id}/source.wav already exists, return its S3 key.
    Else: download media, extract or convert to 16k wav, upload to audio/{media_id}/source.wav, return key.
    """
    source_key = f"audio/{media_id}/source.wav"
    if s3_utils.object_exists(source_key):
        return source_key

    s3_media_key = _get_media_s3_key(media_id)
    with tempfile.TemporaryDirectory() as tmp:
        local_in = os.path.join(tmp, "input" + os.path.splitext(s3_media_key)[1])
        local_wav = os.path.join(tmp, "source.wav")

        s3_utils.download_file(s3_media_key, local_in)

        if _is_video(s3_media_key):
            ffmpeg_utils.extract_audio(local_in, local_wav, sample_rate=16000, mono=True)
        else:
            ffmpeg_utils.convert_audio_to_wav(local_in, local_wav, sample_rate=16000, mono=True)

        s3_utils.upload_file(local_wav, source_key, content_type="audio/wav")

    return source_key
