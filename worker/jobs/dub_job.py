"""
DUB_MEDIA: get original transcript (run TRANSCRIBE if needed), translate, TTS per segment, RVC, merge or upload audio.
Output: video -> dubbed/{media_id}/{lang}.mp4; audio-only -> audio/{media_id}/{lang}.wav
"""
import json
import logging
import os
import tempfile

from worker.utils import s3_utils, ffmpeg_utils, audio_utils, media_preprocess
from worker.ai_models import whisper_model, xtts_model, rvc_model, translator

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def _media_is_video(media_id: str) -> bool:
    keys = s3_utils.list_keys(f"uploads/media/{media_id}/")
    if not keys:
        return False
    return keys[0].lower().endswith(VIDEO_EXTENSIONS)


def _ensure_original_transcript(media_id: str, tmp: str) -> dict:
    original_key = f"transcripts/{media_id}/original.json"
    if s3_utils.object_exists(original_key):
        path = os.path.join(tmp, "original.json")
        s3_utils.download_file(original_key, path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    media_preprocess.ensure_source_audio(media_id)
    audio_path = os.path.join(tmp, "source.wav")
    s3_utils.download_file(f"audio/{media_id}/source.wav", audio_path)
    result = whisper_model.transcribe_audio(audio_path)
    transcript = {
        "media_id": media_id,
        "segments": result["segments"],
        "full_text": result["full_text"],
        "language": result.get("language", "en"),
    }
    s3_utils.upload_json(original_key, transcript)
    return transcript


def run_dub_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job.get("language", "es")
    voice_sample_s3 = job.get("voice_sample")
    is_video = _media_is_video(media_id)
    logger.info("DUB_MEDIA media_id=%s language=%s is_video=%s", media_id, language, is_video)

    with tempfile.TemporaryDirectory() as tmp:
        orig = _ensure_original_transcript(media_id, tmp)
        segments = orig.get("segments", [])
        source_lang = orig.get("language", "en")

        if not segments:
            logger.info("DUB_MEDIA media_id=%s no segments; uploading silent output", media_id)
            if is_video:
                video_key = s3_utils.list_keys(f"uploads/media/{media_id}/")[0]
                video_path = os.path.join(tmp, "video" + os.path.splitext(video_key)[1])
                s3_utils.download_file(video_key, video_path)
                out_mp4 = os.path.join(tmp, "out.mp4")
                ffmpeg_utils.extract_audio(video_path, os.path.join(tmp, "silence.wav"))
                ffmpeg_utils.merge_audio_with_video(video_path, os.path.join(tmp, "silence.wav"), out_mp4)
                s3_utils.upload_file(out_mp4, f"dubbed/{media_id}/{language}.mp4", content_type="video/mp4")
            else:
                s3_utils.upload_bytes(b"", f"audio/{media_id}/{language}.wav", content_type="audio/wav")
            logger.info("DUB_MEDIA media_id=%s uploaded silent output", media_id)
            return

        translated = translator.translate_segments(segments, source_lang, language)
        speaker_wav = None
        if voice_sample_s3:
            speaker_wav = s3_utils.download_to_temp(voice_sample_s3, suffix=".wav")

        segment_wavs = []
        for i, seg in enumerate(translated):
            seg_wav = os.path.join(tmp, "seg_%d.wav" % i)
            xtts_model.generate_speech(
                seg["text"], seg_wav, language=language, speaker_wav_path=speaker_wav
            )
            segment_wavs.append(seg_wav)

        dubbed_audio_path = os.path.join(tmp, "dubbed.wav")
        audio_utils.concat_wav_files(segment_wavs, dubbed_audio_path)

        if speaker_wav and rvc_model.load_rvc():
            rvc_path = os.path.join(tmp, "dubbed_rvc.wav")
            if rvc_model.convert_voice(dubbed_audio_path, rvc_path, speaker_wav):
                dubbed_audio_path = rvc_path

        if is_video:
            video_key = s3_utils.list_keys(f"uploads/media/{media_id}/")[0]
            video_path = os.path.join(tmp, "video" + os.path.splitext(video_key)[1])
            s3_utils.download_file(video_key, video_path)
            output_path = os.path.join(tmp, "output.mp4")
            ffmpeg_utils.merge_audio_with_video(video_path, dubbed_audio_path, output_path)
            out_key = f"dubbed/{media_id}/{language}.mp4"
            s3_utils.upload_file(output_path, out_key, content_type="video/mp4")
            logger.info("DUB_MEDIA media_id=%s uploaded key=%s segments=%d", media_id, out_key, len(segments))
        else:
            out_key = f"audio/{media_id}/{language}.wav"
            s3_utils.upload_file(dubbed_audio_path, out_key, content_type="audio/wav")
            logger.info("DUB_MEDIA media_id=%s uploaded key=%s segments=%d", media_id, out_key, len(segments))
