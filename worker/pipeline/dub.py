"""
Full dubbing pipeline: ASR → prosody extraction → translate → rewrite → TTS (prosody-conditioned) → align → RVC → merge.
Same S3 contract: transcripts/{media_id}/original.json, dubbed/{media_id}/{lang}.mp4 or audio/{media_id}/{lang}.wav.
"""
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from worker.utils import s3_utils, ffmpeg_utils, media_preprocess
from worker.pipeline import asr, translate, rewrite, tts, align, rvc

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def _media_is_video(media_id: str) -> bool:
    keys = s3_utils.list_keys(f"uploads/media/{media_id}/")
    if not keys:
        return False
    return keys[0].lower().endswith(VIDEO_EXTENSIONS)


def _ensure_original_transcript(media_id: str, tmp: str) -> Dict[str, Any]:
    original_key = f"transcripts/{media_id}/original.json"
    if s3_utils.object_exists(original_key):
        path = os.path.join(tmp, "original.json")
        s3_utils.download_file(original_key, path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    media_preprocess.ensure_source_audio(media_id)
    audio_path = os.path.join(tmp, "source.wav")
    s3_utils.download_file(f"audio/{media_id}/source.wav", audio_path)
    # Caller holds GPU session (worker runs DUB_MEDIA inside gpu_session)
    result = asr.transcribe(audio_path)
    transcript = {
        "media_id": media_id,
        "segments": result["segments"],
        "full_text": result["full_text"],
        "language": result.get("language", "en"),
    }
    s3_utils.upload_json(original_key, transcript)
    return transcript


def _slice_segment_audio(source_wav: str, start_sec: float, end_sec: float, out_path: str) -> None:
    """Extract segment [start_sec, end_sec] from source_wav into out_path."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", source_wav,
        "-ss", str(start_sec), "-t", str(max(0.01, end_sec - start_sec)),
        "-ac", "1", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def run_dub(
    media_id: str,
    language: str,
    voice_sample_s3: Optional[str] = None,
    use_prosody: bool = True,
) -> None:
    """
    Run full dubbing pipeline. Output: dubbed/{media_id}/{lang}.mp4 or audio/{media_id}/{lang}.wav.
    use_prosody: if True (default), use original segment audio for prosody in StyleTTS2; RVC applies voice identity.
    """
    is_video = _media_is_video(media_id)
    if is_video:
        out_key = f"dubbed/{media_id}/{language}.mp4"
        min_size = 1024
    else:
        out_key = f"audio/{media_id}/{language}.wav"
        min_size = 256
    if s3_utils.object_exists_and_non_empty(out_key, min_size=min_size):
        logger.info("DUB_MEDIA media_id=%s language=%s output already exists, skipping", media_id, language)
        return

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
                ffmpeg_utils.extract_audio(video_path, os.path.join(tmp, "silence.wav"))
                out_mp4 = os.path.join(tmp, "out.mp4")
                ffmpeg_utils.merge_audio_with_video(video_path, os.path.join(tmp, "silence.wav"), out_mp4)
                s3_utils.upload_file(out_mp4, out_key, content_type="video/mp4")
            else:
                s3_utils.upload_bytes(b"", f"audio/{media_id}/{language}.wav", content_type="audio/wav")
            return

        # CPU: translate then syllable-aware rewrite per segment
        translated = translate.translate_segments(segments, source_lang, language)
        from worker.translation.syllable_counter import count_syllables
        for seg in translated:
            orig_text = seg.get("original_text", "")
            target_syllables = count_syllables(orig_text, source_lang) if orig_text else None
            seg["text"] = rewrite.rewrite(
                seg["text"], language=language, target_syllables=target_syllables
            )

        # Total duration
        if is_video:
            video_key = s3_utils.list_keys(f"uploads/media/{media_id}/")[0]
            video_path = os.path.join(tmp, "video" + os.path.splitext(video_key)[1])
            s3_utils.download_file(video_key, video_path)
            total_duration_sec = ffmpeg_utils.get_audio_duration_seconds(video_path)
        else:
            source_wav = os.path.join(tmp, "source.wav")
            s3_utils.download_file(f"audio/{media_id}/source.wav", source_wav)
            total_duration_sec = ffmpeg_utils.get_audio_duration_seconds(source_wav)
        total_duration_sec = max(total_duration_sec, max((s["end"] for s in translated), default=0.0) + 0.5)

        speaker_wav = None
        if voice_sample_s3:
            raw = s3_utils.download_to_temp(voice_sample_s3, suffix=".wav")
            from worker.utils.reference_audio import ensure_preprocessed_reference
            speaker_wav = ensure_preprocessed_reference(raw, tmp, isolate_voice=True)

        source_wav_path = os.path.join(tmp, "source.wav")
        if not os.path.isfile(source_wav_path) and use_prosody:
            s3_utils.download_file(f"audio/{media_id}/source.wav", source_wav_path)

        segment_wavs = []
        n_seg = len(translated)
        for i, seg in enumerate(translated):
            seg_wav = os.path.join(tmp, "seg_%d.wav" % i)
            style_audio_path = None
            prosody_dict = None
            if use_prosody and os.path.isfile(source_wav_path):
                seg_audio_raw = os.path.join(tmp, "seg_audio_raw_%d.wav" % i)
                _slice_segment_audio(source_wav_path, seg["start"], seg["end"], seg_audio_raw)
                seg_audio = os.path.join(tmp, "seg_audio_%d.wav" % i)
                from worker.utils.reference_audio import preprocess_prosody_segment
                preprocess_prosody_segment(seg_audio_raw, seg_audio)
                style_audio_path = seg_audio if os.path.isfile(seg_audio) else seg_audio_raw
                from worker.prosody.extract import extract_prosody
                prosody_dict = extract_prosody(style_audio_path)

            # Prosody from original segment only; voice identity applied by RVC later (dual reference).
            logger.info("TTS segment %d/%d (generating voice)", i + 1, n_seg)
            tts.generate_speech(
                    seg["text"], seg_wav,
                    language=language,
                    speaker_wav_path=None if style_audio_path else speaker_wav,
                    prosody=prosody_dict,
                    style_audio_path=style_audio_path,
                )
            target_dur = seg["end"] - seg["start"]
            aligned_wav = os.path.join(tmp, "aligned_%d.wav" % i)
            align.align_segment_to_duration(seg_wav, aligned_wav, target_dur)
            segment_wavs.append(aligned_wav)

        # Normalize segment boundaries: back-to-back
        slot_starts, slot_ends = [], []
        for i, s in enumerate(translated):
            end_sec = s["end"]
            start_sec = slot_ends[i - 1] if i else s["start"]
            if end_sec <= start_sec:
                end_sec = start_sec + 0.05
            slot_starts.append(start_sec)
            slot_ends.append(end_sec)
        segment_timeline = [(slot_starts[i], slot_ends[i], segment_wavs[i]) for i in range(len(translated))]
        dubbed_audio_path = os.path.join(tmp, "dubbed.wav")
        align.build_timeline_wav(segment_timeline, total_duration_sec, dubbed_audio_path)

        if speaker_wav:
            try:
                from worker.ai_models import rvc_model
                if rvc_model.load_rvc() is not None:
                    rvc_path = os.path.join(tmp, "dubbed_rvc.wav")
                    if rvc.convert_voice(dubbed_audio_path, rvc_path, speaker_wav):
                        dubbed_audio_path = rvc_path
            except RuntimeError as e:
                logger.warning("RVC not configured (set RVC_MODEL_PATH): %s", e)

        if is_video:
            output_path = os.path.join(tmp, "output.mp4")
            ffmpeg_utils.merge_audio_with_video(video_path, dubbed_audio_path, output_path)
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Merge produced no output file")
            s3_utils.upload_file(output_path, out_key, content_type="video/mp4")
        else:
            if not os.path.exists(dubbed_audio_path) or os.path.getsize(dubbed_audio_path) == 0:
                raise RuntimeError("Dubbed audio file missing or empty")
            s3_utils.upload_file(dubbed_audio_path, out_key, content_type="audio/wav")
        logger.info("DUB_MEDIA media_id=%s uploaded key=%s segments=%d", media_id, out_key, len(segments))
