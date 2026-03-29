"""
Full dubbing pipeline: ASR → prosody extraction → translate (NLLB) → TTS (prosody-conditioned) → align → merge.
Same S3 contract: transcripts/{media_id}/original.json, dubbed/{media_id}/{lang}.mp4 or audio/{media_id}/{lang}.wav.
"""
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from worker.utils.job_logging import log_preview

from worker.utils import s3_utils, ffmpeg_utils, media_preprocess
from worker.pipeline import asr, translate, tts, align

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
        logger.info(
            "DUB_PIPELINE media_id=%s loading existing transcript key=%s",
            media_id,
            original_key,
        )
        s3_utils.download_file(original_key, path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.info(
        "DUB_PIPELINE media_id=%s no original.json — running ASR then uploading %s",
        media_id,
        original_key,
    )
    media_preprocess.ensure_source_audio(media_id)
    audio_path = os.path.join(tmp, "source.wav")
    s3_utils.download_file(f"audio/{media_id}/source.wav", audio_path)
    # Caller holds GPU session (worker runs DUB_MEDIA inside gpu_session)
    t_asr = time.monotonic()
    result = asr.transcribe(audio_path)
    logger.info(
        "DUB_PIPELINE media_id=%s ASR done segments=%d elapsed_sec=%.1f",
        media_id,
        len(result.get("segments", [])),
        time.monotonic() - t_asr,
    )
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
    voice_sample_s3: str,
    skip_if_output_exists: bool = False,
) -> Optional[str]:
    """
    Run full dubbing pipeline. Output: dubbed/{media_id}/{lang}.mp4 or audio/{media_id}/{lang}.wav.
    Returns S3 key of dubbed output on success (for job_completions/{job_id}.json).

    skip_if_output_exists: if True, skip pipeline when output already on S3 (legacy); default False
    so re-dubs always regenerate and overwrite.
    """
    is_video = _media_is_video(media_id)
    if is_video:
        out_key = f"dubbed/{media_id}/{language}.mp4"
        min_size = 1024
    else:
        out_key = f"audio/{media_id}/{language}.wav"
        min_size = 256
    if skip_if_output_exists and s3_utils.object_exists_and_non_empty(out_key, min_size=min_size):
        logger.info(
            "DUB_PIPELINE media_id=%s language=%s skip_if_output_exists=True key=%s — skip",
            media_id,
            language,
            out_key,
        )
        return out_key

    ctx = "media_id=%s language=%s" % (media_id, language)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.monotonic()
        orig = _ensure_original_transcript(media_id, tmp)
        logger.info(
            "DUB_PIPELINE %s phase=transcript_ready segments=%s source_lang=%s elapsed_sec=%.1f",
            ctx,
            len(orig.get("segments", [])),
            orig.get("language", "en"),
            time.monotonic() - t0,
        )
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
            return out_key

        t_tr = time.monotonic()
        translated = translate.translate_segments(
            segments, source_lang, language, log_context=ctx
        )
        logger.info(
            "DUB_PIPELINE %s phase=translate_done elapsed_sec=%.1f (no LLM rewrite; TTS uses NLLB text)",
            ctx,
            time.monotonic() - t_tr,
        )
        n_seg = len(translated)

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

        raw_voice = s3_utils.download_to_temp(voice_sample_s3, suffix=".wav")
        from worker.utils.reference_audio import ensure_preprocessed_reference, preprocess_prosody_segment

        speaker_wav = ensure_preprocessed_reference(raw_voice, tmp, isolate_voice=True)
        if not os.path.isfile(speaker_wav) or os.path.getsize(speaker_wav) == 0:
            raise RuntimeError("voice_sample produced no usable WAV after preprocessing")

        source_wav_path = os.path.join(tmp, "source.wav")
        s3_utils.download_file(f"audio/{media_id}/source.wav", source_wav_path)
        if not os.path.isfile(source_wav_path) or os.path.getsize(source_wav_path) == 0:
            raise RuntimeError("Missing or empty audio/%s/source.wav (required for segment prosody)" % media_id)

        segment_wavs = []
        for i, seg in enumerate(translated):
            seg_wav = os.path.join(tmp, "seg_%d.wav" % i)
            seg_audio_raw = os.path.join(tmp, "seg_audio_raw_%d.wav" % i)
            _slice_segment_audio(source_wav_path, seg["start"], seg["end"], seg_audio_raw)
            seg_audio = os.path.join(tmp, "seg_audio_%d.wav" % i)
            preprocess_prosody_segment(seg_audio_raw, seg_audio)
            style_audio_path = seg_audio if os.path.isfile(seg_audio) and os.path.getsize(seg_audio) > 0 else seg_audio_raw
            if not os.path.isfile(style_audio_path) or os.path.getsize(style_audio_path) == 0:
                raise RuntimeError("Failed to build prosody WAV for segment %d/%d" % (i + 1, n_seg))

            logger.info(
                "DUB_PIPELINE %s TTS segment %d/%d text_preview=%r target_dur_sec=%.3f",
                ctx,
                i + 1,
                n_seg,
                log_preview(seg.get("text", ""), 100),
                seg["end"] - seg["start"],
            )
            t_utt = time.monotonic()
            tts.generate_speech(
                seg["text"],
                seg_wav,
                language=language,
                speaker_wav_path=speaker_wav,
                style_audio_path=style_audio_path,
            )
            logger.info(
                "DUB_PIPELINE %s TTS segment %d/%d generate_elapsed_sec=%.2f",
                ctx,
                i + 1,
                n_seg,
                time.monotonic() - t_utt,
            )
            target_dur = seg["end"] - seg["start"]
            aligned_wav = os.path.join(tmp, "aligned_%d.wav" % i)
            align.align_segment_to_duration(seg_wav, aligned_wav, target_dur)
            segment_wavs.append(aligned_wav)

        logger.info(
            "DUB_PIPELINE %s phase=tts_align_done total_elapsed_sec=%.1f",
            ctx,
            time.monotonic() - t_tts,
        )

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
        t_merge = time.monotonic()
        align.build_timeline_wav(segment_timeline, total_duration_sec, dubbed_audio_path)
        logger.info(
            "DUB_PIPELINE %s phase=timeline_wav built elapsed_sec=%.2f duration_sec=%.2f",
            ctx,
            time.monotonic() - t_merge,
            total_duration_sec,
        )

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
        logger.info(
            "DUB_PIPELINE %s done uploaded_key=%s segments=%d is_video=%s",
            ctx,
            out_key,
            len(segments),
            is_video,
        )
        return out_key
