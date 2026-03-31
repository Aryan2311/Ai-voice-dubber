"""
DUB_MEDIA: get original transcript (run TRANSCRIBE if needed), then overlap
batch translation and XTTS synthesis on the same GPU before merging/uploading audio.
Output: video -> dubbed/{media_id}/{lang}.mp4; audio-only -> audio/{media_id}/{lang}.wav
"""
import json
import logging
import os
import queue
import tempfile
import threading
import time

from worker.utils import s3_utils, ffmpeg_utils, audio_utils
from worker.ai_models import xtts_model, translator
from worker.jobs import transcribe_job

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
OVERLAP_BATCH_SIZE = max(1, int(os.getenv("DUB_OVERLAP_BATCH_SIZE", "4")))
OVERLAP_QUEUE_SIZE = max(1, int(os.getenv("DUB_OVERLAP_QUEUE_SIZE", "2")))
_OVERLAP_SENTINEL = object()


def _media_is_video(media_id: str) -> bool:
    keys = s3_utils.list_keys(f"uploads/media/{media_id}/")
    if not keys:
        return False
    return keys[0].lower().endswith(VIDEO_EXTENSIONS)


def _ensure_original_transcript(media_id: str, tmp: str) -> dict:
    original_key = f"transcripts/{media_id}/original.json"
    logger.info("DUB_MEDIA media_id=%s rerunning TRANSCRIBE inline before dubbing", media_id)
    transcribe_job.run_transcribe_job({"job_type": "TRANSCRIBE", "media_id": media_id})
    path = os.path.join(tmp, "original.json")
    s3_utils.download_file(original_key, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dub_job(job: dict) -> None:
    media_id = job["media_id"]
    language = job.get("language", "es")
    voice_sample_s3 = job.get("voice_sample")
    is_video = _media_is_video(media_id)
    logger.info("DUB_MEDIA media_id=%s language=%s is_video=%s", media_id, language, is_video)

    # Idempotent: skip only when a real output exists (non-empty; avoid skipping on empty/corrupt leftover)
    if is_video:
        out_key = f"dubbed/{media_id}/{language}.mp4"
        min_size = 1024
    else:
        out_key = f"audio/{media_id}/{language}.wav"
        min_size = 256
    if job.get("skip_if_exists") and s3_utils.object_exists_and_non_empty(out_key, min_size=min_size):
        logger.info("DUB_MEDIA media_id=%s language=%s output already exists, skipping", media_id, language)
        return out_key

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
            return out_key

        # Get total duration from source so timeline matches video/audio length
        if is_video:
            video_key = s3_utils.list_keys(f"uploads/media/{media_id}/")[0]
            video_path = os.path.join(tmp, "video" + os.path.splitext(video_key)[1])
            s3_utils.download_file(video_key, video_path)
            total_duration_sec = ffmpeg_utils.get_audio_duration_seconds(video_path)
        else:
            source_wav = os.path.join(tmp, "source.wav")
            s3_utils.download_file(f"audio/{media_id}/source.wav", source_wav)
            total_duration_sec = ffmpeg_utils.get_audio_duration_seconds(source_wav)
        total_duration_sec = max(total_duration_sec, max((s["end"] for s in segments), default=0.0) + 0.5)

        speaker_wav = None
        if voice_sample_s3:
            speaker_wav = s3_utils.download_to_temp(voice_sample_s3, suffix=".wav")

        # Normalize segment boundaries from source timings so TTS can fill in translated text batch-by-batch.
        slot_starts = []
        slot_ends = []
        for i, s in enumerate(segments):
            end_sec = s["end"]
            if i == 0:
                start_sec = s["start"]
            else:
                start_sec = slot_ends[i - 1]
            if end_sec <= start_sec:
                end_sec = start_sec + 0.05  # minimum 50ms slot
            slot_starts.append(start_sec)
            slot_ends.append(end_sec)

        translator.assert_overlap_ready(stage="dub_job", source_lang=source_lang, target_lang=language)
        n_seg = len(segments)
        total_batches = max(1, (n_seg + OVERLAP_BATCH_SIZE - 1) // OVERLAP_BATCH_SIZE)
        logger.info(
            "DUB_OVERLAP enabled=true translation_backend=%s media_id=%s language=%s batch_size=%d queue_size=%d total_segments=%d total_batches=%d overlap_window_possible=%s",
            translator.ACTIVE_TRANSLATION_BACKEND,
            media_id,
            language,
            OVERLAP_BATCH_SIZE,
            OVERLAP_QUEUE_SIZE,
            n_seg,
            total_batches,
            total_batches > 1,
        )

        translated_segments = [None] * n_seg
        segment_wavs = [None] * n_seg
        translated_queue = queue.Queue(maxsize=OVERLAP_QUEUE_SIZE)
        stop_event = threading.Event()
        errors = []
        errors_lock = threading.Lock()

        def _record_error(exc: Exception) -> None:
            with errors_lock:
                errors.append(exc)
            stop_event.set()

        def _queue_put(item) -> bool:
            while not stop_event.is_set():
                try:
                    translated_queue.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    continue
            return False

        def _queue_get():
            while True:
                if stop_event.is_set() and translated_queue.empty():
                    return _OVERLAP_SENTINEL
                try:
                    return translated_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set():
                        return _OVERLAP_SENTINEL

        def _translate_batches():
            try:
                for batch_num, start_idx in enumerate(range(0, n_seg, OVERLAP_BATCH_SIZE), start=1):
                    if stop_event.is_set():
                        return
                    batch = segments[start_idx : start_idx + OVERLAP_BATCH_SIZE]
                    translated_batch = translator.translate_batch(
                        batch,
                        source_lang,
                        language,
                        batch_index=batch_num,
                        total_batches=total_batches,
                    )
                    if not _queue_put((batch_num, start_idx, translated_batch)):
                        return
                    logger.info(
                        "DUB_OVERLAP translated_ready batch=%d/%d start_segment=%d batch_segments=%d",
                        batch_num,
                        total_batches,
                        start_idx + 1,
                        len(translated_batch),
                    )
            except Exception as exc:
                _record_error(exc)
            finally:
                _queue_put(_OVERLAP_SENTINEL)

        def _tts_batches():
            try:
                while True:
                    item = _queue_get()
                    if item is _OVERLAP_SENTINEL:
                        return
                    batch_num, start_idx, translated_batch = item
                    t_batch = time.monotonic()
                    for offset, seg in enumerate(translated_batch):
                        if stop_event.is_set():
                            return
                        seg_idx = start_idx + offset
                        translated_segments[seg_idx] = seg
                        seg_wav = os.path.join(tmp, "seg_%d.wav" % seg_idx)
                        logger.info(
                            "DUB_OVERLAP tts_segment=%d/%d batch=%d/%d",
                            seg_idx + 1,
                            n_seg,
                            batch_num,
                            total_batches,
                        )
                        xtts_model.generate_speech(
                            seg["text"], seg_wav, language=language, speaker_wav_path=speaker_wav
                        )
                        segment_wavs[seg_idx] = seg_wav
                    logger.info(
                        "DUB_OVERLAP tts_batch=%d/%d segments=%d elapsed_sec=%.2f",
                        batch_num,
                        total_batches,
                        len(translated_batch),
                        time.monotonic() - t_batch,
                    )
            except Exception as exc:
                _record_error(exc)

        translate_thread = threading.Thread(target=_translate_batches, name="dub-translate", daemon=True)
        tts_thread = threading.Thread(target=_tts_batches, name="dub-tts", daemon=True)
        t_all = time.monotonic()
        translate_thread.start()
        tts_thread.start()
        translate_thread.join()
        tts_thread.join()

        if errors:
            raise errors[0]
        if any(seg is None for seg in translated_segments):
            raise RuntimeError("Overlap pipeline finished without translating all segments.")
        if any(path is None for path in segment_wavs):
            raise RuntimeError("Overlap pipeline finished without generating all TTS segments.")

        logger.info(
            "DUB_OVERLAP completed media_id=%s language=%s total_batches=%d elapsed_sec=%.2f",
            media_id,
            language,
            total_batches,
            time.monotonic() - t_all,
        )

        segment_timeline = [(slot_starts[i], slot_ends[i], segment_wavs[i]) for i in range(n_seg)]
        dubbed_audio_path = os.path.join(tmp, "dubbed.wav")
        audio_utils.build_timeline_wav(segment_timeline, total_duration_sec, dubbed_audio_path)

        if is_video:
            # video_path already downloaded when we got total_duration_sec
            output_path = os.path.join(tmp, "output.mp4")
            ffmpeg_utils.merge_audio_with_video(video_path, dubbed_audio_path, output_path)
            out_key = f"dubbed/{media_id}/{language}.mp4"
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Merge produced no output file or empty file: %s" % output_path)
            s3_utils.upload_file(output_path, out_key, content_type="video/mp4")
            logger.info("DUB_MEDIA media_id=%s uploaded key=%s segments=%d", media_id, out_key, len(segments))
            return out_key
        else:
            out_key = f"audio/{media_id}/{language}.wav"
            if not os.path.exists(dubbed_audio_path) or os.path.getsize(dubbed_audio_path) == 0:
                raise RuntimeError("Dubbed audio file missing or empty: %s" % dubbed_audio_path)
            s3_utils.upload_file(dubbed_audio_path, out_key, content_type="audio/wav")
            logger.info("DUB_MEDIA media_id=%s uploaded key=%s segments=%d", media_id, out_key, len(segments))
            return out_key
