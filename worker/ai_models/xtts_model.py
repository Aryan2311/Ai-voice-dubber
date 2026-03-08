"""
XTTS v2: load once at worker startup, generate speech from text (optionally with voice clone).
XTTS has a 400-token limit per call; long text is split into chunks and audio concatenated.
"""
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_xtts_model = None
_xtts_speaker_wav = None

# XTTS allows max 400 tokens. Hindi/Devanagari can be ~1–2 tokens per char; keep chunks small.
MAX_CHARS_PER_TTS_CALL = 250


def _split_text_for_xtts(text: str) -> list:
    """Split text into chunks of at most MAX_CHARS_PER_TTS_CALL, at sentence/space boundaries when possible."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= MAX_CHARS_PER_TTS_CALL:
        return [text]
    chunks = []
    rest = text
    while rest:
        if len(rest) <= MAX_CHARS_PER_TTS_CALL:
            chunks.append(rest.strip())
            break
        block = rest[: MAX_CHARS_PER_TTS_CALL + 1]
        # Prefer split at sentence end, then at space
        for sep in (". ", "! ", "? ", "। ", ".\n", "!\n", "?\n", "; ", "\n", " "):
            idx = block.rfind(sep)
            if idx > MAX_CHARS_PER_TTS_CALL // 2:
                chunks.append(rest[: idx + len(sep)].strip())
                rest = rest[idx + len(sep) :].lstrip()
                break
        else:
            # No good break; force split at max length
            chunks.append(rest[:MAX_CHARS_PER_TTS_CALL].strip())
            rest = rest[MAX_CHARS_PER_TTS_CALL:].lstrip()
    return [c for c in chunks if c]


def load_xtts():
    """Load XTTS model once. Keep in memory."""
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    # Accept Coqui CPML non-commercial terms so XTTS loads in non-interactive environments (Docker, CI).
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    try:
        from TTS.api import TTS
        # XTTS v2 model
        _xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        return _xtts_model
    except ImportError as e:
        logger.exception("XTTS import failed: %s", e)
        raise RuntimeError("TTS (Coqui) not installed. pip install TTS") from e
    except Exception as e:
        logger.exception("XTTS load failed: %s", e)
        raise


def _tts_one_chunk(model, text: str, path: str, language: str, speaker_wav_path: str = None) -> None:
    """Generate speech for a single chunk (under token limit)."""
    if speaker_wav_path and os.path.isfile(speaker_wav_path):
        model.tts_to_file(
            text=text,
            file_path=path,
            speaker_wav=speaker_wav_path,
            language=language,
        )
    else:
        model.tts_to_file(text=text, file_path=path, language=language)


def generate_speech(
    text: str,
    output_path: str,
    language: str = "en",
    speaker_wav_path: str = None,
) -> None:
    """
    Generate speech from text. If speaker_wav_path is set, clone that voice.
    Long text is split into chunks (XTTS max 400 tokens) and audio concatenated.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text for TTS")
    model = load_xtts()
    chunks = _split_text_for_xtts(text)
    if len(chunks) == 1:
        _tts_one_chunk(model, chunks[0], output_path, language, speaker_wav_path)
        return
    from worker.utils import audio_utils
    with tempfile.TemporaryDirectory() as tmp:
        wav_paths = []
        for i, chunk in enumerate(chunks):
            seg_path = os.path.join(tmp, "chunk_%d.wav" % i)
            _tts_one_chunk(model, chunk, seg_path, language, speaker_wav_path)
            wav_paths.append(seg_path)
        audio_utils.concat_wav_files(wav_paths, output_path)
