"""
Translation via IndicTrans2.

IndicTrans2 is the active translation backend for transcript translation and
dubbing. The previous Mistral helpers are retained below for future reference,
but they are intentionally not used by the public translation API anymore.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Tuple

from worker.gpu.vram_monitor import get_vram_free_total_mb, log_vram

logger = logging.getLogger(__name__)

ACTIVE_TRANSLATION_BACKEND = "indictrans2"
DEFAULT_TRANSLATION_BATCH_SIZE = max(1, int(os.getenv("TRANSLATION_BATCH_SIZE", "4")))
DEFAULT_OVERLAP_MIN_FREE_VRAM_MB = max(0, int(os.getenv("DUB_OVERLAP_MIN_FREE_VRAM_MB", "1024")))
DEFAULT_STARTUP_SOURCE_LANG = os.getenv("TRANSLATION_STARTUP_SOURCE_LANG", "en")
DEFAULT_STARTUP_TARGET_LANG = os.getenv("TRANSLATION_STARTUP_TARGET_LANG", "hi")
DEFAULT_TRANSLATION_ROUTE = os.getenv("INDICTRANS2_DEFAULT_ROUTE", "en-indic").strip().lower()

DEFAULT_EN_INDIC_MODEL = os.getenv("INDICTRANS2_EN_INDIC_MODEL_ID", "ai4bharat/indictrans2-en-indic-1B")
DEFAULT_INDIC_EN_MODEL = os.getenv("INDICTRANS2_INDIC_EN_MODEL_ID", "ai4bharat/indictrans2-indic-en-1B")
DEFAULT_INDIC_INDIC_MODEL = os.getenv("INDICTRANS2_INDIC_INDIC_MODEL_ID", "ai4bharat/indictrans2-indic-indic-1B")

_ACTIVE_MODEL_IDS = {
    "en-indic": DEFAULT_EN_INDIC_MODEL,
    "indic-en": DEFAULT_INDIC_EN_MODEL,
    "indic-indic": DEFAULT_INDIC_INDIC_MODEL,
}

_translator_processor: Any = None
_translator_tokenizer: Any = None
_translator_model: Any = None
_translator_model_id = ""
_translator_route_key = ""
_translator_lock = threading.Lock()

_LANGUAGE_NAMES = {
    "en": "English",
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

_ISO_TO_FLORES = {
    "en": "eng_Latn",
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}
_FLORES_TO_ISO = {value: key for key, value in _ISO_TO_FLORES.items()}
_SUPPORTED_LANGUAGE_CODES = tuple(sorted(_ISO_TO_FLORES.keys()))


def _hf_auth_kwargs() -> Dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {}
    return {"use_auth_token": token}


def _normalize_language_code(language: str) -> str:
    code = (language or "").strip()
    if not code:
        return "en"
    if code in _FLORES_TO_ISO:
        return _FLORES_TO_ISO[code]
    code = code.lower()
    if code in _ISO_TO_FLORES:
        return code
    if "-" in code:
        code = code.split("-", 1)[0]
    if code in _ISO_TO_FLORES:
        return code
    return code


def _language_name(language: str) -> str:
    code = _normalize_language_code(language)
    return _LANGUAGE_NAMES.get(code, code or "English")


def _flores_code(language: str) -> str:
    code = _normalize_language_code(language)
    flores = _ISO_TO_FLORES.get(code)
    if not flores:
        supported = ", ".join(_SUPPORTED_LANGUAGE_CODES)
        raise ValueError(
            "IndicTrans2 active translator currently supports only English and Indic languages. "
            "Unsupported language=%s. Supported codes: %s" % (language, supported)
        )
    return flores


def _resolve_route(source_lang: str, target_lang: str) -> Tuple[str | None, str, str, str | None, str | None]:
    src = _normalize_language_code(source_lang)
    tgt = _normalize_language_code(target_lang)
    if src == tgt:
        return None, src, tgt, None, None

    src_flores = _flores_code(src)
    tgt_flores = _flores_code(tgt)
    src_is_indic = src != "en"
    tgt_is_indic = tgt != "en"

    if src == "en" and tgt_is_indic:
        return "en-indic", src, tgt, src_flores, tgt_flores
    if src_is_indic and tgt == "en":
        return "indic-en", src, tgt, src_flores, tgt_flores
    if src_is_indic and tgt_is_indic:
        return "indic-indic", src, tgt, src_flores, tgt_flores

    raise ValueError(
        "IndicTrans2 active translator currently supports only English <-> Indic and Indic <-> Indic routes. "
        "Unsupported pair=%s->%s" % (source_lang, target_lang)
    )


def _load_processor():
    global _translator_processor
    if _translator_processor is not None:
        return _translator_processor
    from IndicTransToolkit import IndicProcessor
    _translator_processor = IndicProcessor(inference=True)
    return _translator_processor


def _clear_loaded_model() -> None:
    global _translator_tokenizer, _translator_model, _translator_model_id, _translator_route_key
    if _translator_model is None and _translator_tokenizer is None:
        return
    try:
        import torch

        model = _translator_model
        tokenizer = _translator_tokenizer
        _translator_model = None
        _translator_tokenizer = None
        _translator_model_id = ""
        _translator_route_key = ""
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Could not fully unload active IndicTrans2 model: %s", exc)


def load_translation_model(route_key: str | None = None):
    """Load the active IndicTrans2 model for a route such as en-indic."""
    global _translator_tokenizer, _translator_model, _translator_model_id, _translator_route_key
    desired_route = (route_key or DEFAULT_TRANSLATION_ROUTE).strip().lower()
    if desired_route not in _ACTIVE_MODEL_IDS:
        raise ValueError("Unsupported IndicTrans2 route=%s" % desired_route)

    with _translator_lock:
        if (
            _translator_model is not None
            and _translator_tokenizer is not None
            and _translator_route_key == desired_route
            and _translator_model_id == _ACTIVE_MODEL_IDS[desired_route]
        ):
            return _translator_tokenizer, _translator_model

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _load_processor()
        _clear_loaded_model()

        model_id = _ACTIVE_MODEL_IDS[desired_route]
        auth_kwargs = _hf_auth_kwargs()
        logger.info(
            "TRANSLATION_MODEL backend=%s loading route=%s model_id=%s hf_auth=%s",
            ACTIVE_TRANSLATION_BACKEND,
            desired_route,
            model_id,
            bool(auth_kwargs),
        )
        log_vram()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            **auth_kwargs,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **auth_kwargs,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
            model = model.half()
        model.eval()

        _translator_tokenizer = tokenizer
        _translator_model = model
        _translator_model_id = model_id
        _translator_route_key = desired_route

        logger.info(
            "TRANSLATION_MODEL backend=%s loaded route=%s model_id=%s",
            ACTIVE_TRANSLATION_BACKEND,
            desired_route,
            model_id,
        )
        log_vram()
        return tokenizer, model


def unload_translation_model() -> None:
    """Release the active IndicTrans2 model so downstream GPU stages have free VRAM."""
    with _translator_lock:
        _clear_loaded_model()


def assert_overlap_ready(
    stage: str = "runtime",
    min_free_vram_mb: int = DEFAULT_OVERLAP_MIN_FREE_VRAM_MB,
    source_lang: str = DEFAULT_STARTUP_SOURCE_LANG,
    target_lang: str = DEFAULT_STARTUP_TARGET_LANG,
) -> dict:
    """Require that XTTS + IndicTrans2 can stay resident together with some VRAM headroom."""
    from worker.ai_models import xtts_model

    route_key, src, tgt, _, _ = _resolve_route(source_lang, target_lang)
    xtts_model.load_xtts()
    if route_key:
        load_translation_model(route_key)
    log_vram()

    free_total = get_vram_free_total_mb()
    free_mb = free_total.get("free_mb", 0)
    total_mb = free_total.get("total_mb", 0)
    logger.info(
        "OVERLAP readiness stage=%s backend=%s route=%s source_lang=%s target_lang=%s free_vram_mb=%s total_vram_mb=%s min_free_vram_mb=%s",
        stage,
        ACTIVE_TRANSLATION_BACKEND,
        route_key or "identity",
        src,
        tgt,
        free_mb,
        total_mb,
        min_free_vram_mb,
    )
    if total_mb <= 0:
        raise RuntimeError("CUDA is not available; overlap requires both XTTS and IndicTrans2 on GPU.")
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "Overlap unsupported at stage=%s: free_vram_mb=%s is below required min_free_vram_mb=%s "
            "with XTTS and IndicTrans2 resident together."
            % (stage, free_mb, min_free_vram_mb)
        )
    return free_total


def _log_translation_debug(
    batch_label: str,
    source_lang: str,
    target_lang: str,
    inputs: List[str],
    outputs: List[str],
) -> None:
    for idx, (input_text, output_text) in enumerate(zip(inputs, outputs), start=1):
        logger.info(
            "INDICTRANS2 input batch=%s item=%d source_lang=%s target_lang=%s:\n%s",
            batch_label,
            idx,
            source_lang,
            target_lang,
            input_text,
        )
        logger.info(
            "INDICTRANS2 output batch=%s item=%d source_lang=%s target_lang=%s:\n%s",
            batch_label,
            idx,
            source_lang,
            target_lang,
            (output_text or "").strip(),
        )


def _clean_indictrans_output(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _generate_translations(texts: List[str], source_lang: str, target_lang: str) -> List[str]:
    route_key, src, tgt, src_flores, tgt_flores = _resolve_route(source_lang, target_lang)
    if route_key is None:
        return [text or "" for text in texts]

    from worker.ai_models.translation_postprocess import protect_terms, postprocess

    tokenizer, model = load_translation_model(route_key)
    processor = _load_processor()

    import torch

    normalized_texts = [(text or "").strip() for text in texts]

    protected_texts: List[str] = []
    protected_maps: List[List[str]] = []
    for text in normalized_texts:
        pt, pm = protect_terms(text)
        protected_texts.append(pt)
        protected_maps.append(pm)

    prepared_batch = processor.preprocess_batch(protected_texts, src_lang=src_flores, tgt_lang=tgt_flores)
    inputs = tokenizer(
        prepared_batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    max_chars = max((len(text) for text in protected_texts), default=0)
    max_length = max(64, min(256, max_chars * 3 + 32))

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    translated = processor.postprocess_batch(decoded, lang=tgt_flores)
    raw_outputs = [_clean_indictrans_output(text) for text in translated]
    _log_translation_debug("raw", src, tgt, normalized_texts, raw_outputs)

    outputs = [
        postprocess(src_text, raw_text, pmap, tgt)
        for src_text, raw_text, pmap in zip(normalized_texts, raw_outputs, protected_maps)
    ]
    _log_translation_debug("postprocessed", src, tgt, normalized_texts, outputs)
    return outputs


def translate_text(text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
    src = _normalize_language_code(source_lang)
    tgt = _normalize_language_code(target_lang)
    if src == tgt:
        return text

    decoded = _generate_translations([text], src, tgt)
    translated = decoded[0] if decoded else ""
    return translated or (text or "")


def translate_batch(
    segments: List[Dict],
    source_lang: str = "en",
    target_lang: str = "hi",
    batch_index: int | None = None,
    total_batches: int | None = None,
) -> List[Dict]:
    """Translate one batch of segments with a single IndicTrans2 generate call."""
    src = _normalize_language_code(source_lang)
    tgt = _normalize_language_code(target_lang)
    if not segments:
        return []
    if src == tgt:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]

    label = (
        "%d/%d" % (batch_index, total_batches)
        if batch_index is not None and total_batches is not None
        else (str(batch_index) if batch_index is not None else "single")
    )
    texts = [segment.get("text", "") for segment in segments]
    t0 = time.monotonic()
    decoded = _generate_translations(texts, src, tgt)

    result = []
    for seg, text in zip(segments, decoded):
        result.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": text or seg.get("text", ""),
                "original_text": seg.get("text", ""),
            }
        )

    logger.info(
        "TRANSLATE backend=%s batch=%s source_lang=%s target_lang=%s segments=%d elapsed_sec=%.2f",
        ACTIVE_TRANSLATION_BACKEND,
        label,
        src,
        tgt,
        len(segments),
        time.monotonic() - t0,
    )
    return result


def translate_segments(segments: List[Dict], source_lang: str = "en", target_lang: str = "hi") -> List[Dict]:
    """Translate each segment text. Preserve start/end. If source==target, return segments as-is."""
    src = _normalize_language_code(source_lang)
    tgt = _normalize_language_code(target_lang)
    if src == tgt:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]

    route_key, _, _, _, _ = _resolve_route(src, tgt)
    if route_key:
        load_translation_model(route_key)

    result = []
    total_batches = max(1, (len(segments) + DEFAULT_TRANSLATION_BATCH_SIZE - 1) // DEFAULT_TRANSLATION_BATCH_SIZE)
    t0 = time.monotonic()
    for batch_num, start in enumerate(range(0, len(segments), DEFAULT_TRANSLATION_BATCH_SIZE), start=1):
        batch = segments[start : start + DEFAULT_TRANSLATION_BATCH_SIZE]
        result.extend(
            translate_batch(
                batch,
                src,
                tgt,
                batch_index=batch_num,
                total_batches=total_batches,
            )
        )
    logger.info(
        "TRANSLATE backend=%s all source_lang=%s target_lang=%s segments=%d batches=%d elapsed_sec=%.2f",
        ACTIVE_TRANSLATION_BACKEND,
        src,
        tgt,
        len(segments),
        total_batches,
        time.monotonic() - t0,
    )
    return result


# Kept intentionally for future reuse, but not called by the active pipeline.
_MISTRAL_TRANSLATION_MODEL_ID = os.getenv("MISTRAL_TRANSLATION_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")


def _mistral_clean_translation(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?is)\btext\s*:\s*.*?\btranslation\s*:\s*", "", text)
    text = re.sub(r"(?i)\btranslated text\s*:\s*", "", text)
    text = re.sub(r"(?i)\btranslation\s*:\s*", "", text)
    text = re.sub(r"(?i)\btext\s*:\s*", "", text)
    for prefix in ("Translation:", "Translated text:", "Here is the translation:", "Output:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _mistral_build_prompt(text: str, source_lang: str, target_lang: str) -> str:
    return (
        "You are a professional subtitle translator for spoken video/audio.\n"
        f"Translate the subtitle text from {_language_name(source_lang)} to {_language_name(target_lang)}.\n"
        "Use natural, day-to-day conversational language that sounds like a real native speaker.\n"
        "Avoid overly formal, literary, textbook, or awkward phrasing.\n"
        "Return only the final translated subtitle text.\n"
        "Do not add explanations, notes, quotes, labels, 'Text:', or 'Translation:'.\n"
        "Do not repeat the source text.\n"
        "Do not transliterate the target language into Latin script.\n"
        "Keep the output fully in the target language script whenever possible.\n"
        "Preserve the meaning, tone, and punctuation.\n"
        "Keep names and numbers accurate.\n\n"
        f"Text:\n{text}"
    )
