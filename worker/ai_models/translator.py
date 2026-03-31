"""
Translation via Mistral 7B Instruct.

The model can stay resident alongside XTTS for experimental overlapped dubbing.
"""
import logging
import os
import re
import threading
import time
from typing import List, Dict, Any

from worker.gpu.vram_monitor import get_vram_free_total_mb, log_vram

logger = logging.getLogger(__name__)

DEFAULT_TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
DEFAULT_TRANSLATION_BATCH_SIZE = max(1, int(os.getenv("TRANSLATION_BATCH_SIZE", "4")))
DEFAULT_OVERLAP_MIN_FREE_VRAM_MB = max(0, int(os.getenv("DUB_OVERLAP_MIN_FREE_VRAM_MB", "1024")))

_translator_tokenizer: Any = None
_translator_model: Any = None
_translator_model_id: str = ""
_translator_lock = threading.Lock()

_LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
}


def _language_name(language: str) -> str:
    code = (language or "").strip().lower()
    return _LANGUAGE_NAMES.get(code, code or "English")


def _clean_translation(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Remove any repeated instruction labels or inline source/translation scaffolding the model echoes back.
    text = re.sub(r"(?is)\btext\s*:\s*.*?\btranslation\s*:\s*", "", text)
    text = re.sub(r"(?i)\btranslated text\s*:\s*", "", text)
    text = re.sub(r"(?i)\btranslation\s*:\s*", "", text)
    text = re.sub(r"(?i)\btext\s*:\s*", "", text)
    for prefix in (
        "Translation:",
        "Translated text:",
        "Here is the translation:",
        "Output:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _build_prompt(text: str, source_lang: str, target_lang: str) -> str:
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


def load_translation_model(model_id: str = DEFAULT_TRANSLATION_MODEL):
    """Load quantized Mistral 7B once for translation."""
    global _translator_tokenizer, _translator_model, _translator_model_id
    with _translator_lock:
        if _translator_model is not None and _translator_tokenizer is not None and _translator_model_id == model_id:
            return _translator_tokenizer, _translator_model

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("TRANSLATION_MODEL loading model_id=%s", model_id)
        log_vram()

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        _translator_tokenizer = tokenizer
        _translator_model = model
        _translator_model_id = model_id

        logger.info("TRANSLATION_MODEL loaded model_id=%s", model_id)
        log_vram()
        return tokenizer, model


def unload_translation_model() -> None:
    """Release the translation model so downstream GPU stages have free VRAM."""
    global _translator_tokenizer, _translator_model, _translator_model_id
    if _translator_model is None and _translator_tokenizer is None:
        return
    try:
        import gc
        import torch

        del _translator_model
        del _translator_tokenizer
        _translator_model = None
        _translator_tokenizer = None
        _translator_model_id = ""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("Could not fully unload translation model: %s", e)


def assert_overlap_ready(stage: str = "runtime", min_free_vram_mb: int = DEFAULT_OVERLAP_MIN_FREE_VRAM_MB) -> dict:
    """Require that XTTS + Mistral can stay resident together with some VRAM headroom."""
    from worker.ai_models import xtts_model

    xtts_model.load_xtts()
    load_translation_model()
    log_vram()

    free_total = get_vram_free_total_mb()
    free_mb = free_total.get("free_mb", 0)
    total_mb = free_total.get("total_mb", 0)
    logger.info(
        "OVERLAP readiness stage=%s free_vram_mb=%s total_vram_mb=%s min_free_vram_mb=%s",
        stage,
        free_mb,
        total_mb,
        min_free_vram_mb,
    )
    if total_mb <= 0:
        raise RuntimeError("CUDA is not available; overlap requires both XTTS and Mistral on GPU.")
    if free_mb < min_free_vram_mb:
        raise RuntimeError(
            "True overlap unsupported at stage=%s: free_vram_mb=%s is below required min_free_vram_mb=%s "
            "with XTTS and Mistral resident together."
            % (stage, free_mb, min_free_vram_mb)
        )
    return free_total


def _generate_translations(prompts: List[str], max_new_tokens: int) -> List[str]:
    tokenizer, model = load_translation_model()

    import torch

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[:, prompt_len:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in generated]


def translate_text(text: str, source_lang: str = "en", target_lang: str = "es") -> str:
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    if src == tgt:
        return text

    prompts = [_build_prompt(text, src, tgt)]
    max_new_tokens = max(64, min(256, len((text or "").strip()) * 2 + 32))
    translated = _generate_translations(prompts, max_new_tokens=max_new_tokens)[0]
    return _clean_translation(translated) or (text or "")


def translate_batch(
    segments: List[Dict],
    source_lang: str = "en",
    target_lang: str = "es",
    batch_index: int | None = None,
    total_batches: int | None = None,
) -> List[Dict]:
    """Translate one batch of segments with a single generate call."""
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    if not segments:
        return []
    if src == tgt:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]

    prompts = [_build_prompt(s.get("text", ""), src, tgt) for s in segments]
    max_chars = max(len((s.get("text", "") or "").strip()) for s in segments)
    max_new_tokens = max(64, min(256, max_chars * 2 + 32))
    label = (
        "%d/%d" % (batch_index, total_batches)
        if batch_index is not None and total_batches is not None
        else (str(batch_index) if batch_index is not None else "single")
    )
    t0 = time.monotonic()
    decoded = _generate_translations(prompts, max_new_tokens=max_new_tokens)

    result = []
    for seg, text in zip(segments, decoded):
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": _clean_translation(text) or seg.get("text", ""),
            "original_text": seg.get("text", ""),
        })

    logger.info(
        "TRANSLATE batch=%s source_lang=%s target_lang=%s segments=%d elapsed_sec=%.2f",
        label,
        src,
        tgt,
        len(segments),
        time.monotonic() - t0,
    )
    return result


def translate_segments(segments: List[Dict], source_lang: str = "en", target_lang: str = "es") -> List[Dict]:
    """Translate each segment text. Preserve start/end. If source==target, return segments as-is."""
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    if src == tgt:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]

    result = []
    total_batches = max(1, (len(segments) + DEFAULT_TRANSLATION_BATCH_SIZE - 1) // DEFAULT_TRANSLATION_BATCH_SIZE)
    t0 = time.monotonic()
    load_translation_model()
    for batch_num, start in enumerate(range(0, len(segments), DEFAULT_TRANSLATION_BATCH_SIZE), start=1):
        batch = segments[start : start + DEFAULT_TRANSLATION_BATCH_SIZE]
        result.extend(
            translate_batch(
                batch,
                source_lang,
                target_lang,
                batch_index=batch_num,
                total_batches=total_batches,
            )
        )
    logger.info(
        "TRANSLATE all source_lang=%s target_lang=%s segments=%d batches=%d elapsed_sec=%.2f",
        src,
        tgt,
        len(segments),
        total_batches,
        time.monotonic() - t0,
    )
    return result
