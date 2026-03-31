"""
Qwen2.5-3B-Instruct rewriter for improving machine-translated subtitles.

Takes original English + IndicTrans2 translation, outputs conversational-tone
translation.  Loaded in 4-bit quantization (~2.5 GB VRAM) alongside XTTS and
IndicTrans2.  Controlled by REWRITER_ENABLED env var (default: true).
"""
from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List

from worker.gpu.vram_monitor import log_vram

logger = logging.getLogger(__name__)

REWRITER_ENABLED = os.getenv("REWRITER_ENABLED", "true").strip().lower() in ("1", "true", "yes")
REWRITER_MODEL_ID = os.getenv("REWRITER_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
REWRITER_MAX_NEW_TOKENS = int(os.getenv("REWRITER_MAX_NEW_TOKENS", "150"))

_rewriter_model: Any = None
_rewriter_tokenizer: Any = None
_rewriter_lock = threading.Lock()

_LANG_NAMES: Dict[str, str] = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ur": "Urdu", "as": "Assamese", "or": "Odia",
    "ne": "Nepali",
}


def _hf_auth_kwargs() -> Dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {}
    return {"use_auth_token": token}


def load_rewriter() -> None:
    """Load Qwen2.5-3B-Instruct in 4-bit quantization."""
    global _rewriter_model, _rewriter_tokenizer

    if not REWRITER_ENABLED:
        logger.info("REWRITER disabled via REWRITER_ENABLED env var")
        return

    with _rewriter_lock:
        if _rewriter_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        auth_kwargs = _hf_auth_kwargs()
        logger.info("REWRITER loading model_id=%s quantization=4bit", REWRITER_MODEL_ID)
        log_vram()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            REWRITER_MODEL_ID,
            trust_remote_code=True,
            **auth_kwargs,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            REWRITER_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            **auth_kwargs,
        )
        model.eval()

        _rewriter_tokenizer = tokenizer
        _rewriter_model = model

        logger.info("REWRITER loaded model_id=%s", REWRITER_MODEL_ID)
        log_vram()


def unload_rewriter() -> None:
    """Release the rewriter model and free VRAM."""
    global _rewriter_model, _rewriter_tokenizer
    with _rewriter_lock:
        model = _rewriter_model
        tokenizer = _rewriter_tokenizer
        _rewriter_model = None
        _rewriter_tokenizer = None
        del model, tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def is_loaded() -> bool:
    return _rewriter_model is not None and _rewriter_tokenizer is not None


def _build_prompt(english: str, translated: str, lang_name: str) -> str:
    return (
        f"Rewrite the {lang_name} subtitle to sound natural and conversational — "
        f"like how people actually talk in everyday {lang_name}.\n"
        f"Keep English terms that {lang_name} speakers normally use as-is "
        f"(like subscribe, video, gym, workout, etc).\n"
        f"Don't add or remove any information.\n"
        f"Return ONLY the improved {lang_name} text, nothing else.\n\n"
        f"English: {english}\n"
        f"{lang_name}: {translated}\n"
        f"Rewrite:"
    )


def _clean_output(text: str) -> str:
    """Strip common LLM artifacts."""
    text = (text or "").strip()
    for prefix in ("Rewrite:", "Improved Hindi:", "Improved:", "Hindi:", "Output:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rewrite_batch(
    originals: List[str],
    translations: List[str],
    target_lang: str = "hi",
) -> List[str]:
    """Rewrite translations for conversational tone.

    Falls back to the input translations when the rewriter is disabled,
    unloaded, or produces suspicious output for a given segment.
    """
    if not REWRITER_ENABLED or not is_loaded():
        return translations
    if not originals or not translations:
        return translations

    import torch

    lang_name = _LANG_NAMES.get(target_lang, "Hindi")
    t0 = time.monotonic()

    prompts: List[str] = []
    for eng, trans in zip(originals, translations):
        messages = [{"role": "user", "content": _build_prompt(eng.strip(), trans.strip(), lang_name)}]
        prompts.append(
            _rewriter_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    inputs = _rewriter_tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    device = next(_rewriter_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = _rewriter_model.generate(
            **inputs,
            max_new_tokens=REWRITER_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_rewriter_tokenizer.pad_token_id,
        )

    results: List[str] = []
    for i, (gen_ids, original_trans) in enumerate(zip(output_ids, translations)):
        new_tokens = gen_ids[input_len:]
        decoded = _rewriter_tokenizer.decode(new_tokens, skip_special_tokens=True)
        cleaned = _clean_output(decoded)

        too_short = len(cleaned) < max(5, len(original_trans) * 0.25)
        too_long = len(cleaned) > len(original_trans) * 3.5
        if too_short or too_long or not cleaned:
            logger.warning(
                "REWRITER fallback item=%d reason=%s len_out=%d len_mt=%d",
                i, "short" if too_short else ("long" if too_long else "empty"),
                len(cleaned), len(original_trans),
            )
            results.append(original_trans)
        else:
            results.append(cleaned)

    elapsed = time.monotonic() - t0
    logger.info("REWRITER batch=%d target_lang=%s elapsed_sec=%.2f", len(originals), target_lang, elapsed)
    for i, (orig, trans, rw) in enumerate(zip(originals, translations, results)):
        logger.info("REWRITER item=%d\n  EN: %s\n  MT: %s\n  RW: %s", i, orig.strip(), trans.strip(), rw.strip())

    return results
