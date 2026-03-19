"""
Phi-3-mini for casual/natural language rewrite. CPU only. Mandatory; no stub.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

# Default model; override with REWRITE_LLM_MODEL env (e.g. microsoft/Phi-3-mini-4k-instruct)
DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def get_llm():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = os.getenv("REWRITE_LLM_MODEL", DEFAULT_MODEL)
    logger.info("Loading rewrite LLM: %s (CPU)", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    logger.info("Rewrite LLM loaded.")
    return _model, _tokenizer


def rewrite_with_llm(
    text: str,
    language: str = "hi",
    style_hint: Optional[str] = None,
    target_syllables: Optional[int] = None,
) -> str:
    """
    Rewrite text to sound natural and conversational in the target language.
    If target_syllables is set, instructs the model to use approximately that many syllables (±20%).
    """
    model, tokenizer = get_llm()
    lang_names = {"hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French", "de": "German"}
    lang_name = lang_names.get(language, language)
    extra = ""
    if target_syllables is not None and target_syllables > 0:
        low = max(1, target_syllables - int(target_syllables * 0.2))
        high = target_syllables + int(target_syllables * 0.2)
        extra = f"3. Use approximately {target_syllables} syllables (between {low}-{high}). Keep technical or proper words unchanged.\n"
    prompt = f"""Rewrite the following {lang_name} sentence so that:
1. It sounds conversational and natural.
2. It keeps the same meaning.
{extra}Output only the rewritten sentence, nothing else.

Sentence:
{text}

Rewritten:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with __import__("torch").no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    reply = reply.strip().split("\n")[0].strip()
    return reply if reply else text
