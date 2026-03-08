"""
Translation: MarianMT or similar. Load once, translate transcript segments.
"""
import os
from typing import List, Dict

_translator = None
_translator_lang = None


def load_translation_model(source_lang: str = "en", target_lang: str = "es"):
    """Load MarianMT model for language pair. Keep in memory."""
    global _translator, _translator_lang
    key = f"{source_lang}-{target_lang}"
    if _translator is not None and _translator_lang == key:
        return _translator
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        _translator = (
            MarianTokenizer.from_pretrained(model_name),
            MarianMTModel.from_pretrained(model_name),
        )
        _translator_lang = key
        return _translator
    except Exception as e:
        raise RuntimeError(f"Translator load failed: {e}. pip install transformers torch")


def translate_text(text: str, source_lang: str = "en", target_lang: str = "es") -> str:
    if (source_lang or "en").lower() == (target_lang or "en").lower():
        return text
    tokenizer, model = load_translation_model(source_lang, target_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_segments(segments: List[Dict], source_lang: str = "en", target_lang: str = "es") -> List[Dict]:
    """Translate each segment text. Preserve start/end. If source==target, return segments as-is (no opus-mt-en-en)."""
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    if src == tgt:
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", ""), "original_text": s.get("text", "")}
            for s in segments
        ]
    result = []
    for s in segments:
        result.append({
            "start": s["start"],
            "end": s["end"],
            "text": translate_text(s["text"], source_lang, target_lang),
            "original_text": s.get("text", ""),
        })
    return result
