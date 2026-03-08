"""
Translation: MarianMT. Load once per language pair. If direct model (e.g. es-hi) does not exist,
use pivot via English (es->en, then en->hi) so Hindi and other languages work.
"""
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

_translator: Any = None
_translator_lang: str = ""


def _load_pair(source_lang: str, target_lang: str) -> Tuple[Any, Any]:
    """Load a single MarianMT model pair. Returns (tokenizer, model)."""
    from transformers import MarianMTModel, MarianTokenizer
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def load_translation_model(source_lang: str = "en", target_lang: str = "es"):
    """Load MarianMT for language pair. Uses pivot via English if direct model does not exist."""
    global _translator, _translator_lang
    src = (source_lang or "en").lower()
    tgt = (target_lang or "en").lower()
    key = f"{src}-{tgt}"
    if _translator is not None and _translator_lang == key:
        return _translator

    from transformers import MarianMTModel, MarianTokenizer

    # 1) Try direct model (e.g. opus-mt-es-hi)
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        _translator = (
            MarianTokenizer.from_pretrained(model_name),
            MarianMTModel.from_pretrained(model_name),
        )
        _translator_lang = key
        logger.info("Translator loaded direct: %s", model_name)
        return _translator
    except Exception as e:
        logger.info("Direct model %s-%s not available (%s), trying pivot via English", src, tgt, e)

    # 2) Pivot via English: source -> en -> target (when both differ from en)
    if src == "en":
        # Only need en -> target
        try:
            _translator = _load_pair("en", tgt)
            _translator_lang = key
            logger.info("Translator loaded: en->%s", tgt)
            return _translator
        except Exception as e:
            raise RuntimeError(f"Translator load failed for en->{tgt}: {e}") from e
    if tgt == "en":
        # Only need source -> en
        try:
            _translator = _load_pair(src, "en")
            _translator_lang = key
            logger.info("Translator loaded: %s->en", src)
            return _translator
        except Exception as e:
            raise RuntimeError(f"Translator load failed for {src}->en: {e}") from e

    # Both non-English: load src->en and en->tgt
    try:
        tok_src_en, model_src_en = _load_pair(src, "en")
        tok_en_tgt, model_en_tgt = _load_pair("en", tgt)
        _translator = ("pivot", tok_src_en, model_src_en, tok_en_tgt, model_en_tgt)
        _translator_lang = key
        logger.info("Translator loaded pivot: %s->en->%s", src, tgt)
        return _translator
    except Exception as e:
        raise RuntimeError(f"Translator load failed (pivot {src}->en->{tgt}): {e}. pip install transformers torch") from e


def translate_text(text: str, source_lang: str = "en", target_lang: str = "es") -> str:
    if (source_lang or "en").lower() == (target_lang or "en").lower():
        return text
    trans = load_translation_model(source_lang, target_lang)
    if isinstance(trans, tuple) and len(trans) == 5 and trans[0] == "pivot":
        _, tok_src_en, model_src_en, tok_en_tgt, model_en_tgt = trans
        # source -> English
        inp1 = tok_src_en(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        out1 = model_src_en.generate(**inp1)
        mid_text = tok_src_en.decode(out1[0], skip_special_tokens=True)
        # English -> target
        inp2 = tok_en_tgt(mid_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        out2 = model_en_tgt.generate(**inp2)
        return tok_en_tgt.decode(out2[0], skip_special_tokens=True)
    else:
        tokenizer, model = trans
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
    for s in segments:
        result.append({
            "start": s["start"],
            "end": s["end"],
            "text": translate_text(s["text"], source_lang, target_lang),
            "original_text": s.get("text", ""),
        })
    return result
