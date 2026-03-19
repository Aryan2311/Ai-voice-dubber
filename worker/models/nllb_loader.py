"""
NLLB-200 translation only. CPU. No fallbacks; raises if load or translate fails.
"""
import logging
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None

# NLLB-200 language codes (language_script). ISO 639-1/2/3 -> NLLB code.
# See https://huggingface.co/facebook/nllb-200-distilled-600M
NLLB_LANG_CODES = {
    "en": "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn",
    "hi": "hin_Deva", "hin": "hin_Deva", "hindi": "hin_Deva",
    "es": "spa_Latn", "spa": "spa_Latn", "spanish": "spa_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn", "french": "fra_Latn",
    "de": "deu_Latn", "deu": "deu_Latn", "german": "deu_Latn",
    "it": "ita_Latn", "ita": "ita_Latn", "italian": "ita_Latn",
    "pt": "por_Latn", "por": "por_Latn", "portuguese": "por_Latn",
    "ru": "rus_Cyrl", "rus": "rus_Cyrl", "russian": "rus_Cyrl",
    "ja": "jpn_Jpan", "jpn": "jpn_Jpan", "japanese": "jpn_Jpan",
    "ko": "kor_Hang", "kor": "kor_Hang", "korean": "kor_Hang",
    "zh": "zho_Hans", "zho": "zho_Hans", "chinese": "zho_Hans",
    "cmn": "zho_Hans", "yue": "yue_Hant", "cantonese": "yue_Hant",
    "ar": "arb_Arab", "arb": "arb_Arab", "arabic": "arb_Arab",
    "tr": "tur_Latn", "tur": "tur_Latn", "turkish": "tur_Latn",
    "pl": "pol_Latn", "pol": "pol_Latn", "polish": "pol_Latn",
    "nl": "nld_Latn", "nld": "nld_Latn", "dutch": "nld_Latn",
    "vi": "vie_Latn", "vie": "vie_Latn", "vietnamese": "vie_Latn",
    "th": "tha_Thai", "tha": "tha_Thai", "thai": "tha_Thai",
    "id": "ind_Latn", "ind": "ind_Latn", "indonesian": "ind_Latn",
    "ms": "zsm_Latn", "zsm": "zsm_Latn", "malay": "zsm_Latn",
    "bn": "ben_Beng", "ben": "ben_Beng", "bengali": "ben_Beng",
    "ta": "tam_Taml", "tam": "tam_Taml", "tamil": "tam_Taml",
    "te": "tel_Telu", "tel": "tel_Telu", "telugu": "tel_Telu",
    "mr": "mar_Deva", "mar": "mar_Deva", "marathi": "mar_Deva",
    "gu": "guj_Gujr", "guj": "guj_Gujr", "gujarati": "guj_Gujr",
    "kn": "kan_Knda", "kan": "kan_Knda", "kannada": "kan_Knda",
    "ml": "mal_Mlym", "mal": "mal_Mlym", "malayalam": "mal_Mlym",
    "pa": "pan_Guru", "pan": "pan_Guru", "punjabi": "pan_Guru",
    "ur": "urd_Arab", "urd": "urd_Arab", "urdu": "urd_Arab",
    "fa": "pes_Arab", "pes": "pes_Arab", "persian": "pes_Arab",
    "he": "heb_Hebr", "heb": "heb_Hebr", "hebrew": "heb_Hebr",
    "el": "ell_Grek", "ell": "ell_Grek", "greek": "ell_Grek",
    "hu": "hun_Latn", "hun": "hun_Latn", "hungarian": "hun_Latn",
    "cs": "ces_Latn", "ces": "ces_Latn", "czech": "ces_Latn",
    "ro": "ron_Latn", "ron": "ron_Latn", "romanian": "ron_Latn",
    "sv": "swe_Latn", "swe": "swe_Latn", "swedish": "swe_Latn",
    "da": "dan_Latn", "dan": "dan_Latn", "danish": "dan_Latn",
    "fi": "fin_Latn", "fin": "fin_Latn", "finnish": "fin_Latn",
    "no": "nob_Latn", "nob": "nob_Latn", "norwegian": "nob_Latn",
    "uk": "ukr_Cyrl", "ukr": "ukr_Cyrl", "ukrainian": "ukr_Cyrl",
    "bg": "bul_Cyrl", "bul": "bul_Cyrl", "bulgarian": "bul_Cyrl",
    "hr": "hrv_Latn", "hrv": "hrv_Latn", "croatian": "hrv_Latn",
    "sr": "srp_Cyrl", "srp": "srp_Cyrl", "serbian": "srp_Cyrl",
    "sk": "slk_Latn", "slk": "slk_Latn", "slovak": "slk_Latn",
    "sl": "slv_Latn", "slv": "slv_Latn", "slovenian": "slv_Latn",
    "ca": "cat_Latn", "cat": "cat_Latn", "catalan": "cat_Latn",
    "af": "afr_Latn", "afr": "afr_Latn", "afrikaans": "afr_Latn",
    "sw": "swh_Latn", "swh": "swh_Latn", "swahili": "swh_Latn",
    "am": "amh_Ethi", "amh": "amh_Ethi", "amharic": "amh_Ethi",
    "ne": "npi_Deva", "npi": "npi_Deva", "nepali": "npi_Deva",
    "si": "sin_Sinh", "sin": "sin_Sinh", "sinhala": "sin_Sinh",
    "my": "mya_Mymr", "mya": "mya_Mymr", "burmese": "mya_Mymr",
    "km": "khm_Khmr", "khm": "khm_Khmr", "khmer": "khm_Khmr",
    "lo": "lao_Laoo", "lao": "lao_Laoo",
    "tl": "tgl_Latn", "tgl": "tgl_Latn", "tagalog": "tgl_Latn",
}


def load_nllb() -> Tuple[Any, Any]:
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    name = "facebook/nllb-200-distilled-600M"
    _tokenizer = AutoTokenizer.from_pretrained(name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(name)
    logger.info("NLLB-200 loaded: %s", name)
    return _tokenizer, _model


def _nllb_lang_code(lang: str) -> str:
    """Map language to NLLB code. Raises if unknown."""
    key = (lang or "en").strip().lower()
    code = NLLB_LANG_CODES.get(key)
    if code is None:
        code = NLLB_LANG_CODES.get(key[:2])
    if code is None:
        code = NLLB_LANG_CODES.get(key[:3])
    if code is not None:
        return code
    # Fallback: try eng_Latn for unknown (avoid crash); log warning
    logger.warning("Unknown NLLB language %r, using eng_Latn", lang)
    return "eng_Latn"


def translate_text(text: str, source_lang: str = "en", target_lang: str = "hi") -> str:
    if (source_lang or "en").strip().lower() == (target_lang or "en").strip().lower():
        return text
    tok, model = load_nllb()
    tgt_code = _nllb_lang_code(target_lang)
    forced_bos_id = tok.convert_tokens_to_ids(tgt_code)
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    out = model.generate(**inputs, forced_bos_token_id=forced_bos_id)
    return tok.decode(out[0], skip_special_tokens=True)
