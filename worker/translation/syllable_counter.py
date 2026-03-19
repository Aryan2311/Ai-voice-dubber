"""
Syllable counting for source and target languages. Used for syllable-aware translation/rewrite.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Devanagari and common Indic vowel matras (approximate syllable nuclei)
_DEVANAGARI_VOWELS = set(
    "अआइईउऊऋॠऌॡएऐओऔ"
    "ा ि ी ु ू ृ ॄ ॢ ॣ े ै ो ौ ं ः ्"
)
# Latin/English: use pyphen when available
_pyphen_dics = {}


def count_syllables(text: str, lang: str) -> int:
    """
    Return approximate syllable count for the given text and language code.
    Supports: en (pyphen), hi (Devanagari vowel heuristic), and fallback (space-based).
    """
    if not (text or "").strip():
        return 0
    lang = (lang or "en").lower()[:2]
    if lang == "en":
        return _count_syllables_en(text)
    if lang == "hi":
        return _count_syllables_hi(text)
    # Generic: try pyphen with lang if available, else word count as proxy
    try:
        return _count_syllables_pyphen(text, lang)
    except Exception:
        pass
    return max(1, len(text.split()))


def _count_syllables_en(text: str) -> int:
    try:
        import pyphen
        if "en" not in _pyphen_dics:
            _pyphen_dics["en"] = pyphen.Pyphen(lang="en")
        dic = _pyphen_dics["en"]
        count = 0
        for w in text.split():
            count += dic.inserted(w).count("-") + 1
        return max(1, count)
    except Exception as e:
        logger.debug("pyphen fallback: %s", e)
        return max(1, len(text.split()))


def _count_syllables_hi(text: str) -> int:
    """Devanagari: approximate by vowel count (independent vowels + matras)."""
    count = sum(1 for c in text if c in _DEVANAGARI_VOWELS)
    if count > 0:
        return count
    return max(1, len([w for w in text.split() if w]))


def _count_syllables_pyphen(text: str, lang: str) -> int:
    import pyphen
    if lang not in _pyphen_dics:
        try:
            _pyphen_dics[lang] = pyphen.Pyphen(lang=lang)
        except Exception:
            return max(1, len(text.split()))
    dic = _pyphen_dics[lang]
    count = 0
    for w in text.split():
        count += dic.inserted(w).count("-") + 1
    return max(1, count)


def get_syllable_tolerance(target: int, tolerance_pct: float = 0.2) -> tuple:
    """Return (min_allowed, max_allowed) for target syllable count. Default ±20%."""
    delta = max(1, int(round(target * tolerance_pct)))
    return (max(1, target - delta), target + delta)
