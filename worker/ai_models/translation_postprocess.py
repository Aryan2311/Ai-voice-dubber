"""
Post-processing for machine-translated subtitles.

Two-stage pipeline applied around IndicTrans2 inference:

  1. **Pre-translation** – protect English domain terms with placeholders so the
     NMT model does not literally translate them (e.g. "crow pose").
  2. **Post-translation** – restore placeholders as Devanagari transliterations,
     then swap formal/literary Hindi for everyday colloquial equivalents.

Currently Hindi-specific; add other target-language blocks as needed.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  PRE-TRANSLATION: English term protection
# ═══════════════════════════════════════════════════════════════════════════

_PH_FMT = "XTERM{}X"
_PH_RE = re.compile(r"XTERM(\d+)X")

# ── Multi-word English terms that spoken Hindi uses as-is ──────────────────
_MULTIWORD_TERMS: List[str] = [
    "bench press", "wall walk", "wall walks",
    "hollow body hold", "hollow body holds", "hollow body",
    "frog stand", "crow pose", "pike pushup", "pike pushups",
    "dead hang", "muscle up", "muscle ups",
    "warm up", "cool down",
    "pull up", "pull ups", "push up", "push ups",
    "side plank", "front lever", "back lever", "human flag",
    "pistol squat", "pistol squats",
    "box jump", "box jumps", "jump rope", "battle rope",
    "resistance band", "foam roller", "yoga mat",
    "hip flexor", "hip flexors",
    "weight transfer",
]

# ── Tail words: `<word(s)> <tail>` → protect the whole phrase ─────────────
_COMPOUND_TAILS = frozenset(
    "pose poses stand stands pushup pushups push-up push-ups "
    "pullup pullups pull-up pull-ups squat squats plank planks "
    "crunch crunches press curl curls hold holds walk walks "
    "kick kicks raise raises dip dips row rows lift lifts "
    "stretch stretches twist twists roll rolls swing swings "
    "jump jumps lunge lunges bridge bridges fly flies "
    "deadlift deadlifts snatch snatches handstand headstand".split()
)

# ── Single English words that are ALWAYS said in English in spoken Hindi ──
# IMPORTANT: Only include words that are unambiguous and universally used as
# English loanwords.  Do NOT include context-dependent words like "like"
# (preference vs social-media), "follow" (instructions vs social-media),
# "set" (configure vs exercise-set), "share" (divide vs social-media), etc.
# Those are better handled by the colloquial table or the LLM rewriter.
_SINGLE_PASSTHROUGH = frozenset(
    # Social media — unambiguous terms
    "subscribe channel "
    # Tech — universally used as English in Hindi
    "app website video phone mobile laptop wifi bluetooth email notification "
    # Fitness — gym-culture terms that are never translated
    "gym workout cardio pushup pullup squat plank crunch "
    "deadlift dumbbell barbell treadmill handstand headstand "
    "yoga pilates crossfit hiit "
    # Health
    "protein calories diet supplement"
    .split()
)

_COMPOUND_TAIL_RE = re.compile(
    r"\b([A-Za-z]+(?:[\s-][A-Za-z]+){0,2})\s+("
    + "|".join(re.escape(t) for t in sorted(_COMPOUND_TAILS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def protect_terms(text: str) -> Tuple[str, List[str]]:
    """Replace English domain terms with numbered placeholders.

    Returns ``(modified_text, [original_0, original_1, …])``.
    """
    if not text or not text.strip():
        return text, []

    protected: List[str] = []
    result = text

    def _already_protected(start: int, end: int) -> bool:
        ctx = result[max(0, start - 6): end + 6]
        return "XTERM" in ctx

    # Pass 1 — multi-word terms (longest first)
    for term in sorted(_MULTIWORD_TERMS, key=len, reverse=True):
        pat = re.compile(re.escape(term), re.IGNORECASE)
        for m in reversed(list(pat.finditer(result))):
            if _already_protected(m.start(), m.end()):
                continue
            idx = len(protected)
            protected.append(m.group(0))
            result = result[: m.start()] + _PH_FMT.format(idx) + result[m.end():]

    # Pass 2 — compound patterns (<word> <tail>)
    for m in reversed(list(_COMPOUND_TAIL_RE.finditer(result))):
        if _already_protected(m.start(), m.end()):
            continue
        idx = len(protected)
        protected.append(m.group(0))
        result = result[: m.start()] + _PH_FMT.format(idx) + result[m.end():]

    # Pass 3 — standalone passthrough words
    for term in sorted(_SINGLE_PASSTHROUGH, key=len, reverse=True):
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        for m in reversed(list(pat.finditer(result))):
            if _already_protected(m.start(), m.end()):
                continue
            idx = len(protected)
            protected.append(m.group(0))
            result = result[: m.start()] + _PH_FMT.format(idx) + result[m.end():]

    if protected:
        logger.debug("TERM_PROTECT count=%d src=%r → %r", len(protected), text, result)
    return result, protected


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRANSLITERATION DICTIONARY  (English → Devanagari)
# ═══════════════════════════════════════════════════════════════════════════

_EN_TO_DEVA: Dict[str, str] = {
    # ── Social media / tech ──
    "subscribe": "सब्सक्राइब", "like": "लाइक", "share": "शेयर",
    "comment": "कमेंट", "comments": "कमेंट्स", "post": "पोस्ट",
    "follow": "फॉलो", "unfollow": "अनफॉलो", "block": "ब्लॉक",
    "tag": "टैग", "story": "स्टोरी", "reel": "रील", "feed": "फीड",
    "live": "लाइव", "stream": "स्ट्रीम", "trend": "ट्रेंड",
    "trending": "ट्रेंडिंग", "viral": "वायरल", "content": "कंटेंट",
    "creator": "क्रिएटर", "influencer": "इंफ्लुएंसर",
    "channel": "चैनल", "upload": "अपलोड", "download": "डाउनलोड",
    "update": "अपडेट", "app": "ऐप", "website": "वेबसाइट",
    "online": "ऑनलाइन", "offline": "ऑफलाइन",
    "video": "वीडियो", "audio": "ऑडियो",
    "notification": "नोटिफिकेशन", "phone": "फोन", "mobile": "मोबाइल",
    "laptop": "लैपटॉप", "screen": "स्क्रीन",
    "email": "ईमेल", "login": "लॉगिन", "logout": "लॉगआउट",
    "password": "पासवर्ड", "wifi": "वाईफाई", "bluetooth": "ब्लूटूथ",
    # ── Fitness ──
    "gym": "जिम", "workout": "वर्कआउट", "exercise": "एक्सरसाइज़",
    "set": "सेट", "rep": "रेप", "reps": "रेप्स", "round": "राउंड",
    "cardio": "कार्डियो", "training": "ट्रेनिंग",
    "beginner": "बिगिनर", "intermediate": "इंटरमीडिएट",
    "advanced": "एडवांस्ड",
    "pushup": "पुशअप", "pushups": "पुशअप्स",
    "pullup": "पुलअप", "pullups": "पुलअप्स",
    "push up": "पुश अप", "push ups": "पुश अप्स",
    "pull up": "पुल अप", "pull ups": "पुल अप्स",
    "squat": "स्क्वॉट", "squats": "स्क्वॉट्स",
    "plank": "प्लैंक", "planks": "प्लैंक्स",
    "side plank": "साइड प्लैंक",
    "crunch": "क्रंच", "crunches": "क्रंचेज़",
    "deadlift": "डेडलिफ्ट", "deadlifts": "डेडलिफ्ट्स",
    "dumbbell": "डंबल", "barbell": "बारबेल",
    "treadmill": "ट्रेडमिल", "handstand": "हैंडस्टैंड",
    "headstand": "हेडस्टैंड", "yoga": "योगा", "pilates": "पिलाटीज़",
    "crossfit": "क्रॉसफिट", "hiit": "HIIT",
    "routine": "रूटीन",
    # ── Compound fitness terms ──
    "bench press": "बेंच प्रेस", "wall walk": "वॉल वॉक",
    "wall walks": "वॉल वॉक्स",
    "hollow body hold": "हॉलो बॉडी होल्ड",
    "hollow body holds": "हॉलो बॉडी होल्ड्स",
    "hollow body": "हॉलो बॉडी",
    "frog stand": "फ्रॉग स्टैंड", "crow pose": "क्रो पोज़",
    "pike pushup": "पाइक पुशअप", "pike pushups": "पाइक पुशअप्स",
    "dead hang": "डेड हैंग", "muscle up": "मसल अप",
    "muscle ups": "मसल अप्स", "warm up": "वॉर्म अप",
    "cool down": "कूल डाउन",
    "front lever": "फ्रंट लीवर", "back lever": "बैक लीवर",
    "human flag": "ह्यूमन फ्लैग",
    "pistol squat": "पिस्टल स्क्वॉट",
    "pistol squats": "पिस्टल स्क्वॉट्स",
    "box jump": "बॉक्स जंप", "box jumps": "बॉक्स जंप्स",
    "jump rope": "जंप रोप", "battle rope": "बैटल रोप",
    "resistance band": "रेज़िस्टेंस बैंड", "foam roller": "फोम रोलर",
    "yoga mat": "योगा मैट",
    "hip flexor": "हिप फ्लेक्सर", "hip flexors": "हिप फ्लेक्सर्स",
    "weight transfer": "वेट ट्रांसफर",
    # ── Generic tail words (used when compound lookup has no entry) ──
    "pose": "पोज़", "poses": "पोज़ेज़", "stand": "स्टैंड",
    "hold": "होल्ड", "holds": "होल्ड्स", "press": "प्रेस",
    "stretch": "स्ट्रेच", "twist": "ट्विस्ट", "walk": "वॉक",
    "kick": "किक", "raise": "रेज़", "dip": "डिप", "row": "रो",
    "lift": "लिफ्ट", "swing": "स्विंग", "jump": "जंप",
    "lunge": "लंज", "bridge": "ब्रिज", "fly": "फ्लाई",
    "roll": "रोल", "curl": "कर्ल",
    # ── Health ──
    "diet": "डाइट", "protein": "प्रोटीन", "calories": "कैलोरीज़",
    "carbs": "कार्ब्स", "fat": "फैट", "fiber": "फाइबर",
    "supplement": "सप्लीमेंट", "smoothie": "स्मूदी", "shake": "शेक",
}


def _transliterate(english_term: str) -> str:
    """Best-effort English → Devanagari.  Falls back to the original English."""
    key = english_term.lower().strip()
    hit = _EN_TO_DEVA.get(key)
    if hit:
        return hit
    words = key.split()
    if len(words) > 1:
        parts = [_EN_TO_DEVA.get(w, w) for w in words]
        return " ".join(parts)
    return english_term


def restore_terms(text: str, protected: List[str]) -> str:
    """Replace numbered placeholders with Hindi transliterations."""
    if not protected or not text:
        return text or ""

    def _repl(m: re.Match) -> str:
        idx = int(m.group(1))
        if idx < len(protected):
            return _transliterate(protected[idx])
        return m.group(0)

    result = _PH_RE.sub(_repl, text)
    remaining = _PH_RE.findall(result)
    if remaining:
        logger.warning("TERM_RESTORE leftover placeholders: %s", remaining)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3.  HINDI: formal → colloquial  (post-translation)
# ═══════════════════════════════════════════════════════════════════════════

# ── 3a. Verb-morphing rules ───────────────────────────────────────────────
# When a formal Hindi noun pairs with verb X but the colloquial replacement
# needs verb Y, we map every conjugation form so grammar stays intact.
# e.g. "सदस्यता लेना" → "सब्सक्राइब करना" (all tenses/moods).

_LENA_TO_KARNA: List[Tuple[str, str]] = [
    ("ले लें", "कर लें"),
    ("ले लो", "कर लो"),
    ("ले ली", "कर ली"),
    ("ले लिया", "कर लिया"),
    ("ले लिए", "कर लिए"),
    ("ले लिये", "कर लिये"),
    ("लेना", "करना"),
    ("लेने", "करने"),
    ("लेता", "करता"),
    ("लेती", "करती"),
    ("लेते", "करते"),
    ("लेकर", "करके"),
    ("लेंगे", "करेंगे"),
    ("लेंगी", "करेंगी"),
    ("लेगा", "करेगा"),
    ("लेगी", "करेगी"),
    ("लिया", "किया"),
    ("लिए", "किए"),
    ("लिये", "किये"),
    ("ली", "की"),
    ("लूँ", "करूँ"),
    ("लूं", "करूं"),
    ("लें", "करें"),
    ("लो", "करो"),
    ("ले", "कर"),
]

# Formal nouns that pair with "लेना" but whose colloquial form uses "करना".
_NOUN_VERB_MORPH: List[Tuple[str, str]] = [
    ("सदस्यता", "सब्सक्राइब"),
]


def _apply_verb_morph(text: str) -> str:
    """Handle noun+verb pairs where the colloquial noun needs a different verb."""
    result = text
    for formal_noun, colloquial_noun in _NOUN_VERB_MORPH:
        if formal_noun not in result:
            continue
        for old_verb, new_verb in _LENA_TO_KARNA:
            old_phrase = formal_noun + " " + old_verb
            new_phrase = colloquial_noun + " " + new_verb
            result = result.replace(old_phrase, new_phrase)
        result = result.replace(formal_noun, colloquial_noun)
    return result


# ── 3b. Simple noun-swap rules (verb stays the same or noun is standalone) ─
# Sorted longest-first at module load to prevent partial-match clobbering.
_FORMAL_TO_COLLOQUIAL_HI: List[Tuple[str, str]] = sorted(
    [
        # ── Social-media (same-verb swaps: X करना → Y करना) ──
        ("साझा", "शेयर"),
        ("टिप्पणी", "कमेंट"),
        ("अनुसरण", "फॉलो"),
        ("अधिसूचना", "नोटिफिकेशन"),
        ("सूचनाएं", "नोटिफिकेशन्स"),
        # ── Literary connectors nobody uses in speech ──
        ("तत्पश्चात", "उसके बाद"),
        ("तदुपरांत", "उसके बाद"),
        ("सर्वप्रथम", "सबसे पहले"),
        ("अतएव", "इसलिए"),
        ("किन्तु", "लेकिन"),
        ("किंतु", "लेकिन"),
        ("परन्तु", "लेकिन"),
        ("परंतु", "लेकिन"),
        ("एवं", "और"),
        ("तथा", "और"),
        ("यदि", "अगर"),
        ("अतः", "इसलिए"),
        ("कृपया", "प्लीज़"),
        # ── Formal vocabulary → everyday Hindi ──
        ("आवश्यकता है", "ज़रूरत है"),
        ("आवश्यक है", "ज़रूरी है"),
        ("आवश्यकता", "ज़रूरत"),
        ("आवश्यक", "ज़रूरी"),
        ("प्रारम्भ करें", "शुरू करें"),
        ("प्रारंभ करें", "शुरू करें"),
        ("प्रारम्भ", "शुरुआत"),
        ("प्रारंभ", "शुरुआत"),
        ("आरम्भ करें", "शुरू करें"),
        ("आरंभ करें", "शुरू करें"),
        ("आरम्भ", "शुरुआत"),
        ("आरंभ", "शुरुआत"),
        ("समाप्त करें", "खत्म करें"),
        ("समाप्त", "खत्म"),
        ("अत्यंत", "बहुत"),
        ("अत्यधिक", "बहुत ज़्यादा"),
        ("उत्तम", "बढ़िया"),
        ("श्रेष्ठ", "बेस्ट"),
        ("सुनिश्चित करें", "पक्का करें"),
        ("सुनिश्चित कर", "पक्का कर"),
        ("सुनिश्चित", "पक्का"),
        ("विशेष", "खास"),
        ("सहायता", "मदद"),
        ("कठिन", "मुश्किल"),
        ("सरल", "आसान"),
        ("उद्देश्य", "मकसद"),
        ("विधि", "तरीका"),
        ("व्यायाम", "एक्सरसाइज़"),
        ("अभ्यास करें", "प्रैक्टिस करें"),
        ("अभ्यास कर", "प्रैक्टिस कर"),
        ("अभ्यास", "प्रैक्टिस"),
        ("नौसिखिया", "बिगिनर"),
        ("पर्याप्त", "काफी"),
        ("संपूर्ण", "पूरा"),
        ("वास्तव में", "सच में"),
        ("दूरभाष", "फोन"),
        ("संगणक", "कंप्यूटर"),
        ("अंतर्जाल", "इंटरनेट"),
        ("प्रशिक्षण", "ट्रेनिंग"),
        ("शीर्षासन", "हेडस्टैंड"),
        ("हस्तस्थिति", "हैंडस्टैंड"),
    ],
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def colloquialize_hi(text: str) -> str:
    """Replace formal/literary Hindi with everyday equivalents.

    Two passes:
      1. Verb-morphing rules (noun+verb pairs where the verb must change)
      2. Simple noun-swap rules (verb stays the same or noun is standalone)
    """
    if not text:
        return text
    result = _apply_verb_morph(text)
    for formal, colloquial in _FORMAL_TO_COLLOQUIAL_HI:
        result = result.replace(formal, colloquial)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def postprocess(
    source_text: str,
    translated_text: str,
    protected_terms: List[str],
    target_lang: str,
) -> str:
    """Full post-processing: restore placeholders → language-specific fixes."""
    result = restore_terms(translated_text, protected_terms)

    if target_lang in ("hi", "hin_Deva"):
        result = colloquialize_hi(result)

    result = re.sub(r"\s+", " ", result).strip()
    return result
