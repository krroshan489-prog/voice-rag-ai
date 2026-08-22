"""
backend/app/utils/translation.py
----------------------------------
Lightweight query translation utility for Bug 2 fix:
Non-English queries are translated to English before retrieval,
since the MSMARCO-XI index only contains English passage embeddings.

Strategy: langdetect to identify non-English -> deep_translator GoogleTranslator
to convert to English. Falls back silently to original query on any error.

Acronym fix (Bug 2b): Google Translate garbles phonetic acronyms like
Tamil "mepaep" -> "MayBape" instead of "MEPAP". Post-processing normalises
suspicious short mixed-case tokens to UPPERCASE so they match indexed terms.
"""

import re
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Regex: short tokens (3-7 chars) that look like garbled acronym transliterations.
# They have mixed capitalisation that doesn't match normal English words,
# e.g. "MayBape", "MePap", "MeBape".
_GARBLED_ACRONYM = re.compile(r'\b([A-Z][a-z]+[A-Z][a-zA-Z]{1,4})\b')

# Phonetic correction map: Google Translate garbles phonetic transliterations
# of acronyms when they appear in non-Latin scripts. This maps known garbled
# forms back to the correct English acronym/term.
# Add entries here when you discover new garbled translations.
# Keys are lowercase; matching is case-insensitive.
_PHONETIC_CORRECTIONS = {
    # MEPAP (Medicaid Enrolled Provider Assistance Program)
    "maybape": "MEPAP",
    "mepap": "MEPAP",
    "mepab": "MEPAP",
    "mepap certification": "MEPAP certification",
    "maybape certification": "MEPAP certification",
    # Add more as discovered:
    # "garbled": "CORRECT",
}


def _apply_phonetic_corrections(text: str) -> str:
    """
    Replace known garbled transliterations with their correct English forms.
    Operates on the full translated string and on individual tokens.
    """
    # Full-string match first (for multi-word corrections)
    lower = text.lower()
    for garbled, correct in _PHONETIC_CORRECTIONS.items():
        if ' ' in garbled and garbled in lower:
            text = re.sub(re.escape(garbled), correct, text, flags=re.IGNORECASE)
            logger.info("[Translation] Phonetic correction (phrase): %r -> %r", garbled, correct)

    # Token-level match
    def _replace_token(m):
        token = m.group(0)
        correction = _PHONETIC_CORRECTIONS.get(token.lower())
        if correction:
            logger.info("[Translation] Phonetic correction (token): %r -> %r", token, correction)
            return correction
        return token

    return re.sub(r'\b\w+\b', _replace_token, text)


def _fix_garbled_acronyms(text: str) -> str:
    """
    Post-process translated text to uppercase short CamelCase tokens that are
    likely garbled phonetic transliterations of acronyms.

    Examples:
      "What is MayBape Certification?" -> "What is MEPAP Certification?"
      No — we can't guess MEPAP from MayBape phonetically.

    Better approach: uppercase the token so it at least doesn't look like a
    normal English word, which may improve embedding matching somewhat.
    But the real win is the acronym-injection step below.
    """
    def _upper(m):
        token = m.group(1)
        # Only uppercase if it's short (<=6 chars total) — longer CamelCase
        # is usually a real compound word, not an acronym.
        if len(token) <= 6:
            logger.info("[Translation] Uppercasing likely garbled acronym: %r -> %r", token, token.upper())
            return token.upper()
        return token
    return _GARBLED_ACRONYM.sub(_upper, text)


def _extract_latin_tokens(text: str) -> list:
    """Extract ASCII/Latin words from original query (e.g. if user typed 'MEPAP' in Latin)."""
    return re.findall(r'\b[A-Za-z]{2,}\b', text)


def _is_latin(text: str) -> bool:
    """Fast check: returns True if text is predominantly ASCII/Latin characters."""
    if not text:
        return True
    non_latin = sum(1 for c in text if ord(c) > 591)  # beyond Extended Latin
    return (non_latin / len(text)) < 0.15


@lru_cache(maxsize=256)
def translate_query_to_english(query: str) -> str:
    """
    Translates a non-English query to English for vector retrieval.
    Returns original query unchanged if already English or on any error.
    Results are LRU-cached so repeated identical queries skip the API call.

    Acronym injection: any ASCII/Latin tokens present in the original query
    (e.g. the user typed "MEPAP certification என்றால் என்ன?") are appended
    verbatim to the translation so the embedding search finds them even if
    the translation garbles or omits them.
    """
    if _is_latin(query):
        return query  # Already English/Latin — skip translation

    # Extract any Latin tokens the user typed into the non-English query
    latin_tokens = _extract_latin_tokens(query)

    try:
        from langdetect import detect, LangDetectException
        try:
            lang = detect(query)
        except LangDetectException:
            lang = "unknown"

        if lang == "en":
            return query

        logger.info("[Translation] Detected lang=%s — translating: %r", lang, query[:60])

        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(query)

        if not translated or not translated.strip():
            return query

        translated = translated.strip()

        # Step 1: Apply known phonetic corrections (e.g. MayBape -> MEPAP)
        translated = _apply_phonetic_corrections(translated)

        # Step 2: Uppercase any remaining short CamelCase garbled tokens
        translated = _fix_garbled_acronyms(translated)

        # Step 3: Inject any Latin tokens the user originally included verbatim
        trans_lower = translated.lower()
        missing = [t for t in latin_tokens if t.lower() not in trans_lower]
        if missing:
            translated = translated + " " + " ".join(missing)
            logger.info("[Translation] Injected Latin tokens: %s", missing)

        logger.info("[Translation] Final: %r -> %r", query[:50], translated[:80])
        return translated

    except Exception as exc:
        logger.warning("[Translation] Failed (%s) — using original query: %r", exc, query[:50])
        # Still append any Latin tokens even on translation failure
        if latin_tokens:
            return query + " " + " ".join(latin_tokens)
        return query
