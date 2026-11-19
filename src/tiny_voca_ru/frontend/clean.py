"""Unicode cleanup and stress-mark canonicalization (U+0301 -> ``+``)."""

from __future__ import annotations

import re
import unicodedata

from .symbols import PUNCTUATION

LETTER = "а-яёА-ЯЁ"
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
QUOTES = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "`": "'",
        "´": "'",
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "‒": "–",
        "―": "—",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u2007": " ",
        "℃": "°C",
        "¼": "1/4",
        "½": "1/2",
        "¾": "3/4",
        "⅓": "1/3",
        "⅔": "2/3",
    }
)
_PUNCTUATION = "".join(re.escape(p) for p in PUNCTUATION)
SPOKEN_SIGNS = "%$€₽£¥₴@&=№§°×~≈^"
_UNSPOKEN = re.compile(rf"\S*(?:[0-9A-Za-z{re.escape(SPOKEN_SIGNS)}]|\+(?![{VOWELS}]))\S*")


class NotNormalized(ValueError):
    """Raised when text contains digits, Latin letters, or signs needing normalization."""


def unify(text: str) -> str:
    """Normalize Unicode, quotes, dashes, spaces, ellipsis, and combining acute accents."""
    text = unicodedata.normalize("NFC", text).translate(QUOTES)
    text = re.sub(rf"([{VOWELS}])[\u0301\u0300]", r"+\1", text)  # за́мок -> з+амок
    text = re.sub(
        r"[\u00c0-\u024f]", lambda m: unicodedata.normalize("NFD", m[0])[0], text
    )  # café -> cafe
    text = re.sub(r"[\u0300-\u036f]", "", text)
    return re.sub(r"\.{3,}", "…", text)


def unspoken(text: str) -> list[str]:
    """Return whitespace-delimited tokens that require text normalization."""
    return _UNSPOKEN.findall(text)


def tidy(text: str) -> str:
    """Normalize dashes, hyphens, stray symbols, and whitespace."""
    text = re.sub(r"-{2,}|–", "—", text)
    text = re.sub(rf"(?<![{LETTER}+])-|-(?![{LETTER}+])", " — ", text)  # non-word hyphen -> dash
    text = re.sub(rf"[^\s{LETTER}+\-{_PUNCTUATION}]", " ", text)
    text = re.sub(r"(?<=\S)—", " —", text)
    text = re.sub(r"—(?=\S)", "— ", text)
    text = re.sub(r"(?<=[?!])\.\.(?!\.)", "", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    return re.sub(r"\s+", " ", text).strip()


def clean(text: str) -> str:
    """Clean already-normalized text; raise ``NotNormalized`` on digits/Latin/signs."""
    text = unify(text)
    found = unspoken(text)
    if found:
        raise NotNormalized(f"needs a normaliser: {' '.join(found)}")
    return tidy(text)
