"""Tokenization, phonemization, and Russian sentence splitting."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .clean import clean as _clean
from .g2p import phrase_phones
from .normalize import normalize as _normalize
from .stress import LiteStress, StressProvider, get_provider
from .symbols import BOS, EOS, PHONE_IDS, PHONE_SET, PUNCTUATION, SP, SYMBOL_TO_ID, UNK

Piece = tuple[bool, str | list[list[tuple[str, int]]]]

_PUNCTUATION = "".join(re.escape(p) for p in PUNCTUATION)
_TOKEN = re.compile(rf"[{_PUNCTUATION}]|[^\s{_PUNCTUATION}]+")

# Sentence-splitting regexes kept in sync with web/public/worker.js.
_SPACE = re.compile(r"[\s\ufeff]+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]»“]))\s+(?=["'(\[«„]?[A-ZА-ЯЁ0-9])""")
_NOT_AN_END = re.compile(
    r"(?:\b(?:тыс|млн|млрд|руб|коп|долл|ул|пр|просп|пер|наб|пл|обл|кв|корп|стр|рис|табл|см|ср|напр|"
    r"проф|акад|доц|им|св|гр|тел|др|т\.е|т\.д|т\.п|т\.к|т\.н)|\b[A-ZА-ЯЁ])\.$",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_TEXT_END = re.compile(r"[\s\ufeff]+$")
_CLOSING = "\"'”’)]»"


class NothingToPronounce(ValueError):
    """Raised when text contains no pronounceable words, numbers, or letters."""


@dataclass(frozen=True)
class PhoneSequence:
    """Encoded token sequence ready for ``TinyVoca`` or ONNX inference."""

    symbols: tuple[str, ...]
    ids: tuple[int, ...]
    stress: tuple[int, ...]
    unknown: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.ids)


def stress_provider(stress: str | StressProvider | None = None) -> StressProvider:
    """Resolve a stress provider name (``"lite"``, ``"full"``, ``"none"``) or instance."""
    if stress is not None and not isinstance(stress, str):
        return stress
    return get_provider(stress or "lite")


def accentuate(
    text: str,
    stress: str | StressProvider | None = None,
    normalize: str | None = None,
) -> str:
    """Normalize ``text`` and insert ``+`` stress marks (restoring ``ё``)."""
    return stress_provider(stress).accentuate(_normalize(text, normalize))


def unknown_words(
    text: str,
    stress: str | StressProvider | None = None,
    normalize: str | None = None,
) -> list[str]:
    """Return words that ``stress`` cannot mark without an external model (used in browser)."""
    provider = stress_provider(stress)
    if isinstance(provider, LiteStress):
        return provider.unknown_words(_normalize(text, normalize))
    return []


def pieces(
    text: str,
    stress: str | StressProvider | None = None,
    normalize: str | None = None,
    stressed: bool = False,
) -> list[Piece]:
    """Split ``text`` into ``(is_punctuation, value)`` phrase pieces."""
    if stressed:
        if stress is not None or normalize is not None:
            raise ValueError("stressed text takes no stress provider or normaliser")
        cleaned = _clean(text)
    else:
        cleaned = accentuate(text, stress, normalize)
    out: list[Piece] = []
    words: list[str] = []

    def flush() -> None:
        if words:
            phrase = [w for w in phrase_phones(words) if w]
            if phrase:
                out.append((False, phrase))
            words.clear()

    for token in _TOKEN.findall(cleaned):
        if token in PUNCTUATION:
            flush()
            out.append((True, token))
        else:
            words.append(token)
    flush()
    return out


def transcribe(
    text: str,
    stress: str | StressProvider | None = None,
    normalize: str | None = None,
    stressed: bool = False,
) -> str:
    """Return human-readable IPA with ``ˈ`` before stressed vowels."""
    parts: list[str] = []
    for is_punct, value in pieces(text, stress, normalize, stressed):
        if is_punct:
            assert isinstance(value, str)
            parts.append(value)
            continue
        assert not isinstance(value, str)
        for word in value:
            parts.append("".join(("ˈ" if level else "") + phone for phone, level in word))
    return " ".join(parts)


def tokenize(phrase_pieces: Sequence[Piece]) -> PhoneSequence:
    """Convert ``pieces()`` output into a ``PhoneSequence`` with ``<bos>``/``<sp>``/``<eos>``."""
    symbols: list[str] = [BOS]
    stress_levels: list[int] = [0]
    unknown: list[str] = []
    for is_punct, value in phrase_pieces:
        if is_punct:
            assert isinstance(value, str)
            if symbols[-1] != value:  # collapse repeated punctuation like "!!"
                symbols.append(value)
                stress_levels.append(0)
            continue
        assert not isinstance(value, str)
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress_levels.append(0)
            for phone, level in word:
                if phone not in PHONE_SET:
                    unknown.append(phone)
                    phone = UNK
                symbols.append(phone)
                stress_levels.append(level)
    symbols.append(EOS)
    stress_levels.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return PhoneSequence(
        symbols=tuple(symbols),
        ids=tuple(SYMBOL_TO_ID[s] for s in symbols),
        stress=tuple(stress_levels),
        unknown=tuple(unknown),
    )


def phonemize(
    text: str,
    *,
    stress: str | StressProvider | None = None,
    normalize: str | None = None,
    stressed: bool = False,
) -> PhoneSequence:
    """Normalize, accentuate, and encode Russian text into a ``PhoneSequence``."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    return tokenize(pieces(text, stress=stress, normalize=normalize, stressed=stressed))


encode_text = phonemize


def is_phone(token_ids: Iterable[int]) -> list[bool]:
    """Return True for each token ID representing a speech phone (including ``<unk>``)."""
    return [int(i) in PHONE_IDS for i in token_ids]


def split_sentences(text: str) -> list[str]:
    """Split Russian text into sentences without breaking after abbreviations or initials."""
    parts: list[str] = []
    current = ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(char.isalnum() for char in part)]


def ensure_terminal_punctuation(text: str) -> str:
    """Append a period when ``text`` ends without terminal punctuation."""
    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def split_clause(text: str) -> list[str]:
    """Split an overly long sentence at clause punctuation, or in half by words."""
    clauses = _CLAUSE_END.split(text)
    if len(clauses) > 1:
        return clauses
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]
