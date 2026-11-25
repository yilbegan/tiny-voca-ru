"""Russian stressed orthography to IPA phoneme sequences."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .lexicon import (
    ADJECTIVE_ENDINGS,
    ADJECTIVE_STEMS,
    ENCLITICS,
    KEEP_G,
    NONREDUCING,
    NOUN_ENDINGS,
    NOUN_STEMS,
    PROCLITICS,
    WORDS,
)
from .symbols import OBSTRUENTS, SOFT, VOICED_PARTNER, VOICELESS, VOICELESS_PARTNER, VOWEL_SET

STRESS = "+"
VOWEL_LETTERS: frozenset[str] = frozenset("аеёиоуыэюя")
VOWEL_PHONES: dict[str, str] = {
    "а": "a",
    "я": "a",
    "о": "o",
    "ё": "o",
    "у": "u",
    "ю": "u",
    "э": "e",
    "е": "e",
    "ы": "ɨ",
    "и": "i",
}
CONSONANT_PHONES: dict[str, str] = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "д": "d",
    "ж": "ʐ",
    "з": "z",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "tɕ",
    "ш": "ʂ",
    "щ": "ɕ",
}
SOFTENING: frozenset[str] = frozenset("еёиюяь")
IOTATED: frozenset[str] = frozenset("яюеё")
HUSHING: frozenset[str] = frozenset("жшц")  # always hard: и -> ɨ, unstressed е -> ɨ
SOFT_PHONES: frozenset[str] = frozenset(SOFT.values()) | {"tɕ", "ɕ", "j"}

# Ordered consonant-cluster respellings preserving vowel letter count.
CLUSTERS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"ть?ся$", "ца"),  # учится, учиться
        (r"вств", "ств"),  # чувство, здравствуй
        (r"стск", "ск"),  # туристский
        (r"н[тд](?=ск|ц)", "н"),  # гигантский, голландцы
        (r"нтг", "нг"),  # рентген
        (r"ндш", "нш"),  # ландшафт
        (r"стн", "сн"),  # честный
        (r"здн", "зн"),  # поздно
        (r"стл", "сл"),  # счастливый
        (r"лнц", "нц"),  # солнце
        (r"рдц", "рц"),  # сердце
        (r"[тд]ств", "цтв"),  # детство
        (r"[тд]ск(?=(ий|ая|ое|ие|ого|ому|ом|ой|ую|их|им|ими|а|у|е)?$)", "цк"),  # детский, городской
    )
)
_NOT_LETTER = re.compile(r"[^а-яё+]")
_STRAY_STRESS = re.compile(r"\+(?![аеёиоуыэюя])")
_HARD_E = re.compile(r"[бвгджзклмнпрстфхц]\+?э")


@dataclass
class Unit:
    """One written word or hyphenated component."""

    text: str  # with stress marks
    key: str  # without stress marks
    hyphen: bool  # follows a hyphen: кто-то, скажи-ка
    stressed: bool


@dataclass
class Segment:
    """Intermediate phonetic segment during phonological rule application."""

    phone: str
    letter: str = ""  # source letter for consonants (в is transparent to voicing)
    context: str = ""  # vowels: "soft", "hard", "hush" (after ж ш ц), "" (no preceding C)
    stressed: bool = False
    full: bool = False  # retains quality when unstressed
    last: bool = False  # final vowel of its written word
    initial: bool = False  # first segment of its written word
    final_letter: bool = False  # final segment of its written word
    loan: bool = False  # word contains hard е (respelled э after consonant)
    unit: int = 0

def respell(word: str) -> str:
    """Apply lexicon and cluster respellings, preserving stress by vowel index."""
    plain = word.replace(STRESS, "")
    spelled = WORDS.get(plain) or _stem(plain) or plain
    for pattern, replacement in CLUSTERS:
        spelled = pattern.sub(replacement, spelled)
    if spelled.endswith(("ого", "его")) and plain not in KEEP_G:
        spelled = spelled[:-2] + "во"  # нового, его: genitive г -> в
    return _restress(word, spelled) if spelled != plain else word


def _stem(plain: str) -> str | None:
    for cut in range(len(plain), 0, -1):
        stem, ending = plain[:cut], plain[cut:]
        if stem in NOUN_STEMS and ending in NOUN_ENDINGS:
            return NOUN_STEMS[stem] + ending
        if stem in ADJECTIVE_STEMS and ending in ADJECTIVE_ENDINGS:
            return ADJECTIVE_STEMS[stem] + ending
    return None


def _restress(source: str, target: str) -> str:
    marks: set[int] = set()
    count = 0
    mark = False
    for char in source:
        if char == STRESS:
            mark = True
        elif char in VOWEL_LETTERS:
            if mark:
                marks.add(count)
            count, mark = count + 1, False
    out: list[str] = []
    count = 0
    for char in target:
        if char in VOWEL_LETTERS:
            if count in marks:
                out.append(STRESS)
            count += 1
        out.append(char)
    return "".join(out)


def spell(word: str) -> list[Segment]:
    """Convert letters (with ``+`` stress marks) of one word into raw segments."""
    letters: list[str] = []
    stressed: set[int] = set()
    for char in word:
        if char == STRESS:
            stressed.add(len(letters))
        else:
            letters.append(char)
    if not stressed:
        stressed = {i for i, char in enumerate(letters) if char == "ё"}
    out: list[Segment] = []
    for i, char in enumerate(letters):
        before = letters[i - 1] if i else ""
        after = letters[i + 1] if i + 1 < len(letters) else ""
        if char in CONSONANT_PHONES:
            phone = CONSONANT_PHONES[char]
            if phone in SOFT and after in SOFTENING:
                phone = SOFT[phone]
            out.append(Segment(phone, char))
        elif char in VOWEL_LETTERS:
            quality = VOWEL_PHONES[char]
            if before in CONSONANT_PHONES:
                if before in HUSHING:
                    context = "hard" if char == "э" else "hush"  # э only in respelled loanwords
                    if char == "и":
                        quality = "ɨ"
                else:
                    context = "soft" if out[-1].phone in SOFT_PHONES else "hard"
            elif char in IOTATED or (before in ("ь", "ъ") and char in ("и", "о")):
                out.append(Segment("j"))
                context = "soft"
            else:
                context = ""
            out.append(
                Segment(quality, char, context, i in stressed, char == "ё" and i not in stressed)
            )
    vowels = [s for s in out if s.phone in VOWEL_SET]
    if vowels:
        vowels[-1].last = True
    if out:
        out[0].initial = True
        out[-1].final_letter = True
    return out


def _units(word: str) -> Iterator[Unit]:
    for index, part in enumerate(word.lower().split("-")):
        text = _STRAY_STRESS.sub("", _NOT_LETTER.sub("", part))
        key = text.replace(STRESS, "")
        if key:
            yield Unit(text, key, index > 0, STRESS in text or "ё" in key)


def _groups(units: Sequence[Unit]) -> list[list[Unit]]:
    """Group written units into phonetic words (each stressed host with its clitics)."""
    groups: list[list[Unit]] = []
    pending: list[Unit] = []
    for unit in units:
        if unit.stressed:
            groups.append([*pending, unit])
            pending = []
        elif (
            groups
            and not pending
            and (
                unit.hyphen
                or unit.key in ENCLITICS
                or (groups[-1][-1].stressed and groups[-1][-1].key in PROCLITICS)
            )
        ):
            groups[-1].append(unit)
        else:
            pending.append(unit)
    if pending:
        if groups:
            groups[-1].extend(pending)
        else:
            groups.append(pending)
    return groups


def _segments(units: Sequence[Unit]) -> list[Segment]:
    segments: list[Segment] = []
    for number, unit in enumerate(units):
        text = respell(unit.text)
        spelled = spell(text)
        full = unit.key in NONREDUCING and not unit.hyphen
        loan = _HARD_E.search(text) is not None
        for segment in spelled:
            segment.unit, segment.loan = number, loan
            segment.full = segment.full or (full and segment.phone == "o")
        first = spelled[0] if spelled else None
        if (
            first
            and first.phone in VOWEL_SET
            and segments
            and segments[-1].phone not in VOWEL_SET
            and segments[-1].phone not in SOFT_PHONES
        ):  # к игре -> kɨɡrʲe, в огороде
            first.context = "hard"
            if first.phone == "i":
                first.phone = "ɨ"
        segments.extend(spelled)
    return segments


def soften(segments: list[Segment]) -> None:
    """Regressive palatalization (с сестрой, женщина, есть, песня)."""
    for i in range(len(segments) - 2, -1, -1):
        segment, following = segments[i], segments[i + 1]
        if segment.phone not in SOFT:
            continue
        if following.phone == SOFT[segment.phone] or (
            segment.unit == following.unit
            and (
                (segment.phone == "n" and following.phone in ("tɕ", "ɕ"))
                or (
                    segment.phone in ("s", "z")
                    and following.phone in ("tʲ", "dʲ", "nʲ")
                    and not segment.initial
                    and not segment.loan
                )
            )
        ):
            segment.phone = SOFT[segment.phone]


def velar(segments: list[Segment]) -> None:
    """Soft/hard г before к or ч becomes х (легко, мягкий)."""
    for segment, following in zip(segments, segments[1:], strict=False):
        if (
            segment.phone in ("ɡ", "ɡʲ")
            and following.phone in ("k", "kʲ", "tɕ")
            and segment.unit == following.unit
        ):
            segment.phone = "x" if segment.phone == "ɡ" else "xʲ"


def voicing(segments: list[Segment]) -> None:
    """Regressive voicing assimilation and word-final devoicing (в is transparent)."""
    state: str | None = "voiceless"  # word-final devoicing
    for segment in reversed(segments):
        if segment.phone not in OBSTRUENTS:
            state = None
            continue
        if state == "voiceless":
            segment.phone = VOICELESS_PARTNER.get(segment.phone, segment.phone)
        elif state == "voiced":
            segment.phone = VOICED_PARTNER.get(segment.phone, segment.phone)
        if segment.letter != "в":
            state = "voiceless" if segment.phone in VOICELESS else "voiced"


def merge(segments: list[Segment]) -> list[Segment]:
    """Collapse geminates and assimilated sibilant/affricate clusters within a word.

    Across word/clitic boundaries (different ``unit``), preserve consonant length so
    single-consonant prepositions (``в футболке``, ``к которой``, ``с собой``) are not deleted,
    while still applying place assimilation on sibilant pairs (``с шумом`` -> ``ʂ ʂ``).
    """
    out: list[Segment] = []
    for segment in segments:
        previous = out[-1].phone if out else None
        phone = segment.phone
        if previous is None or previous in VOWEL_SET or phone in VOWEL_SET:
            out.append(segment)
        elif out[-1].unit == segment.unit:
            if phone == previous:
                continue
            if (
                (previous in ("s", "sʲ", "ʂ") and phone in ("tɕ", "ɕ"))
                or (previous in ("s", "sʲ") and phone == "ʂ")
                or (previous in ("z", "zʲ") and phone == "ʐ")
                or (previous in ("t", "tʲ", "d", "dʲ") and phone in ("ts", "tɕ"))
            ):
                out[-1].phone = "ɕ" if phone == "tɕ" and previous in ("s", "sʲ", "ʂ") else phone
            else:
                out.append(segment)
        else:
            if (
                (previous in ("s", "sʲ") and phone == "ʂ")
                or (previous in ("z", "zʲ") and phone == "ʐ")
                or (previous in ("s", "sʲ", "ʂ") and phone == "ɕ")
            ):
                out[-1].phone = phone
            elif previous in ("s", "sʲ", "ʂ") and phone == "tɕ":
                out[-1].phone = "ɕ"
            out.append(segment)
    return out


def reduce(segments: list[Segment]) -> None:
    """Apply Russian unstressed vowel reduction relative to the stressed vowel."""
    vowels = [i for i, s in enumerate(segments) if s.phone in VOWEL_SET]
    stress = next((i for i in vowels if segments[i].stressed), None)
    if stress is None:
        return  # phrase of clitics only
    for k, i in enumerate(vowels):
        segment = segments[i]
        if segment.stressed or segment.full:
            continue
        if i > stress:
            position = "post"
        elif k + 1 < len(vowels) and vowels[k + 1] == stress:
            position = "pre1"
        else:
            position = "pre"
        following = segments[i + 1].phone if i + 1 < len(segments) else ""
        before_final_je = (
            i + 2 < len(segments)
            and segments[i + 1].phone == "j"
            and segments[i + 2].final_letter
            and segments[i + 2].letter == "е"
            and segments[i + 2].unit == segment.unit
        )
        segment.phone = _reduced(segment, position, following in SOFT_PHONES, before_final_je)


def _reduced(
    segment: Segment, position: str, before_soft: bool, before_final_je: bool = False
) -> str:
    quality, context = segment.phone, segment.context
    if quality in ("u", "i", "ɨ"):
        return quality
    if (
        quality == "e"
        and position == "post"
        and (
            (segment.final_letter and context == "hush")
            or (before_final_je and context in ("soft", "hush"))
        )
    ):
        return "ə"
    if context == "soft":
        # ə in final syllable (тётя, вишням), i before soft consonant (память)
        return (
            "ə"
            if quality == "a" and position == "post" and segment.last and not before_soft
            else "i"
        )
    if quality == "e":  # initial э- and hard е in loanwords stay e (эпизод, интернет)
        return {"": "e", "hard": "e"}.get(context, "ɨ")
    if position == "pre1" or (position == "pre" and context == ""):
        return "ɐ"
    return "ə"

def phrase_phones(words: Sequence[str]) -> list[list[tuple[str, int]]]:
    """Convert stressed words of one phrase into phonetic words ``[(phone, stress)]``."""
    out: list[list[tuple[str, int]]] = []
    for group in _groups([unit for word in words for unit in _units(word)]):
        segments = _segments(group)
        velar(segments)
        voicing(segments)
        soften(segments)
        segments = merge(segments)
        reduce(segments)
        if segments:
            out.append([(s.phone, int(s.stressed)) for s in segments])
    return out
