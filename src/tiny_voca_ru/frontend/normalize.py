"""Written Russian normalization: numbers, dates, times, units, acronyms, and Latin text."""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from typing import TypeVar

from . import numbers
from .clean import clean, tidy, unify
from .numbers import CASES, compound_prefix
from .tables import (
    ABBREVIATIONS,
    ACRONYM_WORDS,
    BARE,
    CARDINAL_ENDINGS,
    CENTS,
    DATE_PREPOSITIONS,
    DOTTED,
    FORMS,
    LATIN_LETTERS,
    LATIN_RULES,
    LATIN_WORDS,
    LETTER,
    LETTERS,
    LOCATIVE_PREPOSITIONS,
    MONTH_ANY,
    MONTHS,
    NAME_ENDINGS,
    NOT_NOUNS,
    NOUNS,
    NUMBER_PATTERN,
    NUMERAL_STRESS,
    OBLIQUE_PLURAL,
    ORDINAL_ENDINGS,
    PLACES,
    PREPOSITIONS,
    PROVIDERS,
    PUNCTUATION_CLASS,
    QUEENS,
    SCALES,
    SENTENCE_FINAL,
    SINGULAR_INSTRUMENTAL,
    STRESSED,
    SYMBOLS,
    TODAY,
    UNITS,
    UPPER,
    VERB_ENDINGS,
    VOWELS,
    YEAR_PREPOSITIONS,
    plain_word,
)

_F = TypeVar("_F", bound=Callable[..., str])


def _marked(generate: _F) -> _F:
    @functools.wraps(generate)
    def marked(*args: object, **kwargs: object) -> str:
        return " ".join(
            NUMERAL_STRESS.get(word, word) for word in generate(*args, **kwargs).split(" ")
        )

    return marked  # type: ignore[return-value]


cardinal = _marked(numbers.cardinal)
ordinal = _marked(numbers.ordinal)
decimal = _marked(numbers.decimal)
fraction = _marked(numbers.fraction)
digits = numbers.digits

_NEXT = re.compile(rf"\s+([{LETTER}+]+)")


def _sentence_end(text: str, end: int) -> bool:
    return bool(re.match(rf"\s*$|\s+[{UPPER}A-Z\"(—]", text[end:]))


def _preposition(text: str, start: int) -> str:
    head = text[:start]
    while True:
        m = re.search(r"(?:^|(?<=[\s(\"]))([^\W\d_][\w+-]*)\s+$", head)
        if not m:
            return ""
        word = plain_word(m[1])
        if word not in ("минус", "плюс"):
            return word if word in PREPOSITIONS else ""
        head = head[: m.start()]


def _cases(preposition: str, table: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    if table and preposition in table:
        return table[preposition]
    return PREPOSITIONS.get(preposition, ("nom",))


def _next_word(text: str, end: int) -> str:
    m = _NEXT.match(text, end)
    return plain_word(m[1]) if m else ""


def _restress(word: str, stressed: str) -> str:
    if "+" in word or plain_word(stressed) != word.lower():
        return word
    letters = iter(word)
    return "".join(c if c == "+" else next(letters) for c in stressed)


def noun_form(lemma: str, count: int | str, case: str, stressed: bool = False) -> str:
    """Return declined form of ``lemma`` after ``count`` (int or decimal string) in ``case``."""
    gender, animate, singular, plural, few = (STRESSED if stressed else NOUNS)[lemma]
    i = CASES.index(case)
    if isinstance(count, str):
        return plural[1] if gender == "p" else singular[1]
    one = count % 10 == 1 and count % 100 != 11
    if count and count % 1000 == 0:
        return plural[1]  # тысяча рублей
    if gender == "p":
        return plural[i] if one or case not in ("nom", "acc") else plural[1]
    if case not in ("nom", "acc"):
        return singular[i] if one else plural[i]
    if one:
        return singular[i]
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return plural[1] if case == "acc" and animate and count < 5 else few
    return plural[1]


def _say(count: int | str, lemma: str | None, case: str) -> str:
    if isinstance(count, str):
        whole, _, part = count.partition(",")
        return decimal(whole, part, case)
    gender, animate = NOUNS[lemma][:2] if lemma else ("m", False)
    if gender == "p":
        return cardinal(count, case, "m", animate, plural=True)
    return cardinal(count, case, gender, animate)


def _count_case(count: int | str, lemma: str, word: str, cases: Sequence[str]) -> str | None:
    for case in (*cases, *CASES):
        if noun_form(lemma, count, case) == word:
            return case
    return None


def _value(text: str) -> int | str:
    return text if "," in text else int(text)


def _amount(value: int | str, lemma: str | None, case: str, scale: str | None = None) -> str:
    if scale:
        words = f"{_say(value, scale, case)} {noun_form(scale, value, case, True)}"
        return words + (f" {STRESSED[lemma][3][1]}" if lemma else "")
    if lemma is None:
        return _say(value, None, case)
    return f"{_say(value, lemma, case)} {noun_form(lemma, value, case, True)}"


def _number_words(value: str) -> str:
    if (len(value) > 1 and value.startswith("0")) or len(value) > 15:
        return digits(value)
    return cardinal(int(value))


# --- normalization passes -------------------------------------------------------------------------


def phones(text: str) -> str:
    """Verbalize phone numbers like ``+7 (999) 123-45-67``."""

    def say(m: re.Match[str]) -> str:
        groups = re.findall(r"\d+", m[0])
        if len(groups) == 1:  # 89991234567
            whole = groups[0]
            groups = [whole[:-10], whole[-10:-7], whole[-7:-4], whole[-4:-2], whole[-2:]]
        words = ["плюс"] if m[0].startswith("+") else []
        for group in groups:
            zeros = len(group) - len(group.lstrip("0"))
            words.extend(["ноль"] * zeros)
            if group.lstrip("0"):
                words.append(cardinal(int(group)))
        return " ".join(words)

    separated = r"[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}"
    text = re.sub(rf"(?<![\d+])(?:\+\d{{1,3}}|8){separated}(?!\d)", say, text)
    text = re.sub(r"(?<![\d+])(?:\+7|8)\d{10}(?!\d)", say, text)
    return re.sub(r"(?<![\d.,:-])\d{3}-\d{2}-\d{2}(?![\d-])", say, text)


def groups(text: str) -> str:
    """Collapse thousands separators, normalize decimals to comma, and expand dotted versions."""
    text = re.sub(
        r"(?<![\d,.])\d{1,3}(?: \d{3})+(?![\d,]|\.\d)", lambda m: m[0].replace(" ", ""), text
    )
    text = re.sub(
        r"(?<![\d,.])\d{1,3}(?:\.\d{3}){2,}(?![\d,]|\.\d)", lambda m: m[0].replace(".", ""), text
    )
    text = re.sub(
        r"(?<![\d.])\d+(?:\.\d+){2,}(?!\d|\.\d)",
        lambda m: " точка ".join(_number_words(p) for p in m[0].split(".")),
        text,
    )
    return re.sub(r"(?<![\d,.])(\d+)\.(\d+)(?!\d|\.\d)", r"\1,\2", text)


def dates(text: str) -> str:
    """Verbalize numeric, ISO, and month-name Russian dates."""

    def day_case(m: re.Match[str], ending: str | None = None) -> str:
        preposition = _preposition(m.string, m.start())
        before = re.search(rf"([{LETTER}]+)\s+$", m.string[: m.start()])
        if not preposition and before and before[1].lower() in TODAY:
            return "nom"  # сегодня семнадцатое ноября
        cases = _cases(preposition, DATE_PREPOSITIONS) if preposition else ("gen",)
        readings = [
            case for case, gender in ORDINAL_ENDINGS.get(ending or "", ()) if gender in ("m", "n")
        ]
        return next(
            (case for case in cases if case in readings), readings[0] if readings else cases[0]
        )

    def year_words(m: re.Match[str]) -> str:
        if not m["y"]:
            return ""
        dot = "." if m["g"] == "г." and _sentence_end(m.string, m.end()) else ""
        return f" {ordinal(int(m['y']), 'gen')} года{dot}"

    def numeric(m: re.Match[str]) -> str:
        day, month = int(m["d"]), int(m["m"])
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m[0]
        return f"{ordinal(day, day_case(m), 'n')} {MONTHS[month - 1]}{year_words(m)}"

    def written(m: re.Match[str]) -> str:
        case = day_case(m, m["s"])
        days = " — ".join(ordinal(int(d), case, "n") for d in (m["d"], m["d2"]) if d)
        return f"{days} {m['month']}{year_words(m)}"

    year = r"(?P<y>\d{4})(?!\d)(?:\s*(?P<g>г\.|года\b))?"
    ending = r"(?:-(?P<s>го|е|ое|ого|му|ому|м|ом))?"
    text = re.sub(rf"(?<![\d.])(?P<d>\d{{1,2}})\.(?P<m>\d{{1,2}})\.{year}", numeric, text)
    text = re.sub(
        r"(?<![\d-])(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?![\d-])(?P<g>)", numeric, text
    )  # ISO
    text = re.sub(
        rf"\b(?P<p>[сС]о?|[оО]т)\s+(?P<d>\d{{1,2}})(?=\s+(?:по|до)\s+\d{{1,2}}(?:-[а-я]+)?\s+"
        rf"(?:{'|'.join(MONTHS)})\b)",  # с 1 по 5 мая -> с первого
        lambda m: (
            f"{m['p']} {ordinal(int(m['d']), 'gen', 'n')}" if 1 <= int(m["d"]) <= 31 else m[0]
        ),
        text,
    )
    text = re.sub(
        rf"(?<![\d,.])(?P<d>\d{{1,2}}){ending}(?:\s?[-–—]\s?(?P<d2>\d{{1,2}})(?:-[а-я]+)?)?"
        rf"\s+(?P<month>{'|'.join(MONTHS)})\b(?:\s+{year})?",
        written,
        text,
    )
    return re.sub(rf"(?P<month>\b{MONTH_ANY})\s+{year}", lambda m: m["month"] + year_words(m), text)


def times(text: str) -> str:
    """Verbalize 24-hour times like ``14:30``, ``в 9:05``, ``с 9:00 до 18:00``."""

    def time(m: re.Match[str]) -> str:
        hour, minute, second = int(m["h"]), m["m"], m["s"]
        if hour > 24 or int(minute) > 59 or (second and int(second) > 59):
            return m[0]
        case = _cases(_preposition(m.string, m.start()))[0]
        if minute == "00" and not second:
            return f"{cardinal(hour, case)} {noun_form('час', hour, case, True)}"
        words = [cardinal(hour, case)]
        for part in (minute, second):
            if part:  # в 21:01 -> в двадцать один ноль одна
                minutes = cardinal(int(part), "nom" if case == "acc" else case, "f")
                if part[0] == "0":
                    words.append("ноль")
                words.append(minutes)
        return " ".join(words)

    return re.sub(r"(?<![\d:.,])(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?![\d:])", time, text)


def _after_noun(text: str, start: int) -> bool:
    m = re.search(rf"(?<![\w+-])([{LETTER}+]{{3,}})[ \t]+$", text[:start])
    return (
        bool(m)
        and plain_word(m[1]) not in NOT_NOUNS
        and not plain_word(m[1]).endswith(VERB_ENDINGS)
    )


def years(text: str) -> str:
    """Verbalize years and year ranges (``в 1999 году``, ``2026 г.``, ``1941–1945 гг.``)."""
    forms = "|".join(
        sorted(set(NOUNS["год"][2] + NOUNS["год"][3]) - {"лет"}, key=len, reverse=True)
    )
    text = re.sub(
        rf"\b(?P<p>[сС]о?|[оО]т)\s+(?P<y>\d{{3,4}})(?=\s+(?:по|до)\s+\d{{3,4}}\s*(?:гг?\.|(?:{forms})\b))",
        lambda m: f"{m['p']} {ordinal(int(m['y']), 'gen')}",
        text,
    )

    def year(m: re.Match[str]) -> str:
        noun, suffix = m["noun"], m["suf"]
        preposition = _preposition(m.string, m.start())
        plural_noun = noun == "гг." or noun in NOUNS["год"][3]
        stressed_forms = STRESSED["год"][3] if plural_noun else STRESSED["год"][2]
        plain_forms = NOUNS["год"][3] if plural_noun else NOUNS["год"][2]
        if noun in ("г.", "гг."):
            case = (
                _cases(preposition, YEAR_PREPOSITIONS)[0]
                if preposition
                else "gen"
                if _after_noun(m.string, m.start())
                else "nom"
            )
            word = stressed_forms[CASES.index(case)].replace("л+ет", "год+ов")  # до 1995 гг.
            if _sentence_end(m.string, m.end()):
                word += "."
        else:
            options = [c for c in CASES if plain_forms[CASES.index(c)] == noun]
            case = next(
                (c for c in _cases(preposition, YEAR_PREPOSITIONS) if c in options), options[0]
            )
            word = _restress(noun, stressed_forms[CASES.index(case)])
        plural = bool(suffix) and plural_noun
        year_list = [m["y"]] + ([m["y2"]] if m["y2"] else [])
        return " — ".join(ordinal(int(y), case, "m", plural) for y in year_list) + " " + word

    return re.sub(
        rf"(?<![\d,.])(?P<y>\d{{3,4}})(?:\s?[-–—]\s?(?P<y2>\d{{3,4}}))?(?P<suf>-[а-яё]+)?"
        rf"\s*(?P<noun>гг\.|г\.|(?:{forms})\b)",
        year,
        text,
    )


def _to_roman(n: int) -> str:
    out = ""
    for value, symbol in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while n >= value:
            out += symbol
            n -= value
    return out


def romans(text: str) -> str:
    """Verbalize Roman numerals for centuries, monarchs, and section titles."""
    value_of = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    def roman(token: str) -> int | None:
        token = token.replace("Х", "X")
        total = previous = 0
        for char in reversed(token):
            value = value_of[char]
            total = total - value if value < previous else total + value
            previous = max(previous, value)
        return total if 0 < total < 4000 and _to_roman(total) == token else None

    century_forms = "|".join(sorted(set(NOUNS["век"][2] + NOUNS["век"][3]), key=len, reverse=True))

    def century(m: re.Match[str]) -> str:
        noun = m["noun"]
        preposition = _preposition(m.string, m.start())
        if noun in ("в.", "вв."):
            case = _cases(preposition, YEAR_PREPOSITIONS)[0]
            stressed_century = STRESSED["век"][3] if noun == "вв." else STRESSED["век"][2]
            word = stressed_century[CASES.index(case)]
            if _sentence_end(m.string, m.end()):
                word += "."
        else:
            is_sg = noun in NOUNS["век"][2]
            plain_century = NOUNS["век"][2] if is_sg else NOUNS["век"][3]
            stressed_century = STRESSED["век"][2] if is_sg else STRESSED["век"][3]
            options = [c for c in CASES if plain_century[CASES.index(c)] == noun]
            case = next(
                (c for c in _cases(preposition, YEAR_PREPOSITIONS) if c in options), options[0]
            )
            word = _restress(noun, stressed_century[CASES.index(case)])
        raw_values = [roman(n) for n in (m["n"], m["n2"]) if n]
        if any(v is None for v in raw_values):
            return m[0]
        return " — ".join(ordinal(v, case) for v in raw_values if v is not None) + " " + word

    def name(m: re.Match[str]) -> str:
        word, value = m["name"], roman(m["n"])
        plain = plain_word(word)
        if value is None or not (plain in FORMS or word[0].isupper()):
            return m[0]
        if plain in FORMS:  # глава IV, том II
            lemma = FORMS[plain]
            gender, _, singular = NOUNS[lemma][:3]
            case = next((c for c in CASES if singular[CASES.index(c)] == plain), "nom")
            word = _restress(word, STRESSED[lemma][2][CASES.index(case)])
        else:  # Пётр I, при Екатерине II
            gender = "f" if plain.startswith(QUEENS) else "m"
            endings = NAME_ENDINGS[gender]
            by_ending = next(
                (cases for ending, cases in endings if plain.endswith(ending)), ("nom",)
            )
            by_preposition = _cases(_preposition(m.string, m.start()))
            case = next((c for c in by_preposition if c in by_ending), by_ending[0])
        return f"{word} {ordinal(value, case, gender)}"

    numeral = r"[IVXLCХ]+"
    text = re.sub(
        rf"(?<![\w-])(?P<n>{numeral})(?:\s?[-–—]\s?(?P<n2>{numeral}))?\s+"
        rf"(?P<noun>вв\.|в\.|(?:{century_forms})\b)",
        century,
        text,
    )
    return re.sub(rf"(?P<name>\b[{UPPER}]?[а-яё]{{2,}})[ \t]+(?P<n>[IVX]+)\b(?![\w-])", name, text)


def places(text: str) -> str:
    """Expand place/document abbreviations (``г. Москва``, ``ул. Ленина``, ``д. 5``)."""
    for pattern, lemma, follows in PLACES:
        after = {"name": rf"(?=\s*[{UPPER}])", "number": r"(?=\s*\d)", "any": r"(?=\s*\S)"}[follows]

        def place(m: re.Match[str], lemma: str = lemma) -> str:
            if re.search(r"\d\s*$", m.string[: m.start()]):  # 5 ч. 30 мин. is a unit, not часть
                return m[0]
            case = _cases(_preposition(m.string, m.start()), LOCATIVE_PREPOSITIONS)[0]
            return STRESSED[lemma][2][CASES.index(case)]

        text = re.sub(rf"(?<![\w.-]){pattern}{after}", place, text)
    return text


def abbreviations(text: str) -> str:
    """Expand common Russian abbreviations (``т.е.``, ``проф.``, ``№``)."""
    for pattern, long in ABBREVIATIONS:

        def expand(m: re.Match[str], long: str = long) -> str:
            if re.search(r"\d\s?$", m.string[: m.start()]):  # 75 см. is a unit, not смотри
                return m[0]
            final = long in SENTENCE_FINAL and _sentence_end(m.string, m.end())
            return long + ("." if final else "")

        capitalised = pattern[0].upper() + pattern[1:]  # См. рис. 3, Т.е.
        text = re.sub(rf"(?<![\w.])(?:{pattern}|{capitalised})", expand, text)
    return text


def fractions(text: str) -> str:
    """Expand fractions (``3/4`` -> ``три четвёртых``; ``дом 5/2`` -> ``дом 5 дробь 2``)."""

    def say(m: re.Match[str]) -> str:
        numerator, denominator = int(m[1]), int(m[2])
        preposition = _preposition(m.string, m.start())
        before = re.search(rf"([{LETTER}]+)\s+$", m.string[: m.start()])
        if 0 < numerator < denominator <= 100 and not (before and plain_word(before[1]) in FORMS):
            return fraction(numerator, denominator, _cases(preposition)[0])
        return f"{m[1]} дробь {m[2]}"

    return re.sub(r"(?<![\d/,])(\d+)/(\d+)(?![\d/])", say, text)


def _counting_case(preposition: str, value: int | str) -> str:
    if preposition == "по":  # по одному рублю (dat) vs по пять рублей (acc)
        return "dat" if isinstance(value, int) and value % 10 == 1 and value % 100 != 11 else "acc"
    return _cases(preposition)[0]


def _ordinal_agreeing(
    n: int, lemma: str, word: str, cases: Sequence[str]
) -> tuple[str, str | None]:
    gender, animate, singular, plural = NOUNS[lemma][:4]
    stressed_sg, stressed_pl = STRESSED[lemma][2], STRESSED[lemma][3]
    for stressed_forms, forms, is_plural in (
        (stressed_sg, singular, gender == "p"),
        (stressed_pl, plural, True),
    ):
        for case in (*cases, *CASES):
            if forms[CASES.index(case)] == word:
                return (
                    ordinal(n, case, "m" if is_plural else gender, is_plural, animate),
                    stressed_forms[CASES.index(case)],
                )
    return ordinal(n, cases[0]), None


def _cardinal_ending(n: int, ending: str, preposition: str, word: str) -> bool:
    if ending == "х":
        return n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14)
    if ending == "м":
        return preposition in ("к", "ко")
    if ending == "ю":
        return preposition in ("с", "со", "между", "над", "перед") or word.endswith(
            ("ами", "ями", "ьми")
        )
    return True


def _suffixed(
    n: int,
    ending: str,
    lemma: str | None,
    word: str,
    preposition: str,
) -> tuple[str, str | None]:
    cases = _cases(preposition)
    if ending not in ORDINAL_ENDINGS and ending not in CARDINAL_ENDINGS:
        prefix = compound_prefix(n) if 0 < n < 1000 else numbers.cardinal(n, "gen") + " "
        overlap = max(k for k in range(len(ending) + 1) if prefix.endswith(ending[:k]))
        return prefix + ending[overlap:], None  # 5-летний, 2-хкомнатная
    gender, animate = NOUNS[lemma][:2] if lemma else ("m", False)
    gender = "m" if gender == "p" else gender
    if ending in CARDINAL_ENDINGS and _cardinal_ending(n, ending, preposition, word):
        noun_cases = [c for c in CASES if lemma and noun_form(lemma, n, c) == word]
        for case in [*noun_cases, *cases, "gen", "dat", "prep", "ins"]:
            if case not in ("nom", "acc") and numbers.cardinal(n, case, gender).endswith(ending):
                stressed = noun_form(lemma, n, case, True) if lemma and case in noun_cases else None
                return cardinal(n, case, gender, animate), stressed
    readings = ORDINAL_ENDINGS.get(ending)
    if not readings:
        return cardinal(n), None
    if lemma:
        noun_gender, animate, singular, plural = NOUNS[lemma][:4]
        stressed_sg, stressed_pl = STRESSED[lemma][2], STRESSED[lemma][3]
        for case, reading in readings:
            for stressed_forms, is_plural, forms in (
                (stressed_sg, False, singular),
                (stressed_pl, True, plural),
            ):
                own = "pl" if is_plural else noun_gender
                fits = reading.rstrip("*") == own or (
                    reading == "m" and own == "n" and case not in ("nom", "acc")
                )
                if (
                    fits
                    and forms[CASES.index(case)] == word
                    and (animate or not reading.endswith("*"))
                ):
                    return (
                        ordinal(n, case, "m" if is_plural else noun_gender, is_plural, animate),
                        stressed_forms[CASES.index(case)],
                    )
    filtered = [r for r in readings if not r[1].endswith("*")]
    if ending in ("е", "ое", "ье") and not lemma and n % 10 == 0 and n >= 20:
        filtered = [r for r in filtered if r[1] == "pl"]  # 90-е, 2000-е
    case, reading = next((r for c in cases for r in filtered if r[0] == c), filtered[0])
    return ordinal(n, case, "m" if reading == "pl" else reading, reading == "pl"), None


def _unknown_noun(word: str, count: int | str, cases: Sequence[str]) -> tuple[str | None, str]:
    if len(word) < 4 or not isinstance(count, int):
        return None, "m"
    if count % 10 == 1 and count % 100 != 11:
        if "ins" in cases:
            for ending, gender in SINGULAR_INSTRUMENTAL:
                if word.endswith(ending):
                    return "ins", gender
        return None, "n" if word.endswith("о") else "m"
    for ending, case in OBLIQUE_PLURAL:
        if word.endswith(ending):
            return case, "m"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        if word.endswith(("ов", "ев", "ёв")):
            return "gen", "m"
        if word.endswith(("ы", "и")):
            return None, "f"
    return None, "m"


def _bare_number(
    text: str,
    value: str,
    values: Sequence[int | str],
    preposition: str,
    end: int,
) -> tuple[str, str | None]:
    word = _next_word(text, end)
    lemma = FORMS.get(word)
    last = values[-1]
    cases = _cases(preposition)
    if lemma is None:
        if (
            preposition in ("в", "во")
            and len(values) == 1
            and value.isdigit()
            and isinstance(last, int)
            and 1800 <= last <= 2099
            and not re.match(r"\s*\d", text[end:])
        ):
            return ordinal(last, "prep"), None  # в 1999 (году)
        if (
            len(values) == 1
            and value.isdigit()
            and (len(value) > 15 or (len(value) > 1 and value[0] == "0"))
        ):
            return digits(value), None
        case, gender = _unknown_noun(word, last, cases)
        return (
            " — ".join(
                cardinal(v, case, gender)
                if case and isinstance(v, int)
                else _say(v, None, _counting_case(preposition, v))
                if gender == "m" or not isinstance(v, int)
                else cardinal(v, _counting_case(preposition, v), gender)
                for v in values
            ),
            None,
        )
    if lemma in ("век", "число") and isinstance(last, int):
        said = [_ordinal_agreeing(int(v), lemma, word, cases) for v in values if isinstance(v, int)]
        return " — ".join(words for words, _ in said), said[-1][1]
    case = _count_case(last, lemma, word, (_counting_case(preposition, last), *cases))
    if case is None:
        if isinstance(last, int) and len(values) == 1:  # на 5 этаже -> на пятом этаже
            return _ordinal_agreeing(last, lemma, word, cases)
        return " — ".join(_say(v, lemma, cases[0]) for v in values), None
    return " — ".join(_say(v, lemma, case) for v in values), noun_form(lemma, last, case, True)


def _with_noun(text: str, end: int, said: str, noun: str | None) -> tuple[str, int]:
    m = _NEXT.match(text, end) if noun else None
    if m is None or noun is None:
        return said, end
    return said + text[end : m.start(1)] + _restress(m[1], noun), m.end()


def quantities(text: str) -> str:
    """Verbalize numbers with units, currencies, scales, ordinal endings, and agreeing nouns."""
    units = "|".join(re.escape(u) for u in sorted(set(UNITS) - BARE, key=len, reverse=True))
    pattern = re.compile(
        rf"(?<![\d,])(?P<n>{NUMBER_PATTERN})(?:\s?[-–—]\s?(?P<n2>{NUMBER_PATTERN}))?(?:-(?P<suf>[а-яё]+))?"
        rf"(?:\s?(?P<scale>{'|'.join(SCALES)})\.?(?![{LETTER}]))?"
        rf"(?:\s?(?P<unit>{units})(?P<dot>\.)?(?![{LETTER}A-Za-z])"
        rf"|\s(?P<bare>{'|'.join(BARE)})(?=\s*(?:[{PUNCTUATION_CLASS}]|$)))?"
    )

    def quantity(m: re.Match[str]) -> tuple[str, int]:
        src, end = m.string, m.end()
        unit = m["unit"] or m["bare"]
        scale = SCALES.get(m["scale"]) if m["scale"] else None
        ending = m["suf"]
        values: list[int | str] = [_value(m["n"])] + ([_value(m["n2"])] if m["n2"] else [])
        preposition = _preposition(src, m.start())
        if ending and not unit and not scale:
            if not all(isinstance(v, int) for v in values):
                return m[0], end
            word = _next_word(src, end)
            said = [
                _suffixed(v, ending, FORMS.get(word), word, preposition)
                for v in values
                if isinstance(v, int)
            ]
            return _with_noun(src, end, " — ".join(words for words, _ in said), said[-1][1])
        if not unit and not scale:
            return _with_noun(src, end, *_bare_number(src, m["n"], values, preposition, end))
        lemma, tail = UNITS[unit] if unit else (None, "")
        case = _counting_case(preposition, values[-1]) if not scale else _cases(preposition)[0]
        if preposition in ("в", "во") and (
            (lemma in ("километр", "метр") and re.match(r"\s+от\b", src[end:]))
            or (lemma == "процент" and re.match(r"\s+случа", src[end:]))
        ):
            case = "prep"  # в 5 км от города, в 15% случаев
        dot = "." if m["dot"] and (unit not in DOTTED or _sentence_end(src, end)) else ""
        cents = CENTS.get(lemma) if lemma else None
        if cents and not scale and len(values) == 1 and isinstance(values[0], str):
            whole, part = map(int, values[0].split(","))
            if len(values[0].split(",")[1]) == 2:  # 12,50 руб.
                words = _amount(part, cents, case)
                main = _amount(whole, lemma, case) + (f" {tail}" if tail else "")
                return (f"{main} {words}" if whole else words) + dot, end
        words = [
            _say(v, scale or lemma, case) + (f" {noun_form(scale, v, case, True)}" if scale else "")
            for v in values[:-1]
        ]
        words.append(_amount(values[-1], lemma, case, scale))
        return " — ".join(words) + (f" {tail}" if tail else "") + dot, end

    out: list[str] = []
    done = 0
    for m in pattern.finditer(text):
        said, end = quantity(m)
        out.extend((text[done : m.start()], said))
        done = end
    out.append(text[done:])
    return "".join(out)


def signs(text: str) -> str:
    """Expand numeric signs and leading currency symbols (``-5``, ``+5``, ``~5``, ``$5``)."""
    text = re.sub(r"(?:(?<=[\s(\"])|^)[-–](?=\d)", "минус ", text)
    text = re.sub(r"(?:(?<=[\s(\"])|^)\+(?=\d)", "плюс ", text)
    text = re.sub(r"[~≈]\s?(?=\d)", "около ", text)
    text = re.sub(r"(?<=\d)\s?[×xхX*]\s?(?=\d)", " на ", text)
    return re.sub(
        rf"(?P<c>[$€£₽¥₴])\s?(?P<n>{NUMBER_PATTERN})(?:\s?(?P<s>тыс\.|млн|млрд|трлн))?",
        lambda m: f"{m['n']}{' ' + m['s'] if m['s'] else ''} {m['c']}",
        text,
    )


def acronyms(text: str) -> str:
    """Spell out Cyrillic acronyms (``ФСБ``) while reading pronounceable ones as words (``МИД``)."""

    def word_like(token: str) -> bool:
        vowels = sum(c in VOWELS for c in token)
        if len(token) >= 5:
            return vowels / len(token) >= 0.25
        return len(token) == 4 and vowels > 0 and not re.search(rf"[^{VOWELS}]{{3}}", token)

    def acronym(m: re.Match[str]) -> str:
        token, suffix = m["a"], m["s"] or ""
        key = token.lower()
        before = re.search(rf"(?<![\w+])([{UPPER}]+)\s+$", m.string[: m.start()])
        after = re.match(rf"\s+([{UPPER}]+)(?![\w+])", m.string[m.end() :])
        shouting = any(
            n and word_like(n[1]) and n[1].lower() not in ACRONYM_WORDS for n in (before, after)
        )
        if key in ACRONYM_WORDS or word_like(token) or (shouting and not suffix):  # ЭТО ОЧЕНЬ ВАЖНО
            return key + suffix
        return " ".join(LETTERS[c] for c in key) + (" " + suffix if suffix else "")

    return re.sub(rf"(?<![\w+])(?P<a>[{UPPER}]{{2,}})(?P<s>[а-яё]+)?(?![\w+])", acronym, text)


def latin(text: str) -> str:
    """Transliterate Latin words and acronyms into Cyrillic."""

    def word(token: str) -> str:
        key = token.lower()
        if key in LATIN_WORDS:
            return LATIN_WORDS[key]
        if token.isupper() and len(token) <= 5:
            return " ".join(LATIN_LETTERS[c] for c in key)
        return transliterate(key)

    def words(m: re.Match[str]) -> str:
        if m[0].lower() in LATIN_WORDS:
            return LATIN_WORDS[m[0].lower()]
        return " ".join(word(part.replace("'", "")) for part in m[0].split("-"))

    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z])", " точка ", text)  # example.com
    return re.sub(r"(?<![\w])[A-Za-z]+(?:[-'][A-Za-z]+)*(?![\w])", words, text)


def transliterate(word: str) -> str:
    """Rule-based Latin-to-Cyrillic transliteration for lowercase words."""
    if len(word) > 4 and word.endswith("e") and word[-2] not in "aeiouy":
        word = word[:-1]
    out: list[str] = []
    i = 0
    while i < len(word):
        if word[i] == "c" and word[i + 1 : i + 2] in ("e", "i", "y"):
            out.append("с")
            i += 1
            continue
        if word[i] == "y" and i == 0 and word[1:2] in tuple("aeiou"):
            out.append("й")
            i += 1
            continue
        if word[i] == "e" and i == 0:
            out.append("э")
            i += 1
            continue
        for latin_part, cyrillic in LATIN_RULES:
            if word.startswith(latin_part, i):
                out.append(cyrillic)
                i += len(latin_part)
                break
        else:
            i += 1
    return "".join(out)


def powers(text: str) -> str:
    """Expand ``10^6`` to ``10 в степени 6``."""
    return re.sub(r"(?<![\d,])(\d+)\s?\^\s?(\d+)(?!\d)", r"\1 в степени \2", text)


def symbols(text: str) -> str:
    """Replace remaining spoken symbols with words while keeping ``+`` stress marks."""
    text = re.sub(rf"\+(?=[{VOWELS}])", "\0", text)
    for symbol, words in SYMBOLS.items():
        text = text.replace(symbol, words)
    return text.replace("\0", "+")


def standalone(text: str) -> str:
    """Expand standalone unit abbreviations without a preceding number."""
    for short, lemma in (
        ("тыс.", "тысяча"),
        ("млн", "миллион"),
        ("млрд", "миллиард"),
        ("трлн", "триллион"),
        ("руб.", "рубль"),
        ("коп.", "копейка"),
        ("км", "километр"),
        ("кг", "килограмм"),
    ):

        def expand(m: re.Match[str], short: str = short, long: str = STRESSED[lemma][3][1]) -> str:
            return long + ("." if short.endswith(".") and _sentence_end(m.string, m.end()) else "")

        text = re.sub(rf"(?<![\w.]){re.escape(short)}(?![\w])", expand, text)
    return text


def rules(text: str) -> str:
    """Run all Russian text normalization passes."""
    text = unify(re.sub(r"(?<=\d)(?=[½¼¾⅓⅔])", " и ", text))
    text = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)  # MP3, 4K
    text = re.sub(
        rf"(?<=[{LETTER}A-Za-z])-(?=\d)|(?<=\d)-(?=[A-Za-z])", " ", text
    )  # Ту-154, COVID-19
    text = re.sub(r"[ \t]+", " ", text)
    for step in (
        phones,
        dates,
        times,
        groups,
        years,
        romans,
        abbreviations,
        places,
        signs,
        fractions,
        powers,
        quantities,
        standalone,
        acronyms,
        latin,
        symbols,
    ):
        text = step(text)
    return tidy(text)


def normalize(text: str, provider: str | None = None) -> str:
    """Normalize written Russian into spoken Cyrillic words and punctuation."""
    mode = provider or "rules"
    if mode not in PROVIDERS:
        raise ValueError(f"unknown normaliser {mode!r}; expected one of {PROVIDERS}")
    return clean(text) if mode == "none" else rules(text)
