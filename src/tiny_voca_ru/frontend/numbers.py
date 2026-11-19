"""Russian cardinal, ordinal, decimal, and fractional numeral verbalization."""

from __future__ import annotations

from collections.abc import Callable, Sequence

CASES: tuple[str, ...] = ("nom", "gen", "dat", "acc", "ins", "prep")

ONE: dict[str, tuple[str, ...]] = {
    "m": ("один", "одного", "одному", "один", "одним", "одном"),
    "f": ("одна", "одной", "одной", "одну", "одной", "одной"),
    "n": ("одно", "одного", "одному", "одно", "одним", "одном"),
    "pl": ("одни", "одних", "одним", "одни", "одними", "одних"),
}
FEW: dict[int, tuple[str, ...]] = {
    2: ("два", "двух", "двум", "два", "двумя", "двух"),
    3: ("три", "трёх", "трём", "три", "тремя", "трёх"),
    4: ("четыре", "четырёх", "четырём", "четыре", "четырьмя", "четырёх"),
}
COLLECTIVE: dict[int, tuple[str, ...]] = {
    2: ("двое", "двоих", "двоим", "двое", "двоими", "двоих"),
    3: ("трое", "троих", "троим", "трое", "троими", "троих"),
    4: ("четверо", "четверых", "четверым", "четверо", "четверыми", "четверых"),
}
UNITS: tuple[str, ...] = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
TENS: tuple[str, ...] = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
HUNDREDS: tuple[str, ...] = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)
ZERO: tuple[str, ...] = ("ноль", "ноля", "нолю", "ноль", "нолём", "ноле")

# Scale nouns: (singular cases, plural cases); тысяча is feminine, others masculine.
SCALES: tuple[tuple[int, str], ...] = (
    (10**12, "триллион"),
    (10**9, "миллиард"),
    (10**6, "миллион"),
    (10**3, "тысяча"),
)
SCALE_FORMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "тысяча": (
        ("тысяча", "тысячи", "тысяче", "тысячу", "тысячей", "тысяче"),
        ("тысячи", "тысяч", "тысячам", "тысячи", "тысячами", "тысячах"),
    ),
    **{
        word: (
            (word, word + "а", word + "у", word, word + "ом", word + "е"),
            (word + "ы", word + "ов", word + "ам", word + "ы", word + "ами", word + "ах"),
        )
        for word in ("миллион", "миллиард", "триллион")
    },
}

ORDINAL_UNITS: tuple[str, ...] = (
    "нулев",
    "перв",
    "втор",
    "трет",
    "четвёрт",
    "пят",
    "шест",
    "седьм",
    "восьм",
    "девят",
    "десят",
    "одиннадцат",
    "двенадцат",
    "тринадцат",
    "четырнадцат",
    "пятнадцат",
    "шестнадцат",
    "семнадцат",
    "восемнадцат",
    "девятнадцат",
)
ORDINAL_TENS: tuple[str, ...] = (
    "",
    "",
    "двадцат",
    "тридцат",
    "сороков",
    "пятидесят",
    "шестидесят",
    "семидесят",
    "восьмидесят",
    "девяност",
)
STRESSED: frozenset[str] = frozenset(("нулев", "втор", "шест", "седьм", "восьм", "сороков"))
HARD_ENDINGS: dict[str, tuple[str, ...]] = {
    "m": ("ый", "ого", "ому", "ый", "ым", "ом"),
    "f": ("ая", "ой", "ой", "ую", "ой", "ой"),
    "n": ("ое", "ого", "ому", "ое", "ым", "ом"),
    "pl": ("ые", "ых", "ым", "ые", "ыми", "ых"),
}
SOFT_ENDINGS: dict[str, tuple[str, ...]] = {
    "m": ("ий", "ьего", "ьему", "ий", "ьим", "ьем"),
    "f": ("ья", "ьей", "ьей", "ью", "ьей", "ьей"),
    "n": ("ье", "ьего", "ьему", "ье", "ьим", "ьем"),
    "pl": ("ьи", "ьих", "ьим", "ьи", "ьими", "ьих"),
}


def _index(case: str) -> int:
    return CASES.index(case)


def plural_form(n: int, forms: Sequence[str]) -> str:
    """Select Russian plural form from ``(1, 2-4, 5+)`` (e.g. рубль/рубля/рублей)."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return forms[1]
    return forms[2]


def _i_declension(word: str, case: str) -> str:
    """Decline пять..двадцать/тридцать (``-ью`` in instrumental, ``-и`` in gen/dat/prep)."""
    if case in ("nom", "acc"):
        return word
    if word == "восемь":
        return "восемью" if case == "ins" else "восьми"
    return word[:-1] + ("ью" if case == "ins" else "и")


def _unit(n: int, case: str, gender: str = "m") -> str:
    if n in FEW:
        if gender == "f" and case in ("nom", "acc"):
            return "две" if n == 2 else FEW[n][0]
        return FEW[n][_index(case)]
    return _i_declension(UNITS[n], case)


def _tens(t: int, case: str) -> str:
    if case in ("nom", "acc"):
        return TENS[t]
    if t == 4:
        return "сорока"
    if t == 9:
        return "девяноста"
    if t in (2, 3):
        return _i_declension(TENS[t], case)
    first = _i_declension(UNITS[t], case)  # пятьдесят -> пятидесяти, пятьюдесятью
    return first + ("десятью" if case == "ins" else "десяти")


def _hundreds(h: int, case: str) -> str:
    if case in ("nom", "acc"):
        return HUNDREDS[h]
    if h == 1:
        return "ста"
    tail = {"gen": "сот", "dat": "стам", "ins": "стами", "prep": "стах"}[case]
    return _unit(h, case) + tail


def _group(
    n: int,
    case: str,
    gender: str,
    last: Callable[[int, bool], str] | None = None,
) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_hundreds(hundreds, case))
    if rest >= 20:
        words.append(_tens(rest // 10, case))
        rest %= 10
    if rest:
        if last and rest <= 4:
            words.append(last(rest, bool(words)))
        elif rest == 1:
            words.append(ONE[gender][_index(case)])
        else:
            words.append(_unit(rest, case, gender))
    return words


def _split(n: int) -> list[tuple[int, str | None]]:
    parts: list[tuple[int, str | None]] = []
    for size, noun in SCALES:
        if n >= size:
            count, n = divmod(n, size)
            parts.append((count, noun))
    if n:
        parts.append((n, None))
    return parts


def _scale_noun(count: int, noun: str, case: str) -> str:
    singular, plural = SCALE_FORMS[noun]
    i = _index(case)
    if case in ("nom", "acc"):
        return plural_form(count, (singular[i], singular[1], plural[1]))
    return singular[i] if count % 10 == 1 and count % 100 != 11 else plural[i]


def _scale_words(count: int, noun: str, case: str, leading: bool) -> list[str]:
    gender = "f" if noun == "тысяча" else "m"
    words = [] if count == 1 and leading else _group(count, case, gender)
    return [*words, _scale_noun(count, noun, case)]


def cardinal(
    n: int,
    case: str = "nom",
    gender: str = "m",
    animate: bool = False,
    plural: bool = False,
) -> str:
    """Verbalize cardinal integer ``n`` in the requested Russian case and gender."""
    if n < 0:
        return "минус " + cardinal(-n, case, gender, animate, plural)
    if n == 0:
        return ZERO[_index(case)]
    i = _index(case)
    as_genitive = case == "acc" and animate

    def last(ones: int, compound: bool) -> str:
        if ones == 1:
            form = ONE["pl" if plural else gender]
            return form[1] if as_genitive and (plural or gender == "m") else form[i]
        if plural:
            return COLLECTIVE[ones][1 if as_genitive else i]
        if as_genitive and gender != "n" and not compound:
            return FEW[ones][1]
        return _unit(ones, "nom" if case == "acc" else case, gender)

    words: list[str] = []
    parts = _split(n)
    for count, noun in parts:
        if noun:
            words.extend(_scale_words(count, noun, case, leading=not words))
        else:
            words.extend(
                _group(
                    count,
                    case,
                    gender,
                    lambda ones, compound: last(ones, compound or len(parts) > 1),
                )
            )
    return " ".join(words)


def _adjective(stem: str, case: str, gender: str, plural: bool, animate: bool) -> str:
    key = "pl" if plural else gender
    last = stem.rpartition(" ")[2]
    endings = (SOFT_ENDINGS if last == "трет" else HARD_ENDINGS)[key]
    i = _index(case)
    if case == "acc" and key in ("m", "pl"):
        i = 1 if animate else 0
    ending = endings[i]
    if ending == "ый" and last in STRESSED:
        ending = "ой"
    return stem + ending


def compound_prefix(n: int) -> str:
    """Return combining form for ``0 < n < 1000`` (e.g. двухтысячный, пятилетний)."""
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append("сто" if hundreds == 1 else _hundreds(hundreds, "gen"))
    if rest >= 20:
        words.append("девяносто" if rest // 10 == 9 else _tens(rest // 10, "gen"))
        rest %= 10
    if rest == 1:
        words.append("одно")
    elif rest:
        words.append(_unit(rest, "gen"))
    return "".join(words)


def _ordinal_group(n: int) -> tuple[list[str], str]:
    hundreds, rest = divmod(n, 100)
    words: list[str] = []
    if not rest:
        return words, ("сот" if hundreds == 1 else _hundreds(hundreds, "gen"))
    if hundreds:
        words.append(HUNDREDS[hundreds])
    if rest < 20:
        return words, ORDINAL_UNITS[rest]
    tens, ones = divmod(rest, 10)
    if not ones:
        return words, ORDINAL_TENS[tens]
    words.append(TENS[tens])
    return words, ORDINAL_UNITS[ones]


def ordinal(
    n: int,
    case: str = "nom",
    gender: str = "m",
    plural: bool = False,
    animate: bool = False,
) -> str:
    """Verbalize ordinal ``n`` (e.g. ``1999`` -> ``тысяча девятьсот девяносто девятый``)."""
    if n < 0:
        raise ValueError("ordinal of a negative number")
    if n == 0:
        return _adjective("нулев", case, gender, plural, animate)
    parts = _split(n)
    words: list[str] = []
    for count, noun in parts[:-1]:
        assert noun is not None
        words.extend(_scale_words(count, noun, "nom", leading=not words))
    count, noun = parts[-1]
    if noun:
        prefix = "" if count == 1 else compound_prefix(count)
        stem = prefix + (noun[:-1] if noun == "тысяча" else noun) + "н"
        words.append(_adjective(stem, case, gender, plural, animate))
    else:
        before, stem = _ordinal_group(count)
        words.extend((*before, _adjective(stem, case, gender, plural, animate)))
    return " ".join(words)


def decimal(whole: str, fraction_digits: str, case: str = "nom") -> str:
    """Verbalize a decimal (e.g. ``("3", "05")`` -> ``три целых пять сотых``)."""
    if not fraction_digits:
        return cardinal(int(whole), case)
    words = [cardinal(int(whole), case, "f"), _share("цел", int(whole), case)]
    words.extend(
        (
            cardinal(int(fraction_digits), case, "f"),
            _share(ordinal(10 ** len(fraction_digits))[:-2], int(fraction_digits), case),
        )
    )
    return " ".join(words)


def fraction(numerator: int, denominator: int, case: str = "nom") -> str:
    """Verbalize a fraction (e.g. ``(3, 4)`` -> ``три четвёртых``)."""
    return f"{cardinal(numerator, case, 'f')} {_share(ordinal(denominator)[:-2], numerator, case)}"


def _share(stem: str, count: int, case: str) -> str:
    single = count % 10 == 1 and count % 100 != 11
    if single:
        return _adjective(stem, case, "f", False, False)
    return _adjective(stem, "gen" if case in ("nom", "acc") else case, "f", True, True)


def digits(text: str) -> str:
    """Read a digit string digit by digit (``"007"`` -> ``"ноль ноль семь"``)."""
    return " ".join(UNITS[int(d)] for d in text)
