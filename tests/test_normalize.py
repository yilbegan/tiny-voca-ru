"""Tests for Russian numeral verbalization and text normalization."""

from __future__ import annotations

import random
import re
import unittest
from typing import Any

from tiny_voca_ru.frontend.clean import NotNormalized
from tiny_voca_ru.frontend.normalize import normalize
from tiny_voca_ru.frontend.numbers import (
    CASES,
    cardinal,
    compound_prefix,
    decimal,
    digits,
    fraction,
    ordinal,
)
from tiny_voca_ru.frontend.tables import UNITS

try:
    from num2words import num2words as _num2words  # type: ignore[import-untyped]
except ImportError:
    _num2words: Any = None

N2W_CASE = dict(zip(CASES, "ngdaip", strict=True))
LEADING_ONE = re.compile(
    r"(?:^|(?<=целая )|(?<=целых ))"
    r"(?:один|одна|одно|одну|одного|одной|одному|одним|одном|одни|одних|одними) "
    r"(?=тысяч|миллион|миллиард|триллион)"
)


def _n2w(n: int | str, **kwargs: Any) -> str:
    mapped = {key: N2W_CASE[value] if key == "case" else value for key, value in kwargs.items()}
    return str(_num2words(n, lang="ru", **mapped))


def _groups(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    power = 0
    while n:
        n, group = divmod(n, 1000)
        if group:
            out.append((group, power))
        power += 1
    return out[::-1]


def reference_cardinal(n: int, case: str, gender: str, animate: bool, plural: bool) -> str:
    if n == 0:
        return _n2w(0, case=case)
    *higher, (last, power) = _groups(n)
    words = [_n2w(group * 1000**p, case=case, animate=False) for group, p in higher]
    ones = last % 10 if last % 100 not in (12, 13, 14) else 0
    if power:
        words.append(_n2w(last * 1000**power, case=case, animate=False))
    elif (
        case == "acc" and animate and not plural and gender != "n" and ones in (2, 3, 4) and n >= 20
    ):
        words.append(_n2w(last, case="acc", gender=gender, animate=False))
    elif case == "acc" and not animate and plural and ones in (2, 3, 4):
        words.append(_n2w(last, case="nom", plural=True))
    else:
        words.append(_n2w(last, case=case, gender=gender, animate=animate, plural=plural))
    return LEADING_ONE.sub("", " ".join(words))


def reference_ordinal(n: int, case: str, gender: str, animate: bool, plural: bool) -> str:
    if n == 0:
        return _n2w(0, to="ordinal", case=case, gender=gender, animate=animate, plural=plural)
    *higher, (last, power) = _groups(n)
    words = [_n2w(group * 1000**p, animate=False) for group, p in higher]
    words.append(
        _n2w(
            last * 1000**power,
            to="ordinal",
            case=case,
            gender=gender,
            animate=animate,
            plural=plural,
        )
    )
    return LEADING_ONE.sub("", " ".join(words))


@unittest.skipIf(_num2words is None, "num2words is not installed")
class NumbersAgainstNum2Words(unittest.TestCase):
    def test_random_cardinals_and_ordinals(self) -> None:
        rng = random.Random(2026)
        for _ in range(1500):
            n = rng.randrange(10 ** rng.choice((1, 2, 3, 4, 6, 7, 9, 10, 12, 13)))
            case, gender = rng.choice(CASES), rng.choice("mfn")
            animate, plural = rng.random() < 0.5, rng.random() < 0.3
            with self.subTest(n=n, case=case, gender=gender, animate=animate, plural=plural):
                self.assertEqual(
                    cardinal(n, case, gender, animate, plural),
                    reference_cardinal(n, case, gender, animate, plural),
                )
                self.assertEqual(
                    ordinal(n, case, gender, plural, animate),
                    reference_ordinal(n, case, gender, animate, plural),
                )

    def test_every_form_of_small_numbers(self) -> None:
        for n in (
            *range(0, 125),
            200,
            300,
            400,
            999,
            1000,
            1001,
            2000,
            2002,
            5000,
            21000,
            100000,
            10**6,
        ):
            for case in CASES:
                for gender in "mfn":
                    for animate in (False, True):
                        for plural in (False, True):
                            with self.subTest(
                                n=n, case=case, gender=gender, animate=animate, plural=plural
                            ):
                                self.assertEqual(
                                    cardinal(n, case, gender, animate, plural),
                                    reference_cardinal(n, case, gender, animate, plural),
                                )
                                self.assertEqual(
                                    ordinal(n, case, gender, plural, animate),
                                    reference_ordinal(n, case, gender, animate, plural),
                                )

    def test_num2words_differences(self) -> None:
        self.assertEqual(_n2w(2000, case="acc", animate=True), "двух тысячи")
        self.assertEqual(cardinal(2000, "acc", animate=True), "две тысячи")
        self.assertEqual(_n2w(12000005, to="ordinal"), "двенадцатый миллионов пятый")
        self.assertEqual(ordinal(12000005), "двенадцать миллионов пятый")
        self.assertEqual(_n2w(300005, to="ordinal", case="gen"), "трёхсотого тысяч пятого")
        self.assertEqual(ordinal(300005, "gen"), "триста тысяч пятого")
        self.assertEqual(_n2w(1525, plural=True), "одни тысяча пятьсот двадцать пять")
        self.assertEqual(cardinal(1525, plural=True), "тысяча пятьсот двадцать пять")
        self.assertEqual(_n2w(2000000, gender="f"), "две миллиона")
        self.assertEqual(cardinal(2000000, gender="f"), "два миллиона")
        self.assertEqual(_n2w(2, case="acc", plural=True, animate=False), "двоих")
        self.assertEqual(cardinal(2, "acc", plural=True), "двое")
        self.assertEqual(_n2w(22, case="acc", animate=True), "двадцать двух")
        self.assertEqual(cardinal(22, "acc", animate=True), "двадцать два")
        self.assertEqual(cardinal(2, "acc", animate=True), "двух")
        self.assertEqual(_n2w(1000), "одна тысяча")
        self.assertEqual(cardinal(1000), "тысяча")

    def test_random_decimals(self) -> None:
        rng = random.Random(17)
        for _ in range(300):
            whole = str(rng.randrange(10 ** rng.randint(1, 6)))
            fraction_digits = str(rng.randrange(1, 10**4))
            fraction_digits = fraction_digits.zfill(rng.randint(len(fraction_digits), 5))
            with self.subTest(value=f"{whole},{fraction_digits}"):
                expected = LEADING_ONE.sub("", _n2w(f"{whole}.{fraction_digits}"))
                self.assertEqual(decimal(whole, fraction_digits), expected)


class NumbersTests(unittest.TestCase):
    def test_cardinals(self) -> None:
        self.assertEqual(cardinal(1525), "тысяча пятьсот двадцать пять")
        self.assertEqual(cardinal(1525, "ins"), "тысячей пятьюстами двадцатью пятью")
        self.assertEqual(cardinal(21, "nom", "f"), "двадцать одна")
        self.assertEqual(cardinal(1, "acc", "f", animate=True), "одну")
        self.assertEqual(cardinal(1, "acc", animate=True), "одного")
        self.assertEqual(cardinal(1, plural=True), "одни")
        self.assertEqual(cardinal(-45, "gen"), "минус сорока пяти")
        self.assertEqual(cardinal(4, "gen"), "четырёх")
        self.assertEqual(cardinal(300, "prep"), "трёхстах")
        self.assertEqual(cardinal(10**12 + 1), "триллион один")

    def test_ordinals(self) -> None:
        self.assertEqual(ordinal(1999, "prep"), "тысяча девятьсот девяносто девятом")
        self.assertEqual(ordinal(2000), "двухтысячный")
        self.assertEqual(ordinal(21000), "двадцатиоднотысячный")
        self.assertEqual(ordinal(40000, "gen"), "сорокатысячного")
        self.assertEqual(ordinal(3, "nom", "f"), "третья")
        self.assertEqual(ordinal(23, "gen", "f"), "двадцать третьей")
        self.assertEqual(ordinal(4), "четвёртый")
        self.assertEqual(ordinal(2), "второй")
        self.assertEqual(ordinal(40, "ins"), "сороковым")
        self.assertEqual(ordinal(1, "acc", animate=True), "первого")
        self.assertEqual(ordinal(90, "nom", plural=True), "девяностые")
        with self.assertRaises(ValueError):
            ordinal(-1)

    def test_decimals_fractions_digits_prefixes(self) -> None:
        self.assertEqual(decimal("3", "5"), "три целых пять десятых")
        self.assertEqual(decimal("1", "1"), "одна целая одна десятая")
        self.assertEqual(decimal("3", "5", "gen"), "трёх целых пяти десятых")
        self.assertEqual(decimal("0", "05"), "ноль целых пять сотых")
        self.assertEqual(decimal("12", ""), "двенадцать")
        self.assertEqual(fraction(1, 2), "одна вторая")
        self.assertEqual(fraction(3, 4), "три четвёртых")
        self.assertEqual(fraction(2, 3, "ins"), "двумя третьими")
        self.assertEqual(digits("007"), "ноль ноль семь")
        self.assertEqual(
            [compound_prefix(n) for n in (1, 2, 3, 40, 90, 100, 200, 25)],
            ["одно", "двух", "трёх", "сорока", "девяносто", "сто", "двухсот", "двадцатипяти"],
        )


CASES_: list[tuple[str, str]] = [
    ("У меня 5 рублей.", "У меня пять рублей."),
    ("1 рубль", "один рубль"),
    ("2 рубля", "два рубля"),
    ("21 рубль", "двадцать один рубль"),
    ("11 рублей", "одиннадцать рублей"),
    ("по 5 рублей", "по пять рублей"),
    ("по 1 рублю", "по одному рублю"),
    ("с 5 рублями", "с пятью рублями"),
    ("1525 руб.", "тысяча пятьсот двадцать пять рублей."),
    ("12,50 руб.", "двенадцать рублей пятьдесят копеек."),
    ("0,99 €", "девяносто девять центов"),
    ("$100", "сто долларов"),
    ("Рост малыша уже 102 см.", "Рост малыша уже сто два сантиметра."),
    ("В этом году мы прочитали 32 книги.", "В этом году мы прочитали тридцать две книги."),
    ("Учитель доволен 22 студентами.", "Учитель доволен двадцатью двумя студентами."),
    ("Ошибка встречается в 15% случаев.", "Ошибка встречается в пятнадцати процентах случаев."),
    ("5 тыс. руб.", "пять тысяч рублей."),
    ("1,5 млн руб.", "одна целая пять десятых миллиона рублей."),
    ("Население 146 млн человек.", "Население сто сорок шесть миллионов человек."),
    ("1 000 000 рублей", "миллион рублей"),
    ("в 5 км от города", "в пяти километрах от города"),
    ("100 км/ч", "сто километров в час"),
    ("5 ч. 30 мин.", "пять часов тридцать минут."),
    ("5 ГБ", "пять гигабайт"),
    ("+5°C", "плюс пять градусов Цельсия"),
    ("-10 °C", "минус десять градусов Цельсия"),
    ("Площадь 50 м²", "Площадь пятьдесят квадратных метров"),
    ("2 суток", "двое суток"),
    ("увидел 2 человек", "увидел двух человек"),
    ("Сегодня 17.11.2026.", "Сегодня семнадцатое ноября две тысячи двадцать шестого года."),
    ("17.11.2026", "семнадцатого ноября две тысячи двадцать шестого года"),
    ("2026-11-17", "семнадцатого ноября две тысячи двадцать шестого года"),
    (
        "Он родился 5 мая 1990 г. в Москве.",
        "Он родился пятого мая тысяча девятьсот девяностого года в Москве.",
    ),
    ("до 17 ноября", "до семнадцатого ноября"),
    ("С 1 по 5 мая", "С первого по пятое мая"),
    ("в 1999 году", "в тысяча девятьсот девяносто девятом году"),
    ("в 1941–1945 гг.", "в тысяча девятьсот сорок первом — тысяча девятьсот сорок пятом годах."),
    ("в 1990-х годах", "в тысяча девятьсот девяностых годах"),
    ("14:30", "четырнадцать тридцать"),
    ("в 9:05", "в девять ноль пять"),
    ("с 9:00 до 18:00", "с девяти часов до восемнадцати часов"),
    ("в 21:01", "в двадцать один ноль одна"),
    (
        "+7 (999) 123-45-67",
        "плюс семь девятьсот девяносто девять сто двадцать три сорок пять шестьдесят семь",
    ),
    ("8 800 555-35-35", "восемь восемьсот пятьсот пятьдесят пять тридцать пять тридцать пять"),
    ("5-й класс", "пятый класс"),
    ("в 5-м классе", "в пятом классе"),
    ("на 3-й улице", "на третьей улице"),
    ("в 2-х томах", "в двух томах"),
    ("с 5-ю друзьями", "с пятью друзьями"),
    ("5-летний", "пятилетний"),
    ("2-хкомнатная", "двухкомнатная"),
    ("на 5 этаже", "на пятом этаже"),
    ("XX век", "двадцатый век"),
    ("в XIX веке", "в девятнадцатом веке"),
    ("Пётр I", "Пётр первый"),
    ("при Екатерине II", "при Екатерине второй"),
    ("глава IV", "глава четвёртая"),
    ("3/4 стакана", "три четвёртых стакана"),
    ("½ стакана", "одна вторая стакана"),
    ("д. 5/2", "дом пять дробь два"),
    ("2+2=4", "два плюс два равно четыре"),
    ("3×4", "три на четыре"),
    ("№ 5", "номер пять"),
    ("007", "ноль ноль семь"),
    ("Версия 1.2.3", "Версия один точка два точка три"),
    ("т.е. так", "то есть так"),
    ("и т.д.", "и так далее."),
    ("См. рис. 3", "смотри рисунок три"),
    ("в г. Москве", "в городе Москве"),
    ("на ул. Ленина", "на улице Ленина"),
    ("д. 5, кв. 12", "дом пять, квартира двенадцать"),
    ("школа им. Пушкина", "школа имени Пушкина"),
    ("ФСБ", "эф эс бэ"),
    ("США", "эс шэ а"),
    ("МИД России", "мид России"),
    ("ЭТО ОЧЕНЬ ВАЖНО", "это очень важно"),
    ("iPhone и Google", "айфон и гугл"),
    ("ChatGPT", "чат джи пи ти"),
    ("Python 3.12", "пайтон три целых двенадцать сотых"),
    ("www.example.com", "вэ вэ вэ точка эксампл точка ком"),
    ("user@mail.ru", "юзер собака мейл точка ру"),
    ("COVID-19", "ковид девятнадцать"),
    ("Wi-Fi", "вай-фай"),
    ("café", "кафе"),
    ("«Привет» — сказал он...", '"Привет" — сказал он…'),
    ("за́мок", "з+амок"),
    ("с 21 студентом", "с двадцатью одним студентом"),
    ("перед 31 машиной", "перед тридцатью одной машиной"),
    ("10^6", "десять в степени шесть"),
]


class NormalizeTests(unittest.TestCase):
    def test_cases(self) -> None:
        for written, spoken in CASES_:
            with self.subTest(written=written):
                self.assertEqual(normalize(written).replace("+", ""), spoken.replace("+", ""))

    def test_stress_marks(self) -> None:
        for written, spoken in (
            ("в 2 часа", "в два час+а"),
            ("до 2 часов", "до двух час+ов"),
            ("в 15:00", "в пятнадцать час+ов"),
            ("3 дома", "три д+ома"),
            ("2 тома", "два т+ома"),
            ("В 1999 году", "В тысяча девятьсот девяносто девятом год+у"),
            ("на 5 этаже", "на пятом этаж+е"),
            ("XX века", "двадцатого в+ека"),
            ("глава IV", "глав+а четвёртая"),
            ("с 40 рублями", "с сорок+а рубл+ями"),
            ("с 7 друзьями", "с семь+ю друзьями"),
            ("с 800 рублями", "с восемьюст+ами рубл+ями"),
            ("к 30-му числу", "к тридц+атому числ+у"),
            ("7 тыс. руб.", "семь т+ысяч рубл+ей."),
            ("на ул. Ленина", "на +улице Ленина"),
            ("з+амок и зам+ок", "з+амок и зам+ок"),
            ("за́мок", "з+амок"),
            ("ёлка", "ёлка"),
        ):
            with self.subTest(written=written):
                self.assertEqual(normalize(written), spoken)

    def test_none_only_cleans(self) -> None:
        self.assertEqual(
            normalize("«Что-то» -- и всё!.. За́мок", "none"), '"Что-то" — и всё! З+амок'
        )
        for text, tokens in (
            ("в 2 часа", "2"),
            ("iPhone и MP3", "iPhone MP3"),
            ("рост 5% и №3", "5% №3"),
        ):
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(NotNormalized, f": {re.escape(tokens)}$"),
            ):
                normalize(text, "none")
        with self.assertRaises(ValueError):
            normalize("текст", "morph")

    def test_output_alphabet_and_no_unit_symbol_survives(self) -> None:
        allowed = re.compile(r"^[а-яёА-ЯЁ+\- ,.?!;:\"'()—…]*$")
        for written, _ in CASES_:
            self.assertRegex(normalize(written), allowed)
        for unit in UNITS:
            for text in (f"Цена 5 {unit}.", f"{unit}12,50"):
                self.assertRegex(normalize(text), allowed)

    def test_idempotent(self) -> None:
        extra = [
            "Встреча 17.11.2026 в 14:30 в г. Москве, ул. Ленина, д. 5; "
            "вход 1525 руб. (т.е. дёшево).",
            "В 1990-х гг. ВВП США вырос на 3,5%, а iPhone стоил $599.",
        ]
        for text in [written for written, _ in CASES_] + extra:
            once = normalize(text)
            self.assertEqual(normalize(once), once)
