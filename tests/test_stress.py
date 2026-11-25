"""Tests for Russian lexical stress placement (lite, full, and none providers)."""

from __future__ import annotations

import sys
import unittest
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any

from tiny_voca_ru import frontend
from tiny_voca_ru.frontend import omograph, stress
from tiny_voca_ru.frontend.stress import (
    FullStress,
    LiteStress,
    NoStress,
    accent_input_ids,
    get_provider,
    stress_from_logits,
)

try:
    import onnxruntime  # noqa: F401

    HAVE_MODEL = (
        resources.files("tiny_voca_ru.frontend").joinpath("data").joinpath("accent.onnx").is_file()
    )
except ImportError:
    HAVE_MODEL = False

LOCAL_OMOGRAPH = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "tiny-voca-ru-onnx" / "omograph.onnx"
)
HAVE_LOCAL_OMOGRAPH = HAVE_MODEL and LOCAL_OMOGRAPH.is_file()

PATRONYMICS = (
    "Андр+еевич Алекс+еевна Констант+инович Ром+анович Ив+анович П+авлович +Игоревич "
    "Дм+итриевич Гл+ебович Шостак+ович Абрам+ович Рабин+ович"
)


class WithoutOnnxruntime:
    """Simulate Pyodide where onnxruntime cannot be imported."""

    def __init__(self) -> None:
        self.saved: Any = None

    def __enter__(self) -> None:
        self.saved = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = None  # type: ignore[assignment]
        stress._accent_session.cache_clear()
        stress._predict.cache_clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.saved is None:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = self.saved
        stress._accent_session.cache_clear()
        stress._predict.cache_clear()


class LiteStressTests(unittest.TestCase):
    def setUp(self) -> None:
        provider = get_provider("lite")
        assert isinstance(provider, LiteStress)
        self.lite = provider

    def test_provider_registry(self) -> None:
        self.assertIs(get_provider("lite"), self.lite)
        self.assertEqual(self.lite.name, "lite")
        self.assertIsInstance(get_provider("none"), NoStress)
        self.assertEqual(get_provider("none").accentuate("Замок"), "Замок")
        with self.assertRaises(ValueError):
            get_provider("unknown_provider")

    def test_sentence(self) -> None:
        self.assertEqual(
            self.lite.accentuate("Все уже пришли, а сорок сорок сели на замок."),
            "Вс+е уж+е пришл+и, а с+орок с+орок сел+и на з+амок.",
        )

    def test_homograph_readings(self) -> None:
        self.assertEqual(self.lite.readings("замок"), ["з+амок", "зам+ок"])
        self.assertEqual(self.lite.readings("Мука")[:2], ["м+ука", "мук+а"])
        self.assertEqual(self.lite.readings("уже"), ["уж+е", "+уже"])
        self.assertEqual(self.lite.readings("учителя")[0], "уч+ителя")
        self.assertIn("учител+я", self.lite.readings("учителя"))
        self.assertEqual(self.lite.readings("ведра"), ["вёдра", "ведр+а"])
        self.assertEqual(self.lite.readings("все"), ["вс+е", "всё"])
        self.assertEqual(self.lite.readings("всё"), ["всё"])
        self.assertEqual(self.lite.accentuate("все-таки"), "всё-таки")

    def test_yo_restored(self) -> None:
        self.assertEqual(self.lite.readings("еще"), ["ещё"])
        self.assertEqual(self.lite.accentuate("Еще шел снег."), "Ещё шёл сн+ег.")
        self.assertEqual(self.lite.accentuate("ЕЩЕ"), "ЕЩЁ")
        self.assertEqual(self.lite.accentuate("ёлка"), "ёлка")

    def test_clitics_unstressed(self) -> None:
        self.assertEqual(
            self.lite.accentuate("в доме и на поле, но не за рекой же ли бы"),
            "в д+оме и на п+оле, но не за рек+ой же ли бы",
        )
        self.assertEqual(self.lite.readings("за"), [])
        self.assertEqual(
            self.lite.accentuate("кто-то из-за двери, давай-ка по-русски"),
            "кт+о-то из-за дв+ери, дав+ай-ка по-р+усски",
        )

    def test_ne_takes_stress(self) -> None:
        self.assertEqual(
            self.lite.accentuate("он не был в школе, не было"), "+он н+е был в шк+оле, н+е было"
        )
        self.assertEqual(self.lite.accentuate("не была"), "не был+а")

    def test_stressed_words_untouched(self) -> None:
        self.assertEqual(self.lite.accentuate("зам+ок на двери"), "зам+ок на дв+ери")

    def test_monosyllables(self) -> None:
        self.assertEqual(self.lite.readings("кот"), ["к+от"])
        self.assertEqual(self.lite.accentuate("кот спит"), "к+от сп+ит")

    @unittest.skipUnless(HAVE_MODEL, "onnxruntime or data/accent.onnx missing")
    def test_unknown_word_model(self) -> None:
        self.assertEqual(self.lite.readings("кракозябра"), [])
        out = self.lite.accentuate("Кракозябра пришла.")
        self.assertEqual(out.count("+"), 2)
        self.assertEqual(stress.fold(out), "кракозябра пришла.")
        self.assertEqual(self.lite.unknown_words("Кракозябра пришла."), [])

    def test_unknown_words_without_onnxruntime(self) -> None:
        lite = LiteStress()
        with WithoutOnnxruntime():
            text = "Кракозябра и кракозябра пришли на шмурдяк-то."
            self.assertEqual(lite.unknown_words(text), ["кракозябра", "шмурдяк"])
            self.assertEqual(
                lite.accentuate(text), "Кракозябра и кракозябра пришл+и на шмурдяк-то."
            )
            lite.extra.update({"кракозябра": "кракоз+ябра", "шмурдяк": "шмурд+як"})
            self.assertEqual(lite.unknown_words(text), [])
            self.assertEqual(
                lite.accentuate(text), "Кракоз+ябра и кракоз+ябра пришл+и на шмурд+як-то."
            )
            self.assertEqual(LiteStress().unknown_words("кракоз+ябра и шмурдяк"), ["шмурдяк"])
            self.assertEqual(
                frontend.unknown_words("Кракозябра 2 раза.", stress=LiteStress()), ["кракозябра"]
            )

    def test_custom_predict_callback(self) -> None:
        custom = LiteStress(predict=lambda word: "кракоз+ябра" if word == "кракозябра" else None)
        self.assertEqual(custom.accentuate("Кракозябра пришла."), "Кракоз+ябра пришл+а.")

    def test_preferred_homograph_readings(self) -> None:
        self.assertEqual(
            [w for k, w in stress.PREFERRED.items() if w not in self.lite.readings(k)], []
        )
        self.assertEqual(
            self.lite.accentuate("Потом пора домой, после ужина."),
            "Пот+ом пор+а дом+ой, п+осле +ужина.",
        )
        self.assertEqual(self.lite.readings("потом"), ["пот+ом", "п+отом"])
        for expected in (
            "так+ая",
            "том+у",
            "должн+о",
            "м+ало",
            "н+ачал",
            "п+ередал",
            "светл+о",
            "широк+о",
            "ч+асти",
            "с+оли",
            "б+оли",
            "к+ости",
            "ж+еня",
        ):
            plain = expected.replace("+", "")
            with self.subTest(word=plain):
                self.assertEqual(self.lite.readings(plain)[0], expected)

    def test_standard_patronymics(self) -> None:
        self.assertEqual(self.lite.accentuate(PATRONYMICS.replace("+", "")), PATRONYMICS)

    def test_marks_survive_the_whole_text(self) -> None:
        self.assertEqual(
            frontend.accentuate("Я открыл зам+ок ключом и пошёл домой.", "lite"),
            "+Я откр+ыл зам+ок ключ+ом и пошёл дом+ой.",
        )
        self.assertEqual(frontend.accentuate("в 2 часа", "lite"), "в дв+а час+а")
        self.assertEqual(
            frontend.accentuate("3 дома, 2 тома, свою семью и 7 семьями", "lite"),
            "тр+и д+ома, дв+а т+ома, сво+ю с+емью и семь+ю с+емьями",
        )

    def test_model_io(self) -> None:
        self.assertEqual(accent_input_ids("ёж"), [2, 44, 18, 3])
        logits = [[0, 0, 0], [5, 0, 0], [0, 1, 0], [5, 0, 0], [0, 3, 0], [0, 0, 0]]
        self.assertEqual(stress_from_logits("мама", logits), "мам+а")

    def test_unknown_words_skips_overlong_and_vowelless(self) -> None:
        from unittest.mock import patch

        from tiny_voca_ru.frontend import stress as stress_mod

        provider = stress_mod.LiteStress()
        overlong = "а" * 60
        with patch.object(stress_mod, "_accent_session", lambda: None):
            found = provider.unknown_words(f"Флюмбрастый бвгджз {overlong}")
        self.assertEqual(found, ["флюмбрастый"])


class OmographTokenizerTests(unittest.TestCase):
    def test_wordpiece_ids_basic_and_punctuation(self) -> None:
        ids = omograph.wordpiece_ids("Я открыл зам+ок!")
        self.assertTrue(ids)
        self.assertNotIn(omograph.UNK_ID, ids)
        self.assertEqual(omograph.wordpiece_ids("а" * 105), [omograph.UNK_ID])

    def test_input_ids_wraps_target_and_isolates_sentence(self) -> None:
        text = "Дверной замок давно заржавел. Средневековый замок стоял на холме."
        first_start = text.index("замок")
        second_start = text.rindex("замок")
        first_seqs = omograph.omograph_input_ids(
            text, first_start, first_start + 5, ["з+амок", "зам+ок"]
        )
        second_seqs = omograph.omograph_input_ids(
            text, second_start, second_start + 5, ["з+амок", "зам+ок"]
        )
        self.assertEqual(len(first_seqs), 2)
        for seq in first_seqs + second_seqs:
            self.assertEqual(seq[0], omograph.CLS_ID)
            self.assertEqual(seq[-1], omograph.SEP_ID)
            self.assertIn(omograph.W_OPEN_ID, seq)
            self.assertIn(omograph.W_CLOSE_ID, seq)
            self.assertEqual(seq.count(omograph.SEP_ID), 2)
        # Neighboring sentence words must not leak across the period boundary.
        dvernoy_ids = omograph.wordpiece_ids("дверной")
        srednevek_ids = omograph.wordpiece_ids("средневековый")
        self.assertTrue(all(tid in first_seqs[0] for tid in dvernoy_ids))
        self.assertFalse(all(tid in first_seqs[0] for tid in srednevek_ids))
        self.assertTrue(all(tid in second_seqs[0] for tid in srednevek_ids))
        self.assertFalse(all(tid in second_seqs[0] for tid in dvernoy_ids))

    def test_input_ids_strips_neighbor_plus_and_caps_length(self) -> None:
        long_left = "слово " * 300
        long_right = " текст" * 300
        text = f"{long_left}рубл+ей замок{long_right}"
        start = text.index("замок")
        seqs = omograph.omograph_input_ids(text, start, start + 5, ["з+амок", "зам+ок"])
        for seq in seqs:
            self.assertLessEqual(len(seq), omograph.MAX_LENGTH)
            self.assertEqual(seq[0], omograph.CLS_ID)
            self.assertEqual(seq[-1], omograph.SEP_ID)
            self.assertIn(omograph.W_OPEN_ID, seq)
            self.assertIn(omograph.W_CLOSE_ID, seq)
        # Context '+' before target must be stripped; only the candidate segment keeps '+'.
        sep = seqs[0].index(omograph.SEP_ID)
        plus_id = omograph.wordpiece_ids("+")[0]
        self.assertNotIn(plus_id, seqs[0][:sep])
        self.assertIn(plus_id, seqs[0][sep:])

    def test_choose_reading(self) -> None:
        self.assertEqual(
            omograph.choose_reading(["з+амок", "зам+ок"], [[2.0, -1.0], [-0.5, 3.0]]),
            "зам+ок",
        )
        self.assertEqual(
            omograph.choose_reading(["з+амок", "зам+ок"], [[-1.0, -0.2], [-0.5, -2.0]]),
            "з+амок",
        )
        with self.assertRaises(ValueError):
            omograph.choose_reading([], [])
        with self.assertRaises(ValueError):
            omograph.choose_reading(["з+амок"], [[0.0, 1.0], [1.0, 0.0]])


class FullStressTests(unittest.TestCase):
    def test_ruaccent_provider_removed(self) -> None:
        with self.assertRaises(ValueError):
            get_provider("ruaccent")

    def test_homograph_detection_filters_yo_only_and_explicit_marks(self) -> None:
        lite = LiteStress()
        self.assertEqual(lite.homograph_readings("замок"), ["з+амок", "зам+ок"])
        self.assertEqual(lite.homograph_readings("все"), [])
        self.assertEqual(lite.homograph_readings("ведра"), [])
        self.assertEqual(lite.homograph_readings("села"), ["сел+а", "с+ела"])
        self.assertEqual(lite.homographs("зам+ок на столе"), [])
        spans = lite.homographs("Замок на столе, а замок на холме.")
        self.assertEqual([(s, e) for s, e, _ in spans], [(0, 5), (18, 23)])

    def test_per_occurrence_resolution_with_fake_scorer(self) -> None:
        calls = 0

        def fake_score(sequences: Any) -> list[list[float]]:
            nonlocal calls
            calls += 1
            # First occurrence picks index 1 (зам+ок), second picks index 0 (з+амок).
            return [[0.0, 1.0], [0.0, 5.0]] if calls == 1 else [[0.0, 5.0], [0.0, 1.0]]

        provider = FullStress(score=fake_score)
        self.assertEqual(provider.name, "full")
        self.assertEqual(
            provider.accentuate("Замок заржавел, а замок стоял."),
            "Зам+ок заржав+ел, а з+амок сто+ял.",
        )
        self.assertEqual(calls, 2)

    def test_explicit_marks_and_phrase_rules_preserved(self) -> None:
        provider = FullStress(
            predict=lambda w: "кракоз+ябра" if w == "кракозябра" else None,
            score=lambda seqs: [[0.0, 5.0]] + [[5.0, 0.0]] * (len(seqs) - 1),
        )
        self.assertEqual(
            provider.accentuate("Он не был дома, а зам+ок и кракозябра еще там."),
            "+Он н+е был д+ома, а зам+ок и кракоз+ябра ещё т+ам.",
        )

    def test_missing_local_model_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            FullStress(model_path="/nonexistent/path/omograph.onnx")

    @unittest.skipUnless(HAVE_LOCAL_OMOGRAPH, "local omograph.onnx not present")
    def test_tiny21_int8_omograph_model(self) -> None:
        provider = FullStress(model_path=LOCAL_OMOGRAPH)
        out = provider.accentuate(
            "Дверной замок давно заржавел. Средневековый замок стоял на холме. "
            "В лесу живут рыжие белки. В составе есть белки, жиры и углеводы."
        )
        self.assertIn("зам+ок", out)
        self.assertIn("з+амок", out)
        self.assertIn("б+елки", out)
        self.assertIn("белк+и", out)

        story = provider.accentuate(
            "Она никогда раньше не видела моря и долго стояла у самой воды, не говоря ни слова."
        )
        self.assertIn("м+оря", story)
        self.assertIn("с+амой", story)
        self.assertIn("вод+ы", story)
        self.assertIn("сл+ова", story)

        ticket = frontend.accentuate("Билет стоит 1250 рублей.", stress=provider)
        self.assertIn("ст+оит", ticket)

        manual = provider.accentuate(
            "Знак плюс перед гласной задаёт ударение вручную: "
            "старый зам+ок на вор+отах з+амка, б+елки и белк+и, тв+орог или твор+ог."
        )
        self.assertIn("зам+ок", manual)
        self.assertIn("вор+отах", manual)
        self.assertIn("з+амка", manual)
        self.assertIn("б+елки", manual)
        self.assertIn("белк+и", manual)
        self.assertIn("тв+орог", manual)
        self.assertIn("твор+ог", manual)
