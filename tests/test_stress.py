"""Tests for Russian lexical stress placement (lite, none, and ruaccent providers)."""

from __future__ import annotations

import sys
import unittest
from importlib import resources
from types import TracebackType
from typing import Any

from tiny_voca_ru import frontend
from tiny_voca_ru.frontend import stress
from tiny_voca_ru.frontend.stress import (
    LiteStress,
    NoStress,
    RuaccentStress,
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


def have_ruaccent() -> bool:
    try:
        import ruaccent  # type: ignore[import-not-found]  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(
        try_to_load_from_cache(
            RuaccentStress.REPO,
            "nn/nn_omograph/turbo3.1/model.onnx",
            revision=RuaccentStress.REVISION,
        ),
        str,
    )


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


class RuaccentStressTests(unittest.TestCase):
    def fake(self, output: str) -> RuaccentStress:
        provider = RuaccentStress.__new__(RuaccentStress)
        provider.engine = type("Engine", (), {"process_all": lambda self, text: output})()
        return provider

    def test_output_mapped_onto_text(self) -> None:
        provider = self.fake("Он в+ышел з+а дв+ерь, кт+о-т+о ждал зам+ок ещ+ё.")
        self.assertEqual(
            provider.accentuate("Он вышел за дверь, кто-то ждал з+амок еще…"),
            "Он в+ышел за дв+ерь, кт+о-то ждал з+амок ещё…",
        )
        with self.assertRaises(ValueError):
            self.fake("Он ушёл.").accentuate("Он вышел.")

    def test_standard_patronymics_override(self) -> None:
        provider = self.fake(
            "Андре+евич Алексе+евна Константин+ович Роман+ович Иван+ович Павл+ович Иг+оревич "
            "Дмитри+евич Глеб+ович Шостак+ович Абрам+ович Рабин+ович"
        )
        self.assertEqual(provider.accentuate(PATRONYMICS.replace("+", "")), PATRONYMICS)
        self.assertEqual(self.fake("Иван+ович").accentuate("Иван+ович"), "Иван+ович")

    @unittest.skipUnless(have_ruaccent(), "ruaccent or its turbo3.1 assets not available")
    def test_ruaccent(self) -> None:
        out = get_provider("ruaccent").accentuate(
            "На двери висит замок, а кракозябра еще за дверью."
        )
        self.assertIn("ещё", out)
        self.assertIn(" за ", out)
        self.assertRegex(out, r"кр\+?а\+?к\+?о\+?з")
