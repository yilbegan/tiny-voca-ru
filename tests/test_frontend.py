"""Frontend tokenization, sentence splitting, aligner label mapping, and torch-free import tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

from tiny_voca_ru.config import Config
from tiny_voca_ru.frontend import (
    BOS,
    EOS,
    PHONE_SET,
    SP,
    SYMBOL_TO_ID,
    SYMBOLS,
    UNK,
    NothingToPronounce,
    ensure_terminal_punctuation,
    is_phone,
    phonemize,
    split_clause,
    split_sentences,
    stress,
    tokenize,
)
from tiny_voca_ru.train.align import ALIGNER_LABEL_FALLBACK, ALIGNER_MODEL, AlignmentSettings


class FrontendTests(unittest.TestCase):
    def test_package_root_and_frontend_do_not_import_torch(self) -> None:
        code = (
            "import sys, tiny_voca_ru, tiny_voca_ru.frontend; "
            "seq = tiny_voca_ru.phonemize('Привет, мир!'); "
            "assert len(seq) > 4; "
            "assert 'torch' not in sys.modules"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_inventory_matches_config(self) -> None:
        self.assertEqual(len(SYMBOLS), len(set(SYMBOLS)))
        self.assertEqual(len(SYMBOLS), 61)
        self.assertEqual(Config().vocab_size, len(SYMBOLS))
        with self.assertRaises(ValueError):
            Config(vocab_size=76)

    def test_pieces_become_tokens_with_stress(self) -> None:
        words = [
            [("ɡ", 0), ("ɐ", 0), ("v", 0), ("o", 1), ("rʲ", 0), ("i", 0), ("tʲ", 0)],
            [("ɕ", 0), ("i", 1)],
        ]
        result = tokenize([(False, words), (True, "."), (True, ".")])
        self.assertEqual(result.symbols[0], BOS)
        self.assertEqual(result.symbols[-2:], (".", EOS))
        self.assertIn(SP, result.symbols)
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(sum(result.stress), 2)
        self.assertEqual(sum(is_phone(result.ids)), 9)

    def test_out_of_inventory_phones_and_empty(self) -> None:
        result = tokenize([(False, [[("t", 0), ("ʘ", 0), ("a", 1)]])])
        self.assertEqual(result.unknown, ("ʘ",))
        self.assertIn(UNK, result.symbols)
        with self.assertRaises(NothingToPronounce):
            tokenize([(True, ".")])

    def test_phonemize_real_russian(self) -> None:
        result = phonemize("Съешь же ещё этих мягких французских булок, да выпей чаю!")
        self.assertEqual(result.unknown, ())
        self.assertEqual(result.symbols[0], BOS)
        self.assertEqual(result.symbols[-1], EOS)
        self.assertGreater(sum(result.stress), 4)
        with self.assertRaises(NothingToPronounce):
            phonemize("🙂 !!!")

    def test_stressed_text_ignores_provider(self) -> None:
        with patch.object(
            stress.LiteStress, "accentuate", side_effect=AssertionError("should not run")
        ):
            result = phonemize("В дв+а час+а.", stressed=True)
        self.assertEqual(sum(result.stress), 2)
        with self.assertRaises(ValueError):
            phonemize("В 2 ч+аса.", stressed=True)

    def test_plain_text_keeps_explicit_marks(self) -> None:
        result = phonemize("Я открыл зам+ок ключом и пошёл домой.")
        self.assertEqual(sum(result.stress), 6)
        self.assertNotEqual(result.stress, phonemize("Я открыл замок ключом и пошёл домой.").stress)

    def test_numbers_stay_in_inventory(self) -> None:
        sample = (
            "В 2024 г. в 15:30 проф. Иванов заплатил 12,50 руб. за 3 кг яблок; "
            "МГУ и РЖД — это сокращения."
        )
        result = phonemize(sample)
        self.assertEqual(result.unknown, ())
        self.assertGreater(len(set(result.symbols) & PHONE_SET), 25)

    def test_sentence_splitting_and_terminal_punctuation(self) -> None:
        self.assertEqual(
            split_sentences("В г. Москве ул. Ленина д. 5. Это второй дом! А т.е. третий?"),
            ["В г. Москве ул. Ленина д. 5.", "Это второй дом!", "А т.е. третий?"],
        )
        self.assertEqual(ensure_terminal_punctuation("Привет"), "Привет.")
        self.assertEqual(ensure_terminal_punctuation("Привет!"), "Привет!")
        self.assertEqual(ensure_terminal_punctuation("Привет :)"), "Привет :).")
        self.assertEqual(split_clause("раз, два"), ["раз,", "два"])
        self.assertEqual(split_clause("раз два три"), ["раз два", "три"])
        self.assertEqual(split_clause("слово"), [])

    def test_aligner_constants_calibrated(self) -> None:
        self.assertEqual(ALIGNER_MODEL, "facebook/wav2vec2-lv-60-espeak-cv-ft")
        self.assertEqual(AlignmentSettings().onset_offset_seconds, -0.010)
        self.assertTrue(set(ALIGNER_LABEL_FALLBACK) - {UNK} <= PHONE_SET)
