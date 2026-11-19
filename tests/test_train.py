"""Tests for dataset preparation, CTC alignment, training, and custom-voice fine-tuning."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray

from tiny_voca_ru.checkpoint import load_checkpoint, save_checkpoint
from tiny_voca_ru.config import Config
from tiny_voca_ru.frontend import PHONE_SET, SYMBOLS, UNK, phonemize
from tiny_voca_ru.infer import Synthesizer
from tiny_voca_ru.model import TinyVoca
from tiny_voca_ru.train.align import (
    ALIGNER_RATE,
    ALIGNER_STRIDE,
    AlignmentSettings,
    assemble,
    ctc_viterbi,
    spans_from_states,
)
from tiny_voca_ru.train.dataset import BucketSampler, SpeechDataset, collate_batch, continuous_pitch
from tiny_voca_ru.train.prepare import QualityLimits, prepare_dataset
from tiny_voca_ru.train.train import TrainConfig, load_pretrained_for_finetune, train


def small_config(speakers: tuple[str, ...] = ("F", "M")) -> Config:
    return Config(
        speakers=speakers,
        channels=32,
        heads=2,
        feedforward=64,
        kernel=3,
        encoder_layers=1,
        prosody_layers=1,
        acoustic_layers=1,
        predictor_channels=16,
        cep_causal=16,
        cep_anti=4,
        noise_cep=8,
        max_tokens=128,
        max_frames=256,
    )


class FakeRecognizer:
    """Deterministic in-memory stand-in for PhoneRecognizer in unit tests."""

    def __init__(self, target_phones_by_len: dict[int, list[int]]) -> None:
        self.blank = 0
        self.vocab: dict[str, int] = {p: idx + 1 for idx, p in enumerate(sorted(PHONE_SET | {UNK}))}
        self._targets = target_phones_by_len
        self.next_labels: list[int] = []

    def label(self, symbol: str) -> int:
        val = self.vocab[symbol]
        self.next_labels.append(val)
        return val

    def log_probs(self, wave: NDArray[np.floating[Any]], sample_rate: int) -> NDArray[np.float32]:
        labels = list(self.next_labels)
        self.next_labels.clear()
        seconds = len(wave) / sample_rate
        frames = max(int(round(seconds * ALIGNER_RATE / ALIGNER_STRIDE)), len(labels) * 2 + 4)
        vocab_size = len(self.vocab) + 1
        probs = np.full((frames, vocab_size), -10.0, dtype=np.float32)
        probs[:, self.blank] = -0.05
        # Place each requested phone label cleanly across interior frames with high confidence.
        positions = np.linspace(2, frames - 3, len(labels)).round().astype(int)
        for pos, lbl in zip(positions, labels, strict=True):
            probs[pos, :] = -10.0
            probs[pos, lbl] = -0.01
        return probs


def _voiced_sine(
    duration_sec: float, sr: int = 24000, freq_hz: float = 180.0
) -> NDArray[np.float32]:
    t = np.arange(int(round(duration_sec * sr)), dtype=np.float64) / sr
    wave = 0.25 * np.sin(2 * np.pi * freq_hz * t) + 0.1 * np.sin(4 * np.pi * freq_hz * t)
    # Add short leading/trailing quiet margins.
    margin = int(0.05 * sr)
    wave[:margin] *= 0.001
    wave[-margin:] *= 0.001
    return wave.astype(np.float32)


class TrainingAndFinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_ctc_viterbi_and_assemble(self) -> None:
        log_probs = np.full((12, 5), -10.0, dtype=np.float32)
        log_probs[:, 0] = -0.1
        for t, lbl in ((2, 1), (5, 2), (8, 3)):
            log_probs[t, :] = -10.0
            log_probs[t, lbl] = -0.01
        states = ctc_viterbi(log_probs, np.array([1, 2, 3], dtype=np.int64), blank=0)
        spans = spans_from_states(states, 3)
        self.assertEqual(spans, [(2, 2), (5, 5), (8, 8)])

        symbols = ("<bos>", "d", "o", "m", "<eos>")
        frames = 24
        silent = np.zeros(frames, dtype=bool)
        silent[:2] = True
        silent[-2:] = True
        durations = assemble(
            symbols, spans, frames, silent, hop=256, sample_rate=24000, settings=AlignmentSettings()
        )
        self.assertEqual(int(durations.sum()), frames)
        self.assertGreaterEqual(int(durations[0]), 1)
        self.assertGreaterEqual(int(durations[-1]), 1)
        self.assertTrue(all(int(durations[i]) >= 1 for i in (1, 2, 3)))

    def test_continuous_pitch_filters_octave_jump(self) -> None:
        pitch = torch.tensor([0.0, 0.1, 0.1, 1.5, 0.1, 0.1, 0.0])
        voiced = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        curve, trusted = continuous_pitch(pitch, voiced)
        self.assertEqual(float(trusted[3]), 0.0)
        self.assertAlmostEqual(float(curve[3]), 0.1, places=5)

    def test_prepare_train_and_custom_voice_finetune_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_dir = root / "wavs"
            wav_dir.mkdir()
            rows = [
                ("u1", "u1.wav", "Привет мир.", "alice", "train"),
                ("u2", "u2.wav", "Добрый день.", "alice", "train"),
                ("u3", "u3.wav", "Как дела?", "alice", "validation"),
            ]
            meta_csv = root / "metadata.csv"
            with meta_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "audio", "text", "speaker", "split"])
                for utt_id, fname, text, speaker, split in rows:
                    sf.write(wav_dir / fname, _voiced_sine(0.75), 24000)
                    writer.writerow([utt_id, fname, text, speaker, split])

            base_cfg = small_config(speakers=("alice",))
            fake_rec: Any = FakeRecognizer({})
            manifest = prepare_dataset(
                meta_csv,
                root / "prepared",
                audio_root=wav_dir,
                recognizer=fake_rec,
                base_config=base_cfg,
                limits=QualityLimits(
                    min_phones_per_second=1.0, max_phones_per_second=50.0, min_vowel_voiced=0.0
                ),
            )
            self.assertEqual(manifest["speakers"], ["alice"])
            self.assertEqual(manifest["counts"], {"train": 2, "validation": 1, "rejected": 0})

            ds = SpeechDataset(root / "prepared/train.jsonl", base_cfg)
            self.assertEqual(len(ds), 2)
            sampler = BucketSampler(ds.lengths, frame_budget=500, max_batch=2, seed=0)
            batch = collate_batch([ds[i] for i in sampler.next()])
            self.assertEqual(batch["tokens"].shape[0], len(batch["frame_lengths"]))

            # Create a 2-speaker ("F", "M") pretrained safetensors checkpoint.
            torch.manual_seed(11)
            pretrained_cfg = small_config(speakers=("F", "M"))
            pretrained_model = TinyVoca(pretrained_cfg).eval()
            pretrained_dir = save_checkpoint(root / "pretrained", pretrained_model, step=500)

            # Verify speaker embedding adaptation both with --init-voice and mean fallback.
            target_a = TinyVoca(base_cfg)
            load_pretrained_for_finetune(target_a, pretrained_dir, init_voice="M")
            self.assertTrue(
                torch.equal(target_a.voices.weight[0], pretrained_model.voices.weight[1])
            )
            self.assertTrue(
                torch.equal(
                    target_a.frame_voices.weight[0], pretrained_model.frame_voices.weight[1]
                )
            )

            target_mean = TinyVoca(base_cfg)
            load_pretrained_for_finetune(target_mean, pretrained_dir, init_voice=None)
            self.assertTrue(
                torch.allclose(
                    target_mean.voices.weight[0], pretrained_model.voices.weight.mean(dim=0)
                )
            )

            # Fine-tune onto 1-speaker ("alice",) with --freeze-encoder for 2 steps.
            run_dir = root / "finetune"
            tcfg = TrainConfig(
                steps=2,
                batch_frames=500,
                max_batch=2,
                learning_rate=5e-4,
                warmup=1,
                gan_start=100,
                teacher_hold=1,
                teacher_ramp=1,
                save_every=2,
                log_every=1,
                seed=7,
                freeze_encoder=True,
            )
            inf_dir = train(
                root / "prepared/train.jsonl",
                root / "prepared/validation.jsonl",
                run_dir,
                base_cfg,
                tcfg,
                device="cpu",
                init_from=pretrained_dir,
                init_voice="F",
            )
            self.assertTrue((run_dir / "last.pt").is_file())
            self.assertTrue((run_dir / "best/model.safetensors").is_file())
            self.assertTrue((inf_dir / "model.safetensors").is_file())

            # Encoder weights stayed frozen, while vocoder/duration weights updated.
            ft_model, ft_meta = load_checkpoint(inf_dir)
            self.assertEqual(ft_meta.step, 2)
            self.assertEqual(ft_meta.config.speakers, ("alice",))
            self.assertEqual(ft_meta.symbols, SYMBOLS)
            self.assertTrue(torch.equal(ft_model.tokens.weight, pretrained_model.tokens.weight))
            self.assertFalse(
                torch.equal(ft_model.duration.out.bias, pretrained_model.duration.out.bias)
            )

            # Inference with the fine-tuned 1-speaker safetensors checkpoint works cleanly.
            synth = Synthesizer(inf_dir)
            result = synth.synthesize("Привет, Алиса!", voice="alice")
            self.assertGreater(result.audio.size, 0)
            self.assertEqual(result.sample_rate, 24000)

            # Verify phonemize sanity on metadata text.
            self.assertEqual(phonemize("Привет мир.").unknown, ())
