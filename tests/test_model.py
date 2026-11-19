"""Core model, DSP vocoder, training objective, and gradient tests."""

from __future__ import annotations

import math
import unittest

import torch

from tiny_voca_ru.config import Config
from tiny_voca_ru.frontend.symbols import PHONE_IDS, SYMBOL_TO_ID
from tiny_voca_ru.model import SourceFilter, TinyVoca, cepstrum_response, filtered, pulse_train
from tiny_voca_ru.train.dataset import collate_batch
from tiny_voca_ru.train.losses import Discriminators
from tiny_voca_ru.train.train import (
    TrainConfig,
    TrainingSystem,
    learning_rate,
    optimize_step,
    teacher_probability,
)


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


def synthetic_item(
    config: Config, token_count: int = 8, frames: int = 24, speaker: int = 0
) -> dict[str, object]:
    phones = sorted(PHONE_IDS)
    token_ids = [
        SYMBOL_TO_ID["<bos>"],
        *(phones[i % len(phones)] for i in range(token_count - 2)),
        SYMBOL_TO_ID["<eos>"],
    ]
    stress = [0] * token_count
    durations = [0, *(max(1, frames // (token_count - 2)) for _ in range(token_count - 2)), 0]
    durations[1] += frames - sum(durations)
    return {
        "tokens": torch.tensor(token_ids, dtype=torch.long),
        "stress": torch.tensor(stress, dtype=torch.long),
        "durations": torch.tensor(durations, dtype=torch.long),
        "speaker": speaker,
        "wave": torch.randn(frames * config.hop) * 0.2,
        "mel": torch.randn(frames, config.n_mels),
        "prosody": torch.stack(
            (torch.zeros(frames), torch.ones(frames), torch.zeros(frames)),
            dim=-1,
        ),
        "pitch_trusted": torch.ones(frames),
    }


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_default_model_parameter_count(self) -> None:
        model = TinyVoca(Config())
        params = sum(p.numel() for p in model.parameters())
        self.assertLess(params, 5_000_000)
        self.assertGreater(params, 4_000_000)

    def test_pulse_train_pitch_power_and_phase(self) -> None:
        rate, hop = 24000, 256
        f0 = torch.full((1, 20), 150.0)
        wave = pulse_train(f0, rate, hop)
        self.assertEqual(wave.shape, (1, 20 * hop))
        self.assertAlmostEqual(float(wave.square().mean()), 1.0, delta=0.15)
        shifted = pulse_train(f0, rate, hop, phase=torch.tensor([1.0]))
        self.assertFalse(torch.allclose(shifted, wave))

    def test_stft_filtering_transparent_for_flat_response(self) -> None:
        source = torch.randn(2, 4096)
        window = torch.hann_window(512)
        frames = 4096 // 128 + 1
        flat = torch.ones(2, frames, 257, dtype=torch.complex64)
        out = filtered(source, flat, 512, 128, window)
        self.assertLess(float((out - source).abs().max()), 1e-4)
        with self.assertRaises(ValueError):
            filtered(source, flat[:, :-1], 512, 128, window)

    def test_cepstrum_gives_gain_and_minimum_phase(self) -> None:
        zero = cepstrum_response(torch.zeros(1, 8), torch.zeros(1, 2), 64)
        self.assertLess(float((zero - 1.0).abs().max()), 1e-6)
        causal = torch.tensor([[0.0, 0.5, -0.2, 0.1, 0.0, 0.0, 0.0, 0.0]])
        response = cepstrum_response(causal, None, 64)
        impulse = torch.fft.irfft(response[0], n=64)
        self.assertGreater(float(impulse[:8].square().sum() / impulse.square().sum()), 0.999)

    def test_expand_and_zero_duration_rejection(self) -> None:
        model = TinyVoca(small_config()).eval()
        hidden = torch.randn(1, 4, model.config.channels)
        mask = torch.ones(1, 4, dtype=torch.bool)
        durations = torch.tensor([[0, 3, 2, 0]])
        expanded, frame_mask, lengths = model.expand(hidden, durations, mask, torch.tensor([0]))
        self.assertEqual(expanded.shape, (1, 5, model.config.channels))
        self.assertEqual(int(lengths[0]), 5)
        self.assertTrue(bool(frame_mask.all()))
        with self.assertRaises(ValueError):
            model.expand(hidden, torch.zeros(1, 4, dtype=torch.long), mask, torch.tensor([0]))

    def test_batch_padding_does_not_change_short_example(self) -> None:
        c = small_config()
        model = TinyVoca(c).eval()
        torch.nn.init.zeros_(model.duration.out.weight)
        model.duration.out.bias.data[:] = torch.tensor([1.2, 0.5])
        phones = sorted(PHONE_IDS)
        short = torch.tensor([[phones[0], phones[1], phones[2], 0, 0]])
        long = torch.tensor([[phones[3], phones[4], phones[5], phones[6], phones[7]]])
        tokens = torch.cat((short, long), dim=0)
        stress = torch.zeros_like(tokens)
        lengths = torch.tensor([3, 5])
        speakers = torch.tensor([0, 1])

        alone, alone_len, _ = model.synthesize(
            short[:, :3], stress[:1, :3], torch.tensor([3]), torch.tensor([0]), seed=5
        )
        batched, batch_len, _ = model.synthesize(tokens, stress, lengths, speakers, seed=5)
        self.assertEqual(int(alone_len[0]), int(batch_len[0]))
        self.assertLess(
            float((alone[0, : int(alone_len[0])] - batched[0, : int(batch_len[0])]).abs().max()),
            1e-4,
        )

    def test_source_filter_differentiable_and_respects_lengths(self) -> None:
        c = small_config()
        vocoder = SourceFilter(c)
        hidden = torch.randn(2, 10, c.channels, requires_grad=True)
        f0 = torch.full((2, 10), 180.0)
        lengths = torch.tensor([10, 7])
        wave = vocoder(hidden, f0, lengths)
        wave[1, : 7 * c.hop].square().mean().backward()
        assert hidden.grad is not None
        self.assertEqual(float(hidden.grad[1, 7:].abs().sum()), 0.0)

    def test_training_steps_with_and_without_gan(self) -> None:
        c = small_config()
        system = TrainingSystem(c)
        critics = Discriminators(periods=(2, 3), resolutions=((256, 64),))
        opt = torch.optim.AdamW(system.parameters(), lr=1e-3)
        c_opt = torch.optim.AdamW(critics.parameters(), lr=1e-3)
        batch = collate_batch([synthetic_item(c, 8, 24, 0), synthetic_item(c, 6, 20, 1)])
        cfg = TrainConfig(
            learning_rate=1e-3,
            warmup=2,
            gan_start=1,
            gan_ramp=2,
            crop_samples=4096,
            crops=1,
            teacher_hold=1,
            teacher_ramp=1,
        )
        m0 = optimize_step(system, critics, opt, c_opt, batch, step=0, cfg=cfg)
        m1 = optimize_step(system, critics, opt, c_opt, batch, step=1, cfg=cfg)
        self.assertTrue(math.isfinite(float(m0["loss"])))
        self.assertTrue(math.isfinite(float(m1["loss"])))
        self.assertIn("discriminator", m1)

    def test_schedules_and_pitch_range_identity(self) -> None:
        self.assertEqual(teacher_probability(0, 100, 200, 0.5), 1.0)
        self.assertEqual(teacher_probability(300, 100, 200, 0.5), 0.5)
        self.assertAlmostEqual(learning_rate(0, 1e-3, 10, 1000.0), 1e-4, places=7)
        prosody = torch.randn(1, 10, 3)
        mask = torch.ones(1, 10, dtype=torch.bool)
        self.assertIs(TinyVoca.widen_pitch(prosody, mask, 1.0), prosody)
