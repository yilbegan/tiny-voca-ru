"""Tests for ONNX Exportable rewrite parity and ONNX Runtime multi-length execution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tiny_voca_ru.checkpoint import save_checkpoint
from tiny_voca_ru.config import Config
from tiny_voca_ru.export import Exportable, export_model, export_onnx
from tiny_voca_ru.frontend.symbols import PHONE_IDS
from tiny_voca_ru.model import TinyVoca


def small_model() -> TinyVoca:
    torch.manual_seed(0)
    model = TinyVoca(Config()).eval()
    torch.nn.init.zeros_(model.duration.out.weight)
    model.duration.out.bias.data[:] = torch.tensor([1.5, 1.0])
    torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
    torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    return model


def utterance(length: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    phones = sorted(PHONE_IDS)
    tokens = torch.tensor([[phones[(7 * i) % len(phones)] for i in range(length)]])
    return tokens, (torch.arange(length)[None] % 3)


class ExportTests(unittest.TestCase):
    def test_rewrite_matches_synthesize(self) -> None:
        model = small_model()
        tokens, stress = utterance()
        for enhance in (True, False):
            for voice in (0, 1):
                reference, lengths, details = model.synthesize(
                    tokens,
                    stress,
                    torch.tensor([tokens.shape[1]]),
                    torch.tensor([voice]),
                    speed=1.25,
                    pitch_shift=0.1,
                    pitch_range=1.3,
                    enhance=enhance,
                )
                noise = torch.randn(int(lengths[0]), generator=torch.Generator().manual_seed(0))[
                    None
                ]
                with torch.no_grad():
                    audio, durations = Exportable(model, external_noise=True, enhance=enhance)(
                        tokens,
                        stress,
                        torch.tensor([voice]),
                        torch.tensor([1.25]),
                        torch.tensor([0.1]),
                        torch.tensor([1.3]),
                        noise,
                    )
                self.assertTrue(torch.equal(durations, details["durations"]))
                self.assertEqual(audio.shape, reference.shape)
                self.assertLess(
                    float((audio - reference).abs().max()), 1e-2 * float(reference.abs().max())
                )

    def test_onnx_runs_at_multiple_lengths_and_writes_clean_metadata(self) -> None:
        import onnxruntime as ort

        model = small_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ckpt = save_checkpoint(root / "ckpt", model, step=123)
            onnx_path, meta_path = export_onnx(ckpt, root / "onnx", external_noise=True)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["step"], 123)
            self.assertEqual(meta["voices"], ["F", "M"])
            self.assertEqual(
                set(meta.keys()),
                {
                    "sample_rate",
                    "hop",
                    "voices",
                    "symbols",
                    "max_tokens",
                    "max_frames",
                    "step",
                    "inputs",
                    "outputs",
                },
            )

            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            for length in (5, 32):
                tokens, stress = utterance(length)
                reference, lengths, _ = model.synthesize(
                    tokens, stress, torch.tensor([length]), torch.tensor([1])
                )
                noise = np.zeros((1, model.config.hop * model.config.max_frames), dtype=np.float32)
                noise[0, : int(lengths[0])] = torch.randn(
                    int(lengths[0]), generator=torch.Generator().manual_seed(0)
                ).numpy()
                audio, _ = session.run(
                    None,
                    {
                        "tokens": tokens.numpy(),
                        "stress": stress.numpy(),
                        "voice": np.array([1], dtype=np.int64),
                        "speed": np.array([1.0], dtype=np.float32),
                        "pitch_shift": np.array([0.0], dtype=np.float32),
                        "pitch_range": np.array([1.0], dtype=np.float32),
                        "noise": noise,
                    },
                )
                audio_arr = np.asarray(audio, dtype=np.float32)
                self.assertEqual(audio_arr.shape, tuple(reference.shape))
                self.assertLess(
                    float(np.abs(audio_arr - reference.numpy()).max()),
                    2e-2 * float(reference.abs().max()),
                )

            # Direct export_model without external noise also works.
            direct = export_model(model, root / "direct.onnx", external_noise=False)
            self.assertTrue(direct.is_file())
