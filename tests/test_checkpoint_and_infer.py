"""Tests for safetensors checkpoints, legacy .pt conversion, and PyTorch/ONNX inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from tiny_voca_ru.checkpoint import (
    CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    convert_legacy_pt,
    load_checkpoint,
    save_checkpoint,
)
from tiny_voca_ru.config import Config
from tiny_voca_ru.export import export_onnx
from tiny_voca_ru.frontend import SYMBOLS, NothingToPronounce, phonemize
from tiny_voca_ru.infer import OnnxSynthesizer, Synthesizer, TorchSynthesizer
from tiny_voca_ru.model import TinyVoca


def small_model(speakers: tuple[str, ...] = ("F", "M")) -> TinyVoca:
    torch.manual_seed(0)
    cfg = Config(
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
        max_frames=512,
    )
    model = TinyVoca(cfg).eval()
    torch.nn.init.zeros_(model.duration.out.weight)
    model.duration.out.bias.data[:] = torch.tensor([1.5, 0.5])
    torch.nn.init.normal_(model.vocoder.harmonic.weight, std=0.05)
    torch.nn.init.normal_(model.vocoder.noise.weight, std=0.05)
    return model


class CheckpointAndInferTests(unittest.TestCase):
    def test_safetensors_save_load_and_validation(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp) / "ckpt"
            save_checkpoint(ckpt_dir, model, step=42)
            self.assertTrue((ckpt_dir / WEIGHTS_FILENAME).is_file())
            self.assertTrue((ckpt_dir / CONFIG_FILENAME).is_file())

            loaded, meta = load_checkpoint(ckpt_dir)
            self.assertEqual(meta.step, 42)
            self.assertEqual(meta.symbols, SYMBOLS)
            self.assertEqual(meta.config, model.config)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, loaded.state_dict()[k]), k)

            # Direct .safetensors path also loads cleanly.
            loaded_direct, _ = load_checkpoint(ckpt_dir / WEIGHTS_FILENAME)
            self.assertEqual(loaded_direct.config, model.config)

            # Corrupt symbols rejected.
            bad_cfg = json.loads((ckpt_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
            bad_cfg["symbols"] = list(SYMBOLS[:-1])
            (ckpt_dir / CONFIG_FILENAME).write_text(json.dumps(bad_cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_checkpoint(ckpt_dir)

    def test_convert_legacy_pt(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_pt = root / "legacy.pt"
            torch.save(
                {
                    "format_version": 4,
                    "step": 1000,
                    "config": model.config.to_dict(),
                    "frontend": {"symbols": list(SYMBOLS)},
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                },
                legacy_pt,
            )
            out_dir = convert_legacy_pt(legacy_pt, root / "converted")
            loaded, meta = load_checkpoint(out_dir)
            self.assertEqual(meta.step, 1000)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, loaded.state_dict()[k]), k)

    def test_torch_and_onnx_synthesizers(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = save_checkpoint(root / "ckpt", model, step=7)
            onnx_dir = root / "onnx"
            export_onnx(ckpt_dir, onnx_dir, external_noise=True)

            torch_synth = Synthesizer(ckpt_dir)
            self.assertIsInstance(torch_synth._backend, TorchSynthesizer)
            res_pt = torch_synth.synthesize("Привет, мир! Как дела?", voice="F", pause_seconds=0.1)
            self.assertEqual(res_pt.sample_rate, 24000)
            self.assertEqual(len(res_pt.utterances), 2)
            self.assertGreater(res_pt.audio.size, 0)
            self.assertLessEqual(float(np.abs(res_pt.audio).max()), 0.9901)

            wav_path = root / "out.wav"
            torch_synth.save_wav("Проверка связи.", wav_path, voice="M", stress="none")
            audio_disk, sr_disk = sf.read(wav_path, dtype="float32")
            self.assertEqual(sr_disk, 24000)
            self.assertGreater(len(audio_disk), 0)

            onnx_synth = Synthesizer(onnx_dir)
            self.assertIsInstance(onnx_synth._backend, OnnxSynthesizer)
            res_onnx = onnx_synth.synthesize("Привет, мир!", voice="M")
            self.assertEqual(res_onnx.sample_rate, 24000)
            self.assertGreater(res_onnx.audio.size, 0)

            with self.assertRaises(NothingToPronounce):
                torch_synth.synthesize("!!! 🙂")
            with self.assertRaises(ValueError):
                torch_synth.synthesize("Привет", voice="UNKNOWN")
            with self.assertRaises(ValueError):
                torch_synth.synthesize("Привет", voice=True)
            with self.assertRaises(ValueError):
                torch_synth.synthesize("Привет", pause_seconds=-0.1)
            with self.assertRaises(ValueError):
                torch_synth.synthesize("Привет", peak=0.0)

            # pause_seconds > 0 must not alter per-utterance RNG seeds.
            no_pause = torch_synth.synthesize("Привет! Пока!", voice="F", seed=42, peak=None)
            with_pause = torch_synth.synthesize(
                "Привет! Пока!", voice="F", seed=42, pause_seconds=0.05, peak=None
            )
            gap_samples = int(round(0.05 * torch_synth.sample_rate))
            self.assertEqual(with_pause.audio.size, no_pause.audio.size + gap_samples)
            self.assertTrue(np.array_equal(with_pause.audio[-2400:], no_pause.audio[-2400:]))

            # Direct OnnxSynthesizer.speak_utterance validates speed/pitch_range/bool voice.
            phones = phonemize("Привет.")
            with self.assertRaises(ValueError):
                onnx_synth.speak_utterance(phones, speed=0.0)
            with self.assertRaises(ValueError):
                onnx_synth.speak_utterance(phones, speed=-1.0)
            with self.assertRaises(ValueError):
                onnx_synth.speak_utterance(phones, pitch_range=-0.5)
            with self.assertRaises(ValueError):
                onnx_synth.speak_utterance(phones, voice=False)

            # Nonexistent model path raises FileNotFoundError before backend dispatch.
            with self.assertRaises(FileNotFoundError):
                Synthesizer(root / "does-not-exist")
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(root / "does-not-exist")

    def test_onnx_splits_on_max_frames_with_and_without_external_noise(self) -> None:
        model = small_model()
        long_sentence = (
            "Первый длинный фрагмент речи, второй длинный фрагмент речи, "
            "третий длинный фрагмент речи, четвёртый длинный фрагмент речи."
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = save_checkpoint(root / "ckpt", model, step=1)
            torch_res = Synthesizer(ckpt_dir).synthesize(long_sentence, voice="F")
            self.assertGreater(len(torch_res.utterances), 1)

            for ext_noise in (False, True):
                onnx_dir = root / f"onnx_{ext_noise}"
                export_onnx(ckpt_dir, onnx_dir, external_noise=ext_noise)
                onnx_res = Synthesizer(onnx_dir).synthesize(long_sentence, voice="F")
                self.assertEqual(len(onnx_res.utterances), len(torch_res.utterances))
                self.assertGreater(onnx_res.audio.size, 0)
