"""Tests for safetensors checkpoints, legacy .pt conversion, and PyTorch/ONNX inference."""

from __future__ import annotations

import fnmatch
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from tiny_voca_ru import DEFAULT_HF_REPO_ID
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
from tiny_voca_ru.infer import (
    OnnxSynthesizer,
    Synthesizer,
    TorchSynthesizer,
    _is_hf_repo_id,
)
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

    def test_from_pretrained_local_combined_repo_and_backends(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "hf_repo"
            save_checkpoint(repo_dir, model, step=12)
            export_onnx(repo_dir, repo_dir / "onnx", external_noise=True)

            torch_synth = Synthesizer.from_pretrained(repo_dir, backend="torch")
            self.assertIsInstance(torch_synth._backend, TorchSynthesizer)
            self.assertGreater(torch_synth.synthesize("Привет!", voice="F").audio.size, 0)

            onnx_synth = Synthesizer.from_pretrained(repo_dir, backend="onnx")
            self.assertIsInstance(onnx_synth._backend, OnnxSynthesizer)
            self.assertGreater(onnx_synth.synthesize("Привет!", voice="M").audio.size, 0)

            direct_onnx = OnnxSynthesizer.from_pretrained(repo_dir)
            self.assertGreater(direct_onnx.synthesize("Привет!", voice="F").audio.size, 0)

            direct_torch = TorchSynthesizer.from_pretrained(repo_dir)
            self.assertGreater(direct_torch.synthesize("Привет!", voice="M").audio.size, 0)

            with patch("tiny_voca_ru.infer._has_torch_backend", return_value=False):
                auto_onnx = Synthesizer.from_pretrained(repo_dir, backend="auto")
            self.assertIsInstance(auto_onnx._backend, OnnxSynthesizer)

            with self.assertRaises(ValueError):
                Synthesizer.from_pretrained(
                    repo_dir, backend="torch", providers=["CPUExecutionProvider"]
                )
            with self.assertRaises(ValueError):
                Synthesizer.from_pretrained(repo_dir, backend="invalid")  # type: ignore[arg-type]

    def test_from_pretrained_hf_hub_dispatch_and_fallbacks(self) -> None:
        self.assertEqual(DEFAULT_HF_REPO_ID, "egnch/tiny-voca-ru")
        self.assertTrue(_is_hf_repo_id("egnch/tiny-voca-ru"))
        self.assertFalse(_is_hf_repo_id(Path("egnch/tiny-voca-ru")))
        self.assertFalse(_is_hf_repo_id("./egnch/tiny-voca-ru"))
        self.assertFalse(_is_hf_repo_id("checkpoints/model.onnx"))
        self.assertFalse(_is_hf_repo_id("only-one-segment"))

        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote_full = root / "remote_full"
            save_checkpoint(remote_full, model, step=12)
            export_onnx(remote_full, remote_full / "onnx", external_noise=True)

            remote_onnx_only = root / "remote_onnx_only"
            export_onnx(remote_full, remote_onnx_only, external_noise=True)

            remote_sub = root / "remote_sub"
            save_checkpoint(remote_sub / "v2", model, step=12)
            export_onnx(remote_sub / "v2", remote_sub / "v2", external_noise=True)

            calls: list[dict[str, Any]] = []

            def make_fake_snapshot(source: Path, *, fail_torch_offline: bool = False) -> Any:
                call_idx = 0

                def fake_snapshot_download(
                    *,
                    repo_id: str,
                    allow_patterns: list[str],
                    revision: str | None = None,
                    cache_dir: str | None = None,
                    token: str | bool | None = None,
                    local_files_only: bool = False,
                ) -> str:
                    nonlocal call_idx
                    call_idx += 1
                    calls.append(
                        {
                            "repo_id": repo_id,
                            "allow_patterns": list(allow_patterns),
                            "revision": revision,
                            "cache_dir": cache_dir,
                            "token": token,
                            "local_files_only": local_files_only,
                        }
                    )
                    if (
                        fail_torch_offline
                        and local_files_only
                        and "model.safetensors" in allow_patterns
                    ):

                        class LocalEntryNotFoundError(OSError):
                            pass

                        raise LocalEntryNotFoundError("not cached")
                    out = root / f"snap_{source.name}_{call_idx}"
                    out.mkdir(parents=True, exist_ok=True)
                    for src_file in source.rglob("*"):
                        if not src_file.is_file():
                            continue
                        rel = src_file.relative_to(source).as_posix()
                        if any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                            dest = out / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_file, dest)
                    return str(out)

                return fake_snapshot_download

            with patch(
                "huggingface_hub.snapshot_download", side_effect=make_fake_snapshot(remote_full)
            ):
                # Default repo_id + torch backend downloads only safetensors + config.json
                s_torch = Synthesizer.from_pretrained(
                    backend="torch", revision="v1", cache_dir=root / "cache"
                )
                self.assertIsInstance(s_torch._backend, TorchSynthesizer)
                self.assertEqual(calls[-1]["repo_id"], DEFAULT_HF_REPO_ID)
                self.assertEqual(calls[-1]["revision"], "v1")
                self.assertEqual(calls[-1]["allow_patterns"], ["model.safetensors", "config.json"])
                self.assertGreater(s_torch.synthesize("Привет!", voice="F").audio.size, 0)

                # Direct Synthesizer("egnch/tiny-voca-ru", backend="onnx") downloads only ONNX files
                s_onnx = Synthesizer("egnch/tiny-voca-ru", backend="onnx")
                self.assertIsInstance(s_onnx._backend, OnnxSynthesizer)
                self.assertIn("onnx/model.onnx", calls[-1]["allow_patterns"])
                self.assertNotIn("model.safetensors", calls[-1]["allow_patterns"])
                self.assertGreater(s_onnx.synthesize("Привет!", voice="M").audio.size, 0)

                # Auto backend without torch installed picks ONNX without downloading safetensors
                with patch("tiny_voca_ru.infer._has_torch_backend", return_value=False):
                    s_auto_no_torch = Synthesizer.from_pretrained("egnch/tiny-voca-ru")
                self.assertIsInstance(s_auto_no_torch._backend, OnnxSynthesizer)
                self.assertNotIn("model.safetensors", calls[-1]["allow_patterns"])

                # Local nonexistent paths never touch HF Hub
                calls_before = len(calls)
                with self.assertRaises(FileNotFoundError):
                    Synthesizer(root / "missing")
                with self.assertRaises(FileNotFoundError):
                    Synthesizer("./missing/dir")
                with self.assertRaises(FileNotFoundError):
                    Synthesizer("missing-single-segment")
                self.assertEqual(len(calls), calls_before)

            # Auto backend falls back from torch to ONNX on an ONNX-only HF repo
            with patch(
                "huggingface_hub.snapshot_download",
                side_effect=make_fake_snapshot(remote_onnx_only),
            ):
                s_fallback = Synthesizer.from_pretrained("org/onnx-only-model")
                self.assertIsInstance(s_fallback._backend, OnnxSynthesizer)
                self.assertGreater(s_fallback.synthesize("Привет!", voice="F").audio.size, 0)

            # Offline local_files_only=True falls back to cached ONNX when safetensors isn't cached
            with patch(
                "huggingface_hub.snapshot_download",
                side_effect=make_fake_snapshot(remote_full, fail_torch_offline=True),
            ):
                s_offline = Synthesizer.from_pretrained("egnch/tiny-voca-ru", local_files_only=True)
                self.assertIsInstance(s_offline._backend, OnnxSynthesizer)

            # Subfolder support for both Torch and ONNX
            with patch(
                "huggingface_hub.snapshot_download", side_effect=make_fake_snapshot(remote_sub)
            ):
                s_sub_torch = TorchSynthesizer.from_pretrained("org/sub-model", subfolder="v2")
                self.assertEqual(
                    calls[-1]["allow_patterns"], ["v2/model.safetensors", "v2/config.json"]
                )
                self.assertGreater(s_sub_torch.synthesize("Привет!", voice="F").audio.size, 0)

                s_sub_onnx = OnnxSynthesizer.from_pretrained("org/sub-model", subfolder="v2")
                self.assertEqual(calls[-1]["allow_patterns"], ["v2/model.onnx", "v2/model.json"])
                self.assertGreater(s_sub_onnx.synthesize("Привет!", voice="M").audio.size, 0)

    def test_post_normalization_sentence_boundary_guard(self) -> None:
        model = small_model()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = save_checkpoint(Path(tmp) / "ckpt", model, step=5)
            synth = Synthesizer(ckpt_dir)

            # Sentence-final abbreviations that expand with a period must split cleanly
            # without leaking a mid-utterance period into a single utterance.
            split_cases = [
                "Книги, журналы и т.д. Следующее предложение.",
                "Билет стоит 5 тыс. руб. Это недорого.",
                "Он жил в 1999 г. Потом уехал.",
                "В XIX в. Люди жили иначе.",
                "Книги и т.д. Подробности на рис. 3.",
            ]
            for text in split_cases:
                res = synth.synthesize(text, voice="F")
                self.assertEqual(len(res.utterances), 2, text)
                for utt in res.utterances:
                    interior = utt.symbols[1:-2]
                    self.assertNotIn(".", interior, f"mid-utterance period in {text!r}: {utt}")
                    self.assertIn(utt.symbols[-2], (".", "!", "?", "…"), text)

            # Initials stay inside their sentence while sentence-final year abbreviation splits.
            pushkin = synth.synthesize(
                "А. С. Пушкин родился в 1799 г. Он писал стихи.", voice="F"
            )
            self.assertEqual(len(pushkin.utterances), 2)

            # Unexpanded abbreviations before lowercase and inline quotes stay in one utterance.
            for single_text in (
                "Цена 100 руб. и 50 коп.",
                "Он стал гл. редактором.",
                "На ст. метро Арбатская было тихо.",
                "Он крикнул «Стой!» и побежал.",
                "Он подумал… и ушёл.",
            ):
                single = synth.synthesize(single_text, voice="F")
                self.assertEqual(len(single.utterances), 1, single_text)
                self.assertNotIn(".", single.utterances[0].symbols[1:-2], single_text)
            # No double-normalization: normalize() is called once per raw sentence.
            with patch("tiny_voca_ru.infer.normalize", wraps=__import__(
                "tiny_voca_ru.frontend", fromlist=["normalize"]
            ).normalize) as spy:
                res = synth.synthesize("Цена 100 руб. Второе предложение.", voice="F")
                self.assertEqual(len(res.utterances), 2)
                self.assertEqual(spy.call_count, 1)
