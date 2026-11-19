"""PyTorch and ONNX Runtime inference APIs and CLI."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .frontend import (
    SYMBOLS,
    NothingToPronounce,
    PhoneSequence,
    StressProvider,
    ensure_terminal_punctuation,
    phonemize,
    split_clause,
    split_sentences,
)

if TYPE_CHECKING:
    from .model import TinyVoca

PEAK = 0.99


@dataclass(frozen=True)
class SynthesisResult:
    """Output of a synthesis call."""

    audio: NDArray[np.float32]
    sample_rate: int
    utterances: tuple[PhoneSequence, ...]


def limit_peak(wave: NDArray[np.float32], peak: float = PEAK) -> NDArray[np.float32]:
    """Scale ``wave`` down uniformly when its loudest sample exceeds ``peak``."""
    if wave.size == 0:
        return wave
    loudest = float(np.abs(wave).max())
    return (
        (wave * np.float32(peak / loudest)).astype(np.float32, copy=False)
        if loudest > peak
        else wave
    )


class _BaseSynthesizer:
    sample_rate: int
    voices: tuple[str, ...]
    max_tokens: int
    max_frames: int

    def _voice_index(self, voice: str | int) -> int:
        if isinstance(voice, int):
            if not 0 <= voice < len(self.voices):
                raise ValueError(f"voice index {voice} out of range for {self.voices}")
            return voice
        if voice not in self.voices:
            raise ValueError(f"unknown voice {voice!r}; expected one of {self.voices}")
        return self.voices.index(voice)

    def speak_utterance(
        self,
        phones: PhoneSequence,
        *,
        voice: str | int = 0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.0,
        seed: int = 0,
    ) -> NDArray[np.float32]:
        raise NotImplementedError

    def synthesize(
        self,
        text: str,
        *,
        voice: str | int | None = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.0,
        seed: int = 0,
        stress: str | StressProvider = "lite",
        pause_seconds: float = 0.0,
        peak: float | None = PEAK,
    ) -> SynthesisResult:
        """Synthesize Russian text of any length into a float32 waveform."""
        if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        if not math.isfinite(pitch_shift) or not 0.0 <= pitch_range <= 3.0:
            raise ValueError("pitch_shift must be finite and pitch_range between 0.0 and 3.0")
        selected_voice: str | int = self.voices[0] if voice is None else voice
        limit = self.max_tokens - 12
        waves: list[NDArray[np.float32]] = []
        spoken: list[PhoneSequence] = []
        gap = (
            np.zeros(int(round(pause_seconds * self.sample_rate)), dtype=np.float32)
            if pause_seconds > 0
            else None
        )

        def read(piece: str) -> None:
            try:
                phones = phonemize(piece, stress=stress)
            except NothingToPronounce:
                return
            if len(phones) <= limit:
                try:
                    chunk = self.speak_utterance(
                        phones,
                        voice=selected_voice,
                        speed=speed,
                        pitch_shift=pitch_shift,
                        pitch_range=pitch_range,
                        seed=seed + len(waves),
                    )
                except ValueError as err:
                    if "max_frames" not in str(err):
                        raise
                else:
                    if gap is not None and waves:
                        waves.append(gap)
                    waves.append(chunk)
                    spoken.append(phones)
                    return
            pieces = split_clause(piece)
            if not pieces:
                raise ValueError(f"one word is too long to speak: {piece[:40]!r}")
            for part in pieces:
                read(part)

        for sentence in split_sentences(ensure_terminal_punctuation(text)):
            read(sentence)
        if not waves:
            raise NothingToPronounce("nothing in the text could be pronounced")
        audio = np.concatenate(waves).astype(np.float32, copy=False)
        if peak is not None:
            audio = limit_peak(audio, peak)
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate, utterances=tuple(spoken))

    def save_wav(
        self,
        text: str,
        output_path: str | Path,
        *,
        voice: str | int | None = None,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.0,
        seed: int = 0,
        stress: str | StressProvider = "lite",
        pause_seconds: float = 0.0,
        peak: float | None = PEAK,
    ) -> SynthesisResult:
        """Synthesize ``text`` and save float32 WAV audio to ``output_path``."""
        import soundfile as sf

        result = self.synthesize(
            text,
            voice=voice,
            speed=speed,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            seed=seed,
            stress=stress,
            pause_seconds=pause_seconds,
            peak=peak,
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, result.audio, result.sample_rate, subtype="FLOAT")
        return result


class OnnxSynthesizer(_BaseSynthesizer):
    """Lightweight ONNX Runtime inference engine (no PyTorch dependency)."""

    def __init__(self, path: str | Path, *, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        model_path, meta_path = _resolve_onnx_paths(path)
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        if list(meta.get("symbols", ())) != list(SYMBOLS):
            raise ValueError(
                "ONNX model symbols do not match tiny_voca_ru.frontend.symbols.SYMBOLS"
            )
        self.sample_rate = int(meta["sample_rate"])
        self.hop = int(meta["hop"])
        self.voices = tuple(str(v) for v in meta["voices"])
        self.max_tokens = int(meta["max_tokens"])
        self.max_frames = int(meta["max_frames"])
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def speak_utterance(
        self,
        phones: PhoneSequence,
        *,
        voice: str | int = 0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.0,
        seed: int = 0,
    ) -> NDArray[np.float32]:
        if len(phones) > self.max_tokens:
            raise ValueError("utterance exceeds max_tokens")
        feed: dict[str, Any] = {
            "tokens": np.array([phones.ids], dtype=np.int64),
            "stress": np.array([phones.stress], dtype=np.int64),
            "voice": np.array([self._voice_index(voice)], dtype=np.int64),
            "speed": np.array([speed], dtype=np.float32),
            "pitch_shift": np.array([pitch_shift], dtype=np.float32),
            "pitch_range": np.array([pitch_range], dtype=np.float32),
        }
        if "noise" in self._input_names:
            rng = np.random.default_rng(seed)
            feed["noise"] = rng.standard_normal((1, self.hop * self.max_frames)).astype(np.float32)
        outputs = self.session.run(None, feed)
        audio = np.asarray(outputs[0], dtype=np.float32)[0]
        if not np.isfinite(audio).all():
            raise RuntimeError("ONNX model produced non-finite audio")
        return audio


class TorchSynthesizer(_BaseSynthesizer):
    """PyTorch inference engine backed by a safetensors checkpoint or ``TinyVoca`` instance."""

    def __init__(self, model_or_path: TinyVoca | str | Path, *, device: str = "cpu") -> None:
        if isinstance(model_or_path, (str, Path)):
            from .checkpoint import load_checkpoint

            model, _ = load_checkpoint(model_or_path, device=device)
        else:
            model = model_or_path.to(device).eval()
        self.model = model
        self.sample_rate = model.config.sample_rate
        self.voices = model.config.speakers
        self.max_tokens = model.config.max_tokens
        self.max_frames = model.config.max_frames

    def speak_utterance(
        self,
        phones: PhoneSequence,
        *,
        voice: str | int = 0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.0,
        seed: int = 0,
    ) -> NDArray[np.float32]:
        import torch

        if len(phones) > self.max_tokens:
            raise ValueError("utterance exceeds max_tokens")
        device = next(self.model.parameters()).device
        tokens = torch.tensor([phones.ids], dtype=torch.long, device=device)
        stress = torch.tensor([phones.stress], dtype=torch.long, device=device)
        lengths = torch.tensor([len(phones)], dtype=torch.long, device=device)
        speaker = torch.tensor([self._voice_index(voice)], dtype=torch.long, device=device)
        audio, sample_lengths, _ = self.model.synthesize(
            tokens,
            stress,
            lengths,
            speaker,
            speed=speed,
            pitch_shift=pitch_shift,
            seed=seed,
            pitch_range=pitch_range,
        )
        if not torch.isfinite(audio).all():
            raise RuntimeError("model produced non-finite audio")
        return audio[0, : int(sample_lengths[0])].cpu().numpy().astype(np.float32, copy=False)


def _resolve_onnx_paths(path: str | Path) -> tuple[Path, Path]:
    target = Path(path)
    if target.is_dir():
        model_path = target / "model.onnx"
        meta_path = target / "model.json"
    else:
        model_path = target
        meta_path = target.with_suffix(".json")
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"ONNX export must contain {model_path.name} and {meta_path.name}: {path}"
        )
    return model_path, meta_path


def _is_onnx_path(path: Path) -> bool:
    return path.suffix == ".onnx" or (path.is_dir() and (path / "model.onnx").is_file())


class Synthesizer(_BaseSynthesizer):
    """Auto-detecting TTS synthesizer supporting both safetensors (PyTorch) and ONNX exports."""

    def __init__(self, path: str | Path, *, device: str = "cpu") -> None:
        target = Path(path)
        self._backend: _BaseSynthesizer
        if _is_onnx_path(target):
            self._backend = OnnxSynthesizer(target)
        else:
            self._backend = TorchSynthesizer(target, device=device)
        self.sample_rate = self._backend.sample_rate
        self.voices = self._backend.voices
        self.max_tokens = self._backend.max_tokens
        self.max_frames = self._backend.max_frames

    def speak_utterance(
        self,
        phones: PhoneSequence,
        *,
        voice: str | int = 0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        pitch_range: float = 1.0,
        seed: int = 0,
    ) -> NDArray[np.float32]:
        return self._backend.speak_utterance(
            phones,
            voice=voice,
            speed=speed,
            pitch_shift=pitch_shift,
            pitch_range=pitch_range,
            seed=seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Russian speech with tiny-voca-ru (PyTorch or ONNX)"
    )
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=Path,
        required=True,
        help="Checkpoint dir or ONNX dir/file",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Russian text to synthesize")
    source.add_argument("--text-file", type=Path, help="UTF-8 text file to synthesize")
    parser.add_argument("--output", type=Path, required=True, help="Output WAV path")
    parser.add_argument(
        "--voice", default=None, help="Voice name (default: first speaker in model config)"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--pitch-shift", type=float, default=0.0, help="Log-F0 offset (0.058 ≈ +1 semitone)"
    )
    parser.add_argument(
        "--pitch-range", type=float, default=1.0, help="Pitch range scale around utterance mean"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stress", choices=("lite", "ruaccent", "none"), default="lite")
    parser.add_argument(
        "--pause", type=float, default=0.0, help="Silence gap between sentences in seconds"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (ignored for ONNX backend)")
    args = parser.parse_args()

    synth = Synthesizer(args.model, device=args.device)
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    try:
        result = synth.save_wav(
            text,
            args.output,
            voice=args.voice,
            speed=args.speed,
            pitch_shift=args.pitch_shift,
            pitch_range=args.pitch_range,
            seed=args.seed,
            stress=args.stress,
            pause_seconds=args.pause,
        )
    except NothingToPronounce as err:
        parser.exit(1, f"{err}\n")

    for utt in result.utterances:
        print(" ".join(utt.symbols))
    seconds = len(result.audio) / result.sample_rate
    loudest = float(np.abs(result.audio).max()) if result.audio.size else 0.0
    print(
        f"Saved {seconds:.2f}s ({len(result.utterances)} utterance(s), "
        f"peak={loudest:.4f}) -> {args.output}"
    )


if __name__ == "__main__":
    main()
