"""PyTorch and ONNX Runtime inference APIs and CLI."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from .frontend import (
    SYMBOLS,
    NothingToPronounce,
    PhoneSequence,
    StressProvider,
    ensure_terminal_punctuation,
    normalize,
    phonemize,
    split_clause,
    split_sentences,
)

if TYPE_CHECKING:
    from .model import TinyVoca

PEAK = 0.99
DEFAULT_HF_REPO_ID = "egnch/tiny-voca-ru"
BackendName = Literal["auto", "torch", "onnx"]

_HF_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?/[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)
_LOCAL_MODEL_SUFFIXES = (".onnx", ".safetensors", ".pt", ".pth", ".json")


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


def _validate_synthesis_params(speed: float, pitch_shift: float, pitch_range: float) -> None:
    if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    if (
        not math.isfinite(pitch_shift)
        or not math.isfinite(pitch_range)
        or not 0.0 <= pitch_range <= 3.0
    ):
        raise ValueError("pitch_shift must be finite and pitch_range between 0.0 and 3.0")


class _BaseSynthesizer:
    sample_rate: int
    voices: tuple[str, ...]
    max_tokens: int
    max_frames: int

    def _voice_index(self, voice: str | int) -> int:
        if isinstance(voice, bool):
            raise ValueError(f"voice must be a speaker name or integer index, got {voice!r}")
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
        _validate_synthesis_params(speed, pitch_shift, pitch_range)
        if not math.isfinite(pause_seconds) or pause_seconds < 0.0:
            raise ValueError("pause_seconds must be finite and non-negative")
        if peak is not None and (not math.isfinite(peak) or peak <= 0.0):
            raise ValueError("peak must be finite and positive")
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
                phones = phonemize(piece, stress=stress, normalize="none")
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
                        seed=seed + len(spoken),
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

        for raw_sentence in split_sentences(ensure_terminal_punctuation(text)):
            normalized = normalize(raw_sentence)
            for sentence in split_sentences(normalized):
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

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path = DEFAULT_HF_REPO_ID,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        providers: list[str] | None = None,
        subfolder: str | None = None,
    ) -> OnnxSynthesizer:
        """Load an ONNX synthesizer from a local directory or Hugging Face Hub repository."""
        target = _resolve_pretrained_onnx(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        return cls(target, providers=providers)

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
        _validate_synthesis_params(speed, pitch_shift, pitch_range)
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
        try:
            outputs = self.session.run(None, feed)
        except Exception as err:
            msg = str(err).lower()
            if "noise" in self._input_names and ("broadcast" in msg or "invalid dimensions" in msg):
                raise ValueError(f"utterance exceeds max_frames ({self.max_frames})") from err
            raise
        if len(outputs) > 1 and int(np.asarray(outputs[1]).sum()) > self.max_frames:
            raise ValueError(f"utterance exceeds max_frames ({self.max_frames})")
        audio = np.asarray(outputs[0], dtype=np.float32)[0]
        if not np.isfinite(audio).all():
            raise RuntimeError("ONNX model produced non-finite audio")
        return audio


class TorchSynthesizer(_BaseSynthesizer):
    """PyTorch inference engine backed by a safetensors checkpoint or ``TinyVoca`` instance."""

    def __init__(
        self, model_or_path: TinyVoca | str | Path, *, device: str = "cpu", enhance: bool = True
    ) -> None:
        if isinstance(model_or_path, (str, Path)):
            from .checkpoint import load_checkpoint

            model, _ = load_checkpoint(model_or_path, device=device)
        else:
            model = model_or_path.to(device).eval()
        self.model = model
        self.enhance = enhance
        self.sample_rate = model.config.sample_rate
        self.voices = model.config.speakers
        self.max_tokens = model.config.max_tokens
        self.max_frames = model.config.max_frames

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path = DEFAULT_HF_REPO_ID,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        subfolder: str | None = None,
    ) -> TorchSynthesizer:
        """Load a PyTorch synthesizer from a local directory or Hugging Face Hub repository."""
        target = _resolve_pretrained_torch(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        return cls(target, device=device)

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
            enhance=self.enhance,
        )
        if not torch.isfinite(audio).all():
            raise RuntimeError("model produced non-finite audio")
        return audio[0, : int(sample_lengths[0])].cpu().numpy().astype(np.float32, copy=False)


def _has_torch_backend() -> bool:
    return find_spec("torch") is not None and find_spec("safetensors") is not None


def _has_onnx_backend() -> bool:
    return find_spec("onnxruntime") is not None


def _is_hf_repo_id(value: str | Path) -> bool:
    if isinstance(value, Path):
        return False
    if not value or value.startswith((".", "/", "~")) or "\\" in value:
        return False
    if value.lower().endswith(_LOCAL_MODEL_SUFFIXES):
        return False
    return _HF_REPO_ID_RE.match(value) is not None


def _normalize_subfolder(subfolder: str | None) -> str:
    return subfolder.strip("/") if subfolder else ""


def _has_local_torch_files(target: Path) -> bool:
    if target.is_file():
        return target.suffix == ".safetensors"
    return (
        target.is_dir()
        and (target / "model.safetensors").is_file()
        and (target / "config.json").is_file()
    )


def _find_local_onnx_target(target: Path) -> Path | None:
    if target.is_file():
        return target if target.suffix == ".onnx" else None
    if not target.is_dir():
        return None
    if (target / "model.onnx").is_file() and (target / "model.json").is_file():
        return target
    nested = target / "onnx"
    if (nested / "model.onnx").is_file() and (nested / "model.json").is_file():
        return nested
    return None


def _resolve_onnx_paths(path: str | Path) -> tuple[Path, Path]:
    target = Path(path)
    if target.is_dir():
        if (target / "model.onnx").is_file() or not (target / "onnx" / "model.onnx").is_file():
            model_path = target / "model.onnx"
            meta_path = target / "model.json"
        else:
            model_path = target / "onnx" / "model.onnx"
            meta_path = target / "onnx" / "model.json"
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


def _snapshot_download(
    repo_id: str,
    *,
    allow_patterns: list[str],
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "huggingface-hub is required to download models from Hugging Face Hub; "
            'install with `pip install "tiny-voca-ru[onnx]"` or `pip install "tiny-voca-ru[torch]"`'
        ) from err
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
        local_files_only=local_files_only,
    )
    return Path(local_dir)


def _torch_allow_patterns(subfolder: str) -> list[str]:
    if subfolder:
        return [f"{subfolder}/model.safetensors", f"{subfolder}/config.json"]
    return ["model.safetensors", "config.json"]


def _onnx_allow_patterns(subfolder: str) -> list[str]:
    if subfolder:
        return [f"{subfolder}/model.onnx", f"{subfolder}/model.json"]
    return ["onnx/model.onnx", "onnx/model.json", "model.onnx", "model.json"]


def _resolve_pretrained_torch(
    repo_id: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> Path:
    sub = _normalize_subfolder(subfolder)
    candidate = Path(repo_id)
    if candidate.exists():
        target = candidate / sub if sub else candidate
        if not target.exists():
            raise FileNotFoundError(f"model path does not exist: {target}")
        return target
    if isinstance(repo_id, Path):
        raise FileNotFoundError(f"model path does not exist: {repo_id}")
    snapshot = _snapshot_download(
        repo_id,
        allow_patterns=_torch_allow_patterns(sub),
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    target = snapshot / sub if sub else snapshot
    if not _has_local_torch_files(target):
        raise FileNotFoundError(
            f"safetensors checkpoint (model.safetensors + config.json) not found in {repo_id!r}"
        )
    return target


def _resolve_pretrained_onnx(
    repo_id: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> Path:
    sub = _normalize_subfolder(subfolder)
    candidate = Path(repo_id)
    if candidate.exists():
        target = candidate / sub if sub else candidate
        onnx_target = _find_local_onnx_target(target)
        if onnx_target is None:
            raise FileNotFoundError(f"ONNX export (model.onnx + model.json) not found in: {target}")
        return onnx_target
    if isinstance(repo_id, Path):
        raise FileNotFoundError(f"model path does not exist: {repo_id}")
    snapshot = _snapshot_download(
        repo_id,
        allow_patterns=_onnx_allow_patterns(sub),
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    target = snapshot / sub if sub else snapshot
    onnx_target = _find_local_onnx_target(target)
    if onnx_target is None:
        raise FileNotFoundError(
            f"ONNX export (model.onnx + model.json) not found in Hugging Face repo {repo_id!r}"
        )
    return onnx_target


def _validate_backend_options(backend: BackendName, providers: list[str] | None) -> None:
    if backend not in ("auto", "torch", "onnx"):
        raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'torch', or 'onnx'")
    if backend == "torch" and providers is not None:
        raise ValueError("providers is only supported with the ONNX backend")


def _load_local_backend(
    target: Path,
    *,
    backend: BackendName,
    device: str,
    providers: list[str] | None,
) -> _BaseSynthesizer:
    if backend == "onnx":
        onnx_target = _find_local_onnx_target(target)
        if onnx_target is None:
            raise FileNotFoundError(f"ONNX export (model.onnx + model.json) not found in: {target}")
        return OnnxSynthesizer(onnx_target, providers=providers)
    if backend == "torch":
        return TorchSynthesizer(target, device=device)
    if providers is not None:
        onnx_target = _find_local_onnx_target(target)
        if onnx_target is None:
            raise FileNotFoundError(f"ONNX export (model.onnx + model.json) not found in: {target}")
        return OnnxSynthesizer(onnx_target, providers=providers)
    if _is_onnx_path(target):
        return OnnxSynthesizer(target)
    has_torch_files = _has_local_torch_files(target)
    onnx_target = _find_local_onnx_target(target)
    if has_torch_files and (device != "cpu" or _has_torch_backend()):
        return TorchSynthesizer(target, device=device)
    if onnx_target is not None and _has_onnx_backend():
        return OnnxSynthesizer(onnx_target)
    if has_torch_files:
        return TorchSynthesizer(target, device=device)
    if onnx_target is not None:
        return OnnxSynthesizer(onnx_target)
    return TorchSynthesizer(target, device=device)


def _load_hf_backend(
    repo_id: str,
    *,
    backend: BackendName,
    device: str,
    providers: list[str] | None,
    revision: str | None,
    cache_dir: str | Path | None,
    token: str | bool | None,
    local_files_only: bool,
    subfolder: str | None,
) -> _BaseSynthesizer:
    sub = _normalize_subfolder(subfolder)
    if backend == "torch":
        target = _resolve_pretrained_torch(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=sub,
        )
        return TorchSynthesizer(target, device=device)
    if backend == "onnx":
        onnx_target = _resolve_pretrained_onnx(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=sub,
        )
        return OnnxSynthesizer(onnx_target, providers=providers)

    has_torch = _has_torch_backend()
    has_onnx = _has_onnx_backend()
    if providers is not None:
        if not has_onnx:
            raise ImportError(
                'onnxruntime is required when providers are specified; install "tiny-voca-ru[onnx]"'
            )
        candidates: list[Literal["torch", "onnx"]] = ["onnx"]
    elif device != "cpu":
        if not has_torch:
            raise ImportError(
                "PyTorch and safetensors are required for non-CPU devices; "
                'install "tiny-voca-ru[torch]"'
            )
        candidates = ["torch"]
    else:
        candidates = []
        if has_torch:
            candidates.append("torch")
        if has_onnx:
            candidates.append("onnx")
        if not candidates:
            raise ImportError(
                "No inference backend installed; "
                'install `pip install "tiny-voca-ru[onnx]"` '
                'or `pip install "tiny-voca-ru[torch]"`'
            )

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if candidate == "torch":
                target = _resolve_pretrained_torch(
                    repo_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    subfolder=sub,
                )
                return TorchSynthesizer(target, device=device)
            onnx_target = _resolve_pretrained_onnx(
                repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                subfolder=sub,
            )
            return OnnxSynthesizer(onnx_target, providers=providers)
        except FileNotFoundError as err:
            last_error = err
        except Exception as err:
            if local_files_only and type(err).__name__ == "LocalEntryNotFoundError":
                last_error = err
                continue
            raise
    assert last_error is not None
    raise last_error


class Synthesizer(_BaseSynthesizer):
    """Auto-detecting TTS synthesizer supporting local checkpoints and Hugging Face Hub repos."""

    def __init__(
        self,
        path: str | Path,
        *,
        backend: BackendName = "auto",
        device: str = "cpu",
        providers: list[str] | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        subfolder: str | None = None,
    ) -> None:
        _validate_backend_options(backend, providers)
        sub = _normalize_subfolder(subfolder)
        target = Path(path)
        self._backend: _BaseSynthesizer
        if target.exists():
            resolved = target / sub if sub else target
            if not resolved.exists():
                raise FileNotFoundError(f"model path does not exist: {resolved}")
            self._backend = _load_local_backend(
                resolved, backend=backend, device=device, providers=providers
            )
        elif _is_hf_repo_id(path):
            assert isinstance(path, str)
            self._backend = _load_hf_backend(
                path,
                backend=backend,
                device=device,
                providers=providers,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                subfolder=sub or None,
            )
        else:
            raise FileNotFoundError(f"model path does not exist: {target}")
        self.sample_rate = self._backend.sample_rate
        self.voices = self._backend.voices
        self.max_tokens = self._backend.max_tokens
        self.max_frames = self._backend.max_frames

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path = DEFAULT_HF_REPO_ID,
        *,
        backend: BackendName = "auto",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        providers: list[str] | None = None,
        subfolder: str | None = None,
    ) -> Synthesizer:
        """Load a synthesizer from Hugging Face Hub or a local path."""
        _validate_backend_options(backend, providers)
        target = Path(repo_id)
        if target.exists():
            return cls(
                target,
                backend=backend,
                device=device,
                providers=providers,
                subfolder=subfolder,
            )
        if isinstance(repo_id, Path):
            raise FileNotFoundError(f"model path does not exist: {repo_id}")
        instance = cls.__new__(cls)
        instance._backend = _load_hf_backend(
            repo_id,
            backend=backend,
            device=device,
            providers=providers,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        instance.sample_rate = instance._backend.sample_rate
        instance.voices = instance._backend.voices
        instance.max_tokens = instance._backend.max_tokens
        instance.max_frames = instance._backend.max_frames
        return instance

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
        default=DEFAULT_HF_REPO_ID,
        help=f"Local checkpoint/ONNX path or Hugging Face repo ID (default: {DEFAULT_HF_REPO_ID})",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "onnx"),
        default="auto",
        help="Inference backend (default: auto)",
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
    parser.add_argument("--stress", choices=("lite", "full", "none"), default="lite")
    parser.add_argument(
        "--pause", type=float, default=0.0, help="Silence gap between sentences in seconds"
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device (ignored for ONNX backend)")
    parser.add_argument("--revision", default=None, help="Hugging Face Hub branch, tag, or commit")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hugging Face cache directory")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files without network access",
    )
    args = parser.parse_args()

    synth = Synthesizer(
        args.model,
        backend=args.backend,
        device=args.device,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
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
