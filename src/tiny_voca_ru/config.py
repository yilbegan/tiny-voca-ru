"""Model and acoustic feature configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .frontend.symbols import SYMBOLS


@dataclass(frozen=True)
class Config:
    """Architecture and acoustic settings for ``TinyVoca``."""

    # Acoustic features
    sample_rate: int = 24000
    hop: int = 256
    n_fft: int = 1024
    n_mels: int = 80
    f0_min: float = 50.0
    f0_max: float = 600.0
    speakers: tuple[str, ...] = ("F", "M")
    # Russian phoneme inventory size (tiny_voca_ru.frontend.symbols.SYMBOLS)
    vocab_size: int = 61
    # Text encoder and frame-level transformers
    channels: int = 192
    heads: int = 4
    feedforward: int = 640
    kernel: int = 7
    encoder_layers: int = 4
    prosody_layers: int = 2
    acoustic_layers: int = 5
    dropout: float = 0.1
    predictor_channels: int = 128
    # Source-filter waveform head
    synth_fft: int = 512
    synth_hop: int = 128
    noise_fft: int = 256
    noise_hop: int = 64
    cep_causal: int = 80
    cep_anti: int = 16
    noise_cep: int = 32
    harmonic_limit: float = 0.47
    max_tokens: int = 512
    max_frames: int = 3000
    mel_mean: float = -5.0
    mel_scale: float = 5.0
    energy_mean: float = -4.0
    energy_scale: float = 2.0
    pitch_reference_hz: float = 200.0

    def __post_init__(self) -> None:
        if self.vocab_size != len(SYMBOLS):
            raise ValueError(f"vocab_size must equal the phoneme inventory size {len(SYMBOLS)}")
        if self.channels % self.heads or (self.channels // self.heads) % 2:
            raise ValueError("channels must be divisible by heads with an even head width")
        if not self.speakers or len(set(self.speakers)) != len(self.speakers):
            raise ValueError("speaker names must be unique and nonempty")
        for fft, hop in ((self.synth_fft, self.synth_hop), (self.noise_fft, self.noise_hop)):
            if fft != 4 * hop or self.hop % hop:
                raise ValueError(
                    "synthesis windows need 4x overlap and a hop dividing the acoustic hop"
                )
        if (
            self.cep_causal + self.cep_anti >= self.synth_fft // 2
            or self.noise_cep >= self.noise_fft // 2
        ):
            raise ValueError("cepstrum orders must be well below half the synthesis FFT size")
        if (
            min(self.encoder_layers, self.prosody_layers, self.acoustic_layers, self.kernel) < 1
            or not self.kernel % 2
        ):
            raise ValueError("layer counts must be positive and the kernel odd")
        if min(self.mel_scale, self.energy_scale) <= 0 or not 0 < self.harmonic_limit < 0.5:
            raise ValueError("invalid normalization scale or harmonic limit")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["speakers"] = list(self.speakers)
        return data

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Config:
        return cls(**{**value, "speakers": tuple(value["speakers"])})

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Config:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "config" in data and isinstance(data["config"], dict):
            data = data["config"]
        return cls.from_dict(data)
