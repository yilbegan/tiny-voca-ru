"""ONNX export for single-utterance free-running synthesis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import CheckpointMeta, load_checkpoint
from .frontend.symbols import SYMBOLS
from .model import TinyVoca

INPUTS: tuple[str, ...] = ("tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range")
OUTPUTS: tuple[str, ...] = ("audio", "durations")


def _cepstrum_matrices(positions: list[int], fft: int) -> tuple[Tensor, Tensor]:
    buffer = torch.zeros(len(positions), fft, dtype=torch.float64)
    buffer[torch.arange(len(positions)), torch.tensor(positions)] = 1
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return spectrum.real.float(), spectrum.imag.float()


def _analysis_kernels(window: Tensor) -> tuple[Tensor, Tensor]:
    fft = len(window)
    spectrum = torch.fft.rfft(torch.eye(fft, dtype=torch.float64) * window.double(), dim=-1)
    return spectrum.real.T[:, None].float().contiguous(), spectrum.imag.T[
        :, None
    ].float().contiguous()


def _synthesis_kernel(window: Tensor) -> Tensor:
    fft = len(window)
    bins = fft // 2 + 1
    unit = torch.eye(bins, dtype=torch.complex128)
    real = torch.fft.irfft(unit, n=fft, dim=-1)
    imaginary = torch.fft.irfft(unit * 1j, n=fft, dim=-1)
    return (torch.cat((real, imaginary)) * window.double())[:, None].float().contiguous()


class Exportable(nn.Module):
    """Single-utterance ``TinyVoca`` wrapper rewritten in ONNX-compatible ops."""

    harmonic_real: Tensor
    harmonic_imaginary: Tensor
    noise_real: Tensor
    noise_imaginary: Tensor
    synth_analysis_real: Tensor
    synth_analysis_imaginary: Tensor
    synth_synthesis: Tensor
    synth_envelope: Tensor
    noise_analysis_real: Tensor
    noise_analysis_imaginary: Tensor
    noise_synthesis: Tensor
    noise_envelope: Tensor
    ramp: Tensor

    def __init__(self, model: TinyVoca, external_noise: bool = False) -> None:
        super().__init__()
        self.model = model
        self.external_noise = external_noise
        c = model.config
        v = model.vocoder
        causal = list(range(c.cep_causal))
        anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
        for name, (real, imaginary) in (
            ("harmonic", _cepstrum_matrices(causal + anti, c.synth_fft)),
            ("noise", _cepstrum_matrices(list(range(c.noise_cep)), c.noise_fft)),
        ):
            self.register_buffer(f"{name}_real", real, persistent=False)
            self.register_buffer(f"{name}_imaginary", imaginary, persistent=False)
        for name, window in (("synth", v.synth_window), ("noise", v.noise_window)):
            real, imaginary = _analysis_kernels(window)
            self.register_buffer(f"{name}_analysis_real", real, persistent=False)
            self.register_buffer(f"{name}_analysis_imaginary", imaginary, persistent=False)
            self.register_buffer(f"{name}_synthesis", _synthesis_kernel(window), persistent=False)
            self.register_buffer(
                f"{name}_envelope",
                (window.double() ** 2).float()[None, None].contiguous(),
                persistent=False,
            )
        self.register_buffer(
            "ramp", torch.arange(c.hop, dtype=torch.float32) / c.hop, persistent=False
        )
        self.phase_steps = 1 << 24

    def _filter(
        self, source: Tensor, real: Tensor, imaginary: Tensor, name: str, fft: int, hop: int
    ) -> Tensor:
        samples = source.shape[1]
        padded = F.pad(source[:, None], (fft // 2, fft))
        s_real = F.conv1d(padded, getattr(self, f"{name}_analysis_real"), stride=hop)
        s_imaginary = F.conv1d(padded, getattr(self, f"{name}_analysis_imaginary"), stride=hop)
        real, imaginary = real.transpose(1, 2), imaginary.transpose(1, 2)
        product = torch.cat(
            (s_real * real - s_imaginary * imaginary, s_real * imaginary + s_imaginary * real),
            dim=1,
        )
        wave = F.conv_transpose1d(product, getattr(self, f"{name}_synthesis"), stride=hop)
        envelope = F.conv_transpose1d(
            torch.ones_like(s_real[:, :1]), getattr(self, f"{name}_envelope"), stride=hop
        )
        return (wave / envelope)[:, 0, fft // 2 : fft // 2 + samples]

    @staticmethod
    def _response(cepstrum: Tensor, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        magnitude = torch.exp((cepstrum @ real).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _pulses(self, f0: Tensor) -> Tensor:
        """Float32 pulse train using 24-bit integer phase accumulation across frames."""
        c = self.model.config
        knots = torch.log(f0.clamp_min(1.0))
        knots = torch.cat((knots, knots[:, -1:]), dim=1)
        low, high = knots[:, :-1, None], knots[:, 1:, None]
        frequency = torch.exp(low * (1 - self.ramp) + high * self.ramp)
        within = torch.cumsum(frequency / c.sample_rate, dim=2)
        whole_frame = within[:, :, -1]
        steps = torch.round((whole_frame - torch.floor(whole_frame)) * self.phase_steps).long()
        start = (
            torch.remainder(torch.cumsum(steps, dim=1) - steps, self.phase_steps).to(within.dtype)
            / self.phase_steps
        )
        cycles = within + start[:, :, None]
        theta = (2 * math.pi * (cycles - torch.floor(cycles))).reshape(1, -1)
        frequency = frequency.reshape(1, -1)
        harmonics = (c.harmonic_limit * c.sample_rate / frequency).clamp_min(1.0)
        whole = torch.floor(harmonics)
        denominator = 2 * torch.sin(theta / 2)
        tiny = denominator.abs() < 1e-6
        safe_denom = torch.where(tiny, torch.ones_like(denominator), denominator)
        core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / safe_denom - 0.5)
        wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
        return wave * torch.sqrt(2.0 / harmonics)

    def _vocoder(self, hidden: Tensor, f0: Tensor, total: Tensor, noise: Tensor | None) -> Tensor:
        c = self.model.config
        v = self.model.vocoder
        last = total - 1

        # Harmonic branch
        h = v.harmonic(hidden)
        split = c.cep_causal + c.cep_anti
        steps = v.harmonic_steps
        sub = torch.minimum(
            torch.arange(cast(int, steps * total + 3), device=hidden.device),
            steps * total - 1,
        )
        here = torch.div(sub, steps, rounding_mode="floor")
        following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
        envelope = 0.5 * (
            h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following)
        )
        envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
        gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
        envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
        harmonic_real, harmonic_imaginary = self._response(
            envelope, self.harmonic_real, self.harmonic_imaginary
        )

        # Noise branch
        n = v.noise(hidden)
        steps = v.noise_steps
        sub = torch.minimum(
            torch.arange(cast(int, steps * total + 3), device=hidden.device),
            steps * total - 1,
        )
        shape = (
            n[..., : c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor"))
            * v.noise_lifter
        )
        gain = n[..., c.noise_cep :].flatten(1).index_select(1, sub)[..., None]
        shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
        noise_real, noise_imaginary = self._response(shape, self.noise_real, self.noise_imaginary)

        pulses = self._pulses(f0)
        if noise is None:
            noise = torch.randn_like(pulses)
        else:
            noise = noise.index_select(
                1, torch.arange(cast(int, c.hop * total), device=hidden.device)
            )
        return self._filter(
            pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop
        ) + self._filter(noise, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop)

    def forward(
        self,
        tokens: Tensor,
        stress: Tensor,
        voice: Tensor,
        speed: Tensor,
        pitch_shift: Tensor,
        pitch_range: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        m = self.model
        c = m.config
        token_mask = torch.ones_like(tokens, dtype=torch.bool)
        hidden = m.encoder(
            m.tokens(tokens) + m.stress(stress) + m.voices(voice)[:, None], token_mask
        )

        output = m.duration(hidden, token_mask)
        frames = (
            torch.round((torch.exp(output[..., 0].clamp(0, math.log1p(400))) - 1) / speed)
            .long()
            .clamp_min(1)
        )
        present = m.phone_tokens[tokens] | (output[..., 1] > 0)
        durations = torch.where(present, frames, torch.zeros_like(frames))

        ends = durations.cumsum(1)
        total = ends[0, -1]
        frame = torch.arange(cast(int, total), device=tokens.device)[None]
        index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack(
            (relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1
        )
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + m.frame_position(features) + m.frame_voices(voice)[:, None]
        frame_mask = torch.ones_like(frame, dtype=torch.bool)

        prosody_hidden = m.prosody_stack(expanded, frame_mask)
        prosody = m.prosody_head(prosody_hidden)
        vowel = m.vowel_tokens[tokens].gather(1, index)
        voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
        voiced = (voicing > 0).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(
            1, keepdim=True
        ).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        condition = torch.stack(
            (pitch.clamp(-2.0, 1.5), voiced, prosody[..., 2].clamp(-4.5, 3.0)), dim=-1
        )
        acoustic = m.acoustic_stack(prosody_hidden + m.prosody_projection(condition), frame_mask)
        f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)
        return self._vocoder(acoustic, f0, total, noise), durations


def example_inputs(length: int = 24) -> tuple[Tensor, ...]:
    tokens = torch.randint(5, len(SYMBOLS), (1, length))
    scalar = lambda value: torch.tensor([value], dtype=torch.float32)  # noqa: E731
    return (
        tokens,
        torch.zeros_like(tokens),
        torch.tensor([0]),
        scalar(1.0),
        scalar(0.0),
        scalar(1.0),
    )


def export_model(
    model: TinyVoca,
    path: str | Path,
    *,
    external_noise: bool = False,
    opset: int = 17,
) -> Path:
    """Export an in-memory ``TinyVoca`` instance directly to an ``.onnx`` file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Exportable(model, external_noise).eval()
    inputs: tuple[Tensor, ...] = example_inputs()
    names = list(INPUTS)
    axes: dict[str, dict[int, str]] = {
        "tokens": {1: "tokens"},
        "stress": {1: "tokens"},
        "audio": {1: "samples"},
        "durations": {1: "tokens"},
    }
    if external_noise:
        inputs = (*inputs, torch.randn(1, model.config.hop * model.config.max_frames))
        names.append("noise")
        axes["noise"] = {1: "noise_samples"}
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output),
            input_names=names,
            output_names=list(OUTPUTS),
            dynamic_axes=axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return output


def build_metadata(
    model: TinyVoca, meta: CheckpointMeta | None = None, *, external_noise: bool = False
) -> dict[str, Any]:
    """Build clean ``model.json`` metadata dictionary for an ONNX export."""
    c = model.config
    inputs = [*INPUTS, "noise"] if external_noise else list(INPUTS)
    return {
        "sample_rate": c.sample_rate,
        "hop": c.hop,
        "voices": list(c.speakers),
        "symbols": list(SYMBOLS),
        "max_tokens": c.max_tokens,
        "max_frames": c.max_frames,
        "step": meta.step if meta is not None else None,
        "inputs": inputs,
        "outputs": list(OUTPUTS),
    }


def export_onnx(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    external_noise: bool = False,
    opset: int = 17,
) -> tuple[Path, Path]:
    """Export a safetensors checkpoint directory into ``output_dir/{model.onnx,model.json}``."""
    model, meta = load_checkpoint(checkpoint_dir, device="cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_model(
        model, out_dir / "model.onnx", external_noise=external_noise, opset=opset
    )
    meta_path = out_dir / "model.json"
    meta_path.write_text(
        json.dumps(
            build_metadata(model, meta, external_noise=external_noise), ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return onnx_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a tiny-voca-ru safetensors checkpoint to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory (model.safetensors + config.json)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for model.onnx and model.json"
    )
    parser.add_argument(
        "--external-noise", action="store_true", help="Accept noise waveform as an ONNX input"
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    onnx_path, meta_path = export_onnx(
        args.checkpoint, args.output, external_noise=args.external_noise, opset=args.opset
    )
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB) and {meta_path.name}")


if __name__ == "__main__":
    main()
