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
from .model import (
    _HF_NOISE_AM_DEPTH,
    _PITCH_SPIKE_MAX_DEV,
    _SIBILANT_DB_FLOOR,
    _SIBILANT_DB_SPAN,
    TinyVoca,
)

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

    def __init__(self, model: TinyVoca, external_noise: bool = False, enhance: bool = True) -> None:
        super().__init__()
        self.model = model
        self.external_noise = external_noise
        self.enhance = enhance
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

    def _filter_mixed(
        self,
        source_a: Tensor,
        weight_a: Tensor,
        source_b: Tensor,
        weight_b: Tensor,
        real: Tensor,
        imaginary: Tensor,
        name: str,
        fft: int,
        hop: int,
    ) -> Tensor:
        samples = source_a.shape[1]
        padded = F.pad(torch.cat((source_a, source_b), dim=0)[:, None], (fft // 2, fft))
        s_real = F.conv1d(padded, getattr(self, f"{name}_analysis_real"), stride=hop)
        s_imaginary = F.conv1d(padded, getattr(self, f"{name}_analysis_imaginary"), stride=hop)
        wa, wb = weight_a.transpose(1, 2), weight_b.transpose(1, 2)
        mix_real = s_real[:1] * wa + s_real[1:] * wb
        mix_imaginary = s_imaginary[:1] * wa + s_imaginary[1:] * wb
        real, imaginary = real.transpose(1, 2), imaginary.transpose(1, 2)
        product = torch.cat(
            (
                mix_real * real - mix_imaginary * imaginary,
                mix_real * imaginary + mix_imaginary * real,
            ),
            dim=1,
        )
        wave = F.conv_transpose1d(product, getattr(self, f"{name}_synthesis"), stride=hop)
        envelope = F.conv_transpose1d(
            torch.ones_like(mix_real[:, :1]), getattr(self, f"{name}_envelope"), stride=hop
        )
        return (wave / envelope)[:, 0, fft // 2 : fft // 2 + samples]

    @staticmethod
    def _response(cepstrum: Tensor, real: Tensor, imaginary: Tensor) -> tuple[Tensor, Tensor]:
        magnitude = torch.exp((cepstrum @ real).clamp(-14, 6))
        angle = cepstrum @ imaginary
        return magnitude * torch.cos(angle), magnitude * torch.sin(angle)

    def _pulses_and_phase(self, f0: Tensor) -> tuple[Tensor, Tensor]:
        """Float32 pulse train and wrapped phase using 24-bit integer phase accumulation."""
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
        return wave * torch.sqrt(2.0 / harmonics), theta

    def _vocoder(
        self,
        hidden: Tensor,
        f0: Tensor,
        total: Tensor,
        noise: Tensor | None,
        voicing: Tensor | None = None,
    ) -> Tensor:
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
        noise_gains = n[..., c.noise_cep :]
        v_prob: Tensor | None = None
        if voicing is not None:
            v_prob = torch.sigmoid(voicing)
            noise_gains = v._clean_noise_gains(noise_gains, v_prob)
        steps = v.noise_steps
        sub = torch.minimum(
            torch.arange(cast(int, steps * total + 3), device=hidden.device),
            steps * total - 1,
        )
        shape = (
            n[..., : c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor"))
            * v.noise_lifter
        )
        gain = noise_gains.flatten(1).index_select(1, sub)[..., None]
        shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
        noise_real, noise_imaginary = self._response(shape, self.noise_real, self.noise_imaginary)

        pulses, theta = self._pulses_and_phase(f0)
        if noise is None:
            noise = torch.randn_like(pulses)
        else:
            noise = noise.index_select(
                1, torch.arange(cast(int, c.hop * total), device=hidden.device)
            )

        if v_prob is None:
            harmonic_wave = self._filter(
                pulses, harmonic_real, harmonic_imaginary, "synth", c.synth_fft, c.synth_hop
            )
        else:
            mag2 = harmonic_real.square() + harmonic_imaginary.square()
            e_lo = (mag2 * v.sibilant_lo_weights).sum(dim=-1, keepdim=True)
            e_hi = (mag2 * v.sibilant_hi_weights).sum(dim=-1, keepdim=True)
            ln10 = math.log(10.0)
            ratio_db = (10.0 / ln10) * torch.log((e_hi + 1e-12) / (e_lo + 1e-12))
            sibilance = ((ratio_db - _SIBILANT_DB_FLOOR) / _SIBILANT_DB_SPAN).clamp(0.0, 1.0)
            v_sub = 0.5 * (
                v_prob[..., None].index_select(1, here)
                + v_prob[..., None].index_select(1, following)
            )
            alpha = (
                1.0 - sibilance
            ) * v.vowel_harmonic_curve + sibilance * v_sub * v.sibilant_harmonic_curve
            h_weight = torch.sqrt(alpha.clamp(0.0, 1.0))
            n_weight = torch.sqrt((1.0 - alpha).clamp(0.0, 1.0))

            shift = v.noise_shift
            shifted = torch.cat(
                (noise.index_select(1, v.prefix_flip_index), noise[:, :-shift]), dim=1
            )
            if c.hop % 2 == 0:
                sign = (torch.ones_like(f0)[..., None] * v.sample_alt_sign).reshape(1, -1)
            else:
                sign = (
                    1.0
                    - 2.0
                    * torch.remainder(
                        torch.arange(cast(int, c.hop * total), device=hidden.device), 2
                    ).float()
                )[None]
            v_knots = torch.cat((v_prob, v_prob[:, -1:]), dim=1)
            v_samples = (
                v_knots[:, :-1, None] * (1.0 - self.ramp) + v_knots[:, 1:, None] * self.ramp
            ).reshape(1, -1)
            depth = _HF_NOISE_AM_DEPTH * v_samples
            mod = (1.0 + depth * torch.cos(theta)) / torch.sqrt(1.0 + 0.5 * depth.square())
            hf_noise = shifted * sign * mod
            harmonic_wave = self._filter_mixed(
                pulses,
                h_weight,
                hf_noise,
                n_weight,
                harmonic_real,
                harmonic_imaginary,
                "synth",
                c.synth_fft,
                c.synth_hop,
            )

        return harmonic_wave + self._filter(
            noise, noise_real, noise_imaginary, "noise", c.noise_fft, c.noise_hop
        )

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
        shifted_pitch = pitch + pitch_shift
        if self.enhance:
            frame_idx = torch.arange(cast(int, total), device=tokens.device)
            prev_idx = (frame_idx - 1).clamp_min(0)
            next_idx = torch.minimum(frame_idx + 1, total - 1)
            prev = shifted_pitch.index_select(1, prev_idx)
            nxt = shifted_pitch.index_select(1, next_idx)
            lo = torch.minimum(torch.minimum(prev, shifted_pitch), nxt)
            hi = torch.maximum(torch.maximum(prev, shifted_pitch), nxt)
            med = prev + shifted_pitch + nxt - lo - hi
            shifted_pitch = med + _PITCH_SPIKE_MAX_DEV * torch.tanh(
                (shifted_pitch - med) / _PITCH_SPIKE_MAX_DEV
            )
        f0 = (c.pitch_reference_hz * torch.exp(shifted_pitch)).clamp(c.f0_min, c.f0_max)
        voicing_arg = voicing if self.enhance else None
        return self._vocoder(acoustic, f0, total, noise, voicing_arg), durations


def example_inputs(length: int = 24) -> tuple[Tensor, ...]:
    tokens = torch.randint(5, len(SYMBOLS), (1, length))

    def _scalar(value: float) -> Tensor:
        return torch.tensor([value], dtype=torch.float32)

    return (
        tokens,
        torch.zeros_like(tokens),
        torch.tensor([0]),
        _scalar(1.0),
        _scalar(0.0),
        _scalar(1.0),
    )


def export_model(
    model: TinyVoca,
    path: str | Path,
    *,
    external_noise: bool = False,
    opset: int = 17,
    enhance: bool = True,
) -> Path:
    """Export an in-memory ``TinyVoca`` instance directly to an ``.onnx`` file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = Exportable(model, external_noise, enhance=enhance).eval()
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
    enhance: bool = True,
) -> tuple[Path, Path]:
    """Export a safetensors checkpoint directory into ``output_dir/{model.onnx,model.json}``."""
    model, meta = load_checkpoint(checkpoint_dir, device="cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_model(
        model, out_dir / "model.onnx", external_noise=external_noise, opset=opset, enhance=enhance
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
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable Level-1 DSP post-vocoder enhancement (high-frequency crossover and de-buzz)",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    onnx_path, meta_path = export_onnx(
        args.checkpoint,
        args.output,
        external_noise=args.external_noise,
        opset=args.opset,
        enhance=not args.no_enhance,
    )
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB) and {meta_path.name}")


if __name__ == "__main__":
    main()
