"""Feed-forward transformer TTS model with a pitch-driven source-filter waveform head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config
from .frontend.symbols import PHONE_IDS, SYMBOL_TO_ID, VOWELS


class UtteranceTooLong(ValueError):
    """Raised when predicted or provided frames exceed ``Config.max_frames``."""


def length_mask(lengths: Tensor, size: int | None = None) -> Tensor:
    max_len = int(lengths.max()) if size is None else size
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def rotary_table(size: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    inverse = 1.0 / (
        10000 ** (torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
    )
    angle = torch.arange(size, device=device, dtype=torch.float32)[:, None] * inverse[None]
    return angle.cos(), angle.sin()


def rotate(x: Tensor, table: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = table[0].to(x.dtype), table[1].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings."""

    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)

    def forward(self, x: Tensor, mask: Tensor, table: tuple[Tensor, Tensor]) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            rotate(q, table),
            rotate(k, table),
            v,
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    """Transformer block with a depthwise convolution before the feed-forward MLP."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.channels)
        self.attention = Attention(config.channels, config.heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.channels)
        self.depthwise = nn.Conv1d(
            config.channels,
            config.channels,
            config.kernel,
            padding=config.kernel // 2,
            groups=config.channels,
        )
        self.expand = nn.Linear(config.channels, config.feedforward)
        self.project = nn.Linear(config.feedforward, config.channels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor, table: tuple[Tensor, Tensor]) -> Tensor:
        keep = mask[..., None]
        x = (x + self.dropout(self.attention(self.norm1(x), mask, table))) * keep
        y = self.depthwise((self.norm2(x) * keep).transpose(1, 2)).transpose(1, 2)
        y = self.project(self.dropout(F.gelu(self.expand(y))))
        return (x + self.dropout(y)) * keep


class Stack(nn.Module):
    """Stack of rotary transformer blocks followed by a final LayerNorm."""

    def __init__(self, config: Config, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(config) for _ in range(layers))
        self.norm = nn.LayerNorm(config.channels)
        self.width = config.channels // config.heads

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        table = rotary_table(x.shape[1], self.width, x.device)
        for block in self.blocks:
            x = block(x, mask, table)
        return self.norm(x) * mask[..., None]


class Predictor(nn.Module):
    """Two-layer convolutional variance predictor."""

    def __init__(self, config: Config, outputs: int) -> None:
        super().__init__()
        p = config.predictor_channels
        self.conv1 = nn.Conv1d(config.channels, p, 3, padding=1)
        self.conv2 = nn.Conv1d(p, p, 3, padding=1)
        self.norm1 = nn.LayerNorm(p)
        self.norm2 = nn.LayerNorm(p)
        self.drop = nn.Dropout(config.dropout)
        self.out = nn.Linear(p, outputs)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.conv1((x * mask[..., None]).transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm1(F.gelu(x))) * mask[..., None]
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm2(F.gelu(x)))
        return self.out(x) * mask[..., None]


def pulse_train(
    f0: Tensor,
    sample_rate: int,
    hop: int,
    limit: float = 0.47,
    phase: Tensor | None = None,
) -> Tensor:
    """Band-limited pulse train ``[B, T * hop]`` with unit power for ``f0 [B, T]`` in Hz."""
    frames = f0.shape[1]
    knots = torch.log(f0.double().clamp_min(1.0))
    knots = torch.cat((knots, knots[:, -1:]), dim=1)
    position = torch.arange(frames * hop, device=f0.device, dtype=torch.float64) / hop
    low = position.floor().long()
    weight = position - low
    frequency = torch.exp(knots[:, low] * (1 - weight) + knots[:, low + 1] * weight)
    theta = 2 * math.pi * torch.cumsum(frequency / sample_rate, dim=1)
    if phase is not None:
        theta = theta + phase.double()[:, None]
    theta = torch.remainder(theta, 2 * math.pi)
    harmonics = (limit * sample_rate / frequency).clamp_min(1.0)
    whole = torch.floor(harmonics)
    denominator = 2 * torch.sin(theta / 2)
    tiny = denominator.abs() < 1e-7
    safe_denom = torch.where(tiny, torch.ones_like(denominator), denominator)
    core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / safe_denom - 0.5)
    wave = core + (harmonics - whole) * torch.cos((whole + 1) * theta)
    return (wave * torch.sqrt(2.0 / harmonics)).float()


def filtered(source: Tensor, response: Tensor, fft: int, hop: int, window: Tensor) -> Tensor:
    """Apply time-varying complex STFT filter ``response [B, frames, bins]`` to ``source``."""
    spectrum = torch.stft(
        source, fft, hop, window=window, center=True, pad_mode="constant", return_complex=True
    )
    if spectrum.shape[-1] != response.shape[1]:
        raise ValueError("filter frames do not match the source")
    return torch.istft(
        spectrum * response.transpose(1, 2),
        fft,
        hop,
        window=window,
        center=True,
        length=source.shape[-1],
    )


def cepstrum_response(causal: Tensor, anticausal: Tensor | None, fft: int) -> Tensor:
    """Compute complex frequency response ``exp(rfft(cepstrum))``."""
    buffer = causal.new_zeros(*causal.shape[:-1], fft)
    buffer[..., : causal.shape[-1]] = causal
    if anticausal is not None:
        buffer[..., fft - anticausal.shape[-1] :] = anticausal.flip(-1)
    spectrum = torch.fft.rfft(buffer, dim=-1)
    return torch.polar(torch.exp(spectrum.real.clamp(-14, 6)), spectrum.imag)


class SourceFilter(nn.Module):
    """Harmonic-plus-noise source-filter vocoder driven by frame-level complex cepstra."""

    causal_lifter: Tensor
    anti_lifter: Tensor
    noise_lifter: Tensor
    synth_window: Tensor
    noise_window: Tensor

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.harmonic_steps = config.hop // config.synth_hop
        self.noise_steps = config.hop // config.noise_hop
        self.harmonic = nn.Linear(
            config.channels, config.cep_causal + config.cep_anti + self.harmonic_steps
        )
        self.noise = nn.Linear(config.channels, config.noise_cep + self.noise_steps)
        for head, level in ((self.harmonic, -1.5), (self.noise, -2.5)):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            head.bias.data[0] = level

        def lifter(size: int, first: int) -> Tensor:
            scale = 1.0 / (1.0 + torch.arange(first, first + size, dtype=torch.float32) / 4)
            if first == 0:
                scale[0] = 2.0
            return scale

        self.register_buffer("causal_lifter", lifter(config.cep_causal, 0), persistent=False)
        self.register_buffer("anti_lifter", lifter(config.cep_anti, 1), persistent=False)
        self.register_buffer("noise_lifter", lifter(config.noise_cep, 0), persistent=False)
        self.register_buffer("synth_window", torch.hann_window(config.synth_fft), persistent=False)
        self.register_buffer("noise_window", torch.hann_window(config.noise_fft), persistent=False)

    @staticmethod
    def _take(values: Tensor, index: Tensor) -> Tensor:
        return values.gather(1, index[..., None].expand(-1, -1, values.shape[-1]))

    def _subframes(self, frames: int, lengths: Tensor, steps: int) -> Tensor:
        sub = torch.arange(steps * frames + 3, device=lengths.device)[None].expand(len(lengths), -1)
        return torch.minimum(sub, (steps * lengths - 1).clamp_min(0)[:, None])

    @staticmethod
    def _shape(source: Tensor, response: Tensor, fft: int, hop: int, window: Tensor) -> Tensor:
        return filtered(F.pad(source, (0, fft // 2)), response, fft, hop, window)[
            :, : source.shape[-1]
        ]

    def forward(
        self,
        hidden: Tensor,
        f0: Tensor,
        lengths: Tensor,
        noise: Tensor | None = None,
        phase: Tensor | None = None,
    ) -> Tensor:
        c = self.config
        frames = hidden.shape[1]
        samples = frames * c.hop
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            audible = length_mask(lengths * c.hop, samples)
            last = (lengths - 1).clamp_min(0)[:, None]

            # Harmonic branch: interpolated envelope with per-subframe log gain
            h = self.harmonic(hidden)
            split = c.cep_causal + c.cep_anti
            sub = self._subframes(frames, lengths, self.harmonic_steps)
            here = sub // self.harmonic_steps
            following = torch.minimum((sub + 1) // self.harmonic_steps, last)
            causal = (
                0.5
                * (
                    self._take(h[..., : c.cep_causal], here)
                    + self._take(h[..., : c.cep_causal], following)
                )
                * self.causal_lifter
            )
            anti = (
                0.5
                * (
                    self._take(h[..., c.cep_causal : split], here)
                    + self._take(h[..., c.cep_causal : split], following)
                )
                * self.anti_lifter
            )
            causal = torch.cat(
                (
                    causal[..., :1] + h[..., split:].flatten(1).gather(1, sub)[..., None],
                    causal[..., 1:],
                ),
                dim=-1,
            )
            harmonic_response = cepstrum_response(causal, anti, c.synth_fft)

            # Noise branch: one spectral envelope per frame, four sub-frame gains
            n = self.noise(hidden)
            sub = self._subframes(frames, lengths, self.noise_steps)
            shape = self._take(n[..., : c.noise_cep], sub // self.noise_steps) * self.noise_lifter
            shape = torch.cat(
                (
                    shape[..., :1] + n[..., c.noise_cep :].flatten(1).gather(1, sub)[..., None],
                    shape[..., 1:],
                ),
                dim=-1,
            )
            noise_response = cepstrum_response(shape, None, c.noise_fft)

            # Hold last real pitch through batch padding so padding cannot alter final frames
            f0 = f0.float().gather(
                1, torch.minimum(torch.arange(frames, device=f0.device)[None], last)
            )
            pulses = pulse_train(f0, c.sample_rate, c.hop, c.harmonic_limit, phase) * audible
            if noise is None:
                noise = torch.randn(hidden.shape[0], samples, device=hidden.device)
            wave = self._shape(
                pulses, harmonic_response, c.synth_fft, c.synth_hop, self.synth_window
            ) + self._shape(
                noise * audible, noise_response, c.noise_fft, c.noise_hop, self.noise_window
            )
            return wave * audible


class TinyVoca(nn.Module):
    """4.5M-parameter non-autoregressive Russian TTS model."""

    phone_tokens: Tensor
    vowel_tokens: Tensor

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config = c = config if config is not None else Config()
        self.tokens = nn.Embedding(c.vocab_size, c.channels, padding_idx=0)
        self.stress = nn.Embedding(3, c.channels)
        self.voices = nn.Embedding(len(c.speakers), c.channels)
        self.encoder = Stack(c, c.encoder_layers)
        self.duration = Predictor(c, 2)
        self.frame_voices = nn.Embedding(len(c.speakers), c.channels)
        self.frame_position = nn.Linear(3, c.channels)
        self.prosody_stack = Stack(c, c.prosody_layers)
        self.prosody_head = nn.Linear(c.channels, 3)
        self.prosody_projection = nn.Linear(3, c.channels)
        self.acoustic_stack = Stack(c, c.acoustic_layers)
        self.vocoder = SourceFilter(c)

        phones = torch.zeros(c.vocab_size, dtype=torch.bool)
        phones[sorted(PHONE_IDS)] = True
        self.register_buffer("phone_tokens", phones, persistent=False)

        vowels = torch.zeros(c.vocab_size, dtype=torch.bool)
        vowels[[SYMBOL_TO_ID[v] for v in VOWELS]] = True
        self.register_buffer("vowel_tokens", vowels, persistent=False)

    def encode(
        self, tokens: Tensor, stress: Tensor, lengths: Tensor, speakers: Tensor
    ) -> tuple[Tensor, Tensor]:
        if (
            tokens.ndim != 2
            or stress.shape != tokens.shape
            or lengths.shape != (tokens.shape[0],)
            or speakers.shape != lengths.shape
        ):
            raise ValueError("expected tokens[B,N], stress[B,N], lengths[B], speakers[B]")
        if (
            tokens.shape[1] > self.config.max_tokens
            or bool((lengths < 1).any())
            or bool((lengths > tokens.shape[1]).any())
        ):
            raise ValueError("invalid token length")
        mask = length_mask(lengths, tokens.shape[1])
        x = (self.tokens(tokens) + self.stress(stress) + self.voices(speakers)[:, None]) * mask[
            ..., None
        ]
        return self.encoder(x, mask), mask

    def expand(
        self,
        hidden: Tensor,
        durations: Tensor,
        token_mask: Tensor,
        speakers: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Repeat token states by integer durations and add intra-token position features."""
        if (
            durations.shape != token_mask.shape
            or bool((durations < 0).any())
            or bool((durations[~token_mask] != 0).any())
        ):
            raise ValueError("invalid durations")
        lengths = durations.sum(1)
        if bool((lengths < 1).any()):
            raise ValueError("utterance has no frames")
        if bool((lengths > self.config.max_frames).any()):
            raise UtteranceTooLong("utterance exceeds max_frames; split the input")
        size = int(lengths.max())
        ends = durations.cumsum(1)
        frame = (
            torch.arange(size, device=hidden.device)[None]
            .expand(hidden.shape[0], size)
            .contiguous()
        )
        index = torch.searchsorted(ends, frame, right=True).clamp(max=durations.shape[1] - 1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        relative = (offset.to(hidden.dtype) + 0.5) / owned
        features = torch.stack(
            (relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1
        )
        mask = frame < lengths[:, None]
        expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + self.frame_position(features) + self.frame_voices(speakers)[:, None]
        return expanded * mask[..., None], mask, lengths

    def predict_prosody(self, frames: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.prosody_stack(frames, frame_mask)
        return hidden, self.prosody_head(hidden) * frame_mask[..., None]

    @staticmethod
    def hard_prosody(prediction: Tensor) -> Tensor:
        return torch.stack(
            (
                prediction[..., 0].clamp(-2.0, 1.5),
                (prediction[..., 1] > 0).to(prediction.dtype),
                prediction[..., 2].clamp(-4.5, 3.0),
            ),
            dim=-1,
        )

    def decode(self, prosody_hidden: Tensor, condition: Tensor, frame_mask: Tensor) -> Tensor:
        return self.acoustic_stack(
            (prosody_hidden + self.prosody_projection(condition)) * frame_mask[..., None],
            frame_mask,
        )

    def frequency(self, log_f0: Tensor) -> Tensor:
        c = self.config
        return (c.pitch_reference_hz * torch.exp(log_f0.float())).clamp(c.f0_min, c.f0_max)

    def predict_durations(
        self, hidden: Tensor, mask: Tensor, tokens: Tensor, speed: float = 1.0
    ) -> Tensor:
        output = self.duration(hidden, mask).float()
        frames = (
            torch.round(torch.expm1(output[..., 0].clamp(0, math.log1p(400))) / speed)
            .long()
            .clamp_min(1)
        )
        present = self.phone_tokens[tokens] | (output[..., 1] > 0)
        return torch.where(present, frames, torch.zeros_like(frames)) * mask

    @staticmethod
    def widen_pitch(prosody: Tensor, frame_mask: Tensor, pitch_range: float) -> Tensor:
        if pitch_range == 1.0:
            return prosody
        voiced = ((prosody[..., 1] > 0) & frame_mask).to(prosody.dtype)
        mean = (prosody[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(
            1, keepdim=True
        ).clamp_min(1)
        pitch = mean + (prosody[..., 0] - mean) * pitch_range
        return torch.cat((pitch[..., None], prosody[..., 1:]), dim=-1) * frame_mask[..., None]

    @staticmethod
    def pitch_wobble(
        shape: torch.Size, semitones: float, seed: int, device: torch.device
    ) -> Tensor:
        generator = torch.Generator().manual_seed(seed + 7919)
        fast = torch.randn(shape, generator=generator)
        slow = F.avg_pool1d(
            torch.randn(shape[0], 1, shape[1] + 16, generator=generator), 17, stride=1
        )[:, 0]
        slow = slow / slow.std().clamp_min(1e-6)
        return ((2 * slow + fast) / (5**0.5) * semitones * math.log(2) / 12).to(device)

    @torch.inference_mode()
    def synthesize(
        self,
        tokens: Tensor,
        stress: Tensor,
        lengths: Tensor,
        speakers: Tensor,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        seed: int = 0,
        pitch_range: float = 1.0,
        pitch_jitter: float = 0.0,
        voiced_vowels: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run free-running TTS synthesis returning ``(wave, sample_lengths, details)``."""
        if (
            not math.isfinite(speed)
            or not 0.25 <= speed <= 4
            or not math.isfinite(pitch_shift)
            or not 0 <= pitch_range <= 3
        ):
            raise ValueError(
                "speed must be in [0.25, 4], pitch_shift finite, and pitch_range in [0, 3]"
            )
        hidden, mask = self.encode(tokens, stress, lengths, speakers)
        durations = self.predict_durations(hidden, mask, tokens, speed)
        frames, frame_mask, frame_lengths = self.expand(hidden, durations, mask, speakers)
        prosody_hidden, prosody = self.predict_prosody(frames, frame_mask)
        if voiced_vowels:
            ends = durations.cumsum(1)
            frame = (
                torch.arange(frames.shape[1], device=tokens.device)[None]
                .expand(len(lengths), -1)
                .contiguous()
            )
            owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
            vowel = self.vowel_tokens[tokens].gather(1, owner) & frame_mask
            prosody = torch.cat(
                (
                    prosody[..., :1],
                    torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                    prosody[..., 2:],
                ),
                dim=-1,
            )
        prosody = self.widen_pitch(prosody, frame_mask, pitch_range)
        acoustic = self.decode(prosody_hidden, self.hard_prosody(prosody), frame_mask)
        noise = nn.utils.rnn.pad_sequence(
            [
                torch.randn(
                    int(n) * self.config.hop, generator=torch.Generator().manual_seed(seed + row)
                )
                for row, n in enumerate(frame_lengths)
            ],
            batch_first=True,
        )
        pitch = prosody[..., 0] + pitch_shift
        if pitch_jitter > 0:
            pitch = pitch + self.pitch_wobble(pitch.shape, pitch_jitter, seed, pitch.device)
        wave = self.vocoder(
            acoustic, self.frequency(pitch), frame_lengths, noise=noise.to(hidden.device)
        )
        return (
            wave,
            frame_lengths * self.config.hop,
            {
                "durations": durations,
                "prosody": prosody,
                "frame_mask": frame_mask,
            },
        )
