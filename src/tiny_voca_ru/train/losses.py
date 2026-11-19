"""Multi-resolution spectral losses and waveform GAN discriminators."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1)


def magnitude(wave: Tensor, fft: int, hop: int, window: Tensor) -> Tensor:
    spectrum = torch.stft(
        wave.float(), fft, hop, window=window, center=True, pad_mode="constant", return_complex=True
    )
    return torch.view_as_real(spectrum).square().sum(-1).add(1e-9).sqrt()


def mel_filterbank(sample_rate: int, fft: int, n_mels: int) -> Tensor:
    """Slaney-scale triangular mel filterbank ``[n_mels, fft // 2 + 1]``."""
    step, knee, slope = 200.0 / 3.0, 1000.0, math.log(6.4) / 27.0

    def to_hz(mel: Tensor) -> Tensor:
        return torch.where(
            mel < knee / step, mel * step, knee * torch.exp((mel - knee / step) * slope)
        )

    top = (
        knee / step + math.log(sample_rate / 2 / knee) / slope
        if sample_rate / 2 > knee
        else sample_rate / 2 / step
    )
    points = to_hz(torch.linspace(0, top, n_mels + 2, dtype=torch.float64))
    bins = torch.linspace(0, sample_rate / 2, fft // 2 + 1, dtype=torch.float64)
    rising = (bins[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    falling = (points[2:, None] - bins[None]) / (points[2:] - points[1:-1])[:, None]
    weights = (
        torch.minimum(rising, falling).clamp_min(0) * (2.0 / (points[2:] - points[:-2]))[:, None]
    )
    return weights.float()


class SpectralLoss(nn.Module):
    """Multi-resolution STFT and log-mel distances masked to valid utterance lengths."""

    filters: Tensor

    def __init__(
        self,
        sample_rate: int,
        resolutions: Sequence[tuple[int, int]] = ((512, 128), (1024, 256), (2048, 512)),
        mel: tuple[int, int, int] = (1024, 256, 80),
    ) -> None:
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.mel = tuple(mel)
        for fft in {fft for fft, _ in self.resolutions} | {self.mel[0]}:
            self.register_buffer(f"window_{fft}", torch.hann_window(fft), persistent=False)
        self.register_buffer(
            "filters", mel_filterbank(sample_rate, self.mel[0], self.mel[2]), persistent=False
        )

    def log_mel(self, wave: Tensor) -> Tensor:
        fft, hop, _ = self.mel
        window: Tensor = getattr(self, f"window_{fft}")
        return torch.log((self.filters @ magnitude(wave, fft, hop, window)).clamp_min(1e-5))

    def forward(self, predicted: Tensor, target: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        total = predicted.new_zeros((), dtype=torch.float32)
        for fft, hop in self.resolutions:
            window: Tensor = getattr(self, f"window_{fft}")
            p, t = magnitude(predicted, fft, hop, window), magnitude(target, fft, hop, window)
            mask = (
                torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]
            ).float()[:, None]
            logarithmic = (
                (p.clamp_min(1e-5).log() - t.clamp_min(1e-5).log()).abs() * mask
            ).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
            convergence = (((p - t).square() * mask).sum((1, 2)) + 1e-9).sqrt() / (
                t.square() * mask
            ).sum((1, 2)).sqrt().clamp_min(1.0)
            total = total + logarithmic + convergence.mean()
        hop = self.mel[1]
        p, t = self.log_mel(predicted), self.log_mel(target)
        mask = (torch.arange(p.shape[-1], device=p.device)[None] * hop < lengths[:, None]).float()[
            :, None
        ]
        mel_loss = ((p - t).abs() * mask).sum() / (mask.sum() * p.shape[1]).clamp_min(1)
        return total / len(self.resolutions), mel_loss


class PeriodDiscriminator(nn.Module):
    """Single-period 2D convolutional waveform discriminator."""

    def __init__(self, period: int, channels: Sequence[int] = (1, 32, 96, 192, 384)) -> None:
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0)))
            for a, b in zip(channels[:-1], channels[1:], strict=False)
        )
        self.output = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, wave: Tensor) -> tuple[Tensor, list[Tensor]]:
        wave = F.pad(wave, (0, (-wave.shape[-1]) % self.period))
        x = wave.reshape(wave.shape[0], 1, -1, self.period)
        features: list[Tensor] = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class SpectrogramDiscriminator(nn.Module):
    """Single-resolution 2D STFT magnitude discriminator."""

    window: Tensor

    def __init__(self, fft: int, hop: int, channels: int = 32) -> None:
        super().__init__()
        self.fft = fft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
            ]
        )
        self.output = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, wave: Tensor) -> tuple[Tensor, list[Tensor]]:
        x = magnitude(wave, self.fft, self.hop, self.window).transpose(1, 2)[:, None]
        features: list[Tensor] = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            features.append(x)
        return self.output(x), features


class MultiPeriodDiscriminator(nn.Module):
    """Bank of multi-period waveform discriminators."""

    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()
        self.sub_discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(self, wave: Tensor) -> list[tuple[Tensor, list[Tensor]]]:
        return [d(wave) for d in self.sub_discriminators]


class MultiResolutionDiscriminator(nn.Module):
    """Bank of multi-resolution STFT discriminators."""

    def __init__(
        self, resolutions: Sequence[tuple[int, int]] = ((512, 128), (1024, 256), (2048, 512))
    ) -> None:
        super().__init__()
        self.sub_discriminators = nn.ModuleList(
            SpectrogramDiscriminator(fft, hop) for fft, hop in resolutions
        )

    def forward(self, wave: Tensor) -> list[tuple[Tensor, list[Tensor]]]:
        return [d(wave) for d in self.sub_discriminators]


class Discriminators(nn.Module):
    """Combined multi-period and multi-resolution training critics."""

    def __init__(
        self,
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        resolutions: Sequence[tuple[int, int]] = ((512, 128), (1024, 256), (2048, 512)),
    ) -> None:
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.mrd = MultiResolutionDiscriminator(resolutions)

    def forward(self, wave: Tensor) -> list[tuple[Tensor, list[Tensor]]]:
        return [*self.mpd(wave), *self.mrd(wave)]


def discriminator_loss(
    real: Sequence[tuple[Tensor, list[Tensor]]],
    fake: Sequence[tuple[Tensor, list[Tensor]]],
) -> Tensor:
    total = real[0][0].new_zeros(())
    for (r, _), (f, _) in zip(real, fake, strict=False):
        total = total + (r - 1).square().mean() + f.square().mean()
    return total / len(real)


def generator_gan_loss(
    real: Sequence[tuple[Tensor, list[Tensor]]],
    fake: Sequence[tuple[Tensor, list[Tensor]]],
) -> tuple[Tensor, Tensor]:
    adversarial = fake[0][0].new_zeros(())
    matching = fake[0][0].new_zeros(())
    feature_count = 0
    for (_, rs), (f, fs) in zip(real, fake, strict=False):
        adversarial = adversarial + (f - 1).square().mean()
        for r_feat, f_feat in zip(rs, fs, strict=False):
            matching = matching + F.l1_loss(f_feat, r_feat.detach())
            feature_count += 1
    return adversarial / len(fake), matching / max(feature_count, 1)
