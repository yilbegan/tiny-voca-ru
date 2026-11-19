"""Training and custom-voice fine-tuning loop for ``TinyVoca``."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..checkpoint import load_checkpoint, save_checkpoint
from ..config import Config
from ..model import TinyVoca
from .dataset import BucketSampler, SpeechDataset, collate_batch
from .losses import (
    Discriminators,
    SpectralLoss,
    discriminator_loss,
    generator_gan_loss,
    masked_mean,
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "duration": 1.0,
    "pause": 1.0,
    "pitch": 2.0,
    "pitch_spread": 2.0,
    "voicing": 1.0,
    "energy": 1.0,
    "mel_head": 2.0,
    "stft": 2.0,
    "mel": 10.0,
}
SEMITONES = 12.0 / math.log(2.0)


@dataclass
class TrainConfig:
    """Hyperparameters for from-scratch training or custom-voice fine-tuning."""

    steps: int = 10000
    batch_frames: int = 9000
    max_batch: int = 32
    learning_rate: float = 1e-4
    warmup: int = 500
    lr_half_life: float = 50000.0
    grad_clip: float = 10.0
    gan_start: int = 2000
    gan_ratio: float = 0.7
    gan_ramp: int = 1000
    feature_matching: float = 30.0
    gan_scale_max: float = 2000.0
    crop_samples: int = 8192
    crops: int = 2
    teacher_hold: int = 500
    teacher_ramp: int = 1500
    teacher_final: float = 0.0
    validation_size: int = 32
    save_every: int = 500
    log_every: int = 20
    seed: int = 20261113
    freeze_encoder: bool = False
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def masked_std(value: Tensor, mask: Tensor) -> Tensor:
    count = mask.sum(1).clamp_min(1)
    mean = (value * mask).sum(1) / count
    return (((value - mean[:, None]).square() * mask).sum(1) / count + 1e-8).sqrt()


class TrainingSystem(nn.Module):
    """Wraps ``TinyVoca`` with training-only auxiliary mel head and spectral loss."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.model = TinyVoca(config)
        self.mel_head = nn.Linear(config.channels, config.n_mels)
        self.spectral = SpectralLoss(config.sample_rate)

    def objective(
        self,
        batch: dict[str, Tensor],
        teacher_prob: float,
        weights: dict[str, float] = DEFAULT_WEIGHTS,
        random_phase: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        m, c = self.model, self.model.config
        hidden, token_mask = m.encode(
            batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"]
        )
        output = m.duration(hidden, token_mask).float()
        present = batch["durations"] > 0
        duration = masked_mean(
            (output[..., 0] - torch.log1p(batch["durations"].float())).square(),
            token_mask & present,
        )
        pausable = token_mask & ~m.phone_tokens[batch["tokens"]]
        pause = masked_mean(
            F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"),
            pausable,
        )

        frames, frame_mask, lengths = m.expand(
            hidden, batch["durations"], token_mask, batch["speakers"]
        )
        if not torch.equal(lengths, batch["frame_lengths"]):
            raise ValueError("durations do not sum to the acoustic frame count")
        prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
        prediction, target = prediction.float(), batch["prosody"]
        trusted = batch["pitch_trusted"]
        pitch = masked_mean(
            (prediction[..., 0] - target[..., 0]).abs(),
            frame_mask * (trusted + 0.1 * (1 - trusted)),
        )
        sung = frame_mask * target[..., 1] * trusted
        enough = (sung.sum(1) >= 10).float()
        predicted_spread, recorded_spread = (
            masked_std(prediction[..., 0], sung),
            masked_std(target[..., 0], sung),
        )
        pitch_spread = masked_mean((predicted_spread - recorded_spread).abs(), enough)
        voicing = masked_mean(
            F.binary_cross_entropy_with_logits(
                prediction[..., 1], target[..., 1], reduction="none"
            ),
            frame_mask,
        )
        energy = masked_mean((prediction[..., 2] - target[..., 2]).abs(), frame_mask)

        teacher = torch.rand(len(lengths), device=lengths.device) < teacher_prob
        condition = torch.where(teacher[:, None, None], target, m.hard_prosody(prediction.detach()))
        acoustic = m.decode(prosody_hidden, condition.to(prosody_hidden.dtype), frame_mask)
        mel_target = (batch["mel"] - c.mel_mean) / c.mel_scale
        mel_head = masked_mean(
            (self.mel_head(acoustic).float() - mel_target).abs().mean(-1), frame_mask
        )
        phase = (
            2 * math.pi * torch.rand(len(lengths), device=lengths.device) if random_phase else None
        )
        wave = m.vocoder(acoustic, m.frequency(target[..., 0]), lengths, phase=phase)
        stft, mel = self.spectral(wave, batch["wave"], lengths * c.hop)

        terms: dict[str, Tensor] = {
            "duration": duration,
            "pause": pause,
            "pitch": pitch,
            "pitch_spread": pitch_spread,
            "voicing": voicing,
            "energy": energy,
            "mel_head": mel_head,
            "stft": stft,
            "mel": mel,
        }
        loss = wave.new_zeros(())
        for name, value in terms.items():
            loss = loss + weights[name] * value
        with torch.no_grad():
            predicted = m.predict_durations(hidden, token_mask, batch["tokens"])
            predicted_frames = predicted.sum(1)
            pause_frames = (batch["durations"] * pausable).sum().clamp_min(1)
            terms.update(
                base_loss=loss.detach(),
                teacher_fraction=teacher.float().mean(),
                pitch_error_semitones=SEMITONES
                * masked_mean((prediction[..., 0] - target[..., 0]).abs(), sung),
                pitch_spread_ratio=masked_mean(
                    predicted_spread / recorded_spread.clamp_min(1e-3), enough
                ),
                voicing_accuracy=masked_mean(
                    ((prediction[..., 1] > 0) == target[..., 1].bool()).float(), frame_mask
                ),
                pause_accuracy=masked_mean(((output[..., 1] > 0) == present).float(), pausable),
                pause_length_ratio=(predicted * pausable).sum() / pause_frames,
                duration_length_ratio=(predicted_frames / lengths).mean(),
                duration_length_error=((predicted_frames - lengths).abs() / lengths).mean(),
            )
        return (
            loss,
            terms,
            {"wave": wave, "reconstruction": weights["stft"] * stft + weights["mel"] * mel},
        )


def crops(
    fake: Tensor, real: Tensor, lengths: Tensor, size: int, count: int
) -> tuple[Tensor, Tensor, Tensor]:
    crop_size = min(size, int(lengths.min()))
    start = (
        torch.rand(len(lengths), count, device=fake.device) * (lengths[:, None] - crop_size + 1)
    ).long()
    index = (start[..., None] + torch.arange(crop_size, device=fake.device)).flatten(1)
    return (
        fake.gather(1, index).view(-1, crop_size),
        real.gather(1, index).view(-1, crop_size),
        index,
    )


def learning_rate(step: int, base: float, warmup: int, half_life: float, start: int = 0) -> float:
    return base * min(1.0, (step - start + 1) / max(warmup, 1)) * (0.5 ** (step / half_life))


def teacher_probability(step: int, hold: int, ramp: int, final: float) -> float:
    return 1.0 if step < hold else max(final, 1.0 - (1.0 - final) * (step - hold) / max(ramp, 1))


def load_pretrained_for_finetune(
    model: TinyVoca,
    checkpoint_path: str | Path,
    *,
    init_voice: str | None = None,
) -> dict[str, Any]:
    """Load pretrained safetensors weights into ``model``, adapting speaker embeddings if needed."""
    pretrained, meta = load_checkpoint(checkpoint_path, device="cpu")
    src_state = pretrained.state_dict()
    dst_state = model.state_dict()
    src_speakers = pretrained.config.speakers
    dst_speakers = model.config.speakers

    if init_voice is not None and init_voice not in src_speakers:
        raise ValueError(f"--init-voice {init_voice!r} not in pretrained speakers {src_speakers}")

    copied = 0
    adapted_voices: list[str] = []
    for name, dst_tensor in dst_state.items():
        if name not in src_state:
            raise ValueError(f"missing weight {name!r} in pretrained checkpoint")
        src_tensor = src_state[name]
        if name in ("voices.weight", "frame_voices.weight") and (
            src_speakers != dst_speakers or init_voice is not None
        ):
            merged = dst_tensor.clone()
            fallback = (
                src_tensor[src_speakers.index(init_voice)]
                if init_voice is not None
                else src_tensor.mean(dim=0)
            )
            for dst_idx, speaker in enumerate(dst_speakers):
                if init_voice is None and speaker in src_speakers:
                    merged[dst_idx] = src_tensor[src_speakers.index(speaker)]
                else:
                    merged[dst_idx] = fallback
            dst_state[name] = merged
            adapted_voices.append(name)
        elif src_tensor.shape == dst_tensor.shape:
            dst_state[name] = src_tensor.clone()
            copied += 1
        else:
            raise ValueError(
                f"incompatible shape for {name!r}: "
                f"pretrained {tuple(src_tensor.shape)} vs target {tuple(dst_tensor.shape)}"
            )
    model.load_state_dict(dst_state, strict=True)
    return {
        "source_step": meta.step,
        "source_speakers": list(src_speakers),
        "target_speakers": list(dst_speakers),
        "init_voice": init_voice,
        "copied_tensors": copied,
        "adapted_voice_tensors": adapted_voices,
    }


def optimize_step(
    system: TrainingSystem,
    critics: Discriminators,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    step: int,
    cfg: TrainConfig,
    *,
    lr_start: int = 0,
) -> dict[str, float | int]:
    lr = learning_rate(step, cfg.learning_rate, cfg.warmup, cfg.lr_half_life, lr_start)
    for group in optimizer.param_groups:
        group["lr"] = lr
    critic_lr = learning_rate(step, cfg.learning_rate, 1, cfg.lr_half_life, lr_start)
    for group in critic_optimizer.param_groups:
        group["lr"] = critic_lr

    optimizer.zero_grad(set_to_none=True)
    prob = teacher_probability(step, cfg.teacher_hold, cfg.teacher_ramp, cfg.teacher_final)
    loss, metrics, auxiliary = system.objective(batch, prob, cfg.weights)

    if step >= cfg.gan_start:
        wave = auxiliary["wave"]
        fake, real, index = crops(
            wave,
            batch["wave"],
            batch["frame_lengths"] * system.model.config.hop,
            cfg.crop_samples,
            cfg.crops,
        )
        critic_optimizer.zero_grad(set_to_none=True)
        d_loss = discriminator_loss(critics(real), critics(fake.detach()))
        d_loss.backward()
        nn.utils.clip_grad_norm_(critics.parameters(), 10.0, error_if_nonfinite=True)
        critic_optimizer.step()

        critics.requires_grad_(False)
        try:
            with torch.no_grad():
                real_features = critics(real)
            adversarial, matching = generator_gan_loss(real_features, critics(fake))
            gan = adversarial + cfg.feature_matching * matching
            recon_grad = torch.autograd.grad(auxiliary["reconstruction"], wave, retain_graph=True)[
                0
            ].gather(1, index)
            gan_grad = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, index)
            ramp = min(1.0, (step - cfg.gan_start + 1) / max(cfg.gan_ramp, 1))
            scale = (
                (cfg.gan_ratio * ramp * recon_grad.norm() / gan_grad.norm().clamp_min(1e-12))
                .clamp(max=cfg.gan_scale_max)
                .detach()
            )
            loss = loss + scale * gan
            metrics.update(
                discriminator=d_loss,
                adversarial=adversarial,
                feature_matching=matching,
                gan_scale=scale,
            )
            loss.backward()
        finally:
            critics.requires_grad_(True)
    else:
        loss.backward()

    trainable = [p for p in system.parameters() if p.requires_grad]
    grad = nn.utils.clip_grad_norm_(trainable, cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {
        **{k: float(v.detach()) for k, v in metrics.items()},
        "loss": float(loss.detach()),
        "grad_norm": float(grad),
        "frames": int(batch["frame_lengths"].sum()),
        "utterances": len(batch["frame_lengths"]),
    }


def balanced_indices(dataset: SpeechDataset, limit: int) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        groups[str(row["speaker"])].append(i)
    capped = min(max(limit, len(groups)), len(dataset.rows))
    selected: list[int] = []
    for offset in range(capped):
        for voice in sorted(groups):
            if offset < len(groups[voice]):
                selected.append(groups[voice][offset])
                if len(selected) == capped:
                    return selected
    return selected


@torch.no_grad()
def validate(
    system: TrainingSystem,
    dataset: SpeechDataset,
    indices: Sequence[int],
    device: str | torch.device,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    batch_size: int = 8,
) -> dict[str, float]:
    system.eval()
    order = sorted(indices, key=lambda i: dataset.lengths[i])
    totals: dict[str, float] = defaultdict(float)
    weight = 0
    for start in range(0, len(order), batch_size):
        batch = collate_batch([dataset[i] for i in order[start : start + batch_size]], device)
        size = len(batch["frame_lengths"])
        last_metrics: dict[str, Tensor] = {}
        for name, prob in (("teacher_forced", 1.0), ("predicted", 0.0)):
            _, last_metrics, _ = system.objective(batch, prob, weights, random_phase=False)
            totals[f"validation_{name}_mel"] += float(last_metrics["mel"]) * size
            totals[f"validation_{name}_stft"] += float(last_metrics["stft"]) * size
        for key in (
            "duration",
            "pause_accuracy",
            "pause_length_ratio",
            "pitch_error_semitones",
            "pitch_spread_ratio",
            "voicing_accuracy",
            "duration_length_ratio",
            "duration_length_error",
        ):
            totals["validation_" + key] += float(last_metrics[key]) * size
        weight += size
    system.train()
    return {key: value / max(weight, 1) for key, value in totals.items()}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(
    train_manifest: str | Path,
    val_manifest: str | Path,
    output_dir: str | Path,
    config: Config,
    train_cfg: TrainConfig,
    *,
    device: str = "cpu",
    init_from: str | Path | None = None,
    init_voice: str | None = None,
    resume: str | Path | None = None,
) -> Path:
    """Run training or fine-tuning and save resumable + safetensors inference checkpoints."""
    if init_from is not None and resume is not None:
        raise ValueError("pass either init_from or resume, not both")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    torch.manual_seed(train_cfg.seed)

    train_ds = SpeechDataset(train_manifest, config)
    val_ds = SpeechDataset(val_manifest, config)
    val_indices = balanced_indices(val_ds, train_cfg.validation_size)

    system = TrainingSystem(config).to(dev)
    critics = Discriminators().to(dev)

    if init_from is not None:
        report = load_pretrained_for_finetune(system.model, init_from, init_voice=init_voice)
        print(json.dumps({"event": "init_from", **report}, ensure_ascii=False), flush=True)

    if train_cfg.freeze_encoder:
        for module in (system.model.tokens, system.model.stress, system.model.encoder):
            module.requires_grad_(False)

    trainable_params = [p for p in system.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=train_cfg.learning_rate, betas=(0.8, 0.99), weight_decay=0.01
    )
    critic_optimizer = torch.optim.AdamW(
        critics.parameters(), lr=train_cfg.learning_rate, betas=(0.8, 0.99), weight_decay=0.01
    )
    sampler = BucketSampler(
        train_ds.lengths, train_cfg.batch_frames, train_cfg.max_batch, train_cfg.seed
    )

    start_step = 0
    best_val = math.inf
    if resume is not None:
        snap = torch.load(Path(resume), map_location="cpu", weights_only=True)
        system.load_state_dict(snap["system"])
        critics.load_state_dict(snap["critics"])
        optimizer.load_state_dict(snap["optimizer"])
        critic_optimizer.load_state_dict(snap["critic_optimizer"])
        sampler.load_state(snap["sampler"])
        start_step = int(snap["step"])
        best_val = float(snap.get("best_val", math.inf))

    started = time.time()
    for step in range(start_step, train_cfg.steps):
        batch = collate_batch([train_ds[i] for i in sampler.next()], dev)
        metrics = optimize_step(
            system, critics, optimizer, critic_optimizer, batch, step, train_cfg
        )
        completed = step + 1
        if completed % train_cfg.log_every == 0 or completed == train_cfg.steps:
            print(
                json.dumps(
                    {
                        "step": completed,
                        "elapsed": round(time.time() - started, 1),
                        "loss": round(float(metrics["loss"]), 4),
                        "mel": round(float(metrics["mel"]), 4),
                        "dur": round(float(metrics["duration"]), 4),
                    }
                ),
                flush=True,
            )
        if completed % train_cfg.save_every == 0 or completed == train_cfg.steps:
            val_metrics = validate(system, val_ds, val_indices, dev, train_cfg.weights)
            score = val_metrics.get("validation_predicted_mel", math.inf)
            save_checkpoint(out / "inference", system.model, step=completed)
            if score < best_val:
                best_val = score
                save_checkpoint(out / "best", system.model, step=completed)
            _atomic_save(
                {
                    "step": completed,
                    "best_val": best_val,
                    "config": config.to_dict(),
                    "system": system.state_dict(),
                    "critics": critics.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "sampler": sampler.state(),
                },
                out / "last.pt",
            )
            print(
                json.dumps(
                    {
                        "step": completed,
                        "best_val": round(best_val, 4),
                        **{k: round(v, 4) for k, v in val_metrics.items()},
                    }
                ),
                flush=True,
            )
    return out / "inference"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune tiny-voca-ru on a prepared Russian dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Prepared dataset directory (with manifest.json, train.jsonl, validation.jsonl)",
    )
    parser.add_argument(
        "--train", type=Path, help="Explicit train.jsonl path (overrides --data-dir)"
    )
    parser.add_argument(
        "--validation", type=Path, help="Explicit validation.jsonl path (overrides --data-dir)"
    )
    parser.add_argument(
        "--config", type=Path, help="Optional Config JSON (default: from manifest.json)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output run directory")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Pretrained safetensors checkpoint directory for fine-tuning",
    )
    parser.add_argument(
        "--init-voice",
        type=str,
        default=None,
        help="Pretrained speaker name (e.g. F or M) to initialize new voice embeddings",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from last.pt")
    parser.add_argument(
        "--freeze-encoder", action="store_true", help="Freeze text encoder during fine-tuning"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-frames", type=int, default=TrainConfig.batch_frames)
    parser.add_argument("--max-batch", type=int, default=TrainConfig.max_batch)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--warmup", type=int, default=TrainConfig.warmup)
    parser.add_argument("--gan-start", type=int, default=TrainConfig.gan_start)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    if args.data_dir is None and (args.train is None or args.validation is None):
        parser.error("provide --data-dir or both --train and --validation")

    if args.train is not None and args.validation is not None:
        train_manifest, val_manifest = args.train, args.validation
    else:
        assert args.data_dir is not None
        train_manifest = args.train or (args.data_dir / "train.jsonl")
        val_manifest = args.validation or (args.data_dir / "validation.jsonl")
    if args.config is not None:
        config = Config.load(args.config)
    elif args.data_dir is not None and (args.data_dir / "manifest.json").is_file():
        config = Config.load(args.data_dir / "manifest.json")
    else:
        config = Config()

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_frames=args.batch_frames,
        max_batch=args.max_batch,
        learning_rate=args.learning_rate,
        warmup=args.warmup,
        gan_start=args.gan_start,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        freeze_encoder=args.freeze_encoder,
    )
    train(
        train_manifest,
        val_manifest,
        args.output,
        config,
        train_cfg,
        device=args.device,
        init_from=args.init_from,
        init_voice=args.init_voice,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
