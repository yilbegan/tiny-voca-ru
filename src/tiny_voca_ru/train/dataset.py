"""Training dataset, batch collation, and length-bucketed sampler."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..config import Config
from ..frontend.symbols import PHONE_IDS, VOWEL_IDS


def continuous_pitch(
    pitch: Tensor,
    voiced: Tensor,
    default: float = 0.0,
    window: int = 5,
    tolerance: float = 0.45,
) -> tuple[Tensor, Tensor]:
    """Filter octave jumps and linearly interpolate log-F0 across unvoiced frames."""
    is_voiced = voiced > 0.5
    trusted = is_voiced.clone()
    if int(is_voiced.sum()) >= 3:
        values = pitch[is_voiced]
        padded = torch.nn.functional.pad(
            values[None, None], (window // 2, window // 2), mode="replicate"
        )[0, 0]
        median = padded.unfold(0, window, 1).median(dim=1).values
        trusted[is_voiced.nonzero().squeeze(1)[(values - median).abs() > tolerance]] = False
    index = trusted.nonzero().squeeze(1)
    if index.numel() == 0:
        return torch.full_like(pitch, default), trusted.float()
    position = torch.arange(pitch.numel())
    right = torch.searchsorted(index, position).clamp(max=index.numel() - 1)
    left = (torch.searchsorted(index, position, right=True) - 1).clamp(min=0)
    a, b = index[left], index[right]
    span = (b - a).clamp(min=1).float()
    weight = ((position - a).float() / span).clamp(0, 1)
    curve = pitch[a] * (1 - weight) + pitch[b] * weight
    return curve, trusted.float()


class SpeechDataset:
    """Prepared JSONL dataset backed by per-utterance ``.pt`` acoustic feature files."""

    def __init__(
        self,
        manifest: str | Path,
        config: Config,
        *,
        root: str | Path | None = None,
        cache: bool = True,
        voiced_vowels: bool = True,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        self.config = config
        self.rows: list[dict[str, Any]] = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"empty dataset: {self.manifest}")

        seen: set[str] = set()
        for row in self.rows:
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(f"duplicate ID in manifest: {row_id}")
            seen.add(row_id)
            tokens: list[int] = row["tokens"]
            stress: list[int] = row["stress"]
            durations: list[int] = row["durations"]
            frames = int(row["frames"])
            if len(tokens) != len(durations) or len(tokens) != len(stress):
                raise ValueError(f"token/duration length mismatch: {row_id}")
            if sum(durations) != frames or min(durations) < 0:
                raise ValueError(f"durations do not sum to frame count: {row_id}")
            if any(d < 1 for t, d in zip(tokens, durations, strict=False) if t in PHONE_IDS):
                raise ValueError(f"phoneme without frames: {row_id}")
            if not 0 < len(tokens) <= config.max_tokens or not 0 < frames <= config.max_frames:
                raise ValueError(f"utterance exceeds model limits: {row_id}")
            if (
                max(tokens) >= config.vocab_size
                or min(tokens) < 1
                or row["speaker"] not in config.speakers
            ):
                raise ValueError(f"invalid token or speaker: {row_id}")

        self.lengths: list[int] = [int(r["frames"]) for r in self.rows]
        self._cache: dict[int, dict[str, Any]] | None = {} if cache else None
        self.voiced_vowels = voiced_vowels

    def __len__(self) -> int:
        return len(self.rows)

    def preload(self, indices: Sequence[int] | None = None, workers: int = 8) -> None:
        """Load feature files into memory ahead of training."""
        targets = range(len(self)) if indices is None else indices
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(self.__getitem__, targets))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        row, c = self.rows[index], self.config
        path = (self.root / row["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"feature file escapes dataset root: {row['path']}")
        stored: dict[str, Tensor] = torch.load(path, map_location="cpu", weights_only=True)
        t = int(row["frames"])
        if (
            stored["mel"].shape != (t, c.n_mels)
            or stored["prosody"].shape != (t, 3)
            or stored["wave"].shape != (t * c.hop,)
        ):
            raise ValueError(f"feature dimensions differ from manifest: {path.name}")
        if any(not torch.isfinite(stored[k]).all() for k in ("mel", "prosody", "wave")):
            raise ValueError(f"non-finite features: {path.name}")

        pitch, trusted = continuous_pitch(stored["prosody"][:, 0], stored["prosody"][:, 1])
        voiced = stored["prosody"][:, 1]
        if self.voiced_vowels:
            vowel = torch.repeat_interleave(
                torch.tensor([t_id in VOWEL_IDS for t_id in row["tokens"]]),
                torch.tensor(row["durations"], dtype=torch.long),
            )
            voiced = torch.maximum(voiced, vowel.to(voiced.dtype))

        item: dict[str, Any] = {
            "tokens": torch.tensor(row["tokens"], dtype=torch.long),
            "stress": torch.tensor(row["stress"], dtype=torch.long),
            "durations": torch.tensor(row["durations"], dtype=torch.long),
            "speaker": c.speakers.index(row["speaker"]),
            "wave": stored["wave"],
            "mel": stored["mel"],
            "prosody": torch.stack(
                (pitch, voiced, (stored["prosody"][:, 2] - c.energy_mean) / c.energy_scale), dim=-1
            ),
            "pitch_trusted": trusted,
        }
        if self._cache is not None:
            self._cache[index] = item
        return item


def collate_batch(
    items: Sequence[dict[str, Any]], device: str | torch.device = "cpu"
) -> dict[str, Tensor]:
    """Pad variable-length dataset items into a single training batch."""
    batch: dict[str, Tensor] = {
        key: pad_sequence([x[key] for x in items], batch_first=True).to(device)
        for key in ("tokens", "stress", "durations", "mel", "prosody", "pitch_trusted", "wave")
    }
    batch["token_lengths"] = torch.tensor(
        [x["tokens"].numel() for x in items], dtype=torch.long, device=device
    )
    batch["frame_lengths"] = torch.tensor(
        [x["mel"].shape[0] for x in items], dtype=torch.long, device=device
    )
    batch["speakers"] = torch.tensor([x["speaker"] for x in items], dtype=torch.long, device=device)
    return batch


class BucketSampler:
    """Length-bucketed random batch sampler under a padded-frame budget."""

    def __init__(
        self,
        lengths: Sequence[int],
        frame_budget: int,
        max_batch: int,
        seed: int,
        pool: int = 256,
    ) -> None:
        self.lengths = list(lengths)
        self.frame_budget = frame_budget
        self.max_batch = max_batch
        self.pool = pool
        self.generator = torch.Generator().manual_seed(seed)
        self.pending: list[list[int]] = []

    def next(self) -> list[int]:
        if not self.pending:
            indices = sorted(
                torch.randint(len(self.lengths), (self.pool,), generator=self.generator).tolist(),
                key=lambda i: self.lengths[i],
            )
            batches: list[list[int]] = []
            current: list[int] = []
            for index in indices:
                if current and (
                    len(current) == self.max_batch
                    or (len(current) + 1) * self.lengths[index] > self.frame_budget
                ):
                    batches.append(current)
                    current = []
                current.append(index)
            batches.append(current)
            self.pending = [
                batches[i] for i in torch.randperm(len(batches), generator=self.generator).tolist()
            ]
        return self.pending.pop()

    def state(self) -> dict[str, Any]:
        return {"generator": self.generator.get_state(), "pending": [list(b) for b in self.pending]}

    def load_state(self, state: dict[str, Any]) -> None:
        self.generator.set_state(state["generator"].cpu())
        self.pending = [list(b) for b in state["pending"]]
