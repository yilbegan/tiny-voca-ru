"""Offline wav2vec2 CTC forced alignment for Russian training data."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import Config
from ..frontend.symbols import PHONE_SET, PUNCTUATION, SP, UNK, VOICELESS, VOWEL_SET
from ..frontend.tokenize import PhoneSequence

ALIGNER_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ALIGNER_RATE = 16000
ALIGNER_STRIDE = 320  # 20 ms at 16 kHz
ALIGNER_CENTER = 200  # 12.5 ms frame center
PAUSING: tuple[str, ...] = (",", ".", "?", "!", ";", ":", "—", "…")

# Calibrated Russian inventory -> wav2vec2 eSpeak recognizer label mapping.
ALIGNER_LABEL_FALLBACK: dict[str, str] = {
    UNK: "<unk>",
    "a": "ɑ",
    "ɐ": "a",
    "ə": "ʌ",
    "ɨ": "y",
    "l": "ɭ",
    "lʲ": "ɭʲ",
    "ʂ": "ʃ",
    "ʐ": "ʒ",
    "tɕ": "tʃʲ",
    "zʲ": "z",
    "fʲ": "f",
    "xʲ": "x",
    "ɡʲ": "ɡ",
}


@dataclass(frozen=True)
class AlignmentSettings:
    """Calibrated Russian timing and silence-detection parameters."""

    onset_offset_seconds: float = -0.010  # phone onset relative to first CTC emission
    min_pause_frames: int = 7  # ~75 ms at 24 kHz / 256 hop
    silence_below_peak_db: float = 45.0
    silence_above_floor_db: float = 8.0
    silence_ceiling_below_peak_db: float = 30.0
    energy_window: int = 512

    def to_dict(self) -> dict[str, Any]:
        return {"aligner": ALIGNER_MODEL, **asdict(self)}


def ctc_viterbi(
    log_probs: NDArray[np.floating[Any]], labels: NDArray[np.int64], blank: int = 0
) -> NDArray[np.int64]:
    """NumPy CTC Viterbi alignment returning extended-state indices (token k -> 2k + 1)."""
    frames, count = log_probs.shape[0], len(labels)
    if count < 1:
        raise ValueError("no labels to align")
    states = 2 * count + 1
    extended = np.full(states, blank, dtype=np.int64)
    extended[1::2] = labels
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = extended[3::2] != extended[1:-2:2]
    score = np.full(states, -np.inf, dtype=np.float64)
    score[0] = float(log_probs[0, blank])
    score[1] = float(log_probs[0, extended[1]])
    back = np.zeros((frames, states), dtype=np.int8)
    for t in range(1, frames):
        one = np.concatenate(([-np.inf], score[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], score[:-2]))
        two[~skip] = -np.inf
        options = np.stack((score, one, two))
        choice = options.argmax(0)
        score = options[choice, np.arange(states)] + log_probs[t, extended]
        back[t] = choice
    state = states - 1 if score[states - 1] >= score[states - 2] else states - 2
    if not np.isfinite(score[state]):
        raise ValueError("audio is too short for this phoneme sequence")
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    return path


def spans_from_states(states: NDArray[np.int64], count: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for index in range(count):
        where = np.flatnonzero(states == 2 * index + 1)
        if not where.size:
            raise ValueError("forced alignment did not visit every phoneme")
        spans.append((int(where[0]), int(where[-1])))
    return spans


def token_spans(
    log_probs: NDArray[np.floating[Any]],
    labels: Sequence[int],
    blank: int = 0,
) -> list[tuple[int, int]]:
    """Return inclusive ``(first, last)`` CTC emission frame spans for each label."""
    count = len(labels)
    repeats = sum(1 for a, b in zip(labels[:-1], labels[1:], strict=False) if a == b)
    if log_probs.shape[0] < count + repeats:
        raise ValueError("audio is too short for this phoneme sequence")
    try:
        import torch
        from torchaudio.functional import forced_align
    except ImportError:
        return spans_from_states(
            ctc_viterbi(log_probs, np.asarray(labels, dtype=np.int64), blank), count
        )
    path, _ = forced_align(
        torch.from_numpy(np.ascontiguousarray(log_probs, dtype=np.float32))[None],
        torch.tensor([list(labels)], dtype=torch.long),
        blank=blank,
    )
    spans: list[list[int]] = []
    previous = blank
    for frame, label in enumerate(path[0].tolist()):
        if label != blank and (previous == blank or label != previous):
            spans.append([frame, frame])
        elif label != blank:
            spans[-1][1] = frame
        previous = label
    if len(spans) != count:
        raise ValueError("forced alignment did not visit every phoneme")
    return [(a, b) for a, b in spans]


def frame_energy_db(
    wave: NDArray[np.floating[Any]], hop: int, window: int = 512
) -> NDArray[np.float64]:
    """Compute per-frame RMS energy in dB centered at sample ``i * hop``."""
    frames = len(wave) // hop
    padded = np.pad(np.asarray(wave, dtype=np.float64), (window // 2, window // 2))
    squares = np.concatenate(([0.0], np.cumsum(padded**2)))
    starts = np.arange(frames) * hop
    mean = (squares[starts + window] - squares[starts]) / window
    return 10.0 * np.log10(np.maximum(mean, 1e-12))


def silence_mask(
    level_db: NDArray[np.float64], settings: AlignmentSettings
) -> tuple[NDArray[np.bool_], float]:
    loud = float(np.percentile(level_db, 95))
    floor = float(np.percentile(level_db, 2))
    threshold = min(
        loud - settings.silence_ceiling_below_peak_db,
        max(loud - settings.silence_below_peak_db, floor + settings.silence_above_floor_db),
    )
    return level_db < threshold, float(threshold)


def pause_carrier(symbols: Sequence[str], group: Sequence[int]) -> int:
    """Choose the token index inside a non-phone span that absorbs a detected pause."""
    for wanted in (PAUSING, PUNCTUATION, (SP,)):
        for index in group:
            if symbols[index] in wanted:
                return index
    return group[0]


def assemble(
    symbols: Sequence[str],
    spans: Sequence[tuple[int, int]],
    frames: int,
    silent: NDArray[np.bool_],
    hop: int,
    sample_rate: int,
    settings: AlignmentSettings,
) -> NDArray[np.int64]:
    """Convert CTC emission spans and silence mask into per-token acoustic frame durations."""
    phones = [i for i, s in enumerate(symbols) if s in PHONE_SET or s == UNK]
    count = len(phones)
    if count != len(spans) or count < 1:
        raise ValueError("phoneme count differs from alignment")
    stride = ALIGNER_STRIDE / ALIGNER_RATE
    emission = np.array([first for first, _ in spans], dtype=np.float64) * stride
    seconds = np.arange(frames) * hop / sample_rate
    keep = np.clip(
        np.round((emission + ALIGNER_CENTER / ALIGNER_RATE) * sample_rate / hop).astype(np.int64),
        0,
        frames - 1,
    )
    for k in range(1, count):
        keep[k] = max(keep[k], keep[k - 1] + 1)
    if keep[-1] > frames - 1:
        raise ValueError("audio is too short for this phoneme sequence")
    starts = np.minimum(
        np.searchsorted(seconds, emission + settings.onset_offset_seconds, side="left"), keep
    )
    starts[0] = 0
    for k in range(1, count):
        starts[k] = max(starts[k], keep[k - 1] + 1)
    token = np.empty(frames, dtype=np.int64)
    bounds = np.append(starts, frames)
    for k in range(count):
        token[bounds[k] : bounds[k + 1]] = phones[k]

    quiet = np.flatnonzero(silent[: keep[0]])
    if quiet.size:  # leading silence -> BOS
        token[: quiet[-1] + 1] = 0
    quiet = np.flatnonzero(silent[keep[-1] + 1 :]) + keep[-1] + 1
    if quiet.size:  # trailing silence -> EOS
        token[quiet[0] :] = len(symbols) - 1
    for k in range(count - 1):  # interior pauses -> pause carrier
        left, right = phones[k], phones[k + 1]
        if right == left + 1:
            continue
        quiet = np.flatnonzero(silent[keep[k] + 1 : keep[k + 1]]) + keep[k] + 1
        if quiet.size < settings.min_pause_frames:
            continue
        token[keep[k] + 1 : quiet[0]] = left
        token[quiet[0] : quiet[-1] + 1] = pause_carrier(symbols, range(left + 1, right))
        token[quiet[-1] + 1 : keep[k + 1]] = right

    if np.any(np.diff(token) < 0):
        raise ValueError("alignment is not monotonic")
    durations = np.bincount(token, minlength=len(symbols)).astype(np.int64)
    if int(durations.sum()) != frames or bool(np.any(durations[phones] < 1)):
        raise ValueError("durations do not cover the acoustic frames")
    return durations


def quality(
    symbols: Sequence[str],
    durations: Sequence[int] | NDArray[np.int64],
    spans: Sequence[tuple[int, int]],
    log_probs: NDArray[np.floating[Any]],
    labels: Sequence[int],
    voiced: NDArray[np.bool_] | NDArray[np.floating[Any]],
    level_db: NDArray[np.float64],
    hop: int,
    sample_rate: int,
) -> dict[str, Any]:
    """Compute label-free alignment quality metrics for filtering bad utterances."""
    dur = np.asarray(durations, dtype=np.int64)
    ends = np.cumsum(dur)
    starts = ends - dur
    token_probability = np.array(
        [
            float(np.exp(log_probs[a : b + 1, label]).mean())
            for (a, b), label in zip(spans, labels, strict=False)
        ]
    )
    phone_frames = np.array(
        [int(d) for s, d in zip(symbols, dur, strict=False) if s in PHONE_SET or s == UNK]
    )

    def voiced_share(wanted: Callable[[str], bool]) -> float | None:
        total = 0
        hits = 0.0
        for symbol, start, end in zip(symbols, starts, ends, strict=False):
            if wanted(symbol) and end - start >= 3:
                total += int(end - start - 2)
                hits += float(voiced[start + 1 : end - 1].sum())
        return round(hits / total, 4) if total else None

    pause = np.zeros(len(voiced), dtype=bool)
    for symbol, start, end in zip(symbols, starts, ends, strict=False):
        if symbol not in PHONE_SET and symbol != UNK:
            pause[start:end] = True
    speech_seconds = float((~pause).sum() * hop / sample_rate)
    return {
        "token_probability_mean": round(float(token_probability.mean()), 4),
        "weak_phone_fraction": round(float((token_probability < 0.1).mean()), 4),
        "max_phone_frames": int(phone_frames.max()),
        "one_frame_phone_fraction": round(float((phone_frames == 1).mean()), 4),
        "phones_per_second": round(len(phone_frames) / max(speech_seconds, 1e-6), 2),
        "pause_frames": int(pause.sum()),
        "leading_frames": int(dur[0]),
        "trailing_frames": int(dur[-1]),
        "vowel_voiced": voiced_share(lambda s: s in VOWEL_SET),
        "voiceless_voiced": voiced_share(lambda s: s in VOICELESS),
        "pause_voiced": round(float(voiced[pause].mean()), 4) if pause.any() else None,
        "pause_level_db": round(float(level_db[pause].mean()), 1) if pause.any() else None,
        "speech_level_db": round(float(np.percentile(level_db[~pause], 95)), 1)
        if (~pause).any()
        else None,
    }


class PhoneRecognizer:
    """Pretrained wav2vec2 eSpeak CTC phoneme recognizer used for dataset preparation."""

    def __init__(
        self,
        device: str = "cpu",
        name: str = ALIGNER_MODEL,
        *,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC

        self.torch = torch
        self.device = torch.device(device)
        model: Any = Wav2Vec2ForCTC.from_pretrained(name, local_files_only=local_files_only)
        model = model.to(self.device).eval()
        if self.device.type == "cuda":
            model = model.half()
        self.model = model
        vocab_path = Path(hf_hub_download(name, "vocab.json", local_files_only=local_files_only))
        self.vocabulary: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.blank = int(self.model.config.pad_token_id)
        self.name = name
        missing = [
            p
            for p in sorted(PHONE_SET | {UNK})
            if ALIGNER_LABEL_FALLBACK.get(p, p) not in self.vocabulary
        ]
        if missing:
            raise ValueError(f"recognizer has no label for inventory phones {missing}")

    def label(self, symbol: str) -> int:
        return int(self.vocabulary[ALIGNER_LABEL_FALLBACK.get(symbol, symbol)])

    def log_probs(self, wave: NDArray[np.floating[Any]], sample_rate: int) -> NDArray[np.float32]:
        from scipy.signal import resample_poly

        factor = math.gcd(sample_rate, ALIGNER_RATE)
        audio = resample_poly(
            np.asarray(wave, dtype=np.float64), ALIGNER_RATE // factor, sample_rate // factor
        )
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        with self.torch.inference_mode():
            values = self.torch.from_numpy(audio.astype(np.float32))[None].to(
                self.device, self.model.dtype
            )
            logits = self.model(values).logits[0].float().log_softmax(-1).cpu().numpy()
            return np.asarray(logits, dtype=np.float32)


def align_utterance(
    recognizer: PhoneRecognizer,
    phonemized: PhoneSequence,
    wave: NDArray[np.floating[Any]],
    voiced: NDArray[np.floating[Any]],
    config: Config,
    settings: AlignmentSettings = AlignmentSettings(),
) -> tuple[list[int], dict[str, Any]]:
    """Align one utterance and return ``(durations, qc_metrics)``."""
    symbols = phonemized.symbols
    frames = len(wave) // config.hop
    if len(wave) != frames * config.hop or len(voiced) != frames:
        raise ValueError("waveform length must equal frames * hop")
    labels = [recognizer.label(s) for s in symbols if s in PHONE_SET or s == UNK]
    log_probs = recognizer.log_probs(wave, config.sample_rate)
    spans = token_spans(log_probs, labels, blank=recognizer.blank)
    level_db = frame_energy_db(wave, config.hop, settings.energy_window)
    silent, threshold_db = silence_mask(level_db, settings)
    durations = assemble(symbols, spans, frames, silent, config.hop, config.sample_rate, settings)
    qc = quality(
        symbols,
        durations,
        spans,
        log_probs,
        labels,
        np.asarray(voiced) > 0.5,
        level_db,
        config.hop,
        config.sample_rate,
    )
    qc["silence_threshold_db"] = round(float(threshold_db), 2)
    return durations.tolist(), qc
