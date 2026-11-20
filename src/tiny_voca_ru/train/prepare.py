"""Simple custom-dataset preparation: audio features, G2P, and CTC forced alignment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray

from ..config import Config
from ..frontend import NothingToPronounce, PhoneSequence, StressProvider, phonemize
from .align import ALIGNER_MODEL, AlignmentSettings, PhoneRecognizer, align_utterance

MIN_FRAMES = 32


@dataclass(frozen=True)
class QualityLimits:
    """Alignment quality thresholds for filtering utterances during preparation."""

    max_weak_phone_fraction: float = 0.25
    min_token_probability: float = 0.50
    max_phone_frames: int = 70
    min_vowel_voiced: float = 0.50
    min_phones_per_second: float = 4.0
    max_phones_per_second: float = 30.0


def load_audio(path: str | Path, target_sample_rate: int) -> NDArray[np.float32]:
    """Load audio as finite mono float32 resampled to ``target_sample_rate``."""
    wave, sr = sf.read(Path(path), dtype="float32", always_2d=False)
    audio = np.asarray(wave, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("expected finite mono audio")
    if sr != target_sample_rate:
        from scipy.signal import resample_poly

        factor = math.gcd(sr, target_sample_rate)
        audio = resample_poly(
            audio.astype(np.float64), target_sample_rate // factor, sr // factor
        ).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = (audio * (0.99 / peak)).astype(np.float32)
    if float(np.sqrt(np.mean(audio**2))) < 1e-5:
        raise ValueError("audio is effectively silent")
    return audio


def extract_features(wave: NDArray[np.float32], config: Config) -> dict[str, torch.Tensor]:
    """Extract padded waveform, log-mel spectrogram, log-F0/voicing, and log-RMS energy."""
    import librosa

    c = config
    if wave.ndim != 1 or not np.isfinite(wave).all() or not wave.size:
        raise ValueError("expected finite mono audio")
    frames = math.ceil(len(wave) / c.hop)
    if not MIN_FRAMES <= frames <= c.max_frames:
        raise ValueError(f"audio length {frames} frames outside [{MIN_FRAMES}, {c.max_frames}]")
    if float(np.max(np.abs(wave))) > 1.01 or float(np.sqrt(np.mean(wave**2))) < 1e-5:
        raise ValueError("audio is clipped/out of range or effectively silent")

    padded = np.pad(wave, (0, frames * c.hop - len(wave))).astype(np.float32)
    spectrum = librosa.stft(
        padded, n_fft=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant"
    )[:, :frames]
    filters = librosa.filters.mel(sr=c.sample_rate, n_fft=c.n_fft, n_mels=c.n_mels)
    mel = np.log(np.maximum(filters @ (np.abs(spectrum) ** 2), 1e-5)).T
    f0, voiced, _ = librosa.pyin(
        padded,
        fmin=c.f0_min,
        fmax=c.f0_max,
        sr=c.sample_rate,
        frame_length=c.n_fft,
        hop_length=c.hop,
        center=True,
        pad_mode="constant",
    )
    f0 = np.asarray(f0[:frames], dtype=np.float32)
    voiced_arr = np.asarray(voiced[:frames], dtype=bool)
    pitch = np.where(
        voiced_arr,
        np.log(np.maximum(np.nan_to_num(f0, nan=c.pitch_reference_hz), 1.0) / c.pitch_reference_hz),
        0.0,
    )
    rms = librosa.feature.rms(
        y=padded, frame_length=c.n_fft, hop_length=c.hop, center=True, pad_mode="constant"
    )[0, :frames]
    prosody = np.stack(
        (pitch, voiced_arr.astype(np.float32), np.log(np.maximum(rms, 1e-5))), axis=-1
    )
    return {
        "wave": torch.from_numpy(padded),
        "mel": torch.from_numpy(mel.astype(np.float32)),
        "prosody": torch.from_numpy(prosody.astype(np.float32)),
    }


def check_alignment(
    phonemized: PhoneSequence, qc: dict[str, Any], limits: QualityLimits
) -> list[str]:
    """Return list of QC rejection reasons for one aligned utterance (empty if accepted)."""
    reasons: list[str] = []
    if phonemized.unknown:
        reasons.append("unknown_phoneme")
    if qc["weak_phone_fraction"] > limits.max_weak_phone_fraction:
        reasons.append("weak_alignment")
    if qc["token_probability_mean"] < limits.min_token_probability:
        reasons.append("low_confidence")
    if qc["max_phone_frames"] > limits.max_phone_frames:
        reasons.append("overlong_phone")
    if qc["vowel_voiced"] is not None and qc["vowel_voiced"] < limits.min_vowel_voiced:
        reasons.append("vowels_not_voiced")
    if not limits.min_phones_per_second <= qc["phones_per_second"] <= limits.max_phones_per_second:
        reasons.append("implausible_rate")
    return reasons


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    """Read utterance records from ``metadata.csv`` or ``metadata.jsonl``."""
    source = Path(path)
    if source.suffix == ".csv":
        with source.open(encoding="utf-8-sig", newline="") as handle:
            return [
                {k.strip(): (v or "").strip() for k, v in row.items() if k}
                for row in csv.DictReader(handle)
            ]
    if source.suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                rows.append({str(k): str(v) for k, v in obj.items() if v is not None})
        return rows
    raise ValueError(f"metadata must be .csv or .jsonl, got {source.suffix!r}")


def _assign_splits(
    rows: list[dict[str, Any]], val_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    has_explicit = any(r.get("split") in ("train", "validation", "val") for r in rows)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    if has_explicit:
        for row in rows:
            split = row.pop("split", "train")
            if split in ("validation", "val"):
                val_rows.append(row)
            else:
                train_rows.append(row)
        return train_rows, val_rows

    # Deterministic hash split grouped by text so identical texts never leak across splits.
    text_keys = sorted({r["text_hash"] for r in rows})
    n_val = (
        max(1, int(round(len(text_keys) * val_ratio)))
        if len(text_keys) > 1 and val_ratio > 0
        else 0
    )
    ordered = sorted(text_keys, key=lambda h: hashlib.sha256(f"split:{h}".encode()).hexdigest())
    val_hashes = set(ordered[:n_val])
    for row in rows:
        row.pop("split", None)
        if row["text_hash"] in val_hashes:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def prepare_dataset(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    audio_root: str | Path | None = None,
    stressed: bool = False,
    stress: str | StressProvider = "lite",
    val_ratio: float = 0.05,
    device: str = "cpu",
    aligner_model: str = ALIGNER_MODEL,
    alignment_settings: AlignmentSettings | None = None,
    limits: QualityLimits | None = None,
    recognizer: PhoneRecognizer | None = None,
    base_config: Config | None = None,
) -> dict[str, Any]:
    """Prepare a custom Russian TTS dataset into ``output_dir`` ready for training."""
    meta_path = Path(metadata_path)
    root = Path(audio_root) if audio_root is not None else meta_path.parent
    out = Path(output_dir)
    features_dir = out / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_metadata(meta_path)
    if not raw_rows:
        raise ValueError(f"no rows found in {meta_path}")

    active_settings = alignment_settings if alignment_settings is not None else AlignmentSettings()
    active_limits = limits if limits is not None else QualityLimits()
    speakers = tuple(sorted({row.get("speaker") or "default" for row in raw_rows}))
    template = base_config if base_config is not None else Config()
    config = Config.from_dict({**template.to_dict(), "speakers": list(speakers)})
    active_recognizer = (
        recognizer if recognizer is not None else PhoneRecognizer(device=device, name=aligner_model)
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, raw in enumerate(raw_rows):
        utt_id = raw.get("id") or f"utt_{index:06d}"
        if utt_id in seen_ids:
            rejected.append({"id": utt_id, "reason": "duplicate_id"})
            continue
        seen_ids.add(utt_id)
        try:
            audio_rel = raw.get("audio", "")
            text = raw.get("text", "").strip()
            speaker = raw.get("speaker") or "default"
            if not audio_rel or not text:
                raise ValueError("row must contain nonempty 'audio' and 'text'")
            audio_path = Path(audio_rel) if Path(audio_rel).is_absolute() else root / audio_rel
            wave = load_audio(audio_path, config.sample_rate)
            feats = extract_features(wave, config)
            if stressed and not re.search(r"[+ёЁ]", text):
                raise ValueError(
                    "no stress mark in stressed text; remove --stressed or supply marked text"
                )
            phones = phonemize(text, stressed=True) if stressed else phonemize(text, stress=stress)
            if len(phones) > config.max_tokens:
                raise ValueError(
                    f"phoneme sequence length {len(phones)} exceeds max_tokens {config.max_tokens}"
                )
            durations, qc = align_utterance(
                active_recognizer,
                phones,
                feats["wave"].numpy(),
                feats["prosody"][:, 1].numpy(),
                config,
                active_settings,
            )
            reasons = check_alignment(phones, qc, active_limits)
            if reasons:
                rejected.append({"id": utt_id, "audio": audio_rel, "reasons": reasons, "qc": qc})
                continue

            safe_name = hashlib.sha256(utt_id.encode("utf-8")).hexdigest()[:16] + ".pt"
            feat_rel = f"features/{safe_name}"
            feats["speaker"] = torch.tensor(config.speakers.index(speaker), dtype=torch.long)
            torch.save(feats, out / feat_rel)
            accepted.append(
                {
                    "id": utt_id,
                    "path": feat_rel,
                    "speaker": speaker,
                    "text": text,
                    "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "frames": int(feats["mel"].shape[0]),
                    "tokens": list(phones.ids),
                    "stress": list(phones.stress),
                    "durations": durations,
                    "qc": qc,
                    "split": raw.get("split", ""),
                }
            )
        except (ValueError, RuntimeError, OSError, NothingToPronounce) as err:
            rejected.append(
                {
                    "id": utt_id,
                    "audio": raw.get("audio", ""),
                    "reason": f"{type(err).__name__}: {err}",
                }
            )

    (out / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected) + ("\n" if rejected else ""),
        encoding="utf-8",
    )
    if not accepted:
        raise ValueError(
            f"no utterances passed preparation ({len(rejected)} rejected); check rejected.jsonl"
        )

    train_rows, val_rows = _assign_splits(accepted, val_ratio)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows)
        + ("\n" if train_rows else ""),
        encoding="utf-8",
    )
    (out / "validation.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + ("\n" if val_rows else ""),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "speakers": list(config.speakers),
        "config": config.to_dict(),
        "alignment": active_settings.to_dict(),
        "limits": asdict(active_limits),
        "counts": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "rejected": len(rejected),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a Russian dataset for tiny-voca-ru training/fine-tuning"
    )
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Path to metadata.csv or metadata.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for features and JSONL splits"
    )
    parser.add_argument(
        "--audio-root", type=Path, default=None, help="Base directory for relative audio paths"
    )
    parser.add_argument(
        "--stressed", action="store_true", help="Input text already has '+' stress marks"
    )
    parser.add_argument("--stress", choices=("lite", "full", "none"), default="lite")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation fraction when no split column exists",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--min-token-probability", type=float, default=QualityLimits.min_token_probability
    )
    parser.add_argument(
        "--max-weak-phone-fraction", type=float, default=QualityLimits.max_weak_phone_fraction
    )
    args = parser.parse_args()

    limits = QualityLimits(
        min_token_probability=args.min_token_probability,
        max_weak_phone_fraction=args.max_weak_phone_fraction,
    )
    manifest = prepare_dataset(
        args.metadata,
        args.output,
        audio_root=args.audio_root,
        stressed=args.stressed,
        stress=args.stress,
        val_ratio=args.val_ratio,
        device=args.device,
        limits=limits,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
