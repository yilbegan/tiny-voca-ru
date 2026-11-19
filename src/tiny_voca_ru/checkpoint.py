"""Safetensors checkpoint I/O and legacy ``.pt`` conversion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import Config
from .frontend.symbols import SYMBOLS
from .model import TinyVoca

CHECKPOINT_FORMAT_VERSION = 1
WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class CheckpointMeta:
    """Metadata stored alongside ``model.safetensors`` in ``config.json``."""

    format_version: int
    step: int | None
    symbols: tuple[str, ...]
    config: Config


def _resolve_checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"checkpoint path does not exist: {target}")
    if target.is_dir():
        return target / WEIGHTS_FILENAME, target / CONFIG_FILENAME
    if target.suffix == ".safetensors":
        return target, target.parent / CONFIG_FILENAME
    raise ValueError(f"expected a checkpoint directory or {WEIGHTS_FILENAME} file, got {target}")


def save_checkpoint(path: str | Path, model: TinyVoca, *, step: int | None = None) -> Path:
    """Save ``model`` as ``model.safetensors`` + ``config.json`` inside directory ``path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tmp_weights = directory / f"{WEIGHTS_FILENAME}.tmp"
    save_file(tensors, str(tmp_weights))
    tmp_weights.replace(directory / WEIGHTS_FILENAME)

    meta: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "symbols": list(SYMBOLS),
        "config": model.config.to_dict(),
    }
    tmp_config = directory / f"{CONFIG_FILENAME}.tmp"
    tmp_config.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_config.replace(directory / CONFIG_FILENAME)
    return directory


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TinyVoca, CheckpointMeta]:
    """Load a safetensors checkpoint directory (or ``model.safetensors`` path) for inference."""
    weights_path, config_path = _resolve_checkpoint_paths(path)
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"checkpoint must contain {WEIGHTS_FILENAME} and {CONFIG_FILENAME}: {path}"
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    version = raw.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported format_version {version!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    symbols = raw.get("symbols")
    if symbols != list(SYMBOLS):
        raise ValueError(
            "checkpoint symbol inventory does not match tiny_voca_ru.frontend.symbols.SYMBOLS"
        )
    config = Config.from_dict(raw["config"])
    if config.vocab_size != len(SYMBOLS):
        raise ValueError(f"checkpoint vocab_size {config.vocab_size} != {len(SYMBOLS)}")

    model = TinyVoca(config)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    meta = CheckpointMeta(
        format_version=int(version),
        step=int(raw["step"]) if raw.get("step") is not None else None,
        symbols=tuple(symbols),
        config=config,
    )
    return model, meta


def convert_legacy_pt(source_pt: str | Path, output_dir: str | Path) -> Path:
    """Convert a legacy PyTorch ``.pt`` checkpoint into ``model.safetensors`` + ``config.json``."""
    payload = torch.load(Path(source_pt), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload or "config" not in payload:
        raise ValueError("legacy checkpoint must be a mapping containing 'config' and 'model'")
    frontend_info = payload.get("frontend")
    if (
        isinstance(frontend_info, dict)
        and "symbols" in frontend_info
        and list(frontend_info["symbols"]) != list(SYMBOLS)
    ):
        raise ValueError("legacy checkpoint symbols do not match SYMBOLS")
    config = Config.from_dict(payload["config"])
    model = TinyVoca(config)
    model.load_state_dict(payload["model"], strict=True)
    step = int(payload["step"]) if payload.get("step") is not None else None
    return save_checkpoint(output_dir, model, step=step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a legacy .pt checkpoint to safetensors + config.json"
    )
    parser.add_argument("--source", type=Path, required=True, help="Path to legacy .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint directory")
    args = parser.parse_args()
    out = convert_legacy_pt(args.source, args.output)
    weights = out / WEIGHTS_FILENAME
    print(f"Converted checkpoint to {out} ({weights.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
