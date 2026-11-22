"""tiny-voca-ru: a lightweight 4.5M-parameter Russian text-to-speech model.

Importing the package root or ``tiny_voca_ru.frontend`` never imports PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import Config
from .frontend import (
    SYMBOLS,
    PhoneSequence,
    accentuate,
    normalize,
    phonemize,
    transcribe,
)

if TYPE_CHECKING:
    from .checkpoint import CheckpointMeta, convert_legacy_pt, load_checkpoint, save_checkpoint
    from .export import export_onnx
    from .infer import (
        DEFAULT_HF_REPO_ID,
        OnnxSynthesizer,
        SynthesisResult,
        Synthesizer,
        TorchSynthesizer,
    )
    from .model import TinyVoca

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "DEFAULT_HF_REPO_ID",
    "SYMBOLS",
    "CheckpointMeta",
    "Config",
    "OnnxSynthesizer",
    "PhoneSequence",
    "SynthesisResult",
    "Synthesizer",
    "TinyVoca",
    "TorchSynthesizer",
    "accentuate",
    "convert_legacy_pt",
    "export_onnx",
    "load_checkpoint",
    "normalize",
    "phonemize",
    "save_checkpoint",
    "transcribe",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CheckpointMeta": (".checkpoint", "CheckpointMeta"),
    "convert_legacy_pt": (".checkpoint", "convert_legacy_pt"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
    "save_checkpoint": (".checkpoint", "save_checkpoint"),
    "export_onnx": (".export", "export_onnx"),
    "DEFAULT_HF_REPO_ID": (".infer", "DEFAULT_HF_REPO_ID"),
    "OnnxSynthesizer": (".infer", "OnnxSynthesizer"),
    "SynthesisResult": (".infer", "SynthesisResult"),
    "Synthesizer": (".infer", "Synthesizer"),
    "TorchSynthesizer": (".infer", "TorchSynthesizer"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
