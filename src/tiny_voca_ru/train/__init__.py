"""Custom dataset preparation, CTC alignment, losses, and fine-tuning."""

from __future__ import annotations

from .align import (
    ALIGNER_LABEL_FALLBACK,
    ALIGNER_MODEL,
    AlignmentSettings,
    PhoneRecognizer,
    align_utterance,
)
from .dataset import BucketSampler, SpeechDataset, collate_batch, continuous_pitch
from .losses import (
    Discriminators,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    SpectralLoss,
    discriminator_loss,
    generator_gan_loss,
)
from .prepare import QualityLimits, extract_features, prepare_dataset
from .train import (
    DEFAULT_WEIGHTS,
    TrainConfig,
    TrainingSystem,
    load_pretrained_for_finetune,
    optimize_step,
    train,
    validate,
)

__all__ = [
    "ALIGNER_LABEL_FALLBACK",
    "ALIGNER_MODEL",
    "DEFAULT_WEIGHTS",
    "AlignmentSettings",
    "BucketSampler",
    "Discriminators",
    "MultiPeriodDiscriminator",
    "MultiResolutionDiscriminator",
    "PhoneRecognizer",
    "QualityLimits",
    "SpectralLoss",
    "SpeechDataset",
    "TrainConfig",
    "TrainingSystem",
    "align_utterance",
    "collate_batch",
    "continuous_pitch",
    "discriminator_loss",
    "extract_features",
    "generator_gan_loss",
    "load_pretrained_for_finetune",
    "optimize_step",
    "prepare_dataset",
    "train",
    "validate",
]
