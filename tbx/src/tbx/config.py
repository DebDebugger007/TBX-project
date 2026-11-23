"""Experiment configuration.

The methodology commits to one idea above all others: an experiment is described
by a configuration file, not by edited code. Swapping the training set, the test
set or the image region is a change to a YAML file; the same pipeline runs any of
them. This module defines the schema those files fill in and the loader that
merges an experiment file onto the shared defaults in ``configs/default.yaml``.

Fields that encode the experiment matrix (Appendix B of the IPR):

* ``train`` / ``test``        which NLM set trains and which tests ("montgomery",
                              "shenzhen", or "both" for the mixed-training run)
* ``region``                 "whole" | "lung_cropped" | "lung_masked"
* ``model``                  "classical" | "resnet18" | "densenet121"
* ``eval_mode``              "within" (stratified 5-fold) | "cross" (train one
                              set, test the other once)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

VALID_SETS = {"montgomery", "shenzhen", "both"}
VALID_REGIONS = {"whole", "lung_cropped", "lung_masked"}
VALID_MODELS = {"classical", "resnet18", "densenet121"}
VALID_EVAL = {"within", "cross"}
VALID_CORRECTIONS = {"none", "histeq", "clahe", "zscore"}


@dataclass
class PathsConfig:
    raw_root: str = "data/raw"            # where the NLM downloads land
    processed_root: str = "data/processed"  # resized images + merged masks
    metadata_csv: str = "data/processed/metadata.csv"
    splits_dir: str = "data/processed/splits"
    masks_dir: str = "data/processed/masks"
    outputs_root: str = "outputs"


@dataclass
class DataConfig:
    image_size: int = 224                 # 224 first; 512 only if GPU time allows
    pad_to_square: bool = True            # preserve aspect ratio, pad not stretch
    montgomery_bits: int = 12             # Montgomery is 12-bit; rescaled to [0,1]
    n_folds: int = 5
    seed: int = 42


@dataclass
class FeatureConfig:
    """Classical lung-field features (Jaeger et al., 2014b style)."""
    glcm_distances: list[int] = field(default_factory=lambda: [1, 2, 4])
    glcm_angles_deg: list[int] = field(default_factory=lambda: [0, 45, 90, 135])
    glcm_levels: int = 32                 # grey levels after requantisation
    lbp_radius: int = 3
    lbp_points: int = 24
    intensity_bins: int = 16


@dataclass
class ModelConfig:
    name: str = "resnet18"                # classical | resnet18 | densenet121
    pretrained: bool = True               # ImageNet weights for the CNN
    classical_estimator: str = "logreg"   # logreg | svm (RBF)
    # CNN training
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    freeze_backbone_epochs: int = 3       # warm up the head first
    # Augmentation: mild, and NO horizontal flip (flipping moves the heart and
    # any laterality marker to the wrong side — never happens in a real image).
    aug_rotation_deg: float = 7.0
    aug_brightness: float = 0.1
    aug_crop_scale: float = 0.9


@dataclass
class EvalConfig:
    bootstrap_n: int = 2000               # stratified bootstrap resamples
    ci: float = 0.95
    screening_sensitivity: float = 0.90   # report specificity at 90% sensitivity
    min_subgroup_n: int = 10              # smaller subgroups are descriptive only


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    description: str = ""
    train: str = "montgomery"
    test: str = "montgomery"
    region: str = "whole"
    model: str = "resnet18"
    eval_mode: str = "within"
    n_target_images: int = 0              # few-shot k for Phase 6 (0 = none)
    correction: str = "none"              # none | clahe | histeq | zscore (Phase 6)

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> "ExperimentConfig":
        if self.train not in VALID_SETS:
            raise ValueError(f"train must be one of {VALID_SETS}, got {self.train!r}")
        if self.test not in VALID_SETS:
            raise ValueError(f"test must be one of {VALID_SETS}, got {self.test!r}")
        if self.region not in VALID_REGIONS:
            raise ValueError(f"region must be one of {VALID_REGIONS}, got {self.region!r}")
        if self.model not in VALID_MODELS:
            raise ValueError(f"model must be one of {VALID_MODELS}, got {self.model!r}")
        if self.eval_mode not in VALID_EVAL:
            raise ValueError(f"eval_mode must be one of {VALID_EVAL}, got {self.eval_mode!r}")
        if self.correction not in VALID_CORRECTIONS:
            raise ValueError(
                f"correction must be one of {VALID_CORRECTIONS}, got {self.correction!r}"
                + (" ('mixed' is not implemented yet)" if self.correction == "mixed" else "")
            )
        if self.region != "whole" and self.model == "classical":
            # The classical model is lung-field only by construction; the region
            # switch is a CNN concept. Flag it rather than silently ignoring it.
            raise ValueError(
                "region switching does not apply to the classical model "
                "(it is already restricted to the lung field)"
            )
        if self.eval_mode == "cross" and self.train == self.test and self.train != "both":
            raise ValueError("cross evaluation needs different train and test sets")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(d: dict) -> ExperimentConfig:
    """Build a validated ExperimentConfig from a plain dict."""
    sub = {
        "paths": PathsConfig(**d.pop("paths", {})),
        "data": DataConfig(**d.pop("data", {})),
        "features": FeatureConfig(**d.pop("features", {})),
        "model_cfg": ModelConfig(**d.pop("model_cfg", {})),
        "eval": EvalConfig(**d.pop("eval", {})),
    }
    cfg = ExperimentConfig(**{**d, **sub})
    return cfg.validate()


def load_config(path: str | Path, default_path: str | Path | None = None) -> ExperimentConfig:
    """Load an experiment config, merged onto the shared defaults.

    If ``default_path`` is None we look for ``configs/default.yaml`` next to the
    experiment file. Missing defaults are fine — the dataclass defaults apply.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        exp = yaml.safe_load(fh) or {}

    if default_path is None:
        candidate = path.parent / "default.yaml"
        default_path = candidate if candidate.exists() else None

    base: dict = {}
    if default_path is not None:
        with Path(default_path).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}

    merged = _deep_merge(base, exp)
    return _build(merged)
