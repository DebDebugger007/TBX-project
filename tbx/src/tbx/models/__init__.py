"""Model subpackage.

Imports are lazy so the classical baseline (scikit-learn only) does not drag in
torch. Access ``tbx.models.train_cnn_cv`` and it is imported on demand; the
classical helpers import with no deep-learning dependency.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "CXRDataset": ("tbx.models.datasets", "CXRDataset"),
    "build_transforms": ("tbx.models.datasets", "build_transforms"),
    "build_cnn": ("tbx.models.cnn", "build_cnn"),
    "train_cnn_cv": ("tbx.models.cnn", "train_cnn_cv"),
    "train_classical_cv": ("tbx.models.classical", "train_classical_cv"),
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
