"""Reproducibility helpers.

Every result in this project has to be reproducible from a fixed seed, because a
headline AUC on 138 images can move a lot with a lucky split. Call
``seed_everything`` once at the top of every script.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Parameters
    ----------
    seed:
        The integer seed. The project default is 42 and lives in the config.
    deterministic_torch:
        If True, ask cuDNN for deterministic kernels. This costs a little speed
        but means a rerun gives the same numbers, which matters when you are
        reporting confidence intervals and claiming a difference is real.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is optional for the data-prep and classical stages.
        pass

    return seed
