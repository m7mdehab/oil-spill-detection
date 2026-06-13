"""Reproducibility helpers: seed all RNGs and pin deterministic backends."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs and (optionally) pin determinism.

    Returns the seed so callers can log it. When ``deterministic`` is true the
    cuDNN backend is forced into deterministic mode and benchmarking is
    disabled, which trades a little throughput for reproducible runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
