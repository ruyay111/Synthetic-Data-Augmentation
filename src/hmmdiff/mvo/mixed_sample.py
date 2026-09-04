"""Load regime-free diffusion windows and sample 60-day simple-return paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hmmdiff.constants import N_ASSETS, SEQ_LEN
from .specialist_sample import sample_simple_paths


def load_uncond_windows(path: Path, seq_len: int = SEQ_LEN, n_assets: int = N_ASSETS) -> np.ndarray:
    """Reshape ``generated_data.csv`` to ``(n_windows, seq_len, n_assets)`` log returns."""
    if not path.exists():
        raise FileNotFoundError(f"Regime-free diffusion file not found: {path}")
    flat = pd.read_csv(path).to_numpy(dtype=float)
    if flat.ndim != 2 or flat.shape[1] != n_assets:
        raise ValueError(f"Expected {n_assets} columns, got {flat.shape}")
    if flat.shape[0] % seq_len != 0:
        raise ValueError(f"Row count {flat.shape[0]} is not divisible by seq_len={seq_len}")
    return flat.reshape(flat.shape[0] // seq_len, seq_len, n_assets)


def sample_uncond_simple_paths(
    windows: np.ndarray,
    n_synth: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_simple_paths(windows, n_synth, horizon, rng)
