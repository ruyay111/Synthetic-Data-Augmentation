"""Sample 60-day simple-return paths from specialist pools."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hmmdiff.pools import load_pool


def load_specialist_pools(pools_root: Path, n_regimes: int) -> dict[int, np.ndarray]:
    return {k: load_pool(pools_root, k) for k in range(n_regimes)}


def sample_simple_paths(
    windows: np.ndarray,
    n_synth: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_synth`` contiguous ``horizon``-day slices and convert log → simple returns."""
    pool = np.asarray(windows, dtype=float)
    if pool.ndim != 3:
        raise ValueError(f"windows must be (n_pool, seq_len, n_assets), got {pool.shape}")
    n_pool, seq_len, n_assets = pool.shape
    n_synth = int(n_synth)
    if n_synth <= 0:
        return np.zeros((0, horizon, n_assets), dtype=float)
    if seq_len < horizon:
        raise ValueError(f"seq_len {seq_len} shorter than horizon {horizon}")
    replace = n_synth > n_pool
    idx = rng.choice(n_pool, size=n_synth, replace=replace)
    max_start = seq_len - horizon
    starts = rng.integers(0, max_start + 1, size=n_synth)
    log_paths = np.stack(
        [pool[int(i), int(s) : int(s) + horizon] for i, s in zip(idx, starts)],
        axis=0,
    )
    return np.expm1(log_paths)


def synth_rows_for_share(n_real: int, share: float) -> int:
    """Number of synth rows so ``synth / (real + synth) == share``.

    ``share`` is in ``[0, 1)``. ``share = 0`` returns 0.
    """
    n_real = int(n_real)
    share = float(share)
    if n_real < 1:
        raise ValueError(f"n_real must be >= 1, got {n_real}")
    if share < 0 or share >= 1:
        raise ValueError(f"share must be in [0, 1), got {share}")
    if share == 0.0:
        return 0
    return int(round(share / (1.0 - share) * n_real))


def sample_simple_rows(
    windows: np.ndarray,
    n_rows: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw contiguous horizon slices, flatten, and keep the first ``n_rows`` days."""
    pool = np.asarray(windows, dtype=float)
    if pool.ndim != 3:
        raise ValueError(f"windows must be (n_pool, seq_len, n_assets), got {pool.shape}")
    n_assets = int(pool.shape[-1])
    n_rows = int(n_rows)
    if n_rows <= 0:
        return np.zeros((0, n_assets), dtype=float)
    n_win = int(np.ceil(n_rows / float(horizon)))
    paths = sample_simple_paths(windows, n_win, horizon, rng)
    extra = paths.reshape(-1, n_assets)
    if extra.shape[0] < n_rows:
        raise ValueError(f"Need {n_rows} synth rows, sampled {extra.shape[0]}")
    return extra[:n_rows]


def sample_simple_paths_by_daily_regime(
    pools: dict[int, np.ndarray],
    daily_regimes: np.ndarray,
    n_synth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build ``n_synth`` paths by drawing each day from that day's predicted pool.

    Day ``h`` is sampled independently from ``pools[daily_regimes[h]]``: a random
    window and a random timestep, then converted log → simple. This is not a
    contiguous 60-day slice from one regime.
    """
    daily = np.asarray(daily_regimes, dtype=int).reshape(-1)
    if daily.size < 1:
        raise ValueError("daily_regimes is empty")
    n_synth = int(n_synth)
    first = next(iter(pools.values()))
    n_assets = int(np.asarray(first).shape[-1])
    if n_synth <= 0:
        return np.zeros((0, daily.size, n_assets), dtype=float)

    log_paths = np.empty((n_synth, daily.size, n_assets), dtype=float)
    for h, regime in enumerate(daily.tolist()):
        if regime not in pools:
            raise KeyError(f"No specialist pool for predicted regime {regime}")
        pool = np.asarray(pools[regime], dtype=float)
        if pool.ndim != 3:
            raise ValueError(f"pool {regime} must be (n_pool, seq_len, n_assets), got {pool.shape}")
        n_pool, seq_len, pool_assets = pool.shape
        if pool_assets != n_assets:
            raise ValueError(f"pool {regime} has {pool_assets} assets, expected {n_assets}")
        if n_pool < 1 or seq_len < 1:
            raise ValueError(f"pool {regime} is empty")
        win_idx = rng.integers(0, n_pool, size=n_synth)
        time_idx = rng.integers(0, seq_len, size=n_synth)
        log_paths[:, h, :] = pool[win_idx, time_idx, :]
    return np.expm1(log_paths)
