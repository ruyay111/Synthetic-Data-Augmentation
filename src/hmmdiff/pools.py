"""Load diffusion pools and prepare them for stitching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def pool_dir(pools_root: Path, regime: int) -> Path:
    return pools_root / f"regime_k{regime}"


def load_pool(pools_root: Path, regime: int) -> np.ndarray:
    """One regime's pool, shape ``(n_pool, seq_len, n_channels)``."""
    path = pool_dir(pools_root, regime) / "windows.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/03_train_specialists.sh then "
            "scripts/04_generate_pools.py."
        )
    return np.load(path)


def load_pool_meta(pools_root: Path, regime: int) -> dict[str, Any]:
    path = pool_dir(pools_root, regime) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def zscore_log_returns(raw: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Map raw log returns to HMM ``emission`` units using global (μ, σ)."""
    return (np.asarray(raw, dtype=float) - mu) / sd


def regime_emission_stats(
    emissions: np.ndarray, regimes: np.ndarray, n_regimes: int
) -> dict[int, tuple[float, float]]:
    """Per-regime (mean, std) of HMM emissions on the inner training slice."""
    stats: dict[int, tuple[float, float]] = {}
    emissions = np.asarray(emissions, dtype=float)
    regimes = np.asarray(regimes).astype(int)
    for regime in range(n_regimes):
        subset = emissions[regimes == regime]
        if subset.size == 0:
            raise ValueError(f"Regime {regime} has no observations in the calibration slice.")
        stats[regime] = (float(subset.mean()), float(subset.std()))
    return stats


def affine_calibrate(
    values: np.ndarray, target_mean: float, target_std: float
) -> np.ndarray:
    """Match ``values`` to ``(target_mean, target_std)`` while preserving rank order."""
    values = np.asarray(values, dtype=float)
    src_mean = float(values.mean())
    src_std = float(values.std())
    if src_std < 1e-12:
        return np.full_like(values, target_mean)
    return (values - src_mean) * (target_std / src_std) + target_mean


def load_generated_images(
    pools_root: Path,
    n_regimes: int,
    channel: int = 0,
    seed: int | None = None,
    n_windows: int | None = None,
    *,
    returns: "pd.DataFrame | None" = None,
    zscore_mu: float | None = None,
    zscore_sd: float | None = None,
    calibrate_regimes: dict[int, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Load pools as flat z-scored return arrays keyed by regime."""
    if (zscore_mu is None) ^ (zscore_sd is None):
        raise ValueError("Pass both zscore_mu and zscore_sd, or neither.")

    if returns is not None:
        from .data import a001_log_return_scale

        if channel != 0:
            raise ValueError(
                "Automatic z-scoring from returns only supports channel=0 (A001). "
                "Pass zscore_mu and zscore_sd explicitly for other channels."
            )
        auto_mu, auto_sd = a001_log_return_scale(returns)
        zscore_mu = auto_mu if zscore_mu is None else zscore_mu
        zscore_sd = auto_sd if zscore_sd is None else zscore_sd

    generated: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed) if seed is not None else None
    for regime in range(n_regimes):
        pool = load_pool(pools_root, regime)
        if seed is not None:
            take = min(n_windows or pool.shape[0], pool.shape[0])
            pool = pool[rng.choice(pool.shape[0], size=take, replace=False)]
        elif n_windows is not None:
            pool = pool[: min(n_windows, pool.shape[0])]
        flat = pool[:, :, channel].reshape(-1).astype(float)
        if zscore_mu is not None and zscore_sd is not None:
            flat = zscore_log_returns(flat, zscore_mu, zscore_sd)
        if calibrate_regimes is not None:
            target_mean, target_std = calibrate_regimes[regime]
            flat = affine_calibrate(flat, target_mean, target_std)
        generated[str(regime)] = flat
    return generated


def load_generated_images_ew(
    pools_root: Path,
    n_regimes: int,
    scale: dict[str, Any],
    *,
    seed: int | None = None,
    n_windows: int | None = None,
    calibrate_regimes: dict[int, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Load 10-asset pools and convert them to EW HMM emission units."""
    from .ew import emission_from_log_panel

    generated: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed) if seed is not None else None
    for regime in range(n_regimes):
        pool = load_pool(pools_root, regime)
        if seed is not None:
            take = min(n_windows or pool.shape[0], pool.shape[0])
            pool = pool[rng.choice(pool.shape[0], size=take, replace=False)]
        elif n_windows is not None:
            pool = pool[: min(n_windows, pool.shape[0])]
        flat = emission_from_log_panel(pool, scale).reshape(-1)
        if calibrate_regimes is not None:
            target_mean, target_std = calibrate_regimes[regime]
            flat = affine_calibrate(flat, target_mean, target_std)
        generated[str(regime)] = flat
    return generated


def pools_available(pools_root: Path, n_regimes: int) -> bool:
    return all((pool_dir(pools_root, k) / "windows.npy").exists() for k in range(n_regimes))


def placeholder_pools(
    series: np.ndarray,
    labels: np.ndarray,
    n_regimes: int,
    size: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Stand-in pools drawn with replacement from real per-regime returns."""
    rng = np.random.default_rng(seed)
    pools: dict[str, np.ndarray] = {}
    for regime in range(n_regimes):
        source = series[labels == regime]
        if source.size == 0:
            raise ValueError(f"Regime {regime} has no real observations to resample")
        pools[str(regime)] = rng.choice(source, size=size, replace=True)
    return pools


def pool_demand(regime_paths: list[np.ndarray], n_regimes: int) -> dict[int, int]:
    """How many samples each regime's pool must supply across the given label paths."""
    demand = {k: 0 for k in range(n_regimes)}
    for path in regime_paths:
        for regime in np.asarray(path).astype(int):
            demand[int(regime)] += 1
    return demand


def check_pools(
    generated: dict[str, np.ndarray],
    regime_paths: list[np.ndarray],
    real_series: np.ndarray,
    real_labels: np.ndarray,
    n_regimes: int,
) -> tuple[list[str], dict[str, Any]]:
    """Validation checkpoints 5 and 6.

    Checks that every pool is finite, long enough to satisfy the worst-case demand from the supplied
    regime paths, and that pool variance increases with regime index the way the real per-regime
    variance does. The variance ordering is the real test of whether specialization happened: if the
    specialists all collapsed to the same distribution, the ordering breaks.
    """
    problems: list[str] = []
    demand = pool_demand(regime_paths, n_regimes)

    real_var, pool_var, summary = [], [], {}
    for regime in range(n_regimes):
        pool = generated[str(regime)]
        real_slice = real_series[real_labels == regime]
        rv = float(real_slice.var()) if real_slice.size else float("nan")
        pv = float(pool.var())
        real_var.append(rv)
        pool_var.append(pv)
        summary[str(regime)] = {
            "pool_length": int(pool.size),
            "required": int(demand[regime]),
            "real_var": rv,
            "pool_var": pv,
            "real_mean": float(real_slice.mean()) if real_slice.size else float("nan"),
            "pool_mean": float(pool.mean()),
        }
        if not np.isfinite(pool).all():
            problems.append(f"regime {regime} pool contains non-finite values")
        if pool.size < demand[regime]:
            problems.append(
                f"regime {regime} pool has {pool.size} samples but the regime paths need "
                f"{demand[regime]}; increase n_pool and resample"
            )

    if not _same_ordering(real_var, pool_var):
        problems.append(
            "pool variance ordering does not match the real per-regime ordering "
            f"(real {[round(v, 3) for v in real_var]}, pools {[round(v, 3) for v in pool_var]}); "
            "the specialists may not have specialized"
        )
    return problems, summary


def _same_ordering(a: list[float], b: list[float]) -> bool:
    return list(np.argsort(a)) == list(np.argsort(b))
