"""Loading the sampled diffusion pools into the format the notebook's stitching code expects.

Each specialist is sampled into ``data/pools/regime_k{k}/windows.npy`` of shape
``(n_pool, seq_len, 1)``. The reference pipeline consumes a flat 1-D array per regime, so the windows
are concatenated in order.

Concatenating independent windows leaves a discontinuity every ``seq_len`` steps. This is deliberate:
the reference GAN has the same artifact, because ``recursive_simulator`` concatenates independent
128-step chunks. Matching it keeps the comparison honest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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


def load_generated_images(
    pools_root: Path,
    n_regimes: int,
    channel: int = 0,
    seed: int | None = None,
    n_windows: int | None = None,
) -> dict[str, np.ndarray]:
    """Build the ``generated_images`` dict keyed by regime string, as the notebook uses it.

    Each value is a flat 1-D array of z-scored log returns. With ``seed`` set, ``n_windows`` windows
    are drawn without replacement in a reproducible order, which allows alternative pool
    realizations to be tested without re-running the sampler.
    """
    generated: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed) if seed is not None else None
    for regime in range(n_regimes):
        pool = load_pool(pools_root, regime)
        if seed is not None:
            take = min(n_windows or pool.shape[0], pool.shape[0])
            pool = pool[rng.choice(pool.shape[0], size=take, replace=False)]
        elif n_windows is not None:
            pool = pool[: min(n_windows, pool.shape[0])]
        generated[str(regime)] = pool[:, :, channel].reshape(-1).astype(float)
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
    """Stand-in pools drawn with replacement from the real per-regime returns.

    These exist only so the notebook and its plots can be developed and checked before the
    specialists finish training. They are an upper bound on marginal fidelity and carry no temporal
    structure whatsoever, since every draw is independent. Any result produced from them says nothing
    about the diffusion models.
    """
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
