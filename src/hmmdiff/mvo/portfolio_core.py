"""Markowitz weights, column-stack mixing, and portfolio metrics.

Ported from ruya ``evaluation/utils/portfolio_core.py``. No calendar mix, stress
ranking, or crisis windows. Optional name cap is applied after collapse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(r: np.ndarray, rf: float = 0.0) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - rf / TRADING_DAYS
    sd = float(excess.std(ddof=0))
    if np.isclose(sd, 0.0):
        return float("nan")
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / sd)


def max_drawdown(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def calmar_ratio(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    ann_ret = float(r.mean() * TRADING_DAYS)
    mdd = max_drawdown(r)
    if not np.isfinite(mdd) or np.isclose(mdd, 0.0):
        return float("nan")
    return ann_ret / abs(mdd)


def summarize_portfolio(port_ret: pd.Series, label: str = "ALL", rf: float = 0.0) -> dict:
    arr = np.asarray(port_ret, dtype=float)
    return {
        "bucket": label,
        "sharpe": sharpe_ratio(arr, rf=rf),
        "calmar": calmar_ratio(arr),
        "max_drawdown": max_drawdown(arr),
        "ann_mu": float(np.nanmean(arr) * TRADING_DAYS),
        "ann_sd": float(np.nanstd(arr, ddof=0) * np.sqrt(TRADING_DAYS)),
        "n_obs": int(np.isfinite(arr).sum()),
    }


OBJECTIVES = ("mean_variance", "min_variance", "max_sharpe", "max_return")


def _apply_short_and_budget(weights: np.ndarray, allow_short: bool) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1).copy()
    n = w.size
    if not allow_short:
        w = np.maximum(w, 0.0)
    total = float(w.sum())
    return w / total if not np.isclose(total, 0.0) else np.ones(n) / n


def _closed_form_weights(
    mu: np.ndarray, sigma: np.ndarray, objective: str, allow_short: bool
) -> np.ndarray:
    n = sigma.shape[0]
    inv_s = np.linalg.pinv(sigma)
    ones = np.ones(n)
    if objective == "mean_variance":
        a = float(ones @ inv_s @ mu)
        b = float(ones @ inv_s @ ones)
        lam = (a - 1.0) / b if not np.isclose(b, 0.0) else 0.0
        weights = inv_s @ (mu - lam * ones)
    elif objective == "min_variance":
        weights = inv_s @ ones
    elif objective == "max_sharpe":
        weights = inv_s @ mu
    elif objective == "max_return":
        weights = np.zeros(n)
        weights[int(np.argmax(mu))] = 1.0
        return _apply_short_and_budget(weights, allow_short=False)
    else:
        raise ValueError(f"Unknown objective {objective!r}; expected one of {OBJECTIVES}")
    return _apply_short_and_budget(weights, allow_short=allow_short)


def greedy_max_return_box(
    mu: np.ndarray, w_min: float = 0.0, w_max: float = 1.0
) -> np.ndarray:
    """Fill highest-mean names first under ``sum w = 1`` and box constraints."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    n = int(mu.size)
    lo = float(w_min)
    hi = float(w_max)
    if n * hi < 1.0 - 1e-9:
        raise ValueError(f"max_weight={hi} cannot sum to 1 with {n} assets")
    if n * lo > 1.0 + 1e-9:
        raise ValueError(f"w_min={lo} already exceeds sum 1 with {n} assets")
    w = np.full(n, lo)
    remaining = max(1.0 - n * lo, 0.0)
    for i in np.argsort(-mu):
        add = min(hi - lo, remaining)
        w[int(i)] += add
        remaining -= add
        if remaining <= 1e-12:
            break
    return w


def collapsed_column_means(mixed: np.ndarray, n_assets: int, n_draw: int) -> np.ndarray:
    """Mean return of each traded name, averaging real and synth copy columns."""
    mu = np.asarray(mixed, dtype=float).mean(axis=0)
    n_assets = int(n_assets)
    out = mu[:n_assets].copy()
    n_draw = int(n_draw)
    if n_draw == 0:
        return out
    for i in range(n_assets):
        start = n_assets + i * n_draw
        out[i] = float(np.mean(np.concatenate(([mu[i]], mu[start : start + n_draw]))))
    return out



def project_sum_to_one_box(
    weights: np.ndarray,
    w_min: float = 0.0,
    w_max: float = 1.0,
    n_iter: int = 64,
) -> np.ndarray:
    """Project onto ``sum w = 1`` and ``w_min ≤ w_i ≤ w_max``."""
    w = np.asarray(weights, dtype=float).reshape(-1).copy()
    n = int(w.size)
    if n < 1:
        raise ValueError("weights is empty")
    lo = float(w_min)
    hi = float(w_max)
    if hi + 1e-12 < lo:
        raise ValueError(f"w_max {hi} is below w_min {lo}")
    if n * hi < 1.0 - 1e-9:
        raise ValueError(f"max_weight={hi} cannot sum to 1 with {n} assets")
    if n * lo > 1.0 + 1e-9:
        raise ValueError(f"w_min={lo} already exceeds sum 1 with {n} assets")
    w = np.clip(w, lo, hi)
    for _ in range(int(n_iter)):
        residual = 1.0 - float(w.sum())
        if abs(residual) < 1e-12:
            return w
        if residual > 0:
            mask = (hi - w) > 1e-12
        else:
            mask = (w - lo) > 1e-12
        if not np.any(mask):
            break
        w[mask] += residual / float(mask.sum())
        w = np.clip(w, lo, hi)
    total = float(w.sum())
    return w / total if not np.isclose(total, 0.0) else np.ones(n) / n


def mean_var_weights(
    returns: np.ndarray,
    allow_short: bool = True,
    ridge: float = 1e-6,
    objective: str = "mean_variance",
) -> np.ndarray:
    """Sum-to-1 portfolio weights. Default is mean-variance; no max-weight cap here."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; expected one of {OBJECTIVES}")
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Need a (T, n_assets) return matrix with T >= 2")
    mu = matrix.mean(axis=0)
    n_obs, n_feat = matrix.shape
    if objective == "max_return":
        weights = np.zeros(n_feat)
        weights[int(np.argmax(mu))] = 1.0
        return _apply_short_and_budget(weights, allow_short=False)
    eff_ridge = float(ridge)
    if n_obs <= n_feat:
        avg_var = float(np.mean(np.var(matrix, axis=0, ddof=1))) if n_obs > 1 else 1.0
        eff_ridge = max(eff_ridge, 1e-4 * max(avg_var, 1e-12))
    cov = np.cov(matrix.T, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    cov = 0.5 * (cov + cov.T) + eff_ridge * np.eye(n_feat)
    return _closed_form_weights(mu, cov, objective=objective, allow_short=allow_short)


def tile_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(arr)
    length = int(length)
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    n = int(arr.shape[0])
    if n < 1:
        raise ValueError("cannot tile an empty array")
    if n >= length:
        return arr[:length]
    reps = int(np.ceil(length / n))
    return np.concatenate([arr] * reps, axis=0)[:length]


MIX_MODES = ("column", "row")


def mix_train_with_regime_paths(
    real_train: np.ndarray,
    synth_paths: np.ndarray | None,
    mix_len: int | None = None,
) -> tuple[np.ndarray, int]:
    """Column-stack real lookback with ``n`` synthetic 60×10 paths."""
    real_train = np.asarray(real_train, dtype=float)
    if real_train.ndim != 2:
        raise ValueError(f"real_train must be 2D, got {real_train.shape}")
    n_synth = 0 if synth_paths is None else int(np.asarray(synth_paths).shape[0])
    if n_synth == 0:
        return real_train.copy(), 0
    synth_paths = np.asarray(synth_paths, dtype=float)
    if synth_paths.ndim != 3:
        raise ValueError(f"synth_paths must be (n_synth, H, n_assets), got {synth_paths.shape}")
    if synth_paths.shape[2] != real_train.shape[1]:
        raise ValueError(
            f"Asset mismatch: real {real_train.shape[1]} vs synth {synth_paths.shape[2]}"
        )
    t_mix = min(int(real_train.shape[0]), int(mix_len or real_train.shape[0]))
    if t_mix < 2:
        raise ValueError(f"Need at least 2 mixed rows; got T={t_mix}")
    real_block = real_train[-t_mix:]
    extra_cols = []
    for asset_idx in range(real_train.shape[1]):
        series = np.empty((t_mix, n_synth), dtype=float)
        for path_idx in range(n_synth):
            series[:, path_idx] = tile_to_length(synth_paths[path_idx, :, asset_idx], t_mix)
        extra_cols.append(series)
    return np.hstack([real_block, np.hstack(extra_cols)]), n_synth


def mix_train_row_append(
    real_train: np.ndarray,
    synth_paths: np.ndarray | None,
) -> tuple[np.ndarray, int]:
    """Row-append synthetic days under the real lookback.

    ``synth_paths`` is ``(n_synth, H, n_assets)`` or a 2D block ``(T, n_assets)``.
    The traded universe stays 10 assets. ``n = 0`` / empty extra returns the real
    lookback unchanged.
    """
    real_train = np.asarray(real_train, dtype=float)
    if real_train.ndim != 2:
        raise ValueError(f"real_train must be 2D, got {real_train.shape}")
    if synth_paths is None:
        return real_train.copy(), 0
    extra = np.asarray(synth_paths, dtype=float)
    if extra.size == 0:
        return real_train.copy(), 0
    if extra.ndim == 3:
        extra = extra.reshape(extra.shape[0] * extra.shape[1], extra.shape[2])
    if extra.ndim != 2:
        raise ValueError(f"synth extra must be 2D or 3D, got {extra.shape}")
    if extra.shape[1] != real_train.shape[1]:
        raise ValueError(
            f"Asset mismatch: real {real_train.shape[1]} vs synth {extra.shape[1]}"
        )
    return np.vstack([real_train, extra]), int(extra.shape[0])


def collapse_weights(weights: np.ndarray, n_assets: int, n_draw: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if n_draw == 0:
        total = float(weights[:n_assets].sum())
        return weights[:n_assets] / total if not np.isclose(total, 0.0) else np.ones(n_assets) / n_assets
    collapsed = weights[:n_assets].copy()
    for i in range(n_assets):
        start = n_assets + i * n_draw
        collapsed[i] += weights[start : start + n_draw].sum()
    total = float(collapsed.sum())
    return collapsed / total if not np.isclose(total, 0.0) else np.ones(n_assets) / n_assets
