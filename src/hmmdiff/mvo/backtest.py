"""Rolling 60/60 MVO: hmm-diffusion vs uncondi-diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hmmdiff.constants import HIGH_VOL_REGIMES as CONST_HIGH_VOL, HIGH_VOL_SHARE
from .hmm_forecast import OpenLoopWalk
from .mixed_sample import sample_uncond_simple_paths
from .portfolio_core import (
    MIX_MODES,
    annualized_return,
    annualized_variance,
    calmar_ratio,
    collapse_weights,
    collapsed_column_means,
    greedy_max_return_box,
    mean_var_weights,
    mix_train_row_append,
    mix_train_with_regime_paths,
    project_sum_to_one_box,
    sharpe_ratio,
)
from .specialist_sample import sample_simple_paths, sample_simple_rows, synth_rows_for_share

HIGH_VOL_REGIMES = set(CONST_HIGH_VOL)


def window_vol_bucket(true_labels: np.ndarray, high_vol_regimes: set[int] | None = None) -> str:
    """High vol if at least half of the hold's true labels are in regimes 3 and 4"""
    labels = np.asarray(true_labels, dtype=int).reshape(-1)
    high = HIGH_VOL_REGIMES if high_vol_regimes is None else set(high_vol_regimes)
    share = float(np.isin(labels, list(high)).mean()) if labels.size else 0.0
    return "high vol" if share >= HIGH_VOL_SHARE else "low vol"


def _hold_seed(random_state: int, hold_start: int, n_synth: int) -> int:
    return int(random_state) + int(hold_start) + int(n_synth)


def _weights_for_hold(
    real_lookback: np.ndarray,
    synth_paths: np.ndarray,
    mix_len: int,
    allow_short: bool,
    ridge: float,
    max_weight: float | None,
    objective: str = "mean_variance",
    mix_mode: str = "column",
) -> np.ndarray:
    if mix_mode not in MIX_MODES:
        raise ValueError(f"Unknown mix_mode {mix_mode!r}; expected one of {MIX_MODES}")
    n_assets = real_lookback.shape[1]
    if mix_mode == "row":
        mixed, _n_draw = mix_train_row_append(real_lookback, synth_paths)
        raw = mean_var_weights(mixed, allow_short=allow_short, ridge=ridge, objective=objective)
        collapsed = collapse_weights(raw, n_assets=n_assets, n_draw=0)
        if max_weight is None:
            return collapsed
        w_min = 0.0 if not allow_short else -float(max_weight)
        if objective == "max_return":
            mu10 = mixed.mean(axis=0)
            return greedy_max_return_box(mu10, w_min=w_min, w_max=float(max_weight))
        return project_sum_to_one_box(collapsed, w_min=w_min, w_max=float(max_weight))

    mixed, n_draw = mix_train_with_regime_paths(real_lookback, synth_paths, mix_len=mix_len)
    raw = mean_var_weights(mixed, allow_short=allow_short, ridge=ridge, objective=objective)
    collapsed = collapse_weights(raw, n_assets=n_assets, n_draw=n_draw)
    if max_weight is None:
        return collapsed
    w_min = 0.0 if not allow_short else -float(max_weight)
    if objective == "max_return":
        mu10 = collapsed_column_means(mixed, n_assets, n_draw)
        return greedy_max_return_box(mu10, w_min=w_min, w_max=float(max_weight))
    return project_sum_to_one_box(collapsed, w_min=w_min, w_max=float(max_weight))


def lookback_and_hold(
    simple: pd.DataFrame,
    val_dates: np.ndarray,
    val_start: int,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(lookback × n_assets, hold × n_assets)`` simple returns, or None if incomplete."""
    dates = pd.DatetimeIndex(pd.to_datetime(val_dates))
    start = dates[val_start]
    hold_dates = dates[val_start : val_start + horizon]
    if len(hold_dates) != horizon:
        return None
    prior = simple.loc[simple.index < start]
    if len(prior) < lookback:
        return None
    real_lb = prior.iloc[-lookback:]
    hold = simple.reindex(hold_dates)
    if real_lb.shape[0] != lookback or hold.shape[0] != horizon:
        return None
    if real_lb.isna().any().any() or hold.isna().any().any():
        return None
    return real_lb.to_numpy(dtype=float), hold.to_numpy(dtype=float)


@dataclass
class MVOBacktestResult:
    windows: pd.DataFrame
    summary: pd.DataFrame


def run_mvo_backtest(
    simple: pd.DataFrame,
    val_dates: np.ndarray,
    true_labels: np.ndarray,
    walk: OpenLoopWalk,
    specialist_pools: dict[int, np.ndarray],
    uncond_windows: np.ndarray,
    mix_grid: list[int],
    *,
    lookback: int = 60,
    horizon: int = 60,
    random_state: int = 42,
    allow_short: bool = True,
    ridge: float = 1e-6,
    max_weight: float | None = None,
    objective: str = "mean_variance",
    mix_mode: str = "column",
) -> MVOBacktestResult:
    """Score non-overlapping holds on a date-indexed 10-asset simple-return panel.

    ``true_labels``, ``val_dates``, and ``walk`` share the validation length. Incomplete
    lookback or hold rows are skipped. ``mix_mode='column'`` is the original expanded
    universe; ``mix_mode='row'`` appends synthetic days so the synth **row share**
    equals each ``mix_grid`` value in percent (0, 10, …, 90). For column mode,
    ``mix_grid`` is still a count of extra windows.
    """
    if mix_mode not in MIX_MODES:
        raise ValueError(f"Unknown mix_mode {mix_mode!r}; expected one of {MIX_MODES}")
    simple = simple.copy()
    simple.index = pd.DatetimeIndex(simple.index)
    labels = np.asarray(true_labels, dtype=int).reshape(-1)
    dates = pd.DatetimeIndex(pd.to_datetime(val_dates))
    n_val = int(labels.size)
    if len(dates) != n_val:
        raise ValueError("val_dates length does not match true_labels")
    if walk.open_loop_pi.shape[0] != n_val:
        raise ValueError("walk length does not match true_labels")

    rows: list[dict] = []
    for hold_idx, val_start in enumerate(walk.hold_starts.tolist()):
        sliced = lookback_and_hold(simple, dates, int(val_start), lookback, horizon)
        if sliced is None:
            continue
        real_lb, hold = sliced
        k_star = int(walk.hold_k_star[hold_idx])
        hold_labels = labels[val_start : val_start + horizon]
        bucket = window_vol_bucket(hold_labels)
        start_date = dates[val_start]
        end_date = dates[val_start + horizon - 1]
        if k_star not in specialist_pools:
            raise KeyError(f"No specialist pool for predicted regime {k_star}")

        for n_synth in mix_grid:
            seed = _hold_seed(random_state, val_start, n_synth)
            rng_hmm = np.random.default_rng(seed)
            rng_uncond = np.random.default_rng(seed)
            if mix_mode == "row":
                share = float(n_synth) / 100.0
                n_rows = synth_rows_for_share(lookback, share)
                hmm_paths = sample_simple_rows(
                    specialist_pools[k_star], n_rows, horizon, rng_hmm
                )
                uncond_paths = sample_simple_rows(
                    uncond_windows, n_rows, horizon, rng_uncond
                )
                synth_pct = (
                    0.0
                    if n_rows == 0
                    else 100.0 * n_rows / (lookback + n_rows)
                )
            else:
                hmm_paths = sample_simple_paths(
                    specialist_pools[k_star], n_synth, horizon, rng_hmm
                )
                uncond_paths = sample_uncond_simple_paths(
                    uncond_windows, n_synth, horizon, rng_uncond
                )
                synth_pct = float(n_synth) * 10.0
            for method, paths in (("hmm-diffusion", hmm_paths), ("uncondi-diffusion", uncond_paths)):
                weights = _weights_for_hold(
                    real_lb,
                    paths,
                    mix_len=lookback,
                    allow_short=allow_short,
                    ridge=ridge,
                    max_weight=max_weight,
                    objective=objective,
                    mix_mode=mix_mode,
                )
                port = hold @ weights
                rows.append(
                    {
                        "method": method,
                        "n_synth": int(n_synth),
                        "synth_pct": float(synth_pct),
                        "hold_start": int(val_start),
                        "start_date": start_date,
                        "end_date": end_date,
                        "k_star": k_star,
                        "bucket": bucket,
                        "sharpe": sharpe_ratio(port),
                        "calmar": calmar_ratio(port),
                        "return": annualized_return(port),
                        "variance": annualized_variance(port),
                        "n_obs": int(port.size),
                    }
                )

    windows = pd.DataFrame(rows)
    if windows.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            windows.groupby(["method", "n_synth", "synth_pct", "bucket"], dropna=False)
            .agg(
                n_windows=("sharpe", "size"),
                sharpe_mean=("sharpe", "mean"),
                sharpe_median=("sharpe", "median"),
                calmar_mean=("calmar", "mean"),
                calmar_median=("calmar", "median"),
                return_mean=("return", "mean"),
                return_median=("return", "median"),
                variance_mean=("variance", "mean"),
                variance_median=("variance", "median"),
            )
            .reset_index()
        )
    return MVOBacktestResult(windows=windows, summary=summary)
