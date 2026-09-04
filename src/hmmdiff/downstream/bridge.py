"""Connect stitched returns to downstream price-based evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hmmdiff.config import config_path
from hmmdiff.data import a001_log_return_scale
from hmmdiff import models, stitch

from .data_utils import load_real_data


def load_real_benchmark(
    cfg: dict[str, Any],
    *,
    test_start_year: str = "2014",
    price_col: str = "A001",
) -> pd.DataFrame:
    """Benchmark panel for downstream evaluation (matches pipeline_1219_benchmark_A1)."""
    csv_path = config_path(cfg, "raw_csv")
    frame = load_real_data(str(csv_path), test_start_year=test_start_year, benchmark=True)
    frame[price_col] = frame[price_col].astype(float)
    frame[f"{price_col}_log"] = np.log(frame[price_col])
    return frame


def build_synthetic_price_frame(
    stitched_z_returns: np.ndarray,
    returns: pd.DataFrame,
    *,
    start_price: float,
    price_col: str = "A001",
) -> pd.DataFrame:
    """Integrate z-scored stitched log returns into a positive price series.

    The downstream pipeline expects a DataFrame like ``row0_prices`` from the GAN benchmark:
    columns ``A001`` and ``A001_log``, indexed by row order (not calendar dates).
    """
    log_returns = z_returns_to_log_returns(stitched_z_returns, returns)
    if log_returns.size == 0:
        raise ValueError("stitched_z_returns is empty")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    prices = start_price * np.exp(np.cumsum(log_returns))
    prices = np.maximum(prices, 1e-8)
    frame = pd.DataFrame({price_col: prices})
    frame[f"{price_col}_log"] = np.log(frame[price_col])
    return frame


def build_uncond_synth_price_frame(
    windows: np.ndarray,
    *,
    start_price: float,
    price_col: str = "A001",
    asset_idx: int = 0,
    separate_windows: bool = True,
) -> pd.DataFrame:
    """Integrate regime-free diffusion windows into a price series for the vol forest.

    ``windows`` is ``(n_windows, seq_len, n_assets)`` raw log returns. Each window is
    independent, so a NaN row is inserted between windows when ``separate_windows``
    is true. Rolling TA features then do not cross window boundaries. There is no
    calendar alignment with the real benchmark.
    """
    arr = np.asarray(windows, dtype=float)
    if arr.ndim == 2:
        log_ret = arr
    elif arr.ndim == 3:
        log_ret = arr[:, :, int(asset_idx)]
    else:
        raise ValueError(f"windows must be 2D or 3D, got {arr.shape}")
    if log_ret.size == 0:
        raise ValueError("windows is empty")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    log_col = f"{price_col}_log"
    log_start = float(np.log(start_price))
    chunks: list[pd.DataFrame] = []
    n_windows = int(log_ret.shape[0])
    for i in range(n_windows):
        log_price = log_start + np.cumsum(log_ret[i])
        price = np.maximum(np.exp(log_price), 1e-8)
        chunks.append(pd.DataFrame({price_col: price, log_col: np.log(price)}))
        if separate_windows and i < n_windows - 1:
            chunks.append(pd.DataFrame({price_col: [np.nan], log_col: [np.nan]}))
    return pd.concat(chunks, ignore_index=True)


def z_returns_to_log_returns(stitched_z_returns: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
    """Map HMM emission units back to raw A001 log returns."""
    mu, sd = a001_log_return_scale(returns)
    return np.asarray(stitched_z_returns, dtype=float) * sd + mu


def benchmark_emissions(
    returns: pd.DataFrame,
    benchmark_index: pd.DatetimeIndex,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Z-scored HMM emissions on trading days shared with the downstream benchmark panel."""
    frame = returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    test = frame.loc[frame["split"] == "test"].set_index("date")
    common = benchmark_index.intersection(test.index)
    if common.empty:
        raise ValueError("No overlapping dates between benchmark panel and returns test split.")
    emissions = test.loc[common, "z_return"].to_numpy(dtype=float)
    return emissions, common


@dataclass(frozen=True)
class AlignedDownstreamResult:
    """Synthetic benchmark panel plus the HMM stitch artifacts used to build it."""

    df_synth: pd.DataFrame
    emissions: np.ndarray
    regime_est: np.ndarray
    stitched_z: np.ndarray
    dates: pd.DatetimeIndex


def build_aligned_downstream_synth(
    *,
    returns: pd.DataFrame,
    generated_images: dict[int, np.ndarray],
    supervised: Any,
    init_dist: np.ndarray,
    n_regimes: int,
    df_real: pd.DataFrame,
    price_col: str = "A001",
    state_method: str = "smooth",
    train_emissions: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> AlignedDownstreamResult:
    """Stitch diffusion pools along HMM states on the benchmark period (2014+).

    The supervised HMM is fit on 2001–2014 train only. ``state_method``:

    - ``smooth``: full-sample forward-backward (default; uses future emissions).
    - ``filter``: causal forward filter. Optional ``train_emissions`` are
      filtered first so the 2014+ prior does not use later returns.
    - ``simulate``: sample a path from the transition matrix; no real 2014+ emissions.
    """
    emissions, dates = benchmark_emissions(returns, df_real.index)
    method = str(state_method).lower()
    if method == "smooth":
        _, regime_est = models.estimate_states(supervised, emissions, init_dist, n_regimes)
    elif method == "filter":
        _, regime_est = models.filter_states(
            supervised, emissions, init_dist, train_emissions=train_emissions
        )
    elif method == "simulate":
        sim_rng = rng if rng is not None else np.random.default_rng(0)
        regime_est = models.simulate_regime_path(
            supervised.transmat, init_dist, len(emissions), sim_rng
        )
    else:
        raise ValueError(f"Unknown state_method {state_method!r}; use filter, smooth, or simulate.")
    stitched_z = stitch.stitch(generated_images, regime_est)
    if len(stitched_z) != len(dates):
        raise ValueError(
            f"Stitched length {len(stitched_z)} != aligned benchmark days {len(dates)}."
        )

    start_price = float(df_real.loc[dates[0], price_col])
    first_log_return = float(z_returns_to_log_returns(stitched_z[:1], returns)[0])
    # ``build_synthetic_price_frame`` applies the first return on day 0; back out the prior close.
    start_price = start_price / np.exp(first_log_return)
    frame = build_synthetic_price_frame(
        stitched_z, returns, start_price=start_price, price_col=price_col
    )
    frame.index = dates
    return AlignedDownstreamResult(
        df_synth=frame,
        emissions=emissions,
        regime_est=regime_est,
        stitched_z=stitched_z,
        dates=dates,
    )
