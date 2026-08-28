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
) -> AlignedDownstreamResult:
    """Stitch diffusion pools along HMM states estimated on the benchmark period (2014+).

    The supervised HMM is fit on pre-2014 inner training data only; regimes on the benchmark window
    are *estimated* from real z-scored returns (no Vol_Regime labels on the test period).
    """
    emissions, dates = benchmark_emissions(returns, df_real.index)
    _, regime_est = models.estimate_states(supervised, emissions, init_dist, n_regimes)
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
