"""Rolling mean-variance backtest with mixed real and synthetic days.

PortfolioBacktest wraps lookback construction, mix-share sampling, and
Markowitz scoring for HMM-Diffusion versus unconditional diffusion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from hmmdiff.mvo.backtest import (
    MVOBacktestResult,
    lookback_and_hold,
    run_mvo_backtest,
    window_vol_bucket,
)
from hmmdiff.mvo.hmm_forecast import OpenLoopWalk
from hmmdiff.mvo.scale_check import run_scale_check


class PortfolioBacktest:
    """MVO evaluation of synthetic augmentation on the EW ten-asset panel."""

    def WindowVolBucket(
        self, true_labels: np.ndarray, high_vol_regimes: set[int] | None = None
    ) -> str:
        """
        Label a hold as high vol or low vol.

        Parameters:
        true_labels: numpy.ndarray
            True regime labels on the hold.
        high_vol_regimes: set or None
            Regime ids treated as high volatility.

        Return:
           str, either 'high vol' or 'low vol'.
        """
        return window_vol_bucket(true_labels, high_vol_regimes=high_vol_regimes)

    def LookbackAndHold(
        self,
        simple: pd.DataFrame,
        val_dates: np.ndarray,
        val_start: int,
        lookback: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Real lookback and hold simple-return blocks.

        Parameters:
        simple: pandas.DataFrame
            Date-indexed simple returns.
        val_dates: numpy.ndarray
            Validation calendar.
        val_start: int
            Hold start index.
        lookback: int
            Number of real days.
        horizon: int
            Hold length.

        Return:
           tuple of arrays or None if the window is incomplete.
        """
        return lookback_and_hold(simple, val_dates, val_start, lookback, horizon)

    def RunMvoBacktest(
        self,
        simple: pd.DataFrame,
        val_dates: np.ndarray,
        true_labels: np.ndarray,
        walk: OpenLoopWalk,
        specialist_pools: dict[int, np.ndarray],
        uncond_windows: np.ndarray,
        mix_grid: list[int],
        **kwargs: Any,
    ) -> MVOBacktestResult:
        """
        Score non-overlapping holds on mixed real and synthetic days.

        Parameters:
        simple: pandas.DataFrame
            Date-indexed 10-asset simple returns.
        val_dates: numpy.ndarray
            Validation dates aligned to true_labels.
        true_labels: numpy.ndarray
            True regime labels on validation.
        walk: OpenLoopWalk
            HMM open-loop walk.
        specialist_pools: dict
            Regime index to specialist windows.
        uncond_windows: numpy.ndarray
            Unconditional diffusion windows.
        mix_grid: list
            Mix shares in percent (row mode) or extra-window counts (column mode).

        Return:
           MVOBacktestResult with windows and summary frames.
        """
        return run_mvo_backtest(
            simple,
            val_dates,
            true_labels,
            walk,
            specialist_pools,
            uncond_windows,
            mix_grid,
            **kwargs,
        )

    def RunScaleCheck(self, *args, **kwargs):
        """
        Compare synthetic vs real simple-return scales before MVO.

        Return:
           ScaleCheckResult.
        """
        return run_scale_check(*args, **kwargs)
