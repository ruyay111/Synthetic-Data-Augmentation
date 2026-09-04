"""Utility API functions for price loading, return preprocessing, and train/test splits.

Implementation lives in PriceReturnProcessor. These snake_case functions are the
standalone entry points used by scripts, notebooks, and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hmmdiff.data_collection.data_processor import PriceReturnProcessor, Splits

__all__ = [
    "Splits",
    "a001_log_return_scale",
    "align_diffusion_panel",
    "align_train_diffusion_panel",
    "asset_columns",
    "build_model_frames",
    "build_multivariate_log_returns",
    "build_returns",
    "compute_splits",
    "load_price_panel",
    "load_prices",
    "load_returns",
    "save_returns",
    "split_counts",
    "split_labels",
    "split_series",
    "splits_from_returns",
    "test_mask",
    "train_mask",
    "train_returns",
]


def compute_splits(
    n_total: int, cfg: dict[str, Any], dates: pd.Series | np.ndarray | None = None
) -> Splits:
    return PriceReturnProcessor(cfg).ComputeSplits(n_total, cfg, dates=dates)


def splits_from_returns(returns: pd.DataFrame, cfg: dict[str, Any]) -> Splits:
    return PriceReturnProcessor(cfg).SplitsFromReturns(returns, cfg)


def split_counts(returns: pd.DataFrame) -> dict[str, Any]:
    return PriceReturnProcessor().SplitCounts(returns)


def train_mask(returns: pd.DataFrame) -> np.ndarray:
    return PriceReturnProcessor().TrainMask(returns)


def test_mask(returns: pd.DataFrame) -> np.ndarray:
    return PriceReturnProcessor().TestMask(returns)


def load_prices(cfg: dict[str, Any]) -> pd.Series:
    return PriceReturnProcessor(cfg).LoadPrices(cfg)


def build_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    return PriceReturnProcessor(cfg).BuildReturns(cfg)


def a001_log_return_scale(returns: pd.DataFrame) -> tuple[float, float]:
    return PriceReturnProcessor().A001LogReturnScale(returns)


def save_returns(returns: pd.DataFrame, path: Path) -> None:
    PriceReturnProcessor().SaveReturns(returns, path)


def load_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    return PriceReturnProcessor(cfg).LoadReturns(cfg)


def asset_columns(cfg: dict[str, Any]) -> list[str]:
    return PriceReturnProcessor(cfg).AssetColumns(cfg)


def load_price_panel(cfg: dict[str, Any]) -> pd.DataFrame:
    return PriceReturnProcessor(cfg).LoadPricePanel(cfg)


def build_multivariate_log_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    return PriceReturnProcessor(cfg).BuildMultivariateLogReturns(cfg)


def align_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    return PriceReturnProcessor(cfg).AlignDiffusionPanel(returns, regime_labels, cfg)


def align_train_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    return PriceReturnProcessor(cfg).AlignTrainDiffusionPanel(returns, regime_labels, cfg)


def train_returns(returns: pd.DataFrame) -> np.ndarray:
    return PriceReturnProcessor().TrainReturns(returns)


def split_series(returns: pd.DataFrame, column: str = "z_return") -> tuple[np.ndarray, np.ndarray]:
    return PriceReturnProcessor().SplitSeries(returns, column=column)


def split_labels(returns: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return PriceReturnProcessor().SplitLabels(returns, labels)


def build_model_frames(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return PriceReturnProcessor(cfg).BuildModelFrames(returns, regime_labels, cfg)
