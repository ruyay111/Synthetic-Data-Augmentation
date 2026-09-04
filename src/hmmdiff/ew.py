"""Utility API functions for the equal-weight regime pipeline.

Implementation lives in EqualWeightProcessor. Clustering and diffusion windows
use the full 10-asset overlap (2001-2022). The HMM emission is the same
equal-weight series, but the HMM still trains only on 2001-2014.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hmmdiff.data_collection.equal_weight_processor import EqualWeightProcessor

__all__ = [
    "align_ew_diffusion_panel",
    "build_ew_returns",
    "build_train_test_frames",
    "emission_from_log_panel",
    "equal_weight_return",
    "ew_scale",
    "load_ew_returns",
    "save_ew_returns",
    "split_counts",
    "split_labels",
    "split_series",
    "standardized_asset_returns",
    "test_mask",
    "train_mask",
]


def load_ew_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    return EqualWeightProcessor(cfg).LoadEwReturns(cfg)


def standardized_asset_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    return EqualWeightProcessor(cfg).StandardizedAssetReturns(cfg)


def equal_weight_return(z_panel: pd.DataFrame) -> pd.Series:
    return EqualWeightProcessor().EqualWeightReturn(z_panel)


def split_counts(returns: pd.DataFrame) -> dict[str, Any]:
    return EqualWeightProcessor().SplitCounts(returns)


def build_ew_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    return EqualWeightProcessor(cfg).BuildEwReturns(cfg)


def ew_scale(cfg: dict[str, Any]) -> dict[str, Any]:
    return EqualWeightProcessor(cfg).EwScale(cfg)


def emission_from_log_panel(panel: np.ndarray, scale: dict[str, Any]) -> np.ndarray:
    return EqualWeightProcessor().EmissionFromLogPanel(panel, scale)


def train_mask(returns: pd.DataFrame) -> np.ndarray:
    return EqualWeightProcessor().TrainMask(returns)


def test_mask(returns: pd.DataFrame) -> np.ndarray:
    return EqualWeightProcessor().TestMask(returns)


def split_series(returns: pd.DataFrame, column: str = "z_return") -> tuple[np.ndarray, np.ndarray]:
    return EqualWeightProcessor().SplitSeries(returns, column=column)


def split_labels(returns: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return EqualWeightProcessor().SplitLabels(returns, labels)


def build_train_test_frames(
    returns: pd.DataFrame, regime_labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return EqualWeightProcessor().BuildTrainTestFrames(returns, regime_labels)


def align_ew_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    return EqualWeightProcessor(cfg).AlignEwDiffusionPanel(returns, regime_labels, cfg)


def save_ew_returns(returns: pd.DataFrame, path: Path) -> None:
    EqualWeightProcessor().SaveEwReturns(returns, path)
