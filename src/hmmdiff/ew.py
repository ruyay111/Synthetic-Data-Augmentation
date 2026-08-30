"""Equal-weight regime pipeline helpers.

Isolated from the A001 path. Clustering and diffusion windows use the full 10-asset overlap
(2001–2022). The HMM emission is the same equal-weight series, but the HMM still trains only on
2001–2014. Train/test follows the A001 calendar cutoff (2014-01-03 / 2014-01-06) and there is no
validation slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path
from .data import (
    asset_columns,
    build_multivariate_log_returns,
    build_returns,
    save_returns,
)


def load_ew_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    path = config_path(cfg, "returns")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_label_regimes_ew.py first."
        )
    return pd.read_parquet(path)


def standardized_asset_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-asset z-scored log returns on the ten-asset overlap (ddof=1)."""
    panel = build_multivariate_log_returns(cfg)
    return (panel - panel.mean()) / panel.std()


def equal_weight_return(z_panel: pd.DataFrame) -> pd.Series:
    """Equal-weight average of per-asset standardized returns."""
    return z_panel.mean(axis=1)


def split_counts(returns: pd.DataFrame) -> dict[str, Any]:
    """Train/test counts and dates from the ``split`` column. Train must be a date prefix."""
    n_total = len(returns)
    n_train = int((returns["split"] == "train").sum())
    if n_train == 0:
        raise ValueError("returns has no train rows")
    if not (returns["split"].to_numpy()[:n_train] == "train").all():
        raise ValueError("train rows must form a date prefix")
    dates = pd.to_datetime(returns["date"])
    return {
        "n_returns": n_total,
        "n_train": n_train,
        "n_test": n_total - n_train,
        "start_date": str(dates.iloc[0].date()),
        "train_end_date": str(dates.iloc[n_train - 1].date()),
        "test_start_date": str(dates.iloc[n_train].date()) if n_train < n_total else None,
        "end_date": str(dates.iloc[-1].date()),
    }


def build_ew_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    """Equal-weight standardized returns, tagged with the A001 calendar train/test split.

    Each of the ten assets is z-scored on the full overlap. The equal-weight average is z-scored
    again; that series is both the Vol_Regime input and the HMM emission.
    """
    a001 = build_returns(cfg)
    a001_by_date = a001.copy()
    a001_by_date["date"] = pd.to_datetime(a001_by_date["date"])
    a001_by_date = a001_by_date.set_index("date")

    z_panel = standardized_asset_returns(cfg)
    ew = equal_weight_return(z_panel)
    z_ew = (ew - ew.mean()) / ew.std()

    aligned = a001_by_date.reindex(z_panel.index)
    if aligned["split"].isna().any():
        missing = int(aligned["split"].isna().sum())
        raise ValueError(f"{missing} equal-weight dates are missing from the A001 series.")

    return pd.DataFrame(
        {
            "date": z_panel.index.strftime("%Y-%m-%d"),
            "ew_return": ew.to_numpy(dtype=float),
            "z_return": z_ew.to_numpy(dtype=float),
            "split": aligned["split"].to_numpy(),
        }
    )


def ew_scale(cfg: dict[str, Any]) -> dict[str, Any]:
    """Moments needed to map 10-asset raw log returns back to EW HMM emission units."""
    panel = build_multivariate_log_returns(cfg)
    z_panel = (panel - panel.mean()) / panel.std()
    ew = equal_weight_return(z_panel)
    return {
        "label_source": "equal_weight",
        "asset_columns": asset_columns(cfg),
        "asset_mean": panel.mean().to_numpy(dtype=float).tolist(),
        "asset_std": panel.std().to_numpy(dtype=float).tolist(),
        "ew_mean": float(ew.mean()),
        "ew_std": float(ew.std()),
    }


def emission_from_log_panel(
    panel: np.ndarray, scale: dict[str, Any]
) -> np.ndarray:
    """Map a raw log-return panel ``(..., n_assets)`` to EW z-score emission units."""
    mean = np.asarray(scale["asset_mean"], dtype=float)
    std = np.asarray(scale["asset_std"], dtype=float)
    z = (np.asarray(panel, dtype=float) - mean) / std
    ew = z.mean(axis=-1)
    return (ew - float(scale["ew_mean"])) / float(scale["ew_std"])


def train_mask(returns: pd.DataFrame) -> np.ndarray:
    return returns["split"].to_numpy() == "train"


def test_mask(returns: pd.DataFrame) -> np.ndarray:
    return returns["split"].to_numpy() == "test"


def split_series(returns: pd.DataFrame, column: str = "z_return") -> tuple[np.ndarray, np.ndarray]:
    """Train and test slices of one column."""
    values = returns[column].to_numpy(dtype=float)
    return values[train_mask(returns)], values[test_mask(returns)]


def split_labels(returns: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train and test slices of full-sample EW labels."""
    labels = np.asarray(labels)
    if len(labels) != len(returns):
        raise ValueError(
            f"Regime labels cover {len(labels)} days but the EW series has {len(returns)}."
        )
    return labels[train_mask(returns)].astype(int), labels[test_mask(returns)].astype(int)


def _emission_frame(emissions: np.ndarray, regimes: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"regime": np.asarray(regimes).astype(int), "emission": emissions})
    out["emission_lag"] = out["emission"].shift(1)
    return out[["regime", "emission", "emission_lag"]]


def build_train_test_frames(
    returns: pd.DataFrame, regime_labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HMM frames for the EW pipeline: full train (2001–2014) and test (2014–2022)."""
    train_series, test_series = split_series(returns)
    train_labels, test_labels = split_labels(returns, regime_labels)
    return (
        _emission_frame(train_series, train_labels),
        _emission_frame(test_series, test_labels),
    )


def align_ew_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """Full-sample 10-asset log returns aligned to EW labels by date.

    Clustering labels and diffusion windows both cover 2001–2022. The HMM still trains only on
    2001–2014 via ``build_train_test_frames``.
    Returns ``panel``, ``labels``, ``dates``, and the aligned EW emission series.
    """
    labels = np.asarray(regime_labels)
    if len(labels) != len(returns):
        raise ValueError(
            f"Regime labels cover {len(labels)} days but the EW series has {len(returns)}."
        )

    panel = build_multivariate_log_returns(cfg)
    frame = returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    labeled = pd.DataFrame(
        {
            "date": frame["date"].to_numpy(),
            "regime": labels.astype(int),
            "z_return": frame["z_return"].to_numpy(dtype=float),
        }
    ).set_index("date")
    common = panel.index.intersection(labeled.index)
    if common.empty:
        raise ValueError("No overlapping dates between the multivariate panel and EW labels.")

    aligned_panel = panel.loc[common, asset_columns(cfg)].to_numpy(dtype=float)
    aligned_labels = labeled.loc[common, "regime"].to_numpy(dtype=int)
    aligned_ew = labeled.loc[common, "z_return"].to_numpy(dtype=float)
    return aligned_panel, aligned_labels, common, aligned_ew


def save_ew_returns(returns: pd.DataFrame, path: Path) -> None:
    save_returns(returns, path)
