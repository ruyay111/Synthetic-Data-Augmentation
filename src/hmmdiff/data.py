"""Price loading, return preprocessing, and train/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_path


@dataclass(frozen=True)
class Splits:
    """Outer train/test split and inner train/validation split."""

    n_total: int
    train_end: int
    train_val_split: int

    @property
    def n_train(self) -> int:
        return self.train_end

    @property
    def n_test(self) -> int:
        return self.n_total - self.train_end

    @property
    def n_train_inner(self) -> int:
        return self.train_val_split

    @property
    def n_val(self) -> int:
        return self.train_end - self.train_val_split


def compute_splits(n_total: int, cfg: dict[str, Any]) -> Splits:
    train_end = int(n_total * cfg["data"]["train_fraction"])
    train_val_split = int(train_end * cfg["data"]["train_val_fraction"])
    return Splits(n_total=n_total, train_end=train_end, train_val_split=train_val_split)


def load_prices(cfg: dict[str, Any]) -> pd.Series:
    """Closing price series for the configured column, with the stale leading rows dropped."""
    raw_path = config_path(cfg, "raw_csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw price file not found: {raw_path}")
    frame = pd.read_csv(raw_path)
    frame = frame.set_index(cfg["data"]["date_column"])
    prices = frame[cfg["data"]["price_column"]].iloc[cfg["data"]["skip_rows"] :]
    return prices


def build_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the z-scored log-return frame, tagged with split membership.

    Returns columns ``date``, ``log_return``, ``z_return``, ``split``. ``z_return`` is the notebook's
    ``emission``.
    """
    prices = load_prices(cfg)
    log_return = np.log(prices).diff().dropna()
    # pandas std is ddof=1, matching the notebook.
    z_return = (log_return - log_return.mean()) / log_return.std()

    splits = compute_splits(len(z_return), cfg)
    split_tag = np.where(np.arange(len(z_return)) < splits.train_end, "train", "test")

    return pd.DataFrame(
        {
            "date": z_return.index.astype(str),
            "close": prices.loc[z_return.index].to_numpy(dtype=float),
            "log_return": log_return.to_numpy(dtype=float),
            "z_return": z_return.to_numpy(dtype=float),
            "split": split_tag,
        }
    )


def a001_log_return_scale(returns: pd.DataFrame) -> tuple[float, float]:
    """Global mean and std for A001 log returns used in z-scoring."""
    log_return = returns["log_return"]
    return float(log_return.mean()), float(log_return.std())


def save_returns(returns: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(path, index=False)


def load_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    path = config_path(cfg, "returns")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_label_regimes.py first."
        )
    return pd.read_parquet(path)


def asset_columns(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["data"]["asset_columns"])


def load_price_panel(cfg: dict[str, Any]) -> pd.DataFrame:
    """Closing prices for every diffusion asset, with stale leading rows dropped."""
    raw_path = config_path(cfg, "raw_csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw price file not found: {raw_path}")
    frame = pd.read_csv(raw_path)
    frame = frame.set_index(cfg["data"]["date_column"])
    frame.index = pd.to_datetime(frame.index)
    columns = asset_columns(cfg)
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing price columns in {raw_path}: {missing}")
    return frame[columns].iloc[cfg["data"]["skip_rows"] :]


def build_multivariate_log_returns(cfg: dict[str, Any]) -> pd.DataFrame:
    """Raw log returns for the diffusion asset panel.

    Rows require every asset to have a valid price on both ``t`` and ``t-1``. Several assets start
    later than A001, so this panel is shorter than the univariate HMM series. Diffusion training uses
    the train-split dates that fall inside this overlap; regime labels are mapped by date rather than
    recomputed.
    """
    prices = load_price_panel(cfg)
    log_return = np.log(prices).diff().dropna(how="any")
    return log_return


def align_train_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Train-split multivariate log returns aligned to cached A001 regime labels by date.

    Returns ``panel`` of shape ``(n_days, n_assets)``, ``labels`` of shape ``(n_days,)``, and the
    shared date index.
    """
    panel = build_multivariate_log_returns(cfg)
    train = returns.loc[returns["split"] == "train"].copy()
    train["date"] = pd.to_datetime(train["date"])
    if len(regime_labels) != len(train):
        raise ValueError(
            f"Regime labels cover {len(regime_labels)} days but the training series has {len(train)}."
        )

    label_frame = pd.DataFrame(
        {"date": train["date"].to_numpy(), "regime": regime_labels.astype(int)}
    ).set_index("date")
    common = panel.index.intersection(label_frame.index)
    if common.empty:
        raise ValueError("No overlapping dates between the multivariate panel and training labels.")

    aligned_panel = panel.loc[common, asset_columns(cfg)].to_numpy(dtype=float)
    aligned_labels = label_frame.loc[common, "regime"].to_numpy(dtype=int)
    return aligned_panel, aligned_labels, common


def train_returns(returns: pd.DataFrame) -> np.ndarray:
    """The z-scored training series that ``Vol_Regime`` and the specialists are fit on."""
    return returns.loc[returns["split"] == "train", "z_return"].to_numpy(dtype=float)


def build_model_frames(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train_data and val_data frames for HMM fitting."""
    train_series = train_returns(returns)
    splits = compute_splits(len(returns), cfg)
    cut = splits.train_val_split

    if len(regime_labels) != len(train_series):
        raise ValueError(
            f"Regime labels cover {len(regime_labels)} days but the training series has "
            f"{len(train_series)}."
        )

    def frame(emissions: np.ndarray, regimes: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame({"regime": regimes.astype(int), "emission": emissions})
        out["emission_lag"] = out["emission"].shift(1)
        return out[["regime", "emission", "emission_lag"]]

    train_data = frame(train_series[:cut], regime_labels[:cut])
    val_data = frame(train_series[cut:], regime_labels[cut:])
    return train_data, val_data
