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
    """Train/test split. When ``train_end_date`` is set there is no validation slice
    (``train_val_split`` equals ``train_end``).
    """

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


def compute_splits(
    n_total: int, cfg: dict[str, Any], dates: pd.Series | np.ndarray | None = None
) -> Splits:
    """Index of the first test day.

    If ``data.train_end_date`` is set, the cut is that calendar date and there is no inner
    validation slice. Otherwise fractions ``train_fraction`` then ``train_val_fraction`` apply
    (legacy A001 and the EW mapping from the long A001 calendar).
    """
    train_end_date = cfg["data"].get("train_end_date")
    if train_end_date:
        if dates is None:
            raise ValueError("date-based splits require dates; use splits_from_returns.")
        stamp = pd.Timestamp(train_end_date)
        n_train = int((pd.to_datetime(dates) <= stamp).sum())
        if n_train == 0 or n_train >= n_total:
            raise ValueError(
                f"train_end_date {train_end_date} does not split the series "
                f"(n_train={n_train}, n_total={n_total})."
            )
        return Splits(n_total=n_total, train_end=n_train, train_val_split=n_train)
    train_end = int(n_total * cfg["data"]["train_fraction"])
    train_val_split = int(train_end * cfg["data"]["train_val_fraction"])
    return Splits(n_total=n_total, train_end=train_end, train_val_split=train_val_split)


def splits_from_returns(returns: pd.DataFrame, cfg: dict[str, Any]) -> Splits:
    return compute_splits(len(returns), cfg, dates=returns["date"])


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


def train_mask(returns: pd.DataFrame) -> np.ndarray:
    return returns["split"].to_numpy() == "train"


def test_mask(returns: pd.DataFrame) -> np.ndarray:
    return returns["split"].to_numpy() == "test"


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
    ``emission``. Optional ``data.start_date`` / ``data.end_date`` crop the series before
    z-scoring. When ``data.train_end_date`` is set, train is that calendar prefix.
    """
    prices = load_prices(cfg)
    log_return = np.log(prices).diff().dropna()
    dates = pd.to_datetime(log_return.index)
    start = cfg["data"].get("start_date")
    end = cfg["data"].get("end_date")
    if start:
        log_return = log_return.loc[dates >= pd.Timestamp(start)]
        dates = pd.to_datetime(log_return.index)
    if end:
        log_return = log_return.loc[dates <= pd.Timestamp(end)]
        dates = pd.to_datetime(log_return.index)
    # pandas std is ddof=1, matching the notebook.
    z_return = (log_return - log_return.mean()) / log_return.std()

    splits = compute_splits(len(z_return), cfg, dates=log_return.index)
    split_tag = np.where(np.arange(len(z_return)) < splits.train_end, "train", "test")

    return pd.DataFrame(
        {
            "date": pd.to_datetime(z_return.index).strftime("%Y-%m-%d"),
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

    Rows require every asset to have a valid price on both ``t`` and ``t-1``. With A001 cropped to
    2001–2022 this panel matches the univariate HMM calendar. Regime labels are mapped by date.
    """
    prices = load_price_panel(cfg)
    log_return = np.log(prices).diff().dropna(how="any")
    return log_return


def align_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Full-sample multivariate log returns aligned to A001 regime labels by date.

    Clustering labels and diffusion windows cover the cropped A001 calendar (2001–2022).
    The HMM still trains only on the train split via ``build_model_frames``.
    """
    labels = np.asarray(regime_labels)
    if len(labels) != len(returns):
        raise ValueError(
            f"Regime labels cover {len(labels)} days but the return series has {len(returns)}."
        )

    panel = build_multivariate_log_returns(cfg)
    labeled = pd.DataFrame(
        {
            "date": pd.to_datetime(returns["date"]).to_numpy(),
            "regime": labels.astype(int),
        }
    ).set_index("date")
    common = panel.index.intersection(labeled.index)
    if common.empty:
        raise ValueError("No overlapping dates between the multivariate panel and A001 labels.")

    aligned_panel = panel.loc[common, asset_columns(cfg)].to_numpy(dtype=float)
    aligned_labels = labeled.loc[common, "regime"].to_numpy(dtype=int)
    return aligned_panel, aligned_labels, common


def align_train_diffusion_panel(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Alias for ``align_diffusion_panel`` (windows now use the full labeled sample)."""
    return align_diffusion_panel(returns, regime_labels, cfg)


def train_returns(returns: pd.DataFrame) -> np.ndarray:
    """The z-scored training series used as HMM emissions."""
    return returns.loc[train_mask(returns), "z_return"].to_numpy(dtype=float)


def split_series(returns: pd.DataFrame, column: str = "z_return") -> tuple[np.ndarray, np.ndarray]:
    """Train and test slices of one column."""
    values = returns[column].to_numpy(dtype=float)
    return values[train_mask(returns)], values[test_mask(returns)]


def split_labels(returns: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if len(labels) != len(returns):
        raise ValueError(
            f"Regime labels cover {len(labels)} days but the return series has {len(returns)}."
        )
    return labels[train_mask(returns)].astype(int), labels[test_mask(returns)].astype(int)


def build_model_frames(
    returns: pd.DataFrame, regime_labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HMM frames for train and test. Labels must cover the full return series."""
    del cfg  # split comes from the returns ``split`` column
    labels = np.asarray(regime_labels)
    if len(labels) != len(returns):
        raise ValueError(
            f"Regime labels cover {len(labels)} days but the return series has {len(returns)}."
        )

    def frame(emissions: np.ndarray, regimes: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame({"regime": regimes.astype(int), "emission": emissions})
        out["emission_lag"] = out["emission"].shift(1)
        return out[["regime", "emission", "emission_lag"]]

    emissions = returns["z_return"].to_numpy(dtype=float)
    train_idx = train_mask(returns)
    test_idx = test_mask(returns)
    return frame(emissions[train_idx], labels[train_idx]), frame(
        emissions[test_idx], labels[test_idx]
    )
