"""Price loading, log-return construction, and train/test splits.

PriceReturnProcessor is the data-processing class for a single price series and
the ten-asset panel. Splits holds the calendar cut used by HMM training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hmmdiff.config import config_path


@dataclass(frozen=True)
class Splits:
    """Train/test index cuts for one return series.

    When train_end_date is set there is no validation slice (train_val_split
    equals train_end).
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


class PriceReturnProcessor:
    """Load raw CSV prices, build log returns, and apply the train/test split.

    Bundle the data-processing APIs used before regime labeling and HMM fit.
    Pass the YAML config dict to the constructor; methods read paths and
    column names from that mapping.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        """
        Store an optional pipeline config.

        Parameters:
        cfg: dict or None
            YAML mapping with paths and data keys. Required for I/O methods.

        Return:
           None
        """
        self.cfg = cfg

    def ComputeSplits(
        self,
        n_total: int,
        cfg: dict[str, Any] | None = None,
        dates: pd.Series | np.ndarray | None = None,
    ) -> Splits:
        """
        Index of the first test day.

        If data.train_end_date is set, the cut is that calendar date and there
        is no inner validation slice. Otherwise train_fraction then
        train_val_fraction apply.

        Parameters:
        n_total: int
            Length of the return series.
        cfg: dict or None
            Pipeline config; defaults to self.cfg.
        dates: pandas.Series, numpy.ndarray, or None
            Observation dates, required for a date-based cut.

        Return:
           Splits dataclass with train_end and train_val_split.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("ComputeSplits requires a config dict.")
        train_end_date = cfg["data"].get("train_end_date")
        if train_end_date:
            if dates is None:
                raise ValueError("date-based splits require dates; use SplitsFromReturns.")
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

    def SplitsFromReturns(self, returns: pd.DataFrame, cfg: dict[str, Any] | None = None) -> Splits:
        """
        Train/test cuts from a return frame that has a date column.

        Parameters:
        returns: pandas.DataFrame
            Frame with a date column.
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           Splits dataclass.
        """
        cfg = cfg if cfg is not None else self.cfg
        return self.ComputeSplits(len(returns), cfg, dates=returns["date"])

    def SplitCounts(self, returns: pd.DataFrame) -> dict[str, Any]:
        """
        Train/test counts and dates from the split column.

        Train rows must form a date prefix.

        Parameters:
        returns: pandas.DataFrame
            Frame with date and split columns.

        Return:
           dict with n_returns, n_train, n_test, and calendar strings.
        """
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

    def TrainMask(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Boolean mask of training rows.

        Parameters:
        returns: pandas.DataFrame
            Frame with a split column.

        Return:
           numpy.ndarray of bool.
        """
        return returns["split"].to_numpy() == "train"

    def TestMask(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Boolean mask of test rows.

        Parameters:
        returns: pandas.DataFrame
            Frame with a split column.

        Return:
           numpy.ndarray of bool.
        """
        return returns["split"].to_numpy() == "test"

    def LoadPrices(self, cfg: dict[str, Any] | None = None) -> pd.Series:
        """
        Closing price series for the configured column.

        Stale leading rows are dropped using data.skip_rows.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.Series of closing prices indexed by date.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("LoadPrices requires a config dict.")
        raw_path = config_path(cfg, "raw_csv")
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw price file not found: {raw_path}")
        frame = pd.read_csv(raw_path)
        frame = frame.set_index(cfg["data"]["date_column"])
        prices = frame[cfg["data"]["price_column"]].iloc[cfg["data"]["skip_rows"] :]
        return prices

    def BuildReturns(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Z-scored log-return frame tagged with split membership.

        Columns are date, close, log_return, z_return, split. z_return is the
        HMM emission. Optional data.start_date / data.end_date crop before
        z-scoring.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame of processed returns.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("BuildReturns requires a config dict.")
        prices = self.LoadPrices(cfg)
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
        z_return = (log_return - log_return.mean()) / log_return.std()

        splits = self.ComputeSplits(len(z_return), cfg, dates=log_return.index)
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

    def A001LogReturnScale(self, returns: pd.DataFrame) -> tuple[float, float]:
        """
        Global mean and std of A001 log returns used in z-scoring.

        Parameters:
        returns: pandas.DataFrame
            Frame with a log_return column.

        Return:
           tuple of (mean, std) as float.
        """
        log_return = returns["log_return"]
        return float(log_return.mean()), float(log_return.std())

    def SaveReturns(self, returns: pd.DataFrame, path: Path) -> None:
        """
        Write a return frame to parquet.

        Parameters:
        returns: pandas.DataFrame
            Processed return table.
        path: pathlib.Path
            Output parquet path.

        Return:
           None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        returns.to_parquet(path, index=False)

    def LoadReturns(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Read the processed return parquet from config paths.returns.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame of stored returns.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("LoadReturns requires a config dict.")
        path = config_path(cfg, "returns")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/01_label_regimes.py first."
            )
        return pd.read_parquet(path)

    def AssetColumns(self, cfg: dict[str, Any] | None = None) -> list[str]:
        """
        Configured diffusion asset column names.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           list of str.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("AssetColumns requires a config dict.")
        return list(cfg["data"]["asset_columns"])

    def LoadPricePanel(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Closing prices for every diffusion asset.

        Stale leading rows are dropped.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame of prices, datetime index.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("LoadPricePanel requires a config dict.")
        raw_path = config_path(cfg, "raw_csv")
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw price file not found: {raw_path}")
        frame = pd.read_csv(raw_path)
        frame = frame.set_index(cfg["data"]["date_column"])
        frame.index = pd.to_datetime(frame.index)
        columns = self.AssetColumns(cfg)
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing price columns in {raw_path}: {missing}")
        return frame[columns].iloc[cfg["data"]["skip_rows"] :]

    def BuildMultivariateLogReturns(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Raw log returns for the diffusion asset panel.

        Rows require a valid price on both t and t-1 for every asset.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame of log returns.
        """
        cfg = cfg if cfg is not None else self.cfg
        prices = self.LoadPricePanel(cfg)
        log_return = np.log(prices).diff().dropna(how="any")
        return log_return

    def AlignDiffusionPanel(
        self,
        returns: pd.DataFrame,
        regime_labels: np.ndarray,
        cfg: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Full-sample multivariate log returns aligned to A001 regime labels by date.

        Parameters:
        returns: pandas.DataFrame
            Univariate labeled return series.
        regime_labels: numpy.ndarray
            Integer labels covering returns.
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           tuple of (panel array, aligned labels, common DatetimeIndex).
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("AlignDiffusionPanel requires a config dict.")
        labels = np.asarray(regime_labels)
        if len(labels) != len(returns):
            raise ValueError(
                f"Regime labels cover {len(labels)} days but the return series has {len(returns)}."
            )

        panel = self.BuildMultivariateLogReturns(cfg)
        labeled = pd.DataFrame(
            {
                "date": pd.to_datetime(returns["date"]).to_numpy(),
                "regime": labels.astype(int),
            }
        ).set_index("date")
        common = panel.index.intersection(labeled.index)
        if common.empty:
            raise ValueError("No overlapping dates between the multivariate panel and A001 labels.")

        aligned_panel = panel.loc[common, self.AssetColumns(cfg)].to_numpy(dtype=float)
        aligned_labels = labeled.loc[common, "regime"].to_numpy(dtype=int)
        return aligned_panel, aligned_labels, common

    def AlignTrainDiffusionPanel(
        self,
        returns: pd.DataFrame,
        regime_labels: np.ndarray,
        cfg: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Alias for AlignDiffusionPanel (windows use the full labeled sample).

        Parameters:
        returns: pandas.DataFrame
            Univariate labeled return series.
        regime_labels: numpy.ndarray
            Integer labels covering returns.
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           tuple of (panel array, aligned labels, common DatetimeIndex).
        """
        return self.AlignDiffusionPanel(returns, regime_labels, cfg)

    def TrainReturns(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Z-scored training series used as HMM emissions.

        Parameters:
        returns: pandas.DataFrame
            Frame with z_return and split.

        Return:
           numpy.ndarray of float.
        """
        return returns.loc[self.TrainMask(returns), "z_return"].to_numpy(dtype=float)

    def SplitSeries(
        self, returns: pd.DataFrame, column: str = "z_return"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train and test slices of one column.

        Parameters:
        returns: pandas.DataFrame
            Frame with a split column.
        column: str
            Column to slice.

        Return:
           tuple of (train array, test array).
        """
        values = returns[column].to_numpy(dtype=float)
        return values[self.TrainMask(returns)], values[self.TestMask(returns)]

    def SplitLabels(
        self, returns: pd.DataFrame, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train and test slices of full-sample regime labels.

        Parameters:
        returns: pandas.DataFrame
            Frame with a split column.
        labels: numpy.ndarray
            Labels aligned to returns.

        Return:
           tuple of (train labels, test labels) as int arrays.
        """
        labels = np.asarray(labels)
        if len(labels) != len(returns):
            raise ValueError(
                f"Regime labels cover {len(labels)} days but the return series has {len(returns)}."
            )
        return labels[self.TrainMask(returns)].astype(int), labels[self.TestMask(returns)].astype(int)

    def BuildModelFrames(
        self,
        returns: pd.DataFrame,
        regime_labels: np.ndarray,
        cfg: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        HMM frames for train and test.

        Labels must cover the full return series. The config argument is unused;
        the split comes from the returns split column.

        Parameters:
        returns: pandas.DataFrame
            Frame with z_return and split.
        regime_labels: numpy.ndarray
            Integer labels covering returns.
        cfg: dict or None
            Unused; accepted for call-site compatibility.

        Return:
           tuple of (train DataFrame, test DataFrame) with regime, emission,
           emission_lag.
        """
        del cfg
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
        train_idx = self.TrainMask(returns)
        test_idx = self.TestMask(returns)
        return frame(emissions[train_idx], labels[train_idx]), frame(
            emissions[test_idx], labels[test_idx]
        )
