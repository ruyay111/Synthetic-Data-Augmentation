"""Equal-weight ten-asset return construction.

EqualWeightProcessor standardizes each asset, averages them, and tags the
series with the A001 calendar train/test split. Clustering and diffusion
windows use 2001-2022; the HMM trains only on 2001-2014.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hmmdiff.config import config_path
from hmmdiff.data_collection.data_processor import PriceReturnProcessor


class EqualWeightProcessor:
    """Build and align the equal-weight standardized return series.

    Each of the ten assets is z-scored on the overlap sample. The equal-weight
    average is z-scored again; that series is both the Vol_Regime input and
    the HMM emission.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        """
        Store an optional pipeline config.

        Parameters:
        cfg: dict or None
            YAML mapping with paths and data keys.

        Return:
           None
        """
        self.cfg = cfg
        self.price_processor = PriceReturnProcessor(cfg)

    def LoadEwReturns(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Read the stored equal-weight return parquet.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame of EW returns.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("LoadEwReturns requires a config dict.")
        path = config_path(cfg, "returns")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/01_label_regimes_ew.py first."
            )
        return pd.read_parquet(path)

    def StandardizedAssetReturns(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Per-asset z-scored log returns on the ten-asset overlap (ddof=1).

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame of z-scored log returns.
        """
        cfg = cfg if cfg is not None else self.cfg
        panel = self.price_processor.BuildMultivariateLogReturns(cfg)
        return (panel - panel.mean()) / panel.std()

    def EqualWeightReturn(self, z_panel: pd.DataFrame) -> pd.Series:
        """
        Equal-weight average of per-asset standardized returns.

        Parameters:
        z_panel: pandas.DataFrame
            Z-scored asset returns.

        Return:
           pandas.Series of the cross-sectional mean.
        """
        return z_panel.mean(axis=1)

    def SplitCounts(self, returns: pd.DataFrame) -> dict[str, Any]:
        """
        Train/test counts and dates from the split column.

        Parameters:
        returns: pandas.DataFrame
            EW frame with date and split.

        Return:
           dict of counts and calendar strings.
        """
        return self.price_processor.SplitCounts(returns)

    def BuildEwReturns(self, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Equal-weight standardized returns tagged with the A001 calendar split.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           pandas.DataFrame with date, ew_return, z_return, split.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("BuildEwReturns requires a config dict.")
        a001 = self.price_processor.BuildReturns(cfg)
        a001_by_date = a001.copy()
        a001_by_date["date"] = pd.to_datetime(a001_by_date["date"])
        a001_by_date = a001_by_date.set_index("date")

        z_panel = self.StandardizedAssetReturns(cfg)
        ew = self.EqualWeightReturn(z_panel)
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

    def EwScale(self, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Moments needed to map 10-asset raw log returns back to EW emission units.

        Parameters:
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           dict of asset means/stds and EW mean/std.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("EwScale requires a config dict.")
        panel = self.price_processor.BuildMultivariateLogReturns(cfg)
        z_panel = (panel - panel.mean()) / panel.std()
        ew = self.EqualWeightReturn(z_panel)
        return {
            "label_source": "equal_weight",
            "asset_columns": self.price_processor.AssetColumns(cfg),
            "asset_mean": panel.mean().to_numpy(dtype=float).tolist(),
            "asset_std": panel.std().to_numpy(dtype=float).tolist(),
            "ew_mean": float(ew.mean()),
            "ew_std": float(ew.std()),
        }

    def EmissionFromLogPanel(self, panel: np.ndarray, scale: dict[str, Any]) -> np.ndarray:
        """
        Map a raw log-return panel to EW z-score emission units.

        Parameters:
        panel: numpy.ndarray
            Array with last axis equal to n_assets.
        scale: dict
            Output of EwScale.

        Return:
           numpy.ndarray of EW z-score emissions.
        """
        mean = np.asarray(scale["asset_mean"], dtype=float)
        std = np.asarray(scale["asset_std"], dtype=float)
        z = (np.asarray(panel, dtype=float) - mean) / std
        ew = z.mean(axis=-1)
        return (ew - float(scale["ew_mean"])) / float(scale["ew_std"])

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

    def SplitSeries(
        self, returns: pd.DataFrame, column: str = "z_return"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Train and test slices of one column.

        Parameters:
        returns: pandas.DataFrame
            EW return frame.
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
        Train and test slices of full-sample EW labels.

        Parameters:
        returns: pandas.DataFrame
            EW return frame.
        labels: numpy.ndarray
            Labels aligned to returns.

        Return:
           tuple of (train labels, test labels).
        """
        labels = np.asarray(labels)
        if len(labels) != len(returns):
            raise ValueError(
                f"Regime labels cover {len(labels)} days but the EW series has {len(returns)}."
            )
        return labels[self.TrainMask(returns)].astype(int), labels[self.TestMask(returns)].astype(int)

    def BuildTrainTestFrames(
        self, returns: pd.DataFrame, regime_labels: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        HMM frames for the EW pipeline: train (2001-2014) and test (2014-2022).

        Parameters:
        returns: pandas.DataFrame
            EW return frame.
        regime_labels: numpy.ndarray
            Full-sample labels.

        Return:
           tuple of (train DataFrame, test DataFrame).
        """
        train_series, test_series = self.SplitSeries(returns)
        train_labels, test_labels = self.SplitLabels(returns, regime_labels)
        return (
            _emission_frame(train_series, train_labels),
            _emission_frame(test_series, test_labels),
        )

    def AlignEwDiffusionPanel(
        self,
        returns: pd.DataFrame,
        regime_labels: np.ndarray,
        cfg: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """
        Full-sample 10-asset log returns aligned to EW labels by date.

        Parameters:
        returns: pandas.DataFrame
            EW return frame.
        regime_labels: numpy.ndarray
            Labels covering returns.
        cfg: dict or None
            Pipeline config; defaults to self.cfg.

        Return:
           tuple of panel, labels, dates, and aligned EW emission series.
        """
        cfg = cfg if cfg is not None else self.cfg
        if cfg is None:
            raise ValueError("AlignEwDiffusionPanel requires a config dict.")
        labels = np.asarray(regime_labels)
        if len(labels) != len(returns):
            raise ValueError(
                f"Regime labels cover {len(labels)} days but the EW series has {len(returns)}."
            )

        panel = self.price_processor.BuildMultivariateLogReturns(cfg)
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

        aligned_panel = panel.loc[common, self.price_processor.AssetColumns(cfg)].to_numpy(
            dtype=float
        )
        aligned_labels = labeled.loc[common, "regime"].to_numpy(dtype=int)
        aligned_ew = labeled.loc[common, "z_return"].to_numpy(dtype=float)
        return aligned_panel, aligned_labels, common, aligned_ew

    def SaveEwReturns(self, returns: pd.DataFrame, path: Path) -> None:
        """
        Write the EW return frame to parquet.

        Parameters:
        returns: pandas.DataFrame
            EW return table.
        path: pathlib.Path
            Output parquet path.

        Return:
           None
        """
        self.price_processor.SaveReturns(returns, path)


def _emission_frame(emissions: np.ndarray, regimes: np.ndarray) -> pd.DataFrame:
    """
    Build an HMM emission frame with a lag column.

    Parameters:
    emissions: numpy.ndarray
        Z-scored returns.
    regimes: numpy.ndarray
        Integer regime labels.

    Return:
       pandas.DataFrame with regime, emission, emission_lag.
    """
    out = pd.DataFrame({"regime": np.asarray(regimes).astype(int), "emission": emissions})
    out["emission_lag"] = out["emission"].shift(1)
    return out[["regime", "emission", "emission_lag"]]
