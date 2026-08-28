"""Load real benchmark or intraday futures data."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def load_real_data(
    csv_path,
    test_start_year: str = "2013",
    intraday: bool = False,
    use_active_month: bool = False,
    benchmark: bool = False,
) -> pd.DataFrame:
    """Load real benchmark or intraday futures data."""
    if benchmark:
        frame = pd.read_csv(csv_path)
        frame = frame.rename(columns={"as_of": "timestamp"}).assign(
            timestamp=lambda d: pd.to_datetime(d["timestamp"])
        )
        frame = frame.loc[lambda d: d["timestamp"] >= f"{test_start_year}-01-01"]
        return frame.set_index("timestamp").sort_index()

    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    frame = pd.read_csv(csv_path, names=col_names)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()

    if intraday:
        start_time = datetime.time(9, 30)
        end_time = datetime.time(16, 0)
        frame.loc[frame.index.time < start_time, "close"] = np.nan
        frame.loc[frame.index.time > end_time, "close"] = np.nan
        frame.loc[frame.index.weekday >= 5, "close"] = np.nan
        frame["inter"] = frame.index.to_series().diff().dt.total_seconds()
        frame["close_60"] = np.where(frame["inter"] < 61, frame["close"], np.nan)
        frame["returns"] = np.log(frame["close_60"]).diff().dropna()

    if use_active_month:
        monthly_vol = frame["volume"].resample("M").sum()
        best_month = monthly_vol.idxmax()
        start_of_best = best_month.replace(day=1)
        if start_of_best.month == 12:
            start_of_next = start_of_best.replace(year=start_of_best.year + 1, month=1, day=1)
        else:
            start_of_next = start_of_best.replace(month=start_of_best.month + 1, day=1)
        frame = frame.loc[(frame.index >= start_of_best) & (frame.index < start_of_next)]

    return frame


def compute_returns(frame: pd.DataFrame, log: bool = True, col: str | None = None) -> pd.Series:
    """Simple or log returns from a price column."""
    series = frame[col].dropna() if col else frame
    if log:
        return np.log(series).diff().dropna()
    return series.diff().dropna()
