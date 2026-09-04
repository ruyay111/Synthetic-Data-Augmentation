"""Compare synthetic vs real simple-return scales before MVO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hmmdiff.constants import MEAN_ABS_CAP, NATIVE_STD_HI, NATIVE_STD_LO, RATIO_HI, RATIO_LO


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "max", "p01", "p99")}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p01": float(np.quantile(finite, 0.01)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def source_stats(simple: np.ndarray, native_log: np.ndarray, asset_cols: list[str], source: str) -> pd.DataFrame:
    simple = np.asarray(simple, dtype=float)
    native_log = np.asarray(native_log, dtype=float)
    if simple.ndim == 3:
        simple = simple.reshape(-1, simple.shape[-1])
    if native_log.ndim == 3:
        native_log = native_log.reshape(-1, native_log.shape[-1])
    rows = []
    for i, asset in enumerate(asset_cols):
        native = _quantiles(native_log[:, i])
        conv = _quantiles(simple[:, i])
        rows.append(
            {
                "source": source,
                "asset": asset,
                "native_std": native["std"],
                "simple_mean": conv["mean"],
                "simple_std": conv["std"],
                "simple_min": conv["min"],
                "simple_max": conv["max"],
                "simple_p01": conv["p01"],
                "simple_p99": conv["p99"],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ScaleCheckResult:
    table: pd.DataFrame
    ok: bool
    messages: list[str]


def run_scale_check(
    real_log: np.ndarray,
    pool_log: dict[int, np.ndarray],
    uncond_log: np.ndarray,
    asset_cols: list[str],
) -> ScaleCheckResult:
    """Fail if any source looks like z-scores/prices, or simple-return std is far from real."""
    real_log = np.asarray(real_log, dtype=float)
    real_simple = np.expm1(real_log)
    frames = [source_stats(real_simple, real_log, asset_cols, "real")]

    if pool_log:
        stacked = np.concatenate([np.asarray(v, dtype=float) for v in pool_log.values()], axis=0)
        frames.append(source_stats(np.expm1(stacked), stacked, asset_cols, "hmm-diffusion pools"))

    uncond_log = np.asarray(uncond_log, dtype=float)
    frames.append(source_stats(np.expm1(uncond_log), uncond_log, asset_cols, "uncondi-diffusion"))

    table = pd.concat(frames, ignore_index=True)
    messages: list[str] = []
    real_std = {
        asset: float(table.loc[(table["source"] == "real") & (table["asset"] == asset), "simple_std"].iloc[0])
        for asset in asset_cols
    }

    for _, row in table.iterrows():
        native_std = float(row["native_std"])
        if not (NATIVE_STD_LO <= native_std <= NATIVE_STD_HI):
            messages.append(
                f"{row['source']} {row['asset']}: native log std {native_std:.4g} "
                f"is outside [{NATIVE_STD_LO}, {NATIVE_STD_HI}] (z-score or price scale?)"
            )
        if row["source"] == "real":
            continue
        ref = real_std[str(row["asset"])]
        simple_std = float(row["simple_std"])
        if not np.isfinite(ref) or ref <= 0:
            messages.append(f"real {row['asset']} simple std is not usable as a reference")
            continue
        ratio = simple_std / ref
        if ratio < RATIO_LO or ratio > RATIO_HI:
            messages.append(
                f"{row['source']} {row['asset']}: simple std ratio vs real is {ratio:.3g} "
                f"(flag if < {RATIO_LO}× or > {RATIO_HI}×)"
            )
        if abs(float(row["simple_mean"])) > MEAN_ABS_CAP:
            messages.append(
                f"{row['source']} {row['asset']}: simple-return mean {row['simple_mean']:.4g} "
                f"exceeds |{MEAN_ABS_CAP}|; possible unit mismatch"
            )

    return ScaleCheckResult(table=table, ok=not messages, messages=messages)
