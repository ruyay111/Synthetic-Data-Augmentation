"""Run the fair downstream vol comparison (no notebook UI)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hmmdiff.config import bootstrap_imports, config_path, load_config
from hmmdiff import data, models, pools
from hmmdiff.downstream import (
    build_aligned_downstream_synth,
    build_uncond_synth_price_frame,
    load_real_benchmark,
    train_vol_augmentation,
)
from hmmdiff.mvo import load_uncond_windows
from hmmdiff.regimes import load_labels

warnings.filterwarnings("ignore")
bootstrap_imports()

cfg = load_config()
N_REGIMES = int(cfg["regimes"]["n_regimes"])
PRICE_COL = cfg["data"]["price_column"]
HORIZONS = [1, 10, 21]
ADD_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
N_SEEDS = 5
RANDOM_STATE = 42
UNCOND_CSV = REPO_ROOT / "trained_diffusion_withoutregime" / "generated_data.csv"


def main() -> int:
    returns = data.load_returns(cfg)
    labels = load_labels(config_path(cfg, "regime_labels"))
    train_data, _val_data = data.build_model_frames(returns, labels.labels, cfg)

    pools_root = config_path(cfg, "pools")
    if not pools.pools_available(pools_root, N_REGIMES):
        raise SystemExit(f"pools missing under {pools_root}; run scripts/04_generate_pools.py")

    regime_stats = pools.regime_emission_stats(
        train_data.emission.values, train_data.regime.values, N_REGIMES
    )
    generated_images = pools.load_generated_images(
        pools_root,
        N_REGIMES,
        returns=returns,
        calibrate_regimes=regime_stats,
    )
    init_dist = models.initial_distribution(train_data, N_REGIMES)
    print("[1/4] fitting supervised HMM on inner train")
    supervised = models.fit_supervised_hmm(train_data, N_REGIMES, cfg)

    df_real = load_real_benchmark(cfg, test_start_year="2014", price_col=PRICE_COL)
    print("[2/4] causal-filter stitch")
    aligned = build_aligned_downstream_synth(
        returns=returns,
        generated_images=generated_images,
        supervised=supervised,
        init_dist=init_dist,
        n_regimes=N_REGIMES,
        df_real=df_real,
        price_col=PRICE_COL,
        state_method="filter",
        train_emissions=train_data.emission.values,
    )
    common = df_real.index.intersection(aligned.df_synth.index)
    df_real = df_real.loc[common]
    df_synth = aligned.df_synth.loc[common]
    print(f"aligned {df_synth.index.min().date()} -> {df_synth.index.max().date()}  n={len(df_synth)}")

    uncond_windows = load_uncond_windows(UNCOND_CSV)
    df_uncond = build_uncond_synth_price_frame(
        uncond_windows,
        start_price=float(df_real[PRICE_COL].iloc[0]),
        price_col=PRICE_COL,
    )
    print(f"[3/4] uncond windows {uncond_windows.shape}")

    frames = []
    for source, synth in ("hmm-diffusion", df_synth), ("regime-free", df_uncond):
        for horizon in HORIZONS:
            print("=" * 72)
            print(f"{source}  horizon={horizon}")
            metrics = train_vol_augmentation(
                df_real=df_real,
                df_synth=synth,
                price_col=f"{PRICE_COL}_log",
                horizon=horizon,
                add_grid=ADD_GRID,
                n_seeds=N_SEEDS,
                random_state=RANDOM_STATE,
                plot_summary=False,
                synth_source=source,
            )
            metrics["horizon"] = horizon
            metrics["source"] = source
            frames.append(metrics)

    all_metrics = pd.concat(frames, ignore_index=True)
    out = REPO_ROOT / "test_results" / "downstream_vol_fair.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(out, index=False)
    print(f"[4/4] wrote {out}")

    rf = all_metrics.loc[all_metrics["model"] == "rf"]
    summary = (
        rf.groupby(["source", "horizon", "add_mult"], as_index=False)
        .agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"), mse_mean=("mse", "mean"), qlike_mean=("qlike", "mean"))
    )
    print(summary.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
