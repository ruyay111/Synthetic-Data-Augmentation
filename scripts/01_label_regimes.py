#!/usr/bin/env python3
"""Stage 1: preprocess returns and label volatility regimes.

Covers cells 4-10 of ``Supervised HMMs.ipynb``. Writes two artifacts:

  data/processed/sp500tr_returns.parquet   z-scored log returns tagged with split membership
  data/processed/regime_labels.npz         Vol_Regime output for the training series

Both are inputs to every later stage. The label file is treated as immutable once the specialists are
trained against it, so the script refuses to overwrite an existing cache without ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmmdiff.config import bootstrap_imports, config_path, load_config  # noqa: E402
from hmmdiff.data import build_returns, compute_splits, save_returns, train_returns  # noqa: E402
from hmmdiff.regimes import fit_regimes, load_labels, save_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite an existing regime label cache.",
    )
    return parser.parse_args()


def check_preprocessing(returns, cfg) -> None:
    """Validation checkpoint 1: the splits must match the reference notebook exactly."""
    expected = cfg["reference"]
    splits = compute_splits(len(returns), cfg)
    actual = {
        "n_returns": len(returns),
        "n_train": splits.n_train,
        "n_test": splits.n_test,
        "n_train_inner": splits.n_train_inner,
        "n_val": splits.n_val,
        "start_date": returns["date"].iloc[0],
        "end_date": returns["date"].iloc[-1],
    }
    mismatches = {k: (v, expected[k]) for k, v in actual.items() if v != expected[k]}
    if mismatches:
        lines = [f"  {k}: got {got!r}, reference has {want!r}" for k, (got, want) in mismatches.items()]
        raise SystemExit(
            "Preprocessing does not match the reference notebook:\n" + "\n".join(lines)
        )
    print("[OK] preprocessing matches the reference")
    for key, value in actual.items():
        print(f"       {key}: {value}")


def check_labels(regimes, cfg) -> None:
    """Validation checkpoint 2: regime count and variance ordering."""
    meta = regimes.metadata
    n_expected = cfg["regimes"]["n_regimes"]
    if meta["n_regimes_found"] != n_expected:
        raise SystemExit(
            f"Vol_Regime found {meta['n_regimes_found']} regimes, expected {n_expected}. "
            "The tqdm bar (e.g. 4/4) counts cluster-count candidates tried, not regimes found. "
            "Stage 1 should auto-fallback across penalties; if this still fails, extend "
            "regimes.changepoint_penalty_fallback in configs/default.yaml."
        )
    if not meta["variance_monotone"]:
        raise SystemExit(
            "Regime variances are not increasing in the label index, so regime 0 is not the "
            f"lowest-volatility state: {meta['regime_variances']}"
        )
    print(f"[OK] {meta['n_regimes_found']} regimes, variance increasing in label index")
    print(f"       clustering: {meta.get('clustering_method', '?')}  penalty: {meta.get('changepoint_penalty')}")
    print(f"       changepoints: {meta['n_changepoints']}")
    for k, (count, var) in enumerate(zip(meta["regime_counts"], meta["regime_variances"])):
        share = 100 * count / meta["n_days"]
        print(f"       regime {k}: {count:>5} days ({share:>5.1f}%)  variance {var:.4f}")
    print(f"       average regime length: {json.dumps(meta['average_regime_length'])}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    bootstrap_imports()

    returns_path = config_path(cfg, "returns")
    labels_path = config_path(cfg, "regime_labels")

    print(f"[1/2] preprocessing {config_path(cfg, 'raw_csv')}")
    returns = build_returns(cfg)
    check_preprocessing(returns, cfg)
    save_returns(returns, returns_path)
    print(f"[OK] wrote {returns_path}")

    if labels_path.exists() and not args.force:
        print(f"[SKIP] {labels_path} exists; pass --force to relabel")
        print(
            "       Relabeling invalidates data/processed/regime_windows and every trained "
            "specialist."
        )
        cached = load_labels(labels_path)
        meta = cached.metadata
        print(
            f"       cached: {meta['n_regimes_found']} regimes, "
            f"penalty={meta.get('changepoint_penalty', '?')}, "
            f"changepoints={meta.get('n_changepoints', '?')}"
        )
        return 0

    series = train_returns(returns)
    print(f"[2/2] labeling regimes on {len(series)} training days (this takes a few minutes)")
    regimes = fit_regimes(series, cfg)
    check_labels(regimes, cfg)
    save_labels(regimes, labels_path)
    print(f"[OK] wrote {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
