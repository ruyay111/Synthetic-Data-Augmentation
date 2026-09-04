#!/usr/bin/env python3
"""Stage 1 (EW): equal-weight standardized returns and full-sample regime labels.

Clusters Vol_Regime on the 10-asset equal-weight series from 2001-01-01 to 2022-08-31. The HMM
later trains only on 2001–2014; all five regimes must appear in that train slice. Writes

  data/processed/ew_returns.parquet
  data/processed/regime_labels_ew.npz

Does not touch the A001 artifacts used by scripts/01_label_regimes.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmmdiff.config import bootstrap_imports, config_path, load_config  # noqa: E402
from hmmdiff.data_collection.equal_weight_processor import EqualWeightProcessor  # noqa: E402
from hmmdiff.data_collection.regime_processor import RegimeProcessor  # noqa: E402

DEFAULT_CONFIG = "configs/ew.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to a YAML config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite an existing EW regime label cache.",
    )
    return parser.parse_args()


def check_preprocessing(returns, processor) -> dict:
    expected = processor.cfg["reference"]
    actual = processor.SplitCounts(returns)
    keys = [
        "n_returns",
        "n_train",
        "n_test",
        "start_date",
        "train_end_date",
        "test_start_date",
        "end_date",
    ]
    mismatches = {k: (actual[k], expected[k]) for k in keys if actual[k] != expected[k]}
    if mismatches:
        lines = [f"  {k}: got {got!r}, reference has {want!r}" for k, (got, want) in mismatches.items()]
        raise SystemExit("EW preprocessing does not match configs/ew.yaml:\n" + "\n".join(lines))
    print("[OK] EW preprocessing matches the 10-asset overlap calendar")
    for key in keys:
        print(f"       {key}: {actual[key]}")
    return actual


def check_labels(regimes, cfg, n_train: int) -> None:
    meta = regimes.metadata
    n_expected = cfg["regimes"]["n_regimes"]
    if meta["n_regimes_found"] != n_expected:
        raise SystemExit(
            f"Vol_Regime found {meta['n_regimes_found']} regimes, expected {n_expected}."
        )
    if not meta["variance_monotone"]:
        raise SystemExit(
            "Regime variances are not increasing in the label index, so regime 0 is not the "
            f"lowest-volatility state: {meta['regime_variances']}"
        )
    train_counts = [
        int((regimes.labels[:n_train] == k).sum()) for k in range(n_expected)
    ]
    if any(count == 0 for count in train_counts):
        raise SystemExit(
            f"A regime is missing from the 2001–2014 HMM train slice: {train_counts}"
        )
    print(f"[OK] {meta['n_regimes_found']} regimes on {meta['n_days']} days (2001–2022)")
    print(f"       clustering: {meta.get('clustering_method', '?')}  penalty: {meta.get('changepoint_penalty')}")
    print(f"       changepoints: {meta['n_changepoints']}")
    for k, (count, var) in enumerate(zip(meta["regime_counts"], meta["regime_variances"])):
        share = 100 * count / meta["n_days"]
        print(f"       regime {k}: {count:>5} days ({share:>5.1f}%)  variance {var:.4f}")
    print(f"       train-slice counts (2001–2014): {train_counts}")
    print(f"       average regime length: {json.dumps(meta['average_regime_length'])}")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    bootstrap_imports()
    processor = EqualWeightProcessor(cfg)
    regime_processor = RegimeProcessor()

    returns_path = config_path(cfg, "returns")
    labels_path = config_path(cfg, "regime_labels")

    print(f"[1/2] equal-weight standardized returns from {config_path(cfg, 'raw_csv')}")
    returns = processor.BuildEwReturns(cfg)
    actual = check_preprocessing(returns, processor)
    processor.SaveEwReturns(returns, returns_path)
    print(f"[OK] wrote {returns_path}")

    if labels_path.exists() and not args.force:
        print(f"[SKIP] {labels_path} exists; pass --force to relabel")
        print(
            "       Relabeling invalidates data/processed/regime_windows_ew and every EW "
            "specialist."
        )
        cached = regime_processor.LoadLabels(labels_path)
        meta = cached.metadata
        print(
            f"       cached: {meta['n_regimes_found']} regimes, "
            f"penalty={meta.get('changepoint_penalty', '?')}, "
            f"changepoints={meta.get('n_changepoints', '?')}"
        )
        return 0

    series = returns["z_return"].to_numpy(dtype=float)
    n_train = actual["n_train"]
    print(
        f"[2/2] labeling regimes on {len(series)} overlap days "
        f"(HMM coverage slice is first {n_train} train days)"
    )
    regimes = regime_processor.FitRegimes(series, cfg, coverage_end=n_train)
    scale = processor.EwScale(cfg)
    regimes = replace(regimes, metadata={**regimes.metadata, **scale})
    check_labels(regimes, cfg, n_train)
    regime_processor.SaveLabels(regimes, labels_path)
    print(f"[OK] wrote {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
