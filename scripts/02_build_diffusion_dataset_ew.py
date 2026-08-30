#!/usr/bin/env python3
"""Stage 2 (EW): cut per-regime 10-asset windows using equal-weight labels.

Reads full-sample EW labels (2001–2022) from stage 1 and cuts diffusion windows on the same
overlap. The HMM still trains only on 2001–2014; that split is applied later, not here. Writes

  data/processed/regime_windows_ew/regime_{k}.npy
  data/processed/regime_windows_ew/manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmmdiff.config import config_path, load_config  # noqa: E402
from hmmdiff.ew import align_ew_diffusion_panel, load_ew_returns  # noqa: E402
from hmmdiff.regimes import load_labels  # noqa: E402
from hmmdiff.windows import build_windows, save_windows  # noqa: E402

DEFAULT_CONFIG = "configs/ew.yaml"
MIN_INDEPENDENT_WINDOWS = 8
MOMENT_TOLERANCE = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to a YAML config.")
    return parser.parse_args()


def report(manifest: dict) -> list[str]:
    warnings: list[str] = []
    seq_len = manifest["seq_len"]
    print(
        f"{'regime':>6} {'windows':>8} {'indep':>6} {'days':>6} {'segs':>5} {'>=seq':>6} "
        f"{'tiled':>6} {'EW var':>10} {'win var':>10}"
    )
    for key in sorted(manifest["regimes"], key=int):
        row = manifest["regimes"][key]
        independent = row["n_days"] // seq_len
        print(
            f"{key:>6} {row['n_windows']:>8} {independent:>6} {row['n_days']:>6} "
            f"{row['segments']:>5} {row['segments_long']:>6} {row['segments_tiled']:>6} "
            f"{row['source_var']:>10.2e} {row['window_var']:>10.2e}"
        )
        if row["n_windows"] == 0:
            warnings.append(f"regime {key} produced no windows; it cannot be trained")
            continue
        if independent < MIN_INDEPENDENT_WINDOWS:
            warnings.append(
                f"regime {key} has only {row['n_days']} days, about {independent} independent "
                f"{seq_len}-day windows across {row['segments']} segments; its "
                f"{row['n_windows']} training windows are near-duplicates and the specialist will "
                "largely memorize"
            )
        if row["windows_tiled"] == row["n_windows"]:
            warnings.append(
                f"regime {key} is built entirely from cyclically tiled short segments"
            )
        denominator = max(abs(row["source_var"]), 1e-12)
        if abs(row["window_var"] - row["source_var"]) / denominator > MOMENT_TOLERANCE:
            warnings.append(
                f"regime {key} windowed EW variance {row['window_var']:.3f} differs from the "
                f"source slice {row['source_var']:.3f} by more than "
                f"{MOMENT_TOLERANCE:.0%}; the windows are not representative"
            )
    return warnings


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    returns = load_ew_returns(cfg)
    labels_path = config_path(cfg, "regime_labels")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run scripts/01_label_regimes_ew.py first."
        )
    regimes = load_labels(labels_path)
    panel, labels, dates, ew_series = align_ew_diffusion_panel(returns, regimes.labels, cfg)

    print(
        f"building windows from {len(panel)} EW overlap days "
        f"({dates[0].date()} to {dates[-1].date()}), "
        f"{panel.shape[1]} assets, seq_len={cfg['diffusion']['seq_len']}"
    )
    print(f"assets: {', '.join(cfg['data']['asset_columns'])}")
    windows, manifest = build_windows(
        panel, labels, cfg, diag_series=ew_series, label_name="equal_weight"
    )
    manifest["aligned_start"] = str(dates[0].date())
    manifest["aligned_end"] = str(dates[-1].date())
    manifest["label_source"] = "equal_weight"

    warnings = report(manifest)
    total = sum(int(block.shape[0]) for block in windows.values())
    print(f"total windows: {total}")

    output_dir = config_path(cfg, "regime_windows")
    save_windows(windows, manifest, output_dir)
    print(f"[OK] wrote {output_dir}")

    if any(block.shape[0] == 0 for block in windows.values()):
        for line in warnings:
            print(f"[FAIL] {line}")
        raise SystemExit("At least one regime produced no windows; cannot train all specialists.")
    for line in warnings:
        print(f"[WARN] {line}")
    if not warnings:
        print("[OK] all regimes have adequate, representative windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
