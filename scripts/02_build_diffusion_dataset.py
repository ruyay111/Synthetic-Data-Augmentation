#!/usr/bin/env python3
"""Stage 2: cut per-regime training windows for the diffusion specialists.

Reads the cached regime labels from stage 1 and writes

  data/processed/regime_windows/regime_{k}.npy   (n_windows, seq_len, 1)
  data/processed/regime_windows/manifest.json

The manifest is the diagnostic to read before training: it reports how many windows each regime
yielded, how many came from cyclic tiling of short segments, and whether the windowed moments match
the corresponding slice of the source series.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmmdiff.config import config_path, load_config  # noqa: E402
from hmmdiff.data import load_returns, train_returns  # noqa: E402
from hmmdiff.regimes import load_labels  # noqa: E402
from hmmdiff.windows import build_windows, save_windows  # noqa: E402

# Stride-1 sliding windows overlap almost completely, so the raw window count overstates how much
# independent data a specialist sees. The honest measure is how many non-overlapping seq_len windows
# the regime's days could form. Upstream skips training a regime below 8 windows; we warn there.
MIN_INDEPENDENT_WINDOWS = 8
# Relative tolerance when comparing windowed variance against the source slice.
MOMENT_TOLERANCE = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    return parser.parse_args()


def report(manifest: dict) -> list[str]:
    """Print the per-regime table and return any warnings raised by checkpoint 3."""
    warnings: list[str] = []
    seq_len = manifest["seq_len"]
    print(
        f"{'regime':>6} {'windows':>8} {'indep':>6} {'days':>6} {'segs':>5} {'>=seq':>6} "
        f"{'tiled':>6} {'src var':>8} {'win var':>8}"
    )
    for key in sorted(manifest["regimes"], key=int):
        row = manifest["regimes"][key]
        independent = row["n_days"] // seq_len
        print(
            f"{key:>6} {row['n_windows']:>8} {independent:>6} {row['n_days']:>6} "
            f"{row['segments']:>5} {row['segments_long']:>6} {row['segments_tiled']:>6} "
            f"{row['source_var']:>8.3f} {row['window_var']:>8.3f}"
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
                f"regime {key} windowed variance {row['window_var']:.3f} differs from the source "
                f"slice {row['source_var']:.3f} by more than "
                f"{MOMENT_TOLERANCE:.0%}; the windows are not representative"
            )
    return warnings


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    returns = load_returns(cfg)
    series = train_returns(returns)
    regimes = load_labels(config_path(cfg, "regime_labels"))

    print(f"building windows from {len(series)} training days, seq_len={cfg['diffusion']['seq_len']}")
    windows, manifest = build_windows(series, regimes.labels, cfg)

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
