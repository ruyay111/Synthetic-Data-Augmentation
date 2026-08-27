"""Turn the labeled training series into per-regime training windows for the diffusion specialists.

Univariate adaptation of ``build_regime_window_datasets.py`` from the ruya tree. Windows are cut from
*contiguous* same-regime runs so each one is a real stretch of market history rather than a
concatenation of disjoint days that happen to share a label.

Short runs are handled by cyclic tiling rather than being dropped. Tiling manufactures windows from a
segment shorter than ``seq_len`` by rotating and repeating it, which keeps thin regimes trainable at
the cost of those windows being highly redundant. The manifest reports how many windows came from
tiling so the effect stays visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Half-open ``(start, end, regime)`` runs of constant label."""
    if labels.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for idx, value in enumerate(labels[1:], start=1):
        value = int(value)
        if value != current:
            runs.append((start, idx, current))
            start = idx
            current = value
    runs.append((start, len(labels), current))
    return runs


def sliding_windows(block: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """All ``seq_len`` windows of a block, shape ``(n_windows, seq_len, n_channels)``."""
    n_days, n_channels = block.shape
    if n_days < seq_len:
        return np.empty((0, seq_len, n_channels), dtype=float)
    starts = range(0, n_days - seq_len + 1, stride)
    return np.stack([block[s : s + seq_len] for s in starts], axis=0)


def tiled_windows(block: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Windows built by cyclically repeating a run shorter than ``seq_len``."""
    n_days, n_channels = block.shape
    if n_days == 0:
        return np.empty((0, seq_len, n_channels), dtype=float)
    if n_days >= seq_len:
        return sliding_windows(block, seq_len, stride)
    windows = []
    for offset in range(0, n_days, max(1, stride)):
        rotated = np.concatenate([block[offset:], block[:offset]], axis=0)
        reps = int(np.ceil(seq_len / n_days))
        windows.append(np.concatenate([rotated] * reps, axis=0)[:seq_len])
    return np.stack(windows, axis=0)


def build_windows(
    series: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Cut per-regime windows from the labeled series.

    ``series`` is the 1-D z-scored training series; ``labels`` are its regime labels. Returns the
    windows keyed by regime, shape ``(n_windows, seq_len, 1)``, plus a manifest.
    """
    if len(series) != len(labels):
        raise ValueError(f"series has {len(series)} days but labels have {len(labels)}")

    seq_len = int(cfg["diffusion"]["seq_len"])
    stride = int(cfg["diffusion"]["stride"])
    n_regimes = int(cfg["regimes"]["n_regimes"])
    short_policy = cfg["diffusion"]["short_policy"]
    column = series.reshape(-1, 1)

    per_regime: dict[int, list[np.ndarray]] = {k: [] for k in range(n_regimes)}
    stats = {
        k: {"segments": 0, "segments_long": 0, "segments_tiled": 0, "windows_tiled": 0}
        for k in range(n_regimes)
    }

    for start, end, regime in contiguous_runs(labels):
        if not 0 <= regime < n_regimes:
            raise ValueError(f"Unexpected regime label {regime}")
        block = column[start:end]
        stats[regime]["segments"] += 1
        if len(block) >= seq_len:
            windows = sliding_windows(block, seq_len, stride)
            stats[regime]["segments_long"] += 1
        elif short_policy == "tile":
            windows = tiled_windows(block, seq_len, stride)
            stats[regime]["segments_tiled"] += 1
            stats[regime]["windows_tiled"] += int(windows.shape[0])
        else:
            continue
        if windows.shape[0]:
            per_regime[regime].append(windows)

    stacked = {
        k: (
            np.concatenate(per_regime[k], axis=0)
            if per_regime[k]
            else np.empty((0, seq_len, 1), dtype=float)
        )
        for k in range(n_regimes)
    }
    manifest = _build_manifest(series, labels, stacked, stats, cfg)
    return stacked, manifest


def _build_manifest(
    series: np.ndarray,
    labels: np.ndarray,
    windows: dict[int, np.ndarray],
    stats: dict[int, dict[str, int]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    for k, block in windows.items():
        source = series[labels == k]
        regimes[str(k)] = {
            "n_windows": int(block.shape[0]),
            "n_days": int(source.size),
            **stats[k],
            # Windowed moments should track the source slice; a large gap means the windows are not
            # representative of the regime, usually because tiling dominates.
            "source_mean": float(source.mean()) if source.size else float("nan"),
            "source_var": float(source.var()) if source.size else float("nan"),
            "window_mean": float(block.mean()) if block.size else float("nan"),
            "window_var": float(block.var()) if block.size else float("nan"),
        }
    return {
        "seq_len": int(cfg["diffusion"]["seq_len"]),
        "stride": int(cfg["diffusion"]["stride"]),
        "short_policy": cfg["diffusion"]["short_policy"],
        "n_regimes": int(cfg["regimes"]["n_regimes"]),
        "n_channels": 1,
        "n_days": int(series.size),
        "regimes": regimes,
    }


def save_windows(
    windows: dict[int, np.ndarray], manifest: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for regime, block in windows.items():
        path = output_dir / f"regime_{regime}.npy"
        np.save(path, block)
        manifest["regimes"][str(regime)]["path"] = str(path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/02_build_diffusion_dataset.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
