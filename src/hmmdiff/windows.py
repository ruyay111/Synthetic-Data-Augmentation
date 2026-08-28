"""Turn the labeled training series into per-regime training windows for the diffusion specialists.

Multivariate adaptation of ``build_regime_window_datasets.py`` from the ruya tree. Regime labels come
from A001 (stage 1); each window carries all ten configured assets as channels. Windows are cut from
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
    panel: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Cut per-regime windows from the labeled multivariate panel.

    ``panel`` has shape ``(n_days, n_channels)``; ``labels`` are its regime labels. Returns windows
    keyed by regime with shape ``(n_windows, seq_len, n_channels)``, plus a manifest.
    """
    if panel.ndim != 2:
        raise ValueError(f"Expected panel shape (n_days, n_channels); got {panel.shape}")
    if len(panel) != len(labels):
        raise ValueError(f"panel has {len(panel)} days but labels have {len(labels)}")

    seq_len = int(cfg["diffusion"]["seq_len"])
    stride = int(cfg["diffusion"]["stride"])
    n_regimes = int(cfg["regimes"]["n_regimes"])
    short_policy = cfg["diffusion"]["short_policy"]
    label_channel = cfg["data"]["asset_columns"].index(cfg["data"]["price_column"])

    per_regime: dict[int, list[np.ndarray]] = {k: [] for k in range(n_regimes)}
    stats = {
        k: {"segments": 0, "segments_long": 0, "segments_tiled": 0, "windows_tiled": 0}
        for k in range(n_regimes)
    }

    for start, end, regime in contiguous_runs(labels):
        if not 0 <= regime < n_regimes:
            raise ValueError(f"Unexpected regime label {regime}")
        block = panel[start:end]
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

    n_channels = int(panel.shape[1])
    stacked = {
        k: (
            np.concatenate(per_regime[k], axis=0)
            if per_regime[k]
            else np.empty((0, seq_len, n_channels), dtype=float)
        )
        for k in range(n_regimes)
    }
    manifest = _build_manifest(panel, labels, stacked, stats, cfg, label_channel=label_channel)
    return stacked, manifest


def _build_manifest(
    panel: np.ndarray,
    labels: np.ndarray,
    windows: dict[int, np.ndarray],
    stats: dict[int, dict[str, int]],
    cfg: dict[str, Any],
    label_channel: int,
) -> dict[str, Any]:
    assets = list(cfg["data"]["asset_columns"])
    regimes: dict[str, Any] = {}
    for k, block in windows.items():
        source = panel[labels == k]
        label_source = source[:, label_channel] if source.size else np.empty(0)
        regimes[str(k)] = {
            "n_windows": int(block.shape[0]),
            "n_days": int(source.shape[0]),
            **stats[k],
            # Diagnostics use the A001 channel because that is what stage 1 labels on.
            "source_mean": float(label_source.mean()) if label_source.size else float("nan"),
            "source_var": float(label_source.var()) if label_source.size else float("nan"),
            "window_mean": float(block[:, :, label_channel].mean()) if block.size else float("nan"),
            "window_var": float(block[:, :, label_channel].var()) if block.size else float("nan"),
        }
    return {
        "seq_len": int(cfg["diffusion"]["seq_len"]),
        "stride": int(cfg["diffusion"]["stride"]),
        "short_policy": cfg["diffusion"]["short_policy"],
        "n_regimes": int(cfg["regimes"]["n_regimes"]),
        "n_channels": int(panel.shape[1]),
        "assets": assets,
        "label_asset": cfg["data"]["price_column"],
        "label_channel": int(label_channel),
        "return_units": "raw_log_returns",
        "n_days": int(panel.shape[0]),
        "regimes": regimes,
    }


def save_windows(
    windows: dict[int, np.ndarray], manifest: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for regime, block in windows.items():
        filename = f"regime_{regime}.npy"
        np.save(output_dir / filename, block)
        manifest["regimes"][str(regime)]["path"] = filename
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/02_build_diffusion_dataset.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
