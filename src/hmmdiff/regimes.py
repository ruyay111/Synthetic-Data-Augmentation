"""Volatility regime labeling and the label cache.

Wraps the reference ``Vol_Regime`` (ARCH conditional volatility, PELT changepoints, Wasserstein
segment affinity, self-tuning spectral clustering) and persists its output.

The cache is load-bearing rather than a speed optimization. ``assign_clusters`` selects a cluster
count and assignment by minimizing a rotation-alignment cost with conjugate gradient, which is not
guaranteed to land in the same place on a rerun. The specialists in ``data/processed/regime_windows``
are trained against one specific labeling, so every downstream stage must read that same labeling back
rather than recomputing it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeLabels:
    """Cached output of one ``Vol_Regime`` run."""

    labels: np.ndarray
    changepoints: np.ndarray
    clusters: dict[int, int]
    conversion_dict: dict[int, int]
    volatility: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_regimes(self) -> int:
        return int(self.labels.max()) + 1

    def segment_colors(self) -> list[int]:
        """Regime index per changepoint segment, for the ``axvspan`` plots."""
        return [self.conversion_dict[self.clusters[i]] for i in range(len(self.changepoints) - 1)]


def _penalty_candidates(cfg: dict[str, Any]) -> list[int]:
    primary = int(cfg["regimes"]["changepoint_penalty"])
    fallbacks = [int(p) for p in cfg["regimes"].get("changepoint_penalty_fallback", [])]
    seen = set()
    ordered: list[int] = []
    for pen in [primary, *fallbacks]:
        if pen not in seen:
            ordered.append(pen)
            seen.add(pen)
    return ordered


def _inner_train_cut(train_len: int, cfg: dict[str, Any]) -> int:
    return int(train_len * cfg["data"]["train_val_fraction"])


def _run_vol_regime(train_series: np.ndarray, penalty: int, n_regimes: int) -> tuple[Any, np.ndarray]:
    """Run Vol_Regime with tqdm suppressed so penalty sweeps stay readable."""
    from hmmgan.evaluation import Vol_Regime

    vc = Vol_Regime(train_series)
    vc.get_vol()
    vc.get_changepoints(pen=penalty)
    vc.get_attr()
    with contextlib.redirect_stdout(io.StringIO()):
        os.environ["TQDM_DISABLE"] = "1"
        try:
            vc.assign_clusters(max_clusters=n_regimes)
        finally:
            os.environ.pop("TQDM_DISABLE", None)
    return vc, np.asarray(vc.regime_labels, dtype=float)


def _labels_ok(
    train_series: np.ndarray, labels: np.ndarray, n_regimes: int, cfg: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    found = int(labels.max()) + 1
    variances = [float(np.var(train_series[labels == k])) for k in range(found)]
    inner_cut = _inner_train_cut(len(train_series), cfg)
    inner_counts = np.bincount(labels[:inner_cut].astype(int), minlength=n_regimes)
    info = {
        "n_regimes_found": found,
        "regime_variances": variances,
        "variance_monotone": bool(all(np.diff(variances) > 0)) if found > 1 else True,
        "inner_train_counts": inner_counts.tolist(),
        "all_regimes_in_inner_train": bool(found == n_regimes and (inner_counts > 0).all()),
    }
    ok = (
        found == n_regimes
        and info["variance_monotone"]
        and info["all_regimes_in_inner_train"]
    )
    return ok, info


def fit_regimes(train_series: np.ndarray, cfg: dict[str, Any]) -> RegimeLabels:
    """Run the full ``Vol_Regime`` pipeline on the training series.

    Mirrors notebook cell 6, but sweeps changepoint penalties when the configured value yields
    fewer than ``n_regimes`` on the current platform. The reference algorithm is unchanged; only
    the PELT penalty varies.
    """
    n_regimes = int(cfg["regimes"]["n_regimes"])
    attempts: list[dict[str, Any]] = []
    chosen_penalty: int | None = None
    vc = None
    labels: np.ndarray | None = None

    for penalty in _penalty_candidates(cfg):
        vc, labels = _run_vol_regime(train_series, penalty, n_regimes)
        ok, info = _labels_ok(train_series, labels, n_regimes, cfg)
        attempts.append({"penalty": penalty, "ok": ok, **info})
        if ok:
            chosen_penalty = penalty
            break

    if chosen_penalty is None or vc is None or labels is None:
        lines = [
            "Could not find a changepoint penalty that yields "
            f"{n_regimes} variance-ordered regimes with all regimes in the inner training slice.",
            "Tried:",
        ]
        for row in attempts:
            lines.append(
                f"  pen={row['penalty']:>2}: found={row['n_regimes_found']} "
                f"monotone={row['variance_monotone']} "
                f"inner_train={row['inner_train_counts']}"
            )
        lines.append(
            "Add a candidate to regimes.changepoint_penalty_fallback in configs/default.yaml."
        )
        raise SystemExit("\n".join(lines))

    if chosen_penalty != int(cfg["regimes"]["changepoint_penalty"]):
        print(
            f"[WARN] configured penalty {cfg['regimes']['changepoint_penalty']} did not yield "
            f"{n_regimes} usable regimes on this platform; using penalty {chosen_penalty} instead."
        )

    metadata = _build_metadata(train_series, labels, vc, cfg, chosen_penalty=chosen_penalty)
    metadata["penalty_attempts"] = attempts
    return RegimeLabels(
        labels=labels,
        changepoints=np.asarray(vc.changepoints, dtype=int),
        clusters={int(k): int(v) for k, v in vc.clusters.items()},
        conversion_dict={int(k): int(v) for k, v in vc.conversion_dict.items()},
        volatility=np.asarray(vc.vol, dtype=float),
        metadata=metadata,
    )


def _build_metadata(
    series: np.ndarray, labels: np.ndarray, vc: Any, cfg: dict[str, Any], chosen_penalty: int
) -> dict[str, Any]:
    found = int(labels.max()) + 1
    variances = [float(np.var(series[labels == k])) for k in range(found)]
    counts = [int((labels == k).sum()) for k in range(found)]
    return {
        "n_days": int(len(series)),
        "n_regimes_requested": int(cfg["regimes"]["n_regimes"]),
        "n_regimes_found": found,
        "n_changepoints": int(len(vc.changepoints)),
        "changepoint_penalty": chosen_penalty,
        "changepoint_penalty_configured": int(cfg["regimes"]["changepoint_penalty"]),
        "regime_counts": counts,
        "regime_variances": variances,
        "variance_monotone": bool(all(np.diff(variances) > 0)),
        "average_regime_length": average_regime_length(labels),
    }


def average_regime_length(labels: np.ndarray) -> dict[str, float]:
    """Mean number of consecutive days spent in each regime per visit (notebook cell 9)."""
    regime_series = pd.Series(labels)
    diffs = regime_series.diff()
    switchpoints = diffs.dropna()[diffs != 0].index.tolist() + [regime_series.shape[0]]

    lengths: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for i, switchpoint in enumerate(switchpoints):
        previous = str(int(regime_series.loc[switchpoint - 1]))
        counts[previous] += 1
        lengths[previous] += switchpoints[i] if i == 0 else switchpoints[i] - switchpoints[i - 1]
    return {regime: lengths[regime] / counts[regime] for regime in lengths}


def switchpoints(labels: np.ndarray) -> list[int]:
    """Indices at which the regime changes, with the series length appended (cell 9)."""
    regime_series = pd.Series(labels)
    diffs = regime_series.diff()
    return diffs.dropna()[diffs != 0].index.tolist() + [regime_series.shape[0]]


def save_labels(regimes: RegimeLabels, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        labels=regimes.labels,
        changepoints=regimes.changepoints,
        cluster_keys=np.array(sorted(regimes.clusters), dtype=int),
        cluster_values=np.array(
            [regimes.clusters[k] for k in sorted(regimes.clusters)], dtype=int
        ),
        conversion_keys=np.array(sorted(regimes.conversion_dict), dtype=int),
        conversion_values=np.array(
            [regimes.conversion_dict[k] for k in sorted(regimes.conversion_dict)], dtype=int
        ),
        volatility=regimes.volatility,
        metadata=np.array(json.dumps(regimes.metadata)),
    )


def load_labels(path: Path) -> RegimeLabels:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_label_regimes.py first."
        )
    payload = np.load(path, allow_pickle=False)
    return RegimeLabels(
        labels=payload["labels"],
        changepoints=payload["changepoints"],
        clusters=dict(zip(payload["cluster_keys"].tolist(), payload["cluster_values"].tolist())),
        conversion_dict=dict(
            zip(payload["conversion_keys"].tolist(), payload["conversion_values"].tolist())
        ),
        volatility=payload["volatility"],
        metadata=json.loads(str(payload["metadata"])),
    )


def ewm_volatility(series: np.ndarray, com: float) -> np.ndarray:
    """Exponentially weighted volatility of the return series (notebook cell 7)."""
    return np.nan_to_num(
        np.sqrt(pd.DataFrame({"Column1": series}).ewm(com=com).var()).values
    )


def empirical_transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
    """Row-normalized transition counts over the full label path (notebook cell 20)."""
    counts: dict[str, int] = defaultdict(int)
    for i in range(len(labels) - 1):
        counts[f"{int(labels[i])}->{int(labels[i + 1])}"] += 1
    matrix = np.array(
        [[counts[f"{i}->{j}"] for j in range(n_regimes)] for i in range(n_regimes)], dtype=float
    )
    return matrix / matrix.sum(1, keepdims=True)


def outbound_transition_matrix(transition_matrix: np.ndarray, n_regimes: int) -> np.ndarray:
    """Transition probabilities conditional on leaving the current regime (cell 21).

    The diagonal is zeroed and each row renormalized, which shows how the market moves when regimes
    actually change rather than being dominated by regime persistence.
    """
    rows = []
    for i in range(n_regimes):
        others = [j for j in range(n_regimes) if j != i]
        row = list(transition_matrix[i][others] / transition_matrix[i][others].sum())
        row.insert(i, 0.0)
        rows.append(row)
    return np.array(rows)
