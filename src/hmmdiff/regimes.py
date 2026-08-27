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

import json
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


def fit_regimes(train_series: np.ndarray, cfg: dict[str, Any]) -> RegimeLabels:
    """Run the full ``Vol_Regime`` pipeline on the training series.

    Mirrors notebook cell 6: ``get_vol`` then ``get_changepoints`` then ``get_attr`` then
    ``assign_clusters(max_clusters=n_regimes)``.
    """
    from hmmgan.evaluation import Vol_Regime

    n_regimes = cfg["regimes"]["n_regimes"]
    vc = Vol_Regime(train_series)
    vc.get_vol()
    vc.get_changepoints(pen=cfg["regimes"]["changepoint_penalty"])
    vc.get_attr()
    # max_clusters is an upper bound; the self-tuning search may settle on fewer.
    vc.assign_clusters(max_clusters=n_regimes)

    labels = np.asarray(vc.regime_labels, dtype=float)
    metadata = _build_metadata(train_series, labels, vc, cfg)
    return RegimeLabels(
        labels=labels,
        changepoints=np.asarray(vc.changepoints, dtype=int),
        clusters={int(k): int(v) for k, v in vc.clusters.items()},
        conversion_dict={int(k): int(v) for k, v in vc.conversion_dict.items()},
        volatility=np.asarray(vc.vol, dtype=float),
        metadata=metadata,
    )


def _build_metadata(
    series: np.ndarray, labels: np.ndarray, vc: Any, cfg: dict[str, Any]
) -> dict[str, Any]:
    found = int(labels.max()) + 1
    variances = [float(np.var(series[labels == k])) for k in range(found)]
    counts = [int((labels == k).sum()) for k in range(found)]
    return {
        "n_days": int(len(series)),
        "n_regimes_requested": int(cfg["regimes"]["n_regimes"]),
        "n_regimes_found": found,
        "n_changepoints": int(len(vc.changepoints)),
        "changepoint_penalty": cfg["regimes"]["changepoint_penalty"],
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
