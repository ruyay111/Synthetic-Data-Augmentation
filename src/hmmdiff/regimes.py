"""Volatility regime labeling and cached label I/O."""

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


def _prepare_vol_regime(train_series: np.ndarray, penalty: int) -> Any:
    from hmmgan.evaluation import Vol_Regime

    vc = Vol_Regime(train_series)
    vc.get_vol()
    vc.get_changepoints(pen=penalty)
    vc.get_attr()
    return vc


def _finalize_cluster_assignment(
    vc: Any, train_series: np.ndarray, clusters_assign: dict[int, int]
) -> np.ndarray:
    """Map segment clusters to variance-ordered day labels (same rule as ``assign_clusters``)."""
    cp = vc.changepoints.copy()
    cp[:0] = [0]

    var: dict[int, list[float]] = {}
    for seg_idx, cluster in clusters_assign.items():
        var.setdefault(cluster, []).append(float(np.var(train_series[cp[seg_idx] : cp[seg_idx + 1]])))
    mean_var = {cluster: float(np.mean(values)) for cluster, values in var.items()}
    conversion_dict = {raw: ordered for ordered, raw in enumerate(sorted(mean_var, key=mean_var.get))}

    vc.clusters = clusters_assign
    vc.conversion_dict = conversion_dict
    labels = np.ones(len(train_series), dtype=float)
    for seg_idx, cluster in clusters_assign.items():
        labels[cp[seg_idx] : cp[seg_idx + 1]] = conversion_dict[cluster]
    vc.regime_labels = labels
    return labels


def _assign_spectral(vc: Any, train_series: np.ndarray, n_regimes: int) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        os.environ["TQDM_DISABLE"] = "1"
        try:
            vc.assign_clusters(max_clusters=n_regimes)
        finally:
            os.environ.pop("TQDM_DISABLE", None)
    return np.asarray(vc.regime_labels, dtype=float)


def _assign_kmeans(vc: Any, train_series: np.ndarray, n_regimes: int, seed: int) -> np.ndarray:
    """Force ``n_regimes`` segment clusters via k-means on the rotated spectral embedding."""
    from sklearn.cluster import KMeans

    from hmmgan.evaluation._functions import affinity_to_lap_to_eig, get_rotation_matrix

    _, eigvecs = affinity_to_lap_to_eig(vc.attr)
    embedding = eigvecs[:, -n_regimes:]
    _, rotation = get_rotation_matrix(embedding, n_regimes)
    rotated = embedding.dot(rotation)
    n_segments = rotated.shape[0]
    segment_clusters = KMeans(
        n_clusters=n_regimes, n_init=50, random_state=seed
    ).fit_predict(rotated)
    clusters_assign = {seg_idx: int(segment_clusters[seg_idx]) for seg_idx in range(n_segments)}
    return _finalize_cluster_assignment(vc, train_series, clusters_assign)


def _clustering_methods(cfg: dict[str, Any]) -> tuple[str, ...]:
    requested = str(cfg["regimes"].get("clustering_method", "auto"))
    if requested == "auto":
        return ("spectral", "kmeans")
    if requested in {"spectral", "kmeans"}:
        return (requested,)
    raise ValueError(
        f"Unknown regimes.clustering_method {requested!r}; use auto, spectral, or kmeans."
    )


def _run_vol_regime(
    train_series: np.ndarray,
    penalty: int,
    n_regimes: int,
    cfg: dict[str, Any],
    method: str,
) -> tuple[Any, np.ndarray]:
    vc = _prepare_vol_regime(train_series, penalty)
    seed = int(cfg["regimes"].get("clustering_seed", 0))
    if method == "spectral":
        labels = _assign_spectral(vc, train_series, n_regimes)
    elif method == "kmeans":
        labels = _assign_kmeans(vc, train_series, n_regimes, seed=seed)
    else:
        raise ValueError(f"Unknown clustering method {method!r}")
    return vc, labels


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
    """Run Vol_Regime on the training series and return cached labels."""
    n_regimes = int(cfg["regimes"]["n_regimes"])
    methods = _clustering_methods(cfg)
    attempts: list[dict[str, Any]] = []
    chosen_penalty: int | None = None
    chosen_method: str | None = None
    vc = None
    labels: np.ndarray | None = None

    for penalty in _penalty_candidates(cfg):
        for method in methods:
            vc, labels = _run_vol_regime(train_series, penalty, n_regimes, cfg, method=method)
            ok, info = _labels_ok(train_series, labels, n_regimes, cfg)
            attempts.append({"penalty": penalty, "method": method, "ok": ok, **info})
            if ok:
                chosen_penalty = penalty
                chosen_method = method
                break
        if chosen_penalty is not None:
            break

    if chosen_penalty is None or chosen_method is None or vc is None or labels is None:
        lines = [
            "Could not find a changepoint penalty / clustering method that yields "
            f"{n_regimes} variance-ordered regimes with all regimes in the inner training slice.",
            "Tried:",
        ]
        for row in attempts:
            lines.append(
                f"  pen={row['penalty']:>2} {row['method']:<8} found={row['n_regimes_found']} "
                f"monotone={row['variance_monotone']} "
                f"inner_train={row['inner_train_counts']}"
            )
        lines.append(
            "Add a candidate to regimes.changepoint_penalty_fallback in configs/default.yaml."
        )
        raise SystemExit("\n".join(lines))

    configured_pen = int(cfg["regimes"]["changepoint_penalty"])
    requested_method = str(cfg["regimes"].get("clustering_method", "auto"))
    used_fallback = chosen_penalty != configured_pen or (
        requested_method == "auto" and chosen_method != "spectral"
    )
    if used_fallback:
        print(
            f"[WARN] configured penalty {configured_pen} / clustering {requested_method} "
            f"did not yield {n_regimes} usable regimes; using penalty {chosen_penalty} with "
            f"{chosen_method} clustering instead."
        )

    metadata = _build_metadata(
        train_series, labels, vc, cfg, chosen_penalty=chosen_penalty, chosen_method=chosen_method
    )
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
    series: np.ndarray,
    labels: np.ndarray,
    vc: Any,
    cfg: dict[str, Any],
    chosen_penalty: int,
    chosen_method: str,
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
        "clustering_method": chosen_method,
        "regime_counts": counts,
        "regime_variances": variances,
        "variance_monotone": bool(all(np.diff(variances) > 0)),
        "average_regime_length": average_regime_length(labels),
    }


def average_regime_length(labels: np.ndarray) -> dict[str, float]:
    """Mean consecutive days per regime visit."""
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
    """Exponentially weighted volatility of the return series."""
    return np.nan_to_num(
        np.sqrt(pd.DataFrame({"Column1": series}).ewm(com=com).var()).values
    )


def empirical_transition_matrix(labels: np.ndarray, n_regimes: int) -> np.ndarray:
    """Row-normalized transition counts over the label path."""
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
