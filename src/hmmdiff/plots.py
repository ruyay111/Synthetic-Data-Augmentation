"""Plotting helpers for returns, regimes, and generated series."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _regime_colors() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_prices(close: np.ndarray, title: str = "S&P 500 TR Closing Prices") -> None:
    """Notebook cell 5."""
    plt.figure()
    plt.plot(close)
    plt.title(title)
    plt.show()


def plot_regime_spans(
    series: np.ndarray,
    changepoints: np.ndarray,
    segment_regimes: list[int],
    overlay: np.ndarray | None = None,
    figsize: tuple[int, int] = (15, 5),
    title: str = "Training Returns & Volatility Regimes",
) -> None:
    """Returns with regime-colored background spans."""
    colors = _regime_colors()
    plt.figure(figsize=figsize)
    plt.plot(series)
    if overlay is not None:
        plt.plot(overlay)
    for i in range(len(changepoints) - 1):
        plt.axvspan(
            changepoints[i], changepoints[i + 1], alpha=0.3, color=colors[segment_regimes[i]]
        )
    plt.title(title)
    plt.show()


def plot_regime_labels(labels: np.ndarray, figsize: tuple[int, int] = (15, 4)) -> None:
    """Notebook cell 10."""
    plt.figure(figsize=figsize)
    plt.plot(labels)
    plt.title("Volatility Regime Labels")
    plt.show()


def plot_estimates(
    estimates: np.ndarray,
    truth: np.ndarray,
    title: str,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Estimated vs true regimes."""
    plt.figure(figsize=figsize)
    plt.plot(estimates, label="Estimate")
    plt.plot(truth, label="Truth")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_regime_paths(
    series: dict[str, np.ndarray],
    title: str,
    figsize: tuple[int, int] = (12, 6),
    jiggle: float = 0.12,
) -> None:
    """Step-plot of multiple daily regime series (validation overlay).

    ``jiggle`` offsets each line vertically so overlapping integer paths stay visible.
    Tick labels remain the integer regimes.
    """
    names = list(series)
    n_series = len(names)
    offsets = np.linspace(-jiggle, jiggle, n_series) if n_series > 1 else np.zeros(1)
    ymax = 0.0
    plt.figure(figsize=figsize)
    for offset, name in zip(offsets, names):
        values = np.asarray(series[name], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            ymax = max(ymax, float(finite.max()))
        plt.plot(values + offset, drawstyle="steps-post", label=name)
    plt.legend()
    plt.title(title)
    plt.ylabel("regime")
    plt.xlabel("validation day")
    plt.yticks(np.arange(0, int(np.floor(ymax)) + 1))
    plt.ylim(-0.35 - jiggle, ymax + 0.35 + jiggle)
    plt.show()


MVO_METHOD_PALETTE = {
    "hmm-diffusion": "#ff7f0e",
    "mixed": "#1f77b4",
}
MVO_HUE_ORDER = ("hmm-diffusion", "mixed")
MVO_BUCKET_ORDER = ("high vol", "low vol")
MVO_SYNTH_PCT_MIN = 0
MVO_SYNTH_PCT_MAX = 90
MVO_SYNTH_PCT_STEP = 10


def _mvo_orders(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    hue_order = [m for m in MVO_HUE_ORDER if m in set(frame["method"])]
    col_order = [b for b in MVO_BUCKET_ORDER if b in set(frame["bucket"])]
    if not col_order:
        col_order = sorted(frame["bucket"].dropna().unique().tolist())
    return hue_order, col_order


def _show_all_y_tick_labels(grid, ylabel: str | None = None) -> None:
    """Show y tick numbers and the y-axis label on every facet."""
    grid.tick_params(axis="y", labelleft=True)
    for ax in grid.axes.flat:
        ax.tick_params(axis="y", which="both", labelleft=True)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.yaxis.get_label().set_visible(True)


def _ylim_from_values(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    """Axis limits that cover ``values`` with a small margin."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(finite.min())
    hi = float(finite.max())
    if lo == hi:
        span = max(abs(lo), 1.0) * 0.1
        return lo - span, hi + span
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def _set_per_bucket_ylim(grid, frame: pd.DataFrame, metric: str, col_order: list[str]) -> None:
    """Each high/low vol panel uses that bucket's own value range."""
    for ax, bucket in zip(np.ravel(grid.axes), col_order):
        subset = frame.loc[frame["bucket"] == bucket, metric]
        ax.set_ylim(*_ylim_from_values(subset.to_numpy()))


def _default_mvo_title(metric: str, mean: bool) -> str:
    label = str(metric).replace("_", " ").capitalize()
    prefix = f"Mean {label}" if mean else label
    return f"{prefix} by % of synth"


def _mvo_synth_pct_frame(windows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Keep 0%, 10%, …, 90% mix points. Use stored ``synth_pct`` when present."""
    frame = windows.dropna(subset=[metric]).copy()
    if frame.empty:
        return frame
    if "synth_pct" not in frame.columns:
        if "n_synth" not in frame.columns:
            return frame
        frame["synth_pct"] = frame["n_synth"].astype(float) * MVO_SYNTH_PCT_STEP
    frame["synth_pct"] = np.round(frame["synth_pct"].astype(float)).astype(int)
    return frame.loc[
        (frame["synth_pct"] >= MVO_SYNTH_PCT_MIN)
        & (frame["synth_pct"] <= MVO_SYNTH_PCT_MAX)
    ].copy()


def _apply_mvo_title(
    fig,
    title: str,
    *,
    objective: str | None = None,
    constraint: str | None = None,
) -> None:
    tags = []
    if objective:
        tags.append(f"[objective: {objective}]")
    if constraint:
        tags.append(f"[constraints: {constraint}]")
    if not tags:
        fig.suptitle(title, y=1.03)
        return
    fig.subplots_adjust(top=0.80)
    fig.suptitle(title, y=0.98, fontsize=13)
    fig.text(
        0.5,
        0.905,
        "  ".join(tags),
        ha="center",
        va="top",
        fontsize=11,
        color="0.25",
    )


def plot_mvo_boxes(
    windows: pd.DataFrame,
    metric: str,
    title: str | None = None,
    *,
    objective: str | None = None,
    constraint: str | None = None,
) -> None:
    """Boxplots of a window metric by mix percent, method, and high/low vol."""
    import seaborn as sns

    frame = _mvo_synth_pct_frame(windows, metric)
    if frame.empty:
        return
    hue_order, col_order = _mvo_orders(frame)
    pct_order = sorted(frame["synth_pct"].unique().tolist())
    grid = sns.catplot(
        data=frame,
        x="synth_pct",
        y=metric,
        hue="method",
        hue_order=hue_order,
        col="bucket",
        col_order=col_order,
        kind="box",
        order=pct_order,
        palette=MVO_METHOD_PALETTE,
        sharey=False,
        height=4.2,
        aspect=1.15,
    )
    _apply_mvo_title(
        grid.fig,
        title or _default_mvo_title(metric, mean=False),
        objective=objective,
        constraint=constraint,
    )
    grid.set_axis_labels("% of synth", metric)
    _set_per_bucket_ylim(grid, frame, metric, col_order)
    _show_all_y_tick_labels(grid, ylabel=metric)
    plt.show()


def plot_mvo_means(
    windows: pd.DataFrame,
    metric: str,
    title: str | None = None,
    *,
    objective: str | None = None,
    constraint: str | None = None,
) -> None:
    """Mean lines of a window metric by mix percent, method, and high/low vol."""
    import seaborn as sns

    frame = _mvo_synth_pct_frame(windows, metric)
    if frame.empty:
        return
    hue_order, col_order = _mvo_orders(frame)
    means = (
        frame.groupby(["bucket", "method", "synth_pct"], as_index=False)[metric]
        .mean()
        .sort_values("synth_pct")
    )
    grid = sns.relplot(
        data=means,
        x="synth_pct",
        y=metric,
        hue="method",
        hue_order=hue_order,
        col="bucket",
        col_order=col_order,
        kind="line",
        marker="o",
        palette=MVO_METHOD_PALETTE,
        facet_kws={"sharey": False},
        height=4.2,
        aspect=1.15,
    )
    _apply_mvo_title(
        grid.fig,
        title or _default_mvo_title(metric, mean=True),
        objective=objective,
        constraint=constraint,
    )
    grid.set_axis_labels("% of synth", f"mean {metric}")
    ticks = list(range(MVO_SYNTH_PCT_MIN, MVO_SYNTH_PCT_MAX + 1, MVO_SYNTH_PCT_STEP))
    for ax in grid.axes.flat:
        ax.set_xticks(ticks)
    _set_per_bucket_ylim(grid, means, metric, col_order)
    _show_all_y_tick_labels(grid, ylabel=f"mean {metric}")
    plt.show()


def plot_real_vs_generated(
    real_returns: np.ndarray,
    real_regimes: np.ndarray | None,
    generated_returns: np.ndarray,
    generated_regimes: np.ndarray,
    figsize: tuple[int, int] = (20, 5),
    ylim: tuple[float, float] = (-8, 6),
    generated_title: str = "HMM Diffusion Generated Returns",
    suptitle: str | None = None,
) -> None:
    """Real vs generated returns with regime overlays."""
    fig = plt.figure(figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)

    plt.subplot(1, 2, 1)
    plt.plot(real_returns)
    if real_regimes is not None:
        plt.plot(real_regimes, color="black", label="regime")
    plt.title("Real Returns")
    plt.ylim(*ylim)
    if real_regimes is not None:
        plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(generated_returns)
    plt.plot(generated_regimes, color="black", label="regime")
    plt.title(generated_title)
    plt.ylim(*ylim)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_paper_train_test(
    train_real: np.ndarray,
    train_true_regimes: np.ndarray,
    train_estimated_regimes: np.ndarray,
    test_real: np.ndarray,
    test_true_regimes: np.ndarray,
    test_estimated_regimes: np.ndarray,
    generated_images: dict[str, np.ndarray],
    generated_title: str = "HMM Diffusion Generated Returns",
    ylim: tuple[float, float] = (-8, 6),
) -> tuple[np.ndarray, np.ndarray]:
    """Train and validation real vs stitched generated returns."""
    from . import stitch

    train_generated = stitch.stitch(generated_images, train_estimated_regimes)
    test_generated = stitch.stitch(generated_images, test_estimated_regimes)

    plot_real_vs_generated(
        train_real,
        train_true_regimes,
        train_generated,
        train_estimated_regimes,
        ylim=ylim,
        generated_title=generated_title,
        suptitle="Real and HMM-Diffusion Generated Log Returns, Training Set",
    )
    plot_real_vs_generated(
        test_real,
        test_true_regimes,
        test_generated,
        test_estimated_regimes,
        ylim=ylim,
        generated_title=generated_title,
        suptitle="Real and HMM-Diffusion Generated Log Returns, Test Set",
    )
    return train_generated, test_generated


def plot_elbo(losses: list[float], figsize: tuple[int, int] = (10, 4)) -> None:
    """SVI convergence for the neural HMM, which has no MCMC summary to inspect instead."""
    plt.figure(figsize=figsize)
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss per observation")
    plt.title("Neural HMM Training")
    plt.show()


def regime_moments(series: np.ndarray, labels: np.ndarray, n_regimes: int) -> pd.DataFrame:
    """Per-regime mean and variance of emissions."""
    rows = []
    for regime in range(n_regimes):
        subset = series[labels == regime]
        rows.append(
            {
                "regime": regime,
                "n_days": int(subset.size),
                "mean": float(subset.mean()) if subset.size else float("nan"),
                "variance": float(subset.var()) if subset.size else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def regime_length_table(
    train_data: pd.DataFrame, switchpoints: list[int], train_val_split: int
) -> pd.DataFrame:
    """Regime durations either side of each switch within the training slice (cell 19)."""
    bounds = np.array([0] + list(switchpoints))
    bounds = bounds[bounds <= train_val_split]

    rows = []
    for i in range(1, len(bounds)):
        rows.append(
            {
                "current_regime": train_data.regime.iloc[bounds[i]]
                if bounds[i] < len(train_data)
                else np.nan,
                "previous_regime": train_data.regime.iloc[bounds[i] - 1],
                "previous_regime_length": bounds[i] - bounds[i - 1],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["current_regime_length"] = table.previous_regime_length.shift(-1)
    return table.dropna()
