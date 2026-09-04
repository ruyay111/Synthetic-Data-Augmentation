"""Plotting helpers for returns, regimes, MVO backtests, and mix R^2 grids.

PlotAnalytics in testing_analytics is the class API. Functions here are the
standalone snake_case drawing utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmmdiff.constants import (
    HMM_DIFFUSION_COLOR,
    UNCOND_DIFFUSION_COLOR,
    VOL_MIX_PCT_MAX as CONST_VOL_MIX_PCT_MAX,
)


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
    save_path: Path | str | None = None,
) -> None:
    """Step-plot of multiple daily regime series (validation overlay).

    ``jiggle`` offsets each line vertically so overlapping integer paths stay visible.
    Tick labels remain the integer regimes.
    """
    names = list(series)
    n_series = len(names)
    offsets = np.linspace(-jiggle, jiggle, n_series) if n_series > 1 else np.zeros(1)
    ymax = 0.0
    fig = plt.figure(figsize=figsize)
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
    if save_path is not None:
        _save_figure(fig, save_path)
    plt.show()


MVO_METHOD_PALETTE = {
    "hmm-diffusion": "#ff7f0e",
    "uncondi-diffusion": "#1f77b4",
}
MVO_HUE_ORDER = ("hmm-diffusion", "uncondi-diffusion")
MVO_BUCKET_ORDER = ("high vol", "low vol")
MVO_WINDOW_METRICS = ("sharpe", "calmar", "return", "variance")
MVO_METRIC_LABELS = {
    "sharpe": "Sharpe",
    "calmar": "Calmar",
    "return": "Ann. return",
    "variance": "Ann. variance",
}
MVO_SYNTH_PCT_MIN = 0
MVO_SYNTH_PCT_MAX = 90
MVO_SYNTH_PCT_STEP = 10


def _save_figure(fig, path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = dest.suffix.lstrip(".") or "eps"
    fig.savefig(dest, format=fmt, bbox_inches="tight")
    return dest


def _slug_token(text: str) -> str:
    token = str(text).strip().lower().replace("%", "pct").replace(" ", "-")
    token = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in token)
    while "--" in token:
        token = token.replace("--", "-")
    return token.strip("-._")


def mvo_eps_path(
    root: Path | str,
    metric: str,
    kind: str,
    *,
    objective: str | None = None,
    constraint: str | None = None,
) -> Path:
    """``<root>/<metric>/<kind>_<objective>_<constraint>.eps``."""
    parts = [_slug_token(kind)]
    if objective:
        parts.append(_slug_token(objective))
    if constraint:
        parts.append(_slug_token(constraint))
    return Path(root) / _slug_token(metric) / f"{'_'.join(parts)}.eps"


def _mvo_orders(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    hue_order = [m for m in MVO_HUE_ORDER if m in set(frame["method"])]
    col_order = [b for b in MVO_BUCKET_ORDER if b in set(frame["bucket"])]
    if not col_order:
        col_order = sorted(frame["bucket"].dropna().unique().tolist())
    return hue_order, col_order


def _metric_label(metric: str) -> str:
    key = str(metric)
    if key in MVO_METRIC_LABELS:
        return MVO_METRIC_LABELS[key]
    return key.replace("_", " ")


def _default_mvo_title(metric: str, mean: bool) -> str:
    label = _metric_label(metric)
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
    save_path: Path | str | None = None,
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
        sharey=True,
        height=4.2,
        aspect=1.15,
    )
    _apply_mvo_title(
        grid.fig,
        title or _default_mvo_title(metric, mean=False),
        objective=objective,
        constraint=constraint,
    )
    grid.set_axis_labels("% of synth", _metric_label(metric))
    if save_path is not None:
        _save_figure(grid.fig, save_path)
    plt.show()


def plot_mvo_means(
    windows: pd.DataFrame,
    metric: str,
    title: str | None = None,
    *,
    objective: str | None = None,
    constraint: str | None = None,
    save_path: Path | str | None = None,
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
        facet_kws={"sharey": True},
        height=4.2,
        aspect=1.15,
    )
    _apply_mvo_title(
        grid.fig,
        title or _default_mvo_title(metric, mean=True),
        objective=objective,
        constraint=constraint,
    )
    grid.set_axis_labels("% of synth", f"mean {_metric_label(metric).lower()}")
    ticks = list(range(MVO_SYNTH_PCT_MIN, MVO_SYNTH_PCT_MAX + 1, MVO_SYNTH_PCT_STEP))
    for ax in grid.axes.flat:
        ax.set_xticks(ticks)
    if save_path is not None:
        _save_figure(grid.fig, save_path)
    plt.show()


def export_mvo_plots(
    windows: pd.DataFrame,
    save_root: Path | str,
    *,
    objective: str | None = None,
    constraint: str | None = None,
) -> list[Path]:
    """Box and mean plots for each window metric, saved under ``save_root/<metric>/``."""
    paths: list[Path] = []
    for metric in MVO_WINDOW_METRICS:
        boxes = mvo_eps_path(
            save_root, metric, "boxes", objective=objective, constraint=constraint
        )
        means = mvo_eps_path(
            save_root, metric, "means", objective=objective, constraint=constraint
        )
        plot_mvo_boxes(
            windows,
            metric,
            objective=objective,
            constraint=constraint,
            save_path=boxes,
        )
        plot_mvo_means(
            windows,
            metric,
            objective=objective,
            constraint=constraint,
            save_path=means,
        )
        paths.extend([boxes, means])
    return paths


VOL_HMM_COLOR = HMM_DIFFUSION_COLOR
VOL_UNCOND_COLOR = UNCOND_DIFFUSION_COLOR
VOL_HMM_LABEL = "HMM-Diffusion"
VOL_UNCOND_LABEL = "Unconditional Diffusion"
VOL_R2_DROP_CAP = 1.0


def add_mult_to_synth_pct(add_mult) -> np.ndarray:
    """Map add-multiplier ``m`` to mix share: ``100 * m / (1 + m)``."""
    m = np.asarray(add_mult, dtype=float)
    return 100.0 * m / (1.0 + m)


def _vol_compare_ylim(block: pd.DataFrame) -> tuple[float, float]:
    """Keep HMM / HAR / persist readable; do not follow Uncond crashes to -15."""
    hmm_std = block["hmm_r2_std"].fillna(0.0)
    uncond_std = block["uncond_r2_std"].fillna(0.0)
    hmm_lo = float((block["hmm_r2_mean"] - hmm_std).min())
    hmm_hi = float((block["hmm_r2_mean"] + hmm_std).max())
    persist = float(block["persist"].iloc[0])
    har = float(block["har"].iloc[0])
    uncond_lo = float((block["uncond_r2_mean"] - uncond_std).min())
    uncond_hi = float((block["uncond_r2_mean"] + uncond_std).max())
    core_lo = min(hmm_lo, persist, har)
    core_hi = max(hmm_hi, persist, har, uncond_hi)
    y_lo = min(core_lo, max(uncond_lo, core_lo - VOL_R2_DROP_CAP))
    y_hi = core_hi
    pad = max(0.04, 0.12 * max(y_hi - y_lo, 0.08))
    return y_lo - pad, y_hi + pad


def plot_vol_hmm_vs_uncond(compare: pd.DataFrame, horizons=None) -> None:
    """HMM-Diffusion (orange) vs Unconditional Diffusion (blue) test R² by % of synth."""
    frame = compare.copy()
    if "synth_pct" not in frame.columns:
        frame["synth_pct"] = add_mult_to_synth_pct(frame["add_mult"])
    if horizons is None:
        horizons = list(dict.fromkeys(frame["horizon"].tolist()))
    fig, axes = plt.subplots(1, len(horizons), figsize=(14, 4), sharey=False)
    if len(horizons) == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        block = frame.loc[frame["horizon"] == horizon].sort_values("synth_pct")
        ax.errorbar(
            block["synth_pct"],
            block["hmm_r2_mean"],
            yerr=block["hmm_r2_std"].fillna(0.0),
            marker="o",
            color=VOL_HMM_COLOR,
            label=VOL_HMM_LABEL,
        )
        ax.errorbar(
            block["synth_pct"],
            block["uncond_r2_mean"],
            yerr=block["uncond_r2_std"].fillna(0.0),
            marker="s",
            color=VOL_UNCOND_COLOR,
            label=VOL_UNCOND_LABEL,
        )
        ax.axhline(block["persist"].iloc[0], color="gray", ls=":", label="persist")
        ax.axhline(block["har"].iloc[0], color="tab:green", ls="--", label="HAR")
        y_lo, y_hi = _vol_compare_ylim(block)
        ax.set_ylim(y_lo, y_hi)
        uncond_lo = float((block["uncond_r2_mean"] - block["uncond_r2_std"].fillna(0.0)).min())
        if uncond_lo < y_lo:
            ax.text(
                0.98,
                0.04,
                f"Uncond min $R^2$ = {uncond_lo:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=VOL_UNCOND_COLOR,
            )
        ax.set_title(f"h = {horizon}")
        ax.set_xlabel("% of synth")
        ax.set_xticks(block["synth_pct"].to_numpy())
        ax.set_xticklabels([f"{p:.0f}" for p in block["synth_pct"]])
    axes[0].set_ylabel("Test $R^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [VOL_HMM_LABEL, VOL_UNCOND_LABEL, "persist", "HAR"]
    by_label = dict(zip(labels, handles))
    axes[0].legend(
        [by_label[name] for name in order if name in by_label],
        [name for name in order if name in by_label],
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("HMM-Diffusion vs Unconditional Diffusion: add synthetic, keep all real", y=1.03)
    fig.tight_layout()
    plt.show()


VOL_MIX_PCT_MAX = CONST_VOL_MIX_PCT_MAX


def _mix_frame(frame: pd.DataFrame, pct_max: int) -> pd.DataFrame:
    return frame.loc[frame["synthetic_pct"] <= pct_max].copy()


def _draw_mix_r2(
    ax,
    hmm: pd.DataFrame,
    uncond: pd.DataFrame,
    horizon,
    uncond_label: str,
) -> None:
    a = hmm.loc[hmm["horizon"] == horizon].sort_values("synthetic_pct")
    b = uncond.loc[uncond["horizon"] == horizon].sort_values("synthetic_pct")
    ax.plot(
        a["synthetic_pct"],
        a["r2"],
        marker="o",
        color=VOL_HMM_COLOR,
        label="hmm-diffusion",
    )
    ax.plot(
        b["synthetic_pct"],
        b["r2"],
        marker="s",
        color=VOL_UNCOND_COLOR,
        label=uncond_label,
    )
    y = np.concatenate([a["r2"].to_numpy(dtype=float), b["r2"].to_numpy(dtype=float)])
    y = y[np.isfinite(y)]
    if y.size:
        y_lo, y_hi = float(y.min()), float(y.max())
        pad = max(0.04, 0.08 * max(y_hi - y_lo, 0.1))
        ax.set_ylim(y_lo - pad, min(1.0, y_hi + pad))
    ticks = sorted(set(a["synthetic_pct"].tolist()) | set(b["synthetic_pct"].tolist()))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{p:.0f}" for p in ticks])


def plot_vol_mixture_compare(
    hmm: pd.DataFrame,
    uncond: pd.DataFrame,
    horizons=None,
    *,
    pct_max: int = VOL_MIX_PCT_MAX,
    title: str | None = None,
    uncond_label: str = "uncondi-diffusion",
) -> None:
    """Overlay hmm-diffusion and uncondi-diffusion test R² by % of synth.

    One panel per horizon. Both series share that panel's y-axis so the pair is
    comparable. Mix points above ``pct_max`` (100%) are omitted.
    """
    hmm_f = _mix_frame(hmm, pct_max)
    uncond_f = _mix_frame(uncond, pct_max)
    if horizons is None:
        horizons = list(dict.fromkeys(hmm_f["horizon"].tolist()))
    n_h = len(horizons)
    fig_w = 5.8 if n_h == 1 else 14
    fig, axes = plt.subplots(1, n_h, figsize=(fig_w, 4), sharey=False)
    if n_h == 1:
        axes = [axes]
    for ax, horizon in zip(axes, horizons):
        _draw_mix_r2(ax, hmm_f, uncond_f, horizon, uncond_label)
        ax.set_title(f"h = {int(horizon)}")
        ax.set_xlabel("% of synth")
    axes[0].set_ylabel("Test $R^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, frameon=False)
    fig.suptitle(title or "hmm-diffusion vs uncondi-diffusion", y=1.03)
    fig.tight_layout()
    plt.show()


def _mix_long_frame(
    hmm: pd.DataFrame,
    uncond: pd.DataFrame,
    task: str,
    *,
    hmm_label: str,
    uncond_label: str,
    pct_max: int,
) -> pd.DataFrame:
    a = _mix_frame(hmm, pct_max).assign(method=hmm_label, task=task)
    b = _mix_frame(uncond, pct_max).assign(method=uncond_label, task=task)
    return pd.concat([a, b], ignore_index=True)


MIX_GRID_HEIGHT = 3.2
MIX_GRID_ASPECT = 1.15
MIX_GRID_WIDTH_COLS = 2


def _mixture_relplot(
    long: pd.DataFrame,
    horizons,
    *,
    n_cols: int,
    title: str | None,
    uncond_label: str,
    save_path: Path | str | None,
    col_order: tuple[str, ...] | None = None,
) -> None:
    import seaborn as sns

    hmm_label = "hmm-diffusion"
    palette = {
        hmm_label: MVO_METHOD_PALETTE.get(hmm_label, VOL_HMM_COLOR),
        uncond_label: MVO_METHOD_PALETTE.get(uncond_label, VOL_UNCOND_COLOR),
    }
    aspect = MIX_GRID_ASPECT * MIX_GRID_WIDTH_COLS / max(int(n_cols), 1)
    rel_kwargs: dict = dict(
        data=long,
        x="synthetic_pct",
        y="r2",
        hue="method",
        hue_order=(hmm_label, uncond_label),
        row="horizon",
        row_order=list(horizons),
        kind="line",
        marker="o",
        palette=palette,
        facet_kws={"sharey": False, "sharex": True, "margin_titles": True},
        height=MIX_GRID_HEIGHT,
        aspect=aspect,
    )
    if n_cols > 1:
        rel_kwargs["col"] = "task"
        rel_kwargs["col_order"] = col_order
    with sns.axes_style("whitegrid"):
        grid = sns.relplot(**rel_kwargs)
    grid.set_axis_labels("% of synth", "Test $R^2$")
    if n_cols > 1:
        grid.set_titles(col_template="{col_name}", row_template="h = {row_name}")
    else:
        grid.set_titles(template="h = {row_name}")
    ticks = sorted(long["synthetic_pct"].unique().tolist())
    for ax in grid.axes.flat:
        ax.set_xticks(ticks)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, min(1.0, hi))
    grid.fig.suptitle(title or "hmm-diffusion vs uncondi-diffusion", y=1.02)
    grid.tight_layout()
    if save_path is not None:
        _save_figure(grid.fig, save_path)
    plt.show()


def plot_vol_return_mixture_grid(
    vol_hmm: pd.DataFrame,
    vol_uncond: pd.DataFrame,
    ret_hmm: pd.DataFrame,
    ret_uncond: pd.DataFrame,
    horizons=None,
    *,
    pct_max: int = VOL_MIX_PCT_MAX,
    title: str | None = None,
    uncond_label: str = "uncondi-diffusion",
    save_path: Path | str | None = None,
) -> None:
    """Test R² mix plots in a horizon × task grid, styled like the MVO mean plots.

    Rows are forecast horizons. Column 0 is volatility; column 1 is return.
    Each panel overlays hmm-diffusion and uncondi-diffusion. Y-limits are per
    panel and capped at 1.
    """
    hmm_label = "hmm-diffusion"
    long = pd.concat(
        [
            _mix_long_frame(
                vol_hmm,
                vol_uncond,
                "Volatility",
                hmm_label=hmm_label,
                uncond_label=uncond_label,
                pct_max=pct_max,
            ),
            _mix_long_frame(
                ret_hmm,
                ret_uncond,
                "Return",
                hmm_label=hmm_label,
                uncond_label=uncond_label,
                pct_max=pct_max,
            ),
        ],
        ignore_index=True,
    )
    if horizons is None:
        horizons = list(dict.fromkeys(long["horizon"].tolist()))
    long = long.loc[long["horizon"].isin(list(horizons))].copy()
    _mixture_relplot(
        long,
        horizons,
        n_cols=2,
        title=title,
        uncond_label=uncond_label,
        save_path=save_path,
        col_order=("Volatility", "Return"),
    )


def plot_vol_mixture_column(
    vol_hmm: pd.DataFrame,
    vol_uncond: pd.DataFrame,
    horizons=None,
    *,
    pct_max: int = VOL_MIX_PCT_MAX,
    title: str | None = None,
    uncond_label: str = "uncondi-diffusion",
    save_path: Path | str | None = None,
) -> None:
    """Volatility-only mix R²: one column, same figure width as the 3×2 grid."""
    hmm_label = "hmm-diffusion"
    long = _mix_long_frame(
        vol_hmm,
        vol_uncond,
        "Volatility",
        hmm_label=hmm_label,
        uncond_label=uncond_label,
        pct_max=pct_max,
    )
    if horizons is None:
        horizons = list(dict.fromkeys(long["horizon"].tolist()))
    long = long.loc[long["horizon"].isin(list(horizons))].copy()
    _mixture_relplot(
        long,
        horizons,
        n_cols=1,
        title=title or "hmm-diffusion vs uncondi-diffusion (volatility)",
        uncond_label=uncond_label,
        save_path=save_path,
    )


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
