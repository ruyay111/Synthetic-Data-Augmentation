"""Every figure from the reference notebook.

Figure sizes, axis limits, and titles follow ``Supervised HMMs.ipynb`` so the output is directly
comparable. The only intentional change is the title of the paired generated-returns panel, which
says "Diffusion" instead of "GAN".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _regime_colors() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_prices(close: np.ndarray) -> None:
    """Notebook cell 5."""
    plt.figure()
    plt.plot(close)
    plt.title("S&P 500 TR Closing Prices")
    plt.show()


def plot_regime_spans(
    series: np.ndarray,
    changepoints: np.ndarray,
    segment_regimes: list[int],
    overlay: np.ndarray | None = None,
    figsize: tuple[int, int] = (15, 5),
    title: str = "Training Returns & Volatility Regimes",
) -> None:
    """Returns with regime-colored spans (notebook cells 6 and 7).

    ``overlay`` draws the EWMA volatility on top, as cell 7 does.
    """
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
    """Estimated against true regimes (notebook cells 28 and 32)."""
    plt.figure(figsize=figsize)
    plt.plot(estimates, label="Estimate")
    plt.plot(truth, label="Truth")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_real_vs_generated(
    real_returns: np.ndarray,
    real_regimes: np.ndarray,
    generated_returns: np.ndarray,
    generated_regimes: np.ndarray,
    figsize: tuple[int, int] = (20, 5),
    ylim: tuple[float, float] = (-8, 6),
    generated_title: str = "HMM Diffusion Generated Returns",
) -> None:
    """The paired comparison figure, replacing ``hmmgan1.utils.hmm_gan_plot``.

    Left panel is the observed series with its true regime path; right panel is the stitched
    synthetic series with the HMM's estimated regime path. Shared y-limits make the two panels
    directly comparable, which is the whole point of the figure.

    ``generated_title`` must state what actually produced the right panel. When the specialists have
    not been trained the pools hold resampled real returns, and a panel labelled as diffusion output
    would be simply wrong.
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(real_returns)
    plt.plot(real_regimes, color="black", label="regime")
    plt.title("Real Returns")
    plt.ylim(*ylim)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(generated_returns)
    plt.plot(generated_regimes, color="black", label="regime")
    plt.title(generated_title)
    plt.ylim(*ylim)
    plt.legend()

    plt.show()


def plot_elbo(losses: list[float], figsize: tuple[int, int] = (10, 4)) -> None:
    """SVI convergence for the neural HMM, which has no MCMC summary to inspect instead."""
    plt.figure(figsize=figsize)
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss per observation")
    plt.title("Neural HMM Training")
    plt.show()


def regime_moments(series: np.ndarray, labels: np.ndarray, n_regimes: int) -> pd.DataFrame:
    """Per-regime mean and variance of the observed emissions (notebook cell 18)."""
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
