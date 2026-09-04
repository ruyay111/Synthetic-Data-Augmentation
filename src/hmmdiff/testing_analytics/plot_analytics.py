"""Plotting APIs for regimes, MVO backtests, and mixture R^2 grids.

PlotAnalytics bundles the figure helpers in hmmdiff.plots. Method names are
PascalCase; the underlying standalone plot functions remain snake_case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hmmdiff import plots


class PlotAnalytics:
    """Result figures for the EW pipeline and downstream mix experiments."""

    def PlotPrices(self, close: np.ndarray, title: str = "S&P 500 TR Closing Prices") -> None:
        """
        Line plot of a closing-price series.

        Parameters:
        close: numpy.ndarray
            Price series.
        title: str
            Figure title.

        Return:
           None
        """
        plots.plot_prices(close, title=title)

    def PlotRegimeSpans(
        self,
        series: np.ndarray,
        changepoints: np.ndarray,
        segment_regimes: list[int],
        overlay: np.ndarray | None = None,
        figsize: tuple[int, int] = (15, 5),
        title: str = "Training Returns & Volatility Regimes",
    ) -> None:
        """
        Returns with regime-colored background spans.

        Parameters:
        series: numpy.ndarray
            Return series.
        changepoints: numpy.ndarray
            Segment boundaries.
        segment_regimes: list
            Regime index per segment.
        overlay: numpy.ndarray or None
            Optional second series.
        figsize: tuple
            Figure size.
        title: str
            Figure title.

        Return:
           None
        """
        plots.plot_regime_spans(
            series, changepoints, segment_regimes, overlay=overlay, figsize=figsize, title=title
        )

    def PlotMvoMeans(
        self,
        windows: pd.DataFrame,
        metric: str = "sharpe",
        **kwargs,
    ):
        """
        Mean MVO metric versus synthetic mix share.

        Parameters:
        windows: pandas.DataFrame
            Per-hold MVO rows from PortfolioBacktest.
        metric: str
            Column to plot (sharpe, calmar, return, variance).

        Return:
           matplotlib Axes or seaborn object from plot_mvo_means.
        """
        return plots.plot_mvo_means(windows, metric=metric, **kwargs)

    def PlotVolReturnMixtureGrid(
        self,
        vol_hmm: pd.DataFrame,
        vol_uncond: pd.DataFrame,
        ret_hmm: pd.DataFrame,
        ret_uncond: pd.DataFrame,
        **kwargs,
    ) -> None:
        """
        3x2 grid of volatility and return test R^2 versus mix share.

        Parameters:
        vol_hmm: pandas.DataFrame
            HMM-Diffusion volatility mix results.
        vol_uncond: pandas.DataFrame
            Unconditional diffusion volatility mix results.
        ret_hmm: pandas.DataFrame
            HMM-Diffusion return mix results.
        ret_uncond: pandas.DataFrame
            Unconditional diffusion return mix results.

        Return:
           None
        """
        plots.plot_vol_return_mixture_grid(
            vol_hmm, vol_uncond, ret_hmm, ret_uncond, **kwargs
        )

    def ExportMvoPlots(
        self,
        windows: pd.DataFrame,
        output_dir: Path | str,
        **kwargs,
    ) -> list[Path]:
        """
        Write MVO box and mean plots as EPS files.

        Parameters:
        windows: pandas.DataFrame
            Per-hold MVO rows.
        output_dir: pathlib.Path or str
            Output directory.

        Return:
           list of pathlib.Path written.
        """
        return plots.export_mvo_plots(windows, output_dir, **kwargs)
