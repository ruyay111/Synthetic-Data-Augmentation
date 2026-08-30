"""MVO plot scales must follow each bucket's own calculated values."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

try:
    import seaborn  # noqa: F401
except ImportError:
    seaborn = None

from hmmdiff.plots import _ylim_from_values, plot_mvo_boxes, plot_mvo_means


def _facet_axes():
    import matplotlib.pyplot as plt

    return [ax for ax in plt.gcf().axes if "bucket =" in ax.get_title()]


def _windows(high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    pcts = np.arange(0, 100, 10)
    rows = []
    for pct, h, lo in zip(pcts, high, low):
        for method in ("hmm-diffusion", "mixed"):
            rows.append(
                {
                    "method": method,
                    "n_synth": pct,
                    "synth_pct": pct,
                    "bucket": "high vol",
                    "sharpe": float(h),
                }
            )
            rows.append(
                {
                    "method": method,
                    "n_synth": pct,
                    "synth_pct": pct,
                    "bucket": "low vol",
                    "sharpe": float(lo),
                }
            )
    return pd.DataFrame(rows)


@unittest.skipUnless(seaborn is not None, "seaborn is required for MVO plots")
class PerBucketScaleTests(unittest.TestCase):
    def test_ylim_covers_values(self):
        lo, hi = _ylim_from_values(np.array([-1.79, -0.42]))
        self.assertLessEqual(lo, -1.79)
        self.assertGreaterEqual(hi, -0.42)
        self.assertLess(hi, 0.0)

    @patch("matplotlib.pyplot.show")
    def test_mean_plot_uses_separate_ylims(self, _show):
        import matplotlib.pyplot as plt

        high = np.linspace(-0.4, 2.1, 10)
        low = np.linspace(2.0, 3.7, 10)
        plot_mvo_means(_windows(high, low), "sharpe")
        axes = _facet_axes()
        self.assertGreaterEqual(len(axes), 2)
        high_lim = axes[0].get_ylim()
        low_lim = axes[1].get_ylim()
        self.assertLessEqual(high_lim[0], high.min())
        self.assertGreaterEqual(high_lim[1], high.max())
        self.assertLessEqual(low_lim[0], low.min())
        self.assertGreaterEqual(low_lim[1], low.max())
        self.assertNotAlmostEqual(high_lim[0], low_lim[0], places=2)
        self.assertNotAlmostEqual(high_lim[1], low_lim[1], places=2)
        plotted = []
        for ax in axes:
            ys = []
            for line in ax.lines:
                y = np.asarray(line.get_ydata(), dtype=float)
                if y.size:
                    ys.append(y)
            plotted.append(np.concatenate(ys) if ys else np.array([]))
        np.testing.assert_allclose(np.unique(np.round(plotted[0], 6)), np.unique(np.round(high, 6)))
        np.testing.assert_allclose(np.unique(np.round(plotted[1], 6)), np.unique(np.round(low, 6)))
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_box_plot_uses_separate_ylims(self, _show):
        import matplotlib.pyplot as plt

        high = np.linspace(-2.0, 0.0, 10)
        low = np.linspace(1.0, 4.0, 10)
        plot_mvo_boxes(_windows(high, low), "sharpe")
        axes = _facet_axes()
        self.assertGreaterEqual(len(axes), 2)
        self.assertLess(axes[0].get_ylim()[1], 0.6)
        self.assertGreater(axes[1].get_ylim()[0], 0.6)
        plt.close("all")
