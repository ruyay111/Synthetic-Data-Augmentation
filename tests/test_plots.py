"""MVO high/low vol panels share a y-axis; low vol omits tick labels."""

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

from hmmdiff.plots import (
    VOL_HMM_COLOR,
    VOL_UNCOND_COLOR,
    VOL_MIX_PCT_MAX,
    _vol_compare_ylim,
    add_mult_to_synth_pct,
    export_mvo_plots,
    mvo_eps_path,
    plot_mvo_boxes,
    plot_mvo_means,
    plot_vol_hmm_vs_uncond,
    plot_vol_mixture_compare,
    plot_vol_mixture_column,
    plot_vol_return_mixture_grid,
)


def _facet_axes():
    import matplotlib.pyplot as plt

    return [ax for ax in plt.gcf().axes if "bucket =" in ax.get_title()]


def _windows(high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    pcts = np.arange(0, 100, 10)
    rows = []
    for pct, h, lo in zip(pcts, high, low):
        for method in ("hmm-diffusion", "uncondi-diffusion"):
            rows.append(
                {
                    "method": method,
                    "n_synth": pct,
                    "synth_pct": pct,
                    "bucket": "high vol",
                    "sharpe": float(h),
                    "calmar": float(h) * 0.8,
                    "return": float(h) * 0.1,
                    "variance": float(abs(h) + 0.05),
                }
            )
            rows.append(
                {
                    "method": method,
                    "n_synth": pct,
                    "synth_pct": pct,
                    "bucket": "low vol",
                    "sharpe": float(lo),
                    "calmar": float(lo) * 0.8,
                    "return": float(lo) * 0.1,
                    "variance": float(abs(lo) + 0.05),
                }
            )
    return pd.DataFrame(rows)


@unittest.skipUnless(seaborn is not None, "seaborn is required for MVO plots")
class SharedScaleTests(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_mean_plot_shares_ylim_and_hides_low_vol_yticks(self, _show):
        import matplotlib.pyplot as plt

        high = np.linspace(-0.4, 2.1, 10)
        low = np.linspace(2.0, 3.7, 10)
        plot_mvo_means(_windows(high, low), "sharpe")
        axes = _facet_axes()
        self.assertGreaterEqual(len(axes), 2)
        np.testing.assert_allclose(axes[0].get_ylim(), axes[1].get_ylim())
        self.assertTrue(any(t.get_visible() and t.get_text() for t in axes[0].get_yticklabels()))
        self.assertFalse(any(t.get_visible() and t.get_text() for t in axes[1].get_yticklabels()))
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_box_plot_shares_ylim_and_hides_low_vol_yticks(self, _show):
        import matplotlib.pyplot as plt

        high = np.linspace(-2.0, 0.0, 10)
        low = np.linspace(1.0, 4.0, 10)
        plot_mvo_boxes(_windows(high, low), "sharpe")
        axes = _facet_axes()
        self.assertGreaterEqual(len(axes), 2)
        np.testing.assert_allclose(axes[0].get_ylim(), axes[1].get_ylim())
        self.assertTrue(any(t.get_visible() and t.get_text() for t in axes[0].get_yticklabels()))
        self.assertFalse(any(t.get_visible() and t.get_text() for t in axes[1].get_yticklabels()))
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_return_and_variance_plots_run(self, _show):
        import matplotlib.pyplot as plt

        high = np.linspace(0.05, 0.20, 10)
        low = np.linspace(0.02, 0.08, 10)
        for metric in ("return", "variance"):
            plot_mvo_boxes(_windows(high, low), metric)
            plot_mvo_means(_windows(high, low), metric)
            axes = _facet_axes()
            self.assertGreaterEqual(len(axes), 2)
            plt.close("all")

    def test_mvo_eps_path_uses_metric_subdir(self):
        path = mvo_eps_path(
            "/tmp/plots/ew",
            "sharpe",
            "boxes",
            objective="mean-variance",
            constraint="long-only 30% cap",
        )
        self.assertEqual(
            path.as_posix(),
            "/tmp/plots/ew/sharpe/boxes_mean-variance_long-only-30pct-cap.eps",
        )

    @patch("matplotlib.pyplot.show")
    def test_export_mvo_plots_writes_eps(self, _show):
        import tempfile

        import matplotlib.pyplot as plt

        high = np.linspace(-0.4, 2.1, 10)
        low = np.linspace(2.0, 3.7, 10)
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_mvo_plots(
                _windows(high, low),
                tmp,
                objective="mean-variance",
                constraint="unconstrained",
            )
            self.assertEqual(len(paths), 8)
            for path in paths:
                self.assertTrue(path.exists(), msg=str(path))
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.suffix, ".eps")
            metrics = {p.parent.name for p in paths}
            self.assertEqual(metrics, {"sharpe", "calmar", "return", "variance"})
        plt.close("all")


def _vol_compare_frame() -> pd.DataFrame:
    rows = []
    add_grid = (0.0, 0.25, 0.5, 1.0, 2.0)
    for horizon, hmm, uncond, persist, har in (
        (1, 0.98, 0.97, 0.984, 0.985),
        (10, 0.62, -3.8, 0.668, 0.727),
        (21, -0.2, -15.0, 0.206, 0.319),
    ):
        for i, add_mult in enumerate(add_grid):
            frac = i / (len(add_grid) - 1)
            rows.append(
                {
                    "horizon": horizon,
                    "add_mult": add_mult,
                    "persist": persist,
                    "har": har,
                    "hmm_r2_mean": hmm - 0.05 * frac,
                    "hmm_r2_std": 0.02,
                    "uncond_r2_mean": uncond * frac if horizon > 1 else uncond,
                    "uncond_r2_std": 0.05 if horizon == 1 else 0.4 * frac + 0.05,
                }
            )
    return pd.DataFrame(rows)


class VolComparePlotTests(unittest.TestCase):
    def test_add_mult_to_synth_pct(self):
        np.testing.assert_allclose(add_mult_to_synth_pct(0.0), 0.0)
        np.testing.assert_allclose(add_mult_to_synth_pct(0.25), 20.0)
        np.testing.assert_allclose(add_mult_to_synth_pct(1.0), 50.0)
        np.testing.assert_allclose(add_mult_to_synth_pct(2.0), 100.0 * 2.0 / 3.0)

    def test_ylim_does_not_follow_uncond_crash(self):
        block = _vol_compare_frame()
        block = block.loc[block["horizon"] == 21]
        lo, hi = _vol_compare_ylim(block)
        self.assertGreater(lo, -3.0)
        self.assertGreaterEqual(hi, 0.319)
        self.assertLess(hi, 1.0)

    @patch("matplotlib.pyplot.show")
    def test_vol_compare_labels_colors_and_scale(self, _show):
        import matplotlib.pyplot as plt

        plot_vol_hmm_vs_uncond(_vol_compare_frame())
        axes = plt.gcf().axes
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0].get_xlabel(), "% of synth")
        self.assertEqual(axes[0].get_ylabel(), "Test $R^2$")
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        self.assertEqual(
            labels,
            ["HMM-Diffusion", "Unconditional Diffusion", "persist", "HAR"],
        )
        colors = [line.get_color() for line in axes[0].lines if line.get_marker() in ("o", "s")]
        self.assertIn(VOL_HMM_COLOR, colors)
        self.assertIn(VOL_UNCOND_COLOR, colors)
        self.assertGreater(axes[2].get_ylim()[0], -3.0)
        self.assertGreater(axes[0].get_ylim()[0], 0.8)
        plt.close("all")


def _mixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    pcts = list(range(0, 101, 10))
    hmm_rows = []
    uncond_rows = []
    for horizon, hmm0, uncond0 in ((1, 0.97, 0.96), (10, 0.30, -2.0), (20, -1.2, -8.0)):
        for pct in pcts:
            hmm_rows.append(
                {"horizon": horizon, "synthetic_pct": pct, "r2": hmm0 - 0.01 * (pct / 10)}
            )
            crash = 0.0 if horizon == 1 else -40.0 * max(pct - 80, 0) / 20.0
            uncond_rows.append(
                {
                    "horizon": horizon,
                    "synthetic_pct": pct,
                    "r2": uncond0 - 0.02 * (pct / 10) + crash,
                }
            )
    return pd.DataFrame(hmm_rows), pd.DataFrame(uncond_rows)


class VolMixtureCompareTests(unittest.TestCase):
    @patch("matplotlib.pyplot.show")
    def test_drops_100_and_keeps_90_on_same_ylim(self, _show):
        import matplotlib.pyplot as plt

        hmm, uncond = _mixture_frames()
        plot_vol_mixture_compare(hmm, uncond, horizons=(1, 10, 20))
        axes = plt.gcf().axes
        self.assertEqual(len(axes), 3)
        self.assertEqual(axes[0].get_xlabel(), "% of synth")
        self.assertEqual(axes[0].get_ylabel(), "Test $R^2$")
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        self.assertEqual(labels, ["hmm-diffusion", "uncondi-diffusion"])
        colors = [line.get_color() for line in axes[0].lines]
        self.assertIn(VOL_HMM_COLOR, colors)
        self.assertIn(VOL_UNCOND_COLOR, colors)
        for ax in axes:
            xs = np.concatenate([np.asarray(line.get_xdata(), dtype=float) for line in ax.lines])
            self.assertLessEqual(xs.max(), VOL_MIX_PCT_MAX)
            self.assertIn(90.0, xs)
            self.assertNotIn(100.0, xs)
        h1, h10 = axes[0].get_ylim(), axes[1].get_ylim()
        self.assertGreater(axes[0].get_ylim()[0], 0.5)
        self.assertLessEqual(axes[0].get_ylim()[1], 1.0)
        self.assertLess(h10[0], h1[0])
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_single_horizon_uses_uncondi_label(self, _show):
        import matplotlib.pyplot as plt

        hmm, uncond = _mixture_frames()
        plot_vol_mixture_compare(
            hmm,
            uncond,
            horizons=(1,),
            title="hmm-diffusion vs uncondi-diffusion (return)",
        )
        axes = plt.gcf().axes
        self.assertEqual(len(axes), 1)
        self.assertEqual(axes[0].get_title(), "h = 1")
        labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        self.assertEqual(labels, ["hmm-diffusion", "uncondi-diffusion"])
        self.assertEqual(
            plt.gcf()._suptitle.get_text(),
            "hmm-diffusion vs uncondi-diffusion (return)",
        )
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_vol_return_grid_is_rows_by_task(self, _show):
        import tempfile

        import matplotlib.pyplot as plt

        vol_hmm, vol_uncond = _mixture_frames()
        ret_hmm, ret_uncond = _mixture_frames()
        ret_hmm = ret_hmm.copy()
        ret_uncond = ret_uncond.copy()
        ret_hmm["r2"] = -0.04
        ret_uncond["r2"] = -0.06
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "vol_return_mixture_grid.eps"
            plot_vol_return_mixture_grid(
                vol_hmm,
                vol_uncond,
                ret_hmm,
                ret_uncond,
                horizons=(1, 10, 20),
                save_path=dest,
            )
            self.assertTrue(dest.exists())
            self.assertGreater(dest.stat().st_size, 0)
            self.assertEqual(dest.suffix, ".eps")
        axes = [ax for ax in plt.gcf().axes if ax.lines]
        self.assertEqual(len(axes), 6)
        titles = " ".join(ax.get_title() for ax in plt.gcf().axes)
        self.assertIn("Volatility", titles)
        self.assertIn("Return", titles)
        self.assertIn("% of synth", [ax.get_xlabel() for ax in axes])
        self.assertTrue(any("R^2" in (ax.get_ylabel() or "") for ax in axes))
        fig = plt.gcf()
        leg = fig.legends[0] if fig.legends else axes[0].get_legend()
        labels = [t.get_text() for t in leg.get_texts()]
        self.assertEqual(labels, ["hmm-diffusion", "uncondi-diffusion"])
        self.assertGreater(axes[0].get_ylim()[0], 0.5)
        self.assertLessEqual(axes[0].get_ylim()[1], 1.0)
        self.assertLess(axes[1].get_ylim()[1], 0.5)
        plt.close("all")

    @patch("matplotlib.pyplot.show")
    def test_vol_column_matches_grid_width(self, _show):
        import tempfile

        import matplotlib.pyplot as plt

        vol_hmm, vol_uncond = _mixture_frames()
        ret_hmm, ret_uncond = _mixture_frames()
        plot_vol_return_mixture_grid(
            vol_hmm, vol_uncond, ret_hmm, ret_uncond, horizons=(1, 10, 20)
        )
        width_2col = float(plt.gcf().get_size_inches()[0])
        plt.close("all")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "vol_mixture_grid.eps"
            plot_vol_mixture_column(
                vol_hmm,
                vol_uncond,
                horizons=(1, 10, 20),
                save_path=dest,
            )
            self.assertTrue(dest.exists())
            self.assertGreater(dest.stat().st_size, 0)
        axes = [ax for ax in plt.gcf().axes if ax.lines]
        self.assertEqual(len(axes), 3)
        width_1col = float(plt.gcf().get_size_inches()[0])
        self.assertAlmostEqual(width_1col, width_2col, delta=0.75)
        self.assertLessEqual(axes[0].get_ylim()[1], 1.0)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
