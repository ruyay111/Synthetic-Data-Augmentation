"""Unit tests for downstream synth price helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hmmdiff.downstream.bridge import build_uncond_synth_price_frame
from hmmdiff.downstream.prediction import (
    FEATURE_COLS,
    RETURN_FEATURE_COLS,
    RETURN_TARGET_COL,
    VOL_FEATURE_COLS,
    VOL_LAG_FEATURE_COLS,
    VOL_TARGET_COL,
    build_advanced_return_features,
    build_advanced_vol_features,
    build_vol_lag_features,
    train_model_and_evaluate_advanced_return_mixture,
    train_vol_augmentation,
)
from hmmdiff.models import HMMFit, filter_states, simulate_regime_path

try:
    import ta  # noqa: F401
except ImportError:
    ta = None


class UncondSynthPriceTests(unittest.TestCase):
    def test_separates_windows_with_nan(self):
        windows = np.zeros((2, 8, 10))
        windows[0, :, 0] = 0.01
        windows[1, :, 0] = -0.01
        frame = build_uncond_synth_price_frame(windows, start_price=100.0)
        self.assertEqual(len(frame), 17)
        self.assertEqual(int(frame["A001"].isna().sum()), 1)
        self.assertGreater(float(frame["A001"].iloc[7]), 100.0)
        self.assertLess(float(frame["A001"].iloc[-1]), 100.0)

    def test_concat_without_gap(self):
        windows = np.zeros((2, 4, 10))
        windows[:, :, 0] = 0.001
        frame = build_uncond_synth_price_frame(
            windows, start_price=100.0, separate_windows=False
        )
        self.assertEqual(len(frame), 8)
        self.assertFalse(bool(frame["A001"].isna().any()))


def _log_price_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=n)
    log_price = np.log(100.0) + np.cumsum(returns)
    index = pd.bdate_range("2014-01-06", periods=n)
    return pd.DataFrame({"A001_log": log_price}, index=index)


class VolLagAugmentationTests(unittest.TestCase):
    def test_vol_lag_features_have_no_price_ta(self):
        feat = build_vol_lag_features(_log_price_frame(250, 0), "A001_log", horizon=21)
        self.assertEqual(list(VOL_LAG_FEATURE_COLS), ["RV_21", "RV_21_lag5", "RV_21_lag21", "AbsRet_21"])
        for col in ("MA_21", "RSI", "MACD"):
            self.assertNotIn(col, feat.columns)
        self.assertGreater(len(feat), 50)

    def test_augmentation_keeps_all_real_and_adds(self):
        real = _log_price_frame(300, 0)
        synth = _log_price_frame(300, 1)
        metrics = train_vol_augmentation(
            real,
            synth,
            price_col="A001_log",
            horizon=10,
            add_grid=(0.0, 1.0),
            n_seeds=2,
            n_estimators=8,
            plot_summary=False,
            random_state=0,
        )
        rf = metrics.loc[metrics["model"] == "rf"]
        n_real = int(rf["n_real_train"].iloc[0])
        self.assertTrue((rf["n_real_train"] == n_real).all())
        self.assertTrue((rf.loc[rf["add_mult"] == 0.0, "n_synth_added"] == 0).all())
        self.assertTrue((rf.loc[rf["add_mult"] == 1.0, "n_synth_added"] == n_real).all())
        self.assertIn("persist", set(metrics["model"]))
        self.assertIn("har", set(metrics["model"]))


@unittest.skipUnless(ta is not None, "ta is required for return mixture features")
class ReturnMixtureTests(unittest.TestCase):
    def test_return_target_is_forward_simple_return(self):
        log_px = _log_price_frame(250, 0)
        prices = pd.DataFrame({"A001": np.exp(log_px["A001_log"].to_numpy())}, index=log_px.index)
        feat = build_advanced_return_features(prices, "A001", horizon=1)
        self.assertEqual(list(FEATURE_COLS), list(VOL_FEATURE_COLS))
        self.assertEqual(list(FEATURE_COLS), list(RETURN_FEATURE_COLS))
        self.assertIn(RETURN_TARGET_COL, feat.columns)
        for col in FEATURE_COLS:
            self.assertIn(col, feat.columns)
        self.assertNotIn("MA_21", FEATURE_COLS)
        self.assertIn("Vol_63", FEATURE_COLS)
        vol_feat = build_advanced_vol_features(prices, "A001", horizon=1)
        for col in FEATURE_COLS:
            self.assertIn(col, vol_feat.columns)
        self.assertIn(VOL_TARGET_COL, vol_feat.columns)
        t = feat.index[50]
        nxt = prices.index[prices.index.get_loc(t) + 1]
        expected = float(prices.loc[nxt, "A001"] / prices.loc[t, "A001"] - 1.0)
        self.assertAlmostEqual(float(feat.loc[t, RETURN_TARGET_COL]), expected, places=10)

    def test_return_mixture_runs_and_drops_empty_synth(self):
        log_px = _log_price_frame(280, 0)
        prices = pd.DataFrame({"A001": np.exp(log_px["A001_log"].to_numpy())}, index=log_px.index)
        synth = pd.DataFrame(
            {"A001": np.exp(_log_price_frame(280, 1)["A001_log"].to_numpy())},
            index=prices.index,
        )
        metrics = train_model_and_evaluate_advanced_return_mixture(
            prices,
            synth,
            price_col="A001",
            horizon=1,
            mix_grid=(0, 20),
            plot_summary=False,
            random_state=0,
        )
        self.assertEqual(set(metrics["synthetic_pct"]), {0, 20})
        self.assertTrue(np.isfinite(metrics["r2"]).all())


class FilterSimulateTests(unittest.TestCase):
    def test_filter_states_length_and_simulate_path(self):
        n_regimes = 3
        transmat = np.full((n_regimes, n_regimes), 0.1)
        np.fill_diagonal(transmat, 0.8)
        fit = HMMFit(
            name="toy",
            transmat=transmat,
            mu=np.array([-1.0, 0.0, 1.0]),
            sigma=np.array([0.5, 0.5, 0.5]),
        )
        init = np.array([0.2, 0.6, 0.2])
        obs = np.zeros(40)
        probs, states = filter_states(fit, obs, init)
        self.assertEqual(probs.shape, (40, 3))
        self.assertEqual(len(states), 40)
        rng = np.random.default_rng(1)
        path = simulate_regime_path(transmat, init, 40, rng)
        self.assertEqual(len(path), 40)
        self.assertTrue(np.all((path >= 0) & (path < 3)))


if __name__ == "__main__":
    unittest.main()
