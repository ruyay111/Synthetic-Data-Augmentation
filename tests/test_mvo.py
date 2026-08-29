"""Unit tests for MVO helpers (no HMM MCMC)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hmmdiff.mvo.backtest import lookback_and_hold, window_vol_bucket
from hmmdiff.mvo.hmm_forecast import (
    filter_forward,
    forecast_regime_probability_path,
    onehot_average_argmax,
    walk_open_loop,
)
from hmmdiff.mvo.portfolio_core import (
    collapse_weights,
    greedy_max_return_box,
    mean_var_weights,
    mix_train_with_regime_paths,
    project_sum_to_one_box,
)
from hmmdiff.mvo.scale_check import run_scale_check
from hmmdiff.mvo.specialist_sample import sample_simple_paths, sample_simple_paths_by_daily_regime


class ForecastTests(unittest.TestCase):
    def test_open_loop_is_matrix_power(self):
        p = np.array([[0.7, 0.3], [0.2, 0.8]])
        pi = np.array([1.0, 0.0])
        path = forecast_regime_probability_path(pi, p, 3)
        cur = pi
        for h in range(3):
            cur = cur @ p
            np.testing.assert_allclose(path[h], cur)

    def test_filter_uses_emissions_open_loop_does_not(self):
        p = np.eye(2)
        mu = np.array([0.0, 5.0])
        sigma = np.array([0.5, 0.5])
        pi = np.array([0.5, 0.5])
        open_loop = forecast_regime_probability_path(pi, p, 2)
        filtered = filter_forward(pi, np.array([5.0, 5.0]), p, mu, sigma)
        np.testing.assert_allclose(open_loop[0], pi)
        self.assertGreater(filtered[-1, 1], filtered[-1, 0])
        self.assertGreater(filtered[-1, 1], open_loop[-1, 1])

    def test_walk_handoff_changes_next_start(self):
        rng = np.random.default_rng(0)
        p = np.array([[0.9, 0.1], [0.1, 0.9]])
        mu = np.array([-1.0, 1.0])
        sigma = np.array([0.3, 0.3])
        init = np.array([0.5, 0.5])
        inner = rng.normal(-1.0, 0.3, size=20)
        val = np.concatenate([rng.normal(1.0, 0.3, size=8), rng.normal(-1.0, 0.3, size=8)])
        walk = walk_open_loop(inner, val, p, mu, sigma, init, horizon=8, step=8)
        self.assertEqual(len(walk.hold_starts), 2)
        self.assertFalse(np.allclose(walk.pi_before_hold[1], walk.pi_before_hold[0]))
        # Open-loop on hold 0 must not depend on hold 0 emissions.
        path0 = forecast_regime_probability_path(walk.pi_before_hold[0], p, 8)
        np.testing.assert_allclose(walk.open_loop_pi[:8], path0)
        perturbed = val.copy()
        perturbed[:8] = 0.0
        walk2 = walk_open_loop(inner, perturbed, p, mu, sigma, init, horizon=8, step=8)
        np.testing.assert_allclose(walk2.open_loop_pi[:8], walk.open_loop_pi[:8])
        self.assertFalse(np.allclose(walk2.filtered_end_pi[0], walk.filtered_end_pi[0]))

    def test_hold_k_star_is_window_average_argmax(self):
        rng = np.random.default_rng(1)
        p = np.array([[0.6, 0.4], [0.3, 0.7]])
        mu = np.array([-1.0, 1.0])
        sigma = np.array([0.3, 0.3])
        init = np.array([0.9, 0.1])
        inner = rng.normal(-1.0, 0.3, size=12)
        val = rng.normal(0.0, 0.3, size=16)
        walk = walk_open_loop(inner, val, p, mu, sigma, init, horizon=8, step=8)
        for i, start in enumerate(walk.hold_starts.tolist()):
            occupancy = onehot_average_argmax(
                walk.daily_argmax[start : start + 8].astype(int), 2
            )
            self.assertEqual(int(walk.hold_k_star[i]), occupancy)
            np.testing.assert_array_equal(
                walk.window_argmax[start : start + 8], occupancy
            )

    def test_onehot_average_argmax(self):
        self.assertEqual(onehot_average_argmax(np.array([1, 1, 2, 2, 2]), 5), 2)


class MixTests(unittest.TestCase):
    def test_n0_unchanged(self):
        real = np.arange(12, dtype=float).reshape(6, 2)
        mixed, n = mix_train_with_regime_paths(real, None, mix_len=6)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(mixed, real)

    def test_column_stack_and_collapse(self):
        real = np.ones((6, 2))
        synth = np.full((1, 6, 2), 2.0)
        mixed, n = mix_train_with_regime_paths(real, synth, mix_len=6)
        self.assertEqual(n, 1)
        self.assertEqual(mixed.shape, (6, 4))
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        collapsed = collapse_weights(weights, n_assets=2, n_draw=1)
        np.testing.assert_allclose(collapsed, np.array([0.4, 0.6]))

    def test_mean_var_sum_to_one_shorts_allowed(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, size=(60, 10))
        w = mean_var_weights(r, allow_short=True)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_min_variance_and_max_sharpe_sum_to_one(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.001, 0.01, size=(80, 10))
        for objective in ("min_variance", "max_sharpe"):
            w = mean_var_weights(r, allow_short=True, objective=objective)
            self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_max_return_picks_highest_mean(self):
        r = np.zeros((20, 3))
        r[:, 1] = 0.02
        w = mean_var_weights(r, allow_short=False, objective="max_return")
        np.testing.assert_allclose(w, np.array([0.0, 1.0, 0.0]))

    def test_greedy_max_return_box(self):
        mu = np.array([0.01, 0.05, 0.03, 0.04, 0.00])
        w = greedy_max_return_box(mu, w_min=0.0, w_max=0.3)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        np.testing.assert_allclose(w, np.array([0.1, 0.3, 0.3, 0.3, 0.0]))

    def test_project_sum_to_one_box(self):
        w = project_sum_to_one_box(np.array([2.0, 0.0, -1.0, 0.0]), w_min=0.0, w_max=0.5)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertTrue(np.all(w >= -1e-12))
        self.assertTrue(np.all(w <= 0.5 + 1e-12))
        equal = project_sum_to_one_box(np.full(10, 0.1), w_min=0.0, w_max=0.3)
        np.testing.assert_allclose(equal, np.full(10, 0.1), atol=1e-10)

    def test_sample_log_to_simple(self):
        log = np.zeros((4, 8, 3))
        log[..., 0] = np.log(1.01)
        rng = np.random.default_rng(0)
        simple = sample_simple_paths(log, n_synth=2, horizon=5, rng=rng)
        self.assertEqual(simple.shape, (2, 5, 3))
        np.testing.assert_allclose(simple[..., 0], 0.01)

    def test_sample_by_daily_regime(self):
        pools = {
            k: np.full((4, 6, 2), np.log(1.0 + 0.01 * (k + 1))) for k in range(3)
        }
        daily = np.array([0, 2, 1, 2])
        rng = np.random.default_rng(0)
        simple = sample_simple_paths_by_daily_regime(pools, daily, n_synth=3, rng=rng)
        self.assertEqual(simple.shape, (3, 4, 2))
        np.testing.assert_allclose(simple[:, 0, :], 0.01)
        np.testing.assert_allclose(simple[:, 1, :], 0.03)
        np.testing.assert_allclose(simple[:, 2, :], 0.02)
        np.testing.assert_allclose(simple[:, 3, :], 0.03)
        empty = sample_simple_paths_by_daily_regime(pools, daily, n_synth=0, rng=rng)
        self.assertEqual(empty.shape, (0, 4, 2))


class BucketAndScaleTests(unittest.TestCase):
    def test_high_low_vol(self):
        self.assertEqual(window_vol_bucket(np.array([0, 1, 2, 1, 0, 2])), "low vol")
        self.assertEqual(window_vol_bucket(np.array([3, 3, 4, 2, 1, 0])), "high vol")
        self.assertEqual(window_vol_bucket(np.array([3, 3, 3, 0, 0, 0])), "high vol")

    def test_lookback_and_hold(self):
        idx = pd.date_range("2007-01-02", periods=200, freq="B")
        simple = pd.DataFrame(np.ones((200, 10)), index=idx)
        val_dates = idx[80:140].to_numpy()
        out = lookback_and_hold(simple, val_dates, 0, lookback=60, horizon=60)
        self.assertIsNotNone(out)
        lb, hold = out
        self.assertEqual(lb.shape, (60, 10))
        self.assertEqual(hold.shape, (60, 10))

    def test_scale_check_passes_log_returns(self):
        rng = np.random.default_rng(0)
        assets = [f"A{i:03d}" for i in range(10)]
        real = rng.normal(0.0002, 0.01, size=(200, 10))
        pools = {k: rng.normal(0.0002, 0.012, size=(8, 16, 10)) for k in range(2)}
        uncond = rng.normal(0.0003, 0.011, size=(8, 16, 10))
        result = run_scale_check(real, pools, uncond, assets)
        self.assertTrue(result.ok, msg=result.messages)

    def test_scale_check_flags_zscores(self):
        rng = np.random.default_rng(0)
        assets = [f"A{i:03d}" for i in range(10)]
        real = rng.normal(0.0002, 0.01, size=(200, 10))
        pools = {0: rng.normal(0.0, 1.0, size=(8, 16, 10))}
        uncond = rng.normal(0.0003, 0.011, size=(8, 16, 10))
        result = run_scale_check(real, pools, uncond, assets)
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
