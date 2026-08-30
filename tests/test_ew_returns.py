"""Unit tests for the equal-weight return construction (no Vol_Regime)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hmmdiff.config import load_config
from hmmdiff.data import build_model_frames, build_returns, compute_splits, train_returns
from hmmdiff.ew import (
    align_ew_diffusion_panel,
    build_ew_returns,
    build_train_test_frames,
    emission_from_log_panel,
    ew_scale,
    split_counts,
)


class EWReturnsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("configs/ew.yaml")
        cls.returns = build_ew_returns(cls.cfg)

    def test_overlap_calendar_and_split(self):
        actual = split_counts(self.returns)
        expected = self.cfg["reference"]
        for key in expected:
            self.assertEqual(actual[key], expected[key], key)
        self.assertFalse("n_val" in actual)
        self.assertTrue((self.returns["split"].iloc[: actual["n_train"]] == "train").all())
        self.assertTrue((self.returns["split"].iloc[actual["n_train"] :] == "test").all())

    def test_z_return_is_standardized_equal_weight(self):
        z = self.returns["z_return"].to_numpy(dtype=float)
        ew = self.returns["ew_return"]
        np.testing.assert_allclose(z.mean(), 0.0, atol=1e-12)
        reconstructed = ((ew - ew.mean()) / ew.std()).to_numpy(dtype=float)
        np.testing.assert_allclose(z, reconstructed)
        self.assertFalse(np.isnan(z).any())

    def test_ew_is_mean_of_ten_asset_z_scores(self):
        from hmmdiff.ew import equal_weight_return, standardized_asset_returns

        z_panel = standardized_asset_returns(self.cfg)
        self.assertEqual(list(z_panel.columns), self.cfg["data"]["asset_columns"])
        np.testing.assert_allclose(
            self.returns["ew_return"].to_numpy(),
            equal_weight_return(z_panel).to_numpy(),
        )

    def test_diffusion_windows_use_full_sample_hmm_uses_train(self):
        counts = split_counts(self.returns)
        dummy_labels = np.arange(len(self.returns), dtype=int) % 5
        panel, labels, dates, ew_series = align_ew_diffusion_panel(
            self.returns, dummy_labels, self.cfg
        )
        self.assertEqual(len(panel), counts["n_returns"])
        self.assertEqual(len(labels), counts["n_returns"])
        self.assertEqual(len(ew_series), counts["n_returns"])
        self.assertEqual(str(dates[0].date()), counts["start_date"])
        self.assertEqual(str(dates[-1].date()), counts["end_date"])
        self.assertGreater(str(dates[-1].date()), counts["train_end_date"])

        train_data, test_data = build_train_test_frames(self.returns, dummy_labels)
        self.assertEqual(len(train_data), counts["n_train"])
        self.assertEqual(len(test_data), counts["n_test"])
        self.assertEqual(len(train_data) + len(test_data), len(panel))

    def test_train_test_frames_have_no_validation_slice(self):
        n_train = int((self.returns["split"] == "train").sum())
        dummy_labels = np.zeros(len(self.returns), dtype=int)
        dummy_labels[n_train // 2] = 1
        train_data, test_data = build_train_test_frames(self.returns, dummy_labels)
        self.assertEqual(len(train_data), n_train)
        self.assertEqual(len(test_data), len(self.returns) - n_train)
        np.testing.assert_allclose(
            train_data.emission.to_numpy(),
            self.returns.loc[self.returns["split"] == "train", "z_return"].to_numpy(),
        )

    def test_emission_roundtrip_from_log_panel(self):
        from hmmdiff.data import build_multivariate_log_returns

        panel = build_multivariate_log_returns(self.cfg).to_numpy(dtype=float)
        scale = ew_scale(self.cfg)
        emission = emission_from_log_panel(panel, scale)
        np.testing.assert_allclose(emission, self.returns["z_return"].to_numpy())

    def test_a001_pipeline_helpers_are_unchanged(self):
        a001_cfg = load_config()
        returns = build_returns(a001_cfg)
        splits = compute_splits(len(returns), a001_cfg)
        self.assertEqual(splits.n_train, a001_cfg["reference"]["n_train"])
        self.assertEqual(splits.n_train_inner, a001_cfg["reference"]["n_train_inner"])
        self.assertEqual(len(train_returns(returns)), splits.n_train)
        dummy = np.zeros(splits.n_train, dtype=int)
        train_data, val_data = build_model_frames(returns, dummy, a001_cfg)
        self.assertEqual(len(train_data), splits.n_train_inner)
        self.assertEqual(len(val_data), splits.n_val)


if __name__ == "__main__":
    unittest.main()
