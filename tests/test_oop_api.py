"""Tests for the OOP class APIs (no Vol_Regime, no diffusion training)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hmmdiff.constants import MIX_GRID, N_ASSETS, SEQ_LEN, VOL_MIX_PCT_MAX
from hmmdiff.data_collection.data_processor import PriceReturnProcessor, Splits
from hmmdiff.data_collection.equal_weight_processor import EqualWeightProcessor
from hmmdiff.data_collection.window_processor import WindowProcessor
from hmmdiff.model_design.base_model import (
    BaseDiffusionModel,
    BaseHiddenMarkovModel,
)
from hmmdiff.model_design.hmm_models import NeuralHmm, SupervisedHmm
from hmmdiff.model_design.path_stitcher import PathStitcher
from hmmdiff.testing_analytics.portfolio_backtest import PortfolioBacktest


class ConstantsTests(unittest.TestCase):
    def test_mix_grid_excludes_one_hundred(self):
        self.assertEqual(VOL_MIX_PCT_MAX, 90)
        self.assertEqual(MIX_GRID[-1], 90)
        self.assertNotIn(100, MIX_GRID)

    def test_window_shape_constants(self):
        self.assertEqual(SEQ_LEN, 128)
        self.assertEqual(N_ASSETS, 10)


class PriceReturnProcessorTests(unittest.TestCase):
    def test_fraction_splits(self):
        cfg = {"data": {"train_fraction": 0.75, "train_val_fraction": 1.0}}
        splits = PriceReturnProcessor(cfg).ComputeSplits(100, cfg)
        self.assertIsInstance(splits, Splits)
        self.assertEqual(splits.n_train, 75)
        self.assertEqual(splits.n_test, 25)

    def test_split_counts_requires_train_prefix(self):
        frame = pd.DataFrame(
            {
                "date": ["2001-01-01", "2001-01-02", "2001-01-03"],
                "split": ["train", "train", "test"],
            }
        )
        counts = PriceReturnProcessor().SplitCounts(frame)
        self.assertEqual(counts["n_train"], 2)
        self.assertEqual(counts["n_test"], 1)


class EqualWeightProcessorTests(unittest.TestCase):
    def test_equal_weight_is_row_mean(self):
        z_panel = pd.DataFrame({"a": [1.0, 3.0], "b": [3.0, 1.0]})
        series = EqualWeightProcessor().EqualWeightReturn(z_panel)
        np.testing.assert_allclose(series.to_numpy(), [2.0, 2.0])


class WindowProcessorTests(unittest.TestCase):
    def test_contiguous_runs(self):
        labels = np.array([0, 0, 1, 1, 1, 0])
        runs = WindowProcessor().ContiguousRuns(labels)
        self.assertEqual(runs, [(0, 2, 0), (2, 5, 1), (5, 6, 0)])


class PathStitcherTests(unittest.TestCase):
    def test_stitch_consumes_pools_in_order(self):
        generated = {"0": np.array([10.0, 11.0]), "1": np.array([20.0])}
        path = np.array([0, 1, 0])
        out = PathStitcher().Stitch(generated, path)
        np.testing.assert_allclose(out, [10.0, 20.0, 11.0])


class InheritanceTests(unittest.TestCase):
    def test_hmm_and_diffusion_share_base(self):
        self.assertTrue(issubclass(SupervisedHmm, BaseHiddenMarkovModel))
        self.assertTrue(issubclass(NeuralHmm, BaseHiddenMarkovModel))
        self.assertTrue(issubclass(BaseHiddenMarkovModel, object))
        self.assertTrue(issubclass(BaseDiffusionModel, object))


class PortfolioBacktestTests(unittest.TestCase):
    def test_window_vol_bucket(self):
        labels = np.array([0, 0, 0, 0])
        self.assertEqual(PortfolioBacktest().WindowVolBucket(labels), "low vol")
        labels = np.array([3, 3, 4, 4])
        self.assertEqual(PortfolioBacktest().WindowVolBucket(labels), "high vol")


if __name__ == "__main__":
    unittest.main()
