"""Unit tests for downstream synth price helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hmmdiff.downstream.bridge import build_uncond_synth_price_frame


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


if __name__ == "__main__":
    unittest.main()
