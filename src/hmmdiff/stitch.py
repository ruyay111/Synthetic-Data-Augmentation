"""Utility API functions for stitching specialist pools along a regime path.

Implementation lives in PathStitcher.
"""

from __future__ import annotations

from hmmdiff.model_design.path_stitcher import backtest_table, stitch

__all__ = ["backtest_table", "stitch"]
