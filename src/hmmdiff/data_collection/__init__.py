"""Data collection and cleaning.

PriceReturnProcessor loads the raw CSV and builds splits.
EqualWeightProcessor builds the ten-asset equal-weight series used by both generators.
RegimeProcessor assigns volatility-regime labels.
WindowProcessor cuts contiguous same-regime diffusion windows.
PoolProcessor loads and calibrates sampled specialist pools.
"""

from hmmdiff.data_collection.data_processor import PriceReturnProcessor, Splits
from hmmdiff.data_collection.equal_weight_processor import EqualWeightProcessor
from hmmdiff.data_collection.pool_processor import PoolProcessor
from hmmdiff.data_collection.regime_processor import RegimeLabels, RegimeProcessor
from hmmdiff.data_collection.window_processor import WindowProcessor

__all__ = [
    "EqualWeightProcessor",
    "PoolProcessor",
    "PriceReturnProcessor",
    "RegimeLabels",
    "RegimeProcessor",
    "Splits",
    "WindowProcessor",
]
