"""Downstream volatility forecasting."""

from .bridge import (
    AlignedDownstreamResult,
    build_aligned_downstream_synth,
    build_synthetic_price_frame,
    load_real_benchmark,
)
from .data_utils import compute_returns, load_real_data
from .prediction import (
    train_model_and_evaluate,
    train_model_and_evaluate_advanced_volatility,
    train_model_and_evaluate_advanced_volatility_mixture,
)

__all__ = [
    "AlignedDownstreamResult",
    "build_aligned_downstream_synth",
    "build_synthetic_price_frame",
    "compute_returns",
    "load_real_benchmark",
    "load_real_data",
    "train_model_and_evaluate",
    "train_model_and_evaluate_advanced_volatility",
    "train_model_and_evaluate_advanced_volatility_mixture",
]
