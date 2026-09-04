"""Downstream volatility and return forecasting.

ForecastEvaluator in testing_analytics is the class API. Functions here remain
the standalone snake_case entry points used by notebooks.
"""

from .bridge import (
    AlignedDownstreamResult,
    build_aligned_downstream_synth,
    build_synthetic_price_frame,
    build_uncond_synth_price_frame,
    load_real_benchmark,
)
from .data_utils import compute_returns, load_real_data
from .prediction import (
    FEATURE_COLS,
    RETURN_FEATURE_COLS,
    RETURN_TARGET_COL,
    RF_KWARGS,
    VOL_FEATURE_COLS,
    VOL_LAG_FEATURE_COLS,
    VOL_TARGET_COL,
    build_advanced_return_features,
    build_advanced_vol_features,
    build_vol_lag_features,
    train_model_and_evaluate,
    train_model_and_evaluate_advanced_return_mixture,
    train_model_and_evaluate_advanced_volatility,
    train_model_and_evaluate_advanced_volatility_mixture,
    train_vol_augmentation,
)

__all__ = [
    "AlignedDownstreamResult",
    "FEATURE_COLS",
    "RETURN_FEATURE_COLS",
    "RETURN_TARGET_COL",
    "RF_KWARGS",
    "VOL_FEATURE_COLS",
    "VOL_LAG_FEATURE_COLS",
    "VOL_TARGET_COL",
    "build_advanced_return_features",
    "build_advanced_vol_features",
    "build_aligned_downstream_synth",
    "build_synthetic_price_frame",
    "build_uncond_synth_price_frame",
    "build_vol_lag_features",
    "compute_returns",
    "load_real_benchmark",
    "load_real_data",
    "train_model_and_evaluate",
    "train_model_and_evaluate_advanced_return_mixture",
    "train_model_and_evaluate_advanced_volatility",
    "train_model_and_evaluate_advanced_volatility_mixture",
    "train_vol_augmentation",
]
