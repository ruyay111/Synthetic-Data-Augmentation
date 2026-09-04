"""Mean-variance backtest helpers for hmm-diffusion vs uncondi-diffusion.

PortfolioBacktest in testing_analytics is the class API. Functions here remain
the standalone snake_case entry points used by notebooks.
"""

from .backtest import run_mvo_backtest, window_vol_bucket
from .hmm_forecast import (
    filter_forward,
    forecast_regime_probability_path,
    onehot_average_argmax,
    walk_open_loop,
)
from .mixed_sample import load_uncond_windows
from .portfolio_core import (
    calmar_ratio,
    collapse_weights,
    mean_var_weights,
    mix_train_row_append,
    mix_train_with_regime_paths,
    project_sum_to_one_box,
    sharpe_ratio,
    summarize_portfolio,
)
from .scale_check import run_scale_check
from .specialist_sample import (
    load_specialist_pools,
    sample_simple_paths,
    sample_simple_paths_by_daily_regime,
    sample_simple_rows,
    synth_rows_for_share,
)

__all__ = [
    "calmar_ratio",
    "collapse_weights",
    "filter_forward",
    "forecast_regime_probability_path",
    "load_specialist_pools",
    "load_uncond_windows",
    "mean_var_weights",
    "mix_train_row_append",
    "mix_train_with_regime_paths",
    "onehot_average_argmax",
    "project_sum_to_one_box",
    "run_mvo_backtest",
    "window_vol_bucket",
    "run_scale_check",
    "sample_simple_paths",
    "sample_simple_paths_by_daily_regime",
    "sample_simple_rows",
    "sharpe_ratio",
    "summarize_portfolio",
    "synth_rows_for_share",
    "walk_open_loop",
]
