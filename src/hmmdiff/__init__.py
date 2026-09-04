"""Synthetic data augmentation for portfolio optimization and financial forecasting.

Subpackages:
    data_collection  -- PriceReturnProcessor, EqualWeightProcessor, RegimeProcessor
    model_design     -- HMM and diffusion class hierarchy, PathStitcher
    testing_analytics -- PlotAnalytics, PortfolioBacktest, ForecastEvaluator
"""

from .config import bootstrap_imports, config_path, load_config, resolve, REPO_ROOT

__all__ = ["bootstrap_imports", "config_path", "load_config", "resolve", "REPO_ROOT"]
