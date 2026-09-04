"""Volatility and return forecast evaluation with mixed real/synthetic rows.

ForecastEvaluator wraps the shared random-forest mix protocol used in
Downstream-Volatility.ipynb.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from hmmdiff.constants import RETURN_FEATURE_WINDOW
from hmmdiff.downstream.prediction import (
    train_model_and_evaluate,
    train_model_and_evaluate_advanced_return_mixture,
    train_model_and_evaluate_advanced_volatility_mixture,
)


class ForecastEvaluator:
    """Train forests on mixed rows and score R^2 on a held-out real test set."""

    def EvaluateElementary(
        self,
        real_data,
        synthetic_data=None,
        window: int = RETURN_FEATURE_WINDOW,
        target: str = "return",
    ) -> None:
        """
        Next-day return or volatility forecast with linear regression.

        Parameters:
        real_data: array-like
            Real return series.
        synthetic_data: array-like or None
            Optional synthetic series.
        window: int
            Lag window.
        target: str
            'return' or 'volatility'.

        Return:
           None (prints test MSE and R^2).
        """
        train_model_and_evaluate(
            real_data, synthetic_data=synthetic_data, window=window, target=target
        )

    def EvaluateVolatilityMixture(
        self, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Advanced volatility forest over a mix-share grid.

        Return:
           pandas.DataFrame of test R^2 by mix percent and horizon.
        """
        return train_model_and_evaluate_advanced_volatility_mixture(*args, **kwargs)

    def EvaluateReturnMixture(self, *args, **kwargs) -> Any:
        """
        Advanced return forest over a mix-share grid.

        Return:
           mix-result object from the return mixture trainer.
        """
        return train_model_and_evaluate_advanced_return_mixture(*args, **kwargs)
