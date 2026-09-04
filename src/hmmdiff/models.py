"""Utility API functions for HMM variants and shared evaluation helpers.

Implementation of the class hierarchy lives in model_design.hmm_models.
These snake_case functions remain the standalone entry points.
"""

from __future__ import annotations

from hmmdiff.model_design.hmm_models import (
    FITTERS,
    HMMFit,
    accuracy_report,
    estimate_states,
    filter_states,
    fit_markov_switching,
    fit_neural_hmm,
    fit_semi_supervised_hmm,
    fit_supervised_hmm,
    initial_distribution,
    simulate_regime_path,
)

__all__ = [
    "FITTERS",
    "HMMFit",
    "accuracy_report",
    "estimate_states",
    "filter_states",
    "fit_markov_switching",
    "fit_neural_hmm",
    "fit_semi_supervised_hmm",
    "fit_supervised_hmm",
    "initial_distribution",
    "simulate_regime_path",
]
