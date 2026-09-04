"""Utility API functions for volatility regime labeling and cached label I/O.

Implementation lives in RegimeProcessor. These snake_case functions re-export
the labeling helpers used by scripts and notebooks.
"""

from __future__ import annotations

from hmmdiff.data_collection.regime_processor import (
    RegimeLabels,
    average_regime_length,
    empirical_transition_matrix,
    ewm_volatility,
    fit_regimes,
    load_labels,
    outbound_transition_matrix,
    save_labels,
    switchpoints,
)

__all__ = [
    "RegimeLabels",
    "average_regime_length",
    "empirical_transition_matrix",
    "ewm_volatility",
    "fit_regimes",
    "load_labels",
    "outbound_transition_matrix",
    "save_labels",
    "switchpoints",
]
