"""Stitch per-regime pools along a regime path.

PathStitcher draws one value per step from the specialist pool for that
step's regime. backtest_table compares real vs generated moments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stitch(
    generated_images: dict[str, np.ndarray], regime_path: np.ndarray
) -> np.ndarray:
    """Draw one value per step from the pool for that step's regime."""
    counters: dict[str, int] = {key: 0 for key in generated_images}
    output = np.empty(len(regime_path), dtype=float)

    for step, regime in enumerate(np.asarray(regime_path).astype(int)):
        key = str(int(regime))
        if key not in generated_images:
            raise KeyError(f"No pool for regime {key}; pools cover {sorted(generated_images)}")
        pool = generated_images[key]
        index = counters[key]
        if index >= pool.size:
            raise IndexError(
                f"Pool for regime {key} exhausted after {pool.size} draws at step {step}. "
                "Increase sampling.n_pool and rerun scripts/04_generate_pools.py."
            )
        output[step] = pool[index]
        counters[key] = index + 1

    return output


def backtest_table(
    real_emissions: np.ndarray,
    real_regimes: np.ndarray,
    generated: np.ndarray,
    estimated_regimes: np.ndarray,
    n_regimes: int,
) -> pd.DataFrame:
    """Per-regime real vs generated mean and variance."""
    frame = pd.DataFrame(
        {
            "regime": np.asarray(real_regimes).astype(int),
            "emission": np.asarray(real_emissions, dtype=float),
            "gen_regime": np.asarray(estimated_regimes).astype(int),
            "gen_emission": np.asarray(generated, dtype=float),
        }
    )
    rows = []
    for regime in range(n_regimes):
        subset = frame[frame.regime == regime]
        rows.append(
            {
                "regime": regime,
                "n": len(subset),
                "real_mean": subset.emission.mean(),
                "gen_mean": subset.gen_emission.mean(),
                "real_var": subset.emission.var(),
                "gen_var": subset.gen_emission.var(),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


class PathStitcher:
    """Concatenate specialist pool draws along an HMM-forecasted regime path."""

    def Stitch(
        self, generated_images: dict[str, np.ndarray], regime_path: np.ndarray
    ) -> np.ndarray:
        """
        Draw one value per step from the pool for that step's regime.

        Parameters:
        generated_images: dict
            Regime string to flat pool array.
        regime_path: numpy.ndarray
            Integer regime sequence.

        Return:
           numpy.ndarray of stitched emissions.
        """
        return stitch(generated_images, regime_path)

    def BacktestTable(
        self,
        real_emissions: np.ndarray,
        real_regimes: np.ndarray,
        generated: np.ndarray,
        estimated_regimes: np.ndarray,
        n_regimes: int,
    ) -> pd.DataFrame:
        """
        Per-regime real vs generated mean and variance.

        Parameters:
        real_emissions: numpy.ndarray
            Real HMM emissions.
        real_regimes: numpy.ndarray
            True regime labels.
        generated: numpy.ndarray
            Stitched synthetic emissions.
        estimated_regimes: numpy.ndarray
            Predicted regime path used for stitching.
        n_regimes: int
            Number of regimes.

        Return:
           pandas.DataFrame indexed by regime.
        """
        return backtest_table(
            real_emissions, real_regimes, generated, estimated_regimes, n_regimes
        )

