"""Assembling a synthetic return series from per-regime pools, and the paired comparison figure.

Reimplements ``hmmgan1.utils.hmm_gan_plot`` without its TensorFlow import (see
``third_party/VENDORED.md``). The stitching rule is unchanged: walk the regime path and, at each step,
take the next unused value from that regime's pool, tracked by a per-regime counter.

The counters are what tie the synthetic series to the HMM. Nothing about the pools is conditioned on
the regime path; the path only decides which pool each step is drawn from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stitch(
    generated_images: dict[str, np.ndarray], regime_path: np.ndarray
) -> np.ndarray:
    """Draw one value per step from the pool of the regime assigned to that step.

    ``regime_path`` is either the true regime labels (the oracle stitch of notebook cell 15) or the
    HMM's smoothed state estimates (cells 29, 33, and their counterparts in the other variants).

    Raises if a pool runs dry, which means the pools were sampled too small for this path.
    """
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
    """Per-regime real vs generated mean and variance (notebook cell 34).

    Rows are grouped by the *true* regime, while the generated values were drawn according to the
    HMM's *estimated* regime. The comparison therefore reflects both generator quality and state
    estimation accuracy, exactly as in the reference.
    """
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
