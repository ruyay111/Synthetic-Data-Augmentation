"""Causal open-loop regime forecast and post-hold emission filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(pi: np.ndarray) -> np.ndarray:
    pi = np.asarray(pi, dtype=float).reshape(-1)
    total = float(pi.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(pi.shape, 1.0 / pi.size)
    return pi / total


def gaussian_likelihood(emission: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Likelihood of a scalar emission under each regime. ``sigma`` is std, as in hmmgan."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-8, None)
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    return coef * np.exp(-0.5 * ((float(emission) - mu) / sigma) ** 2)


def forecast_regime_probability_path(
    pi0: np.ndarray,
    transmat: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Return ``(H, K)`` with row h = pi0 @ P^{h+1}. No emission update."""
    current = _normalize(pi0)
    p = np.asarray(transmat, dtype=float)
    if current.shape[0] != p.shape[0] or p.shape[0] != p.shape[1]:
        raise ValueError("pi0 / transmat shape mismatch")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    path = np.zeros((horizon, current.shape[0]), dtype=float)
    for h in range(horizon):
        current = _normalize(current @ p)
        path[h] = current
    return path


def filter_forward(
    pi_prev: np.ndarray,
    emissions: np.ndarray,
    transmat: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Causal filter on ``emissions``, starting from filtered ``pi_prev`` (the day before).

    Row t of the result is π after observing ``emissions[t]``. Uses ``π @ P`` so the
    transition convention matches ``forecast_regime_probability_path``.
    """
    p = np.asarray(transmat, dtype=float)
    observations = np.asarray(emissions, dtype=float).reshape(-1)
    current = _normalize(pi_prev)
    out = np.zeros((len(observations), current.size), dtype=float)
    for t, emission in enumerate(observations):
        predicted = _normalize(current @ p)
        likelihood = gaussian_likelihood(emission, mu, sigma)
        updated = predicted * likelihood
        current = _normalize(updated)
        out[t] = current
    return out


def onehot_average_argmax(daily_states: np.ndarray, n_regimes: int) -> int:
    """Average one-hot daily labels, then argmax."""
    states = np.asarray(daily_states, dtype=int).reshape(-1)
    if states.size == 0:
        raise ValueError("daily_states is empty")
    counts = np.bincount(states, minlength=n_regimes).astype(float)
    return int(counts.argmax())


@dataclass
class OpenLoopWalk:
    """Open-loop path on the validation slice, plus post-hold filtered handoff."""

    open_loop_pi: np.ndarray
    daily_argmax: np.ndarray
    window_argmax: np.ndarray
    hold_starts: np.ndarray
    hold_k_star: np.ndarray  # window occupancy argmax; used by hmm-diffusion MVO
    pi_before_hold: np.ndarray
    filtered_end_pi: np.ndarray


def walk_open_loop(
    inner_emissions: np.ndarray,
    val_emissions: np.ndarray,
    transmat: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    init_dist: np.ndarray,
    horizon: int = 60,
    step: int = 60,
) -> OpenLoopWalk:
    """Filter inner train, then roll 60-day open-loop holds on validation.

    MVO must use ``open_loop_pi`` / ``hold_k_star`` from before the post-hold update.
    ``hold_k_star`` is the 60-day occupancy average of open-loop daily argmax labels.
    hmm-diffusion samples contiguous windows from that one specialist pool.
    """
    n_regimes = int(np.asarray(transmat).shape[0])
    val_emissions = np.asarray(val_emissions, dtype=float).reshape(-1)
    inner_emissions = np.asarray(inner_emissions, dtype=float).reshape(-1)

    pi = _normalize(init_dist)
    if inner_emissions.size:
        pi = filter_forward(pi, inner_emissions, transmat, mu, sigma)[-1]

    n_val = int(val_emissions.size)
    open_loop_pi = np.full((n_val, n_regimes), np.nan)
    daily_argmax = np.full(n_val, np.nan)
    window_argmax = np.full(n_val, np.nan)
    hold_starts: list[int] = []
    hold_k_star: list[int] = []
    pi_before: list[np.ndarray] = []
    filtered_end: list[np.ndarray] = []

    t = 0
    while t + horizon <= n_val:
        path = forecast_regime_probability_path(pi, transmat, horizon)
        daily = path.argmax(axis=1).astype(int)
        k_star = onehot_average_argmax(daily, n_regimes)
        open_loop_pi[t : t + horizon] = path
        daily_argmax[t : t + horizon] = daily
        window_argmax[t : t + horizon] = k_star
        hold_starts.append(t)
        hold_k_star.append(k_star)
        pi_before.append(pi.copy())
        filtered = filter_forward(pi, val_emissions[t : t + horizon], transmat, mu, sigma)
        pi = filtered[-1]
        filtered_end.append(pi.copy())
        t += step

    return OpenLoopWalk(
        open_loop_pi=open_loop_pi,
        daily_argmax=daily_argmax,
        window_argmax=window_argmax,
        hold_starts=np.asarray(hold_starts, dtype=int),
        hold_k_star=np.asarray(hold_k_star, dtype=int),
        pi_before_hold=np.stack(pi_before, axis=0) if pi_before else np.zeros((0, n_regimes)),
        filtered_end_pi=np.stack(filtered_end, axis=0) if filtered_end else np.zeros((0, n_regimes)),
    )
