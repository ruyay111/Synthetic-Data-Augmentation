"""The four HMM variants of the reference notebook, behind one interface.

Each variant estimates the same three things from the labeled training frame: a transition matrix and
per-regime Gaussian emission parameters. Once fitted they are interchangeable, because state
estimation always runs through the same ``forward_backward`` call.

"Supervised" here means the regime labels are observed during fitting, so this is not Baum-Welch.
The labels come from ``Vol_Regime``, which is a separate volatility-clustering pipeline, not from the
HMM itself.

The three NumPyro variants are fitted with NUTS and summarized by posterior means, matching the
reference. The neural variant is fitted with Pyro SVI and a delta guide, so its "posterior" is a point
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class HMMFit:
    """Fitted parameters, in the form ``forward_backward`` consumes."""

    name: str
    transmat: np.ndarray  # (K, K)
    mu: np.ndarray  # (K,)
    sigma: np.ndarray  # (K,)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"mean": self.mu, "sigma": self.sigma},
            index=pd.Index(range(len(self.mu)), name="regime"),
        )


def _mcmc_fit(model: Callable, name: str, cfg: dict[str, Any], *model_args) -> HMMFit:
    """Run NUTS on a NumPyro model and reduce the posterior to its means."""
    import jax.numpy as jnp  # noqa: F401  (imported for side effects / availability check)
    from jax import random
    from numpyro.infer import MCMC, NUTS

    hmm_cfg = cfg["hmm"]
    mcmc = MCMC(
        NUTS(model),
        num_warmup=hmm_cfg["num_warmup"],
        num_samples=hmm_cfg["num_samples"],
        num_chains=hmm_cfg["num_chains"],
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(hmm_cfg["rng_seed"]), *model_args)

    posterior = mcmc.get_samples()
    return HMMFit(
        name=name,
        transmat=np.asarray(posterior["probs_x"].mean(0)),
        mu=np.asarray(posterior["probs_mu"].mean(0)),
        sigma=np.asarray(posterior["probs_sigma"].mean(0)),
        diagnostics={"mcmc": mcmc},
    )


def fit_supervised_hmm(train_data: pd.DataFrame, n_regimes: int, cfg: dict[str, Any]) -> HMMFit:
    """Notebook cell 24. Both regimes and emissions are observed."""
    import jax.numpy as jnp
    from hmmgan.hmm import supervised_hmm

    data = jnp.array(train_data.values, dtype=jnp.float32)
    return _mcmc_fit(supervised_hmm, "Supervised HMM", cfg, data, n_regimes)


def fit_markov_switching(train_data: pd.DataFrame, n_regimes: int, cfg: dict[str, Any]) -> HMMFit:
    """Notebook cell 36. Emission mean is regressed on the previous emission.

    The first row is dropped because ``emission_lag`` is NaN there.
    """
    import jax.numpy as jnp
    from hmmgan.hmm import markov_switching_model

    data = jnp.array(train_data.values[1:], dtype=jnp.float32)
    return _mcmc_fit(markov_switching_model, "Markov Switching Model", cfg, data, n_regimes)


def fit_semi_supervised_hmm(
    train_data: pd.DataFrame, n_regimes: int, cfg: dict[str, Any]
) -> HMMFit:
    """Notebook cell 49. The first half keeps its labels, the second half is unlabeled.

    Emissions in the unlabeled half contribute through a forward-algorithm factor, so the transition
    matrix is informed by data whose states were never observed.
    """
    import jax.numpy as jnp
    from hmmgan.hmm import semi_supervised_hmm

    cut = int(cfg["hmm"]["semi_supervised_fraction"] * train_data.shape[0])
    supervised_data = jnp.array(train_data.values[:cut, 1], dtype=jnp.float32)
    supervised_labels = jnp.array(train_data.values[:cut, 0], dtype=jnp.int32)
    unsupervised_data = jnp.array(train_data.values[cut:, 1], dtype=jnp.float32)
    return _mcmc_fit(
        semi_supervised_hmm,
        "Semi-Supervised HMM",
        cfg,
        supervised_data,
        supervised_labels,
        unsupervised_data,
        n_regimes,
    )


def fit_neural_hmm(train_data: pd.DataFrame, n_regimes: int, cfg: dict[str, Any]) -> HMMFit:
    """Notebook cells 62-65. Emissions come from a small network instead of free parameters.

    Fitted with SVI under an ``AutoDelta`` guide restricted to the ``probs_`` sites, so the transition
    matrix is a MAP point estimate and the emission parameters are read straight off the network.
    """
    import pyro
    import torch
    from pyro import poutine
    from pyro.infer import SVI, TraceEnum_ELBO
    from pyro.infer.autoguide import AutoDelta
    from pyro.optim import Adam

    from hmmgan.hmm import Emitter, neural_hmm

    neural_cfg = cfg["hmm"]["neural"]
    # Seeds the emitter initialization as well as SVI, so the whole fit is reproducible.
    pyro.set_rng_seed(int(neural_cfg["seed"]))
    emissions = torch.tensor(train_data.emission.values)
    states = torch.tensor(train_data.regime.values)
    emitter = Emitter(
        neural_cfg["z_dim"], neural_cfg["hidden_dim"], neural_cfg["emission_dim"]
    )

    guide = AutoDelta(
        poutine.block(neural_hmm, expose_fn=lambda msg: msg["name"].startswith("probs_"))
    )
    svi = SVI(neural_hmm, guide, Adam({"lr": neural_cfg["learning_rate"]}), TraceEnum_ELBO())

    pyro.clear_param_store()
    losses = []
    for _ in range(neural_cfg["n_steps"]):
        losses.append(svi.step(emissions, emitter, n_regimes, states) / len(emissions))

    with torch.no_grad():
        params = [emitter(torch.tensor([i], dtype=torch.float32)) for i in range(n_regimes)]
    return HMMFit(
        name="Neural HMM",
        transmat=pyro.get_param_store()["AutoDelta.probs_z"].detach().numpy(),
        mu=np.array([p[0].item() for p in params]),
        sigma=np.array([p[1].item() for p in params]),
        diagnostics={"elbo_losses": losses, "emitter": emitter},
    )


FITTERS: dict[str, Callable[[pd.DataFrame, int, dict[str, Any]], HMMFit]] = {
    "supervised": fit_supervised_hmm,
    "markov_switching": fit_markov_switching,
    "semi_supervised": fit_semi_supervised_hmm,
    "neural": fit_neural_hmm,
}


def initial_distribution(train_data: pd.DataFrame, n_regimes: int) -> np.ndarray:
    """Empirical regime frequencies of the training frame (notebook cell 26).

    Reindexed over all ``n_regimes`` so the vector stays length ``K`` even when a regime is absent
    from the slice. The reference indexes only the observed regimes, which silently produces a short
    vector and a shape error downstream; volatility regimes cluster in time, so a temporal split can
    easily strand one.
    """
    counts = train_data.regime.value_counts(normalize=True)
    return counts.reindex(range(n_regimes), fill_value=0.0).sort_index().values


def estimate_states(
    fit: HMMFit, observations: np.ndarray, init_dist: np.ndarray, n_regimes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed state posteriors ``(T, K)`` and their argmax ``(T,)``."""
    from hmmgan.state_estimation import forward_backward

    return forward_backward(
        init_dist=init_dist,
        observations=observations,
        transition_matrix=fit.transmat,
        mu=fit.mu,
        sigma=fit.sigma,
        num_states=n_regimes,
    )


def accuracy_report(
    truth: np.ndarray,
    estimates: np.ndarray,
    probabilities: np.ndarray,
    n_regimes: int,
    top_k: int = 2,
) -> dict[str, float]:
    """Accuracy and top-k accuracy (notebook cells 27 and 31).

    ``labels`` is passed explicitly so the top-k score stays well defined when a slice happens not to
    contain every regime.
    """
    from sklearn.metrics import accuracy_score, top_k_accuracy_score

    return {
        "accuracy": accuracy_score(truth, estimates),
        f"top_{top_k}_accuracy": top_k_accuracy_score(
            y_true=truth, y_score=probabilities, k=top_k, labels=list(range(n_regimes))
        ),
    }
