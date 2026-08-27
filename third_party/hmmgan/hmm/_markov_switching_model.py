import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.handlers import mask
from numpyro.contrib.control_flow import scan


def markov_switching_model(
    data: jnp.array, 
    num_states: int,
    include_prior: bool = True
):
    """
    NumPyro implementation of Markov Switching Model

    Args
    ====
    data: jnp.array
        Three-dimensional array containing emissions, 
        emission lags, and volatility regime labels

    num_states: int
        Number of volatility regimes being modeled

    include_prior: bool
        Whether or not to mask prior transition distribution
    """
    with mask(mask=include_prior):
        # transition probability prior modeled by Dirichlet distribution
        probs_x = numpyro.sample(
            'probs_x',
            dist.Dirichlet(
                1 / num_states * jnp.ones((num_states, num_states))
            ).to_event(1),
        )

    # normal distribution serving as the prior for the 
    # mean of the emissions distributions
    probs_mu = numpyro.sample(
        'probs_mu',
        dist.Normal(
            jnp.zeros(num_states), jnp.ones(num_states)
        ).to_event(1)
    )

    # normal distribution serving as the prior for the probabilistic 
    # coefficient multiplied by the prior time step's emission
    probs_beta = numpyro.sample(
        'probs_beta',
        dist.Normal(
            jnp.zeros(num_states), jnp.ones(num_states)
        ).to_event(1),
    )

    # half cauchy distribution, which constrains values to be positive,
    # serving as the prior for the variance parameter of the emissions distributions
    probs_sigma = numpyro.sample(
        'probs_sigma',
        dist.HalfCauchy(jnp.ones(num_states)).to_event(1),
    )

    states = jnp.array(data[:, 0], dtype=jnp.int32)
    emissions = jnp.array(data[:, 1], dtype=jnp.float32)
    emission_lags = jnp.array(data[:, 2], dtype=jnp.float32)
    
    # transition function that is fed to NumPyro's `scan` function and 
    # essentially replaces a `for` loop. does sampling and conditioning
    # of transition and emission values and parameters at each time step
    def transition_fn(carry, y):
        x_prev, t = carry
        with mask(mask=True):
            # sample transition probability and condition distribution's
            # parameters on observed volatility regime
            x = numpyro.sample(
                'x',
                dist.Categorical(probs_x[x_prev]),
                infer={'enumerate': 'parallel'},
                obs=y[0]
            )

        # compute mean of emission distribution using markov switching equation
        loc = probs_mu[x] + probs_beta[x] * emission_lags[t]

        with numpyro.plate('emission_plate', 1, dim=-1):
            # sample emission and condition distribution's parameters 
            # on observed emission at given time step
            y = numpyro.sample(
                'y',
                dist.Normal(
                    loc, probs_sigma[x]
                ),
                obs=emissions[t]
            )

        # sampling returns a 0-dimensional value, so an extra
        # dimension must be added to the `x` value for downstream compatibility
        x = x[...,None]

        return (x, t+1), None

    # initialize hidden state and loop through observed data
    x_init = jnp.zeros((1,), dtype=jnp.int32)
    scan(transition_fn, (x_init, 0), states[..., None])
