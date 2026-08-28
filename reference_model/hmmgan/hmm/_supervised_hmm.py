import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.handlers import mask
from numpyro.contrib.control_flow import scan


def supervised_hmm(
    data: jnp.array,
    num_states: int, 
    include_prior: bool = True
):
    """
    NumPyro implementation of supervised HMM

    Args
    ====
    data: jnp.array
        Two-dimensional array of emissions and volatility regime labels

    num_states: int
        Number of volatility regimes

    include_prior: bool
        Whether or not to mask transition distributions
    """
    with mask(mask=include_prior):
        # Dirichlet transition prior
        probs_x = numpyro.sample(
            'probs_x',
            dist.Dirichlet(
                1 / num_states * jnp.ones((num_states, num_states))
            ).to_event(1),
        )

    # normal distribution serving as prior for emissions distribution mean parameter
    probs_mu = numpyro.sample(
        'probs_mu',
        dist.Normal(
            jnp.zeros(num_states), jnp.ones(num_states)
        ).to_event(1)
    )

    # half cauchy distribution, which constrains values to be positive,
    # server as prior for emissions distribution variance parameter
    probs_sigma = numpyro.sample(
        'probs_sigma',
        dist.HalfCauchy(jnp.ones(num_states)).to_event(1)
    )

    # splitting input data into two separate arrays because
    # regime labels and emissions have different dtypes
    emissions = jnp.array(data[:, 1], dtype=jnp.float32)
    states = jnp.array(data[:, 0], dtype=jnp.int32)

    def transition_fn(carry, y):
        x_prev, t = carry
        with mask(mask=True):
            # sample transition from categorical distribution 
            # and condition this distribution's parameters on 
            # volatility regime label
            x = numpyro.sample(
                'x',
                dist.Categorical(probs_x[x_prev]),
                infer={'enumerate': 'parallel'},
                obs=y[0]
            )

        with numpyro.plate('emission_plate', 1, dim=-1):
            # sample emission from normal distribution and 
            # condition this distribution's parameters on 
            # observed emission for the given time step
            y = numpyro.sample(
                'y',
                dist.Normal(
                    probs_mu[x], probs_sigma[x]
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
    