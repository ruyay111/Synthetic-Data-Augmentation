import jax.numpy as jnp

from jax import random, lax
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist

from numpyro.handlers import mask
from numpyro.contrib.control_flow import scan


def log_pdf(
    x: float,
    loc: jnp.array,
    scale: jnp.array
) -> float:
    """
    Calculate log probability of observation 
    given normal distribution parameters

    Args
    ====
    x: float
        Emission

    loc: jnp.array
        Array of normal distribution mean parameters used
        for calculating log probability of observation

    scale: jnp.array
        Array of normal distribution variance parameters used 
        for calculating log probability of observation
        
    Returns
    =======
    float
        Log probability of observation
    """
    return jnp.log(
        (1 / jnp.sqrt(2 * jnp.pi * scale**2))
        * jnp.exp(
            -1 * ((x - loc)**2 / (2 * scale**2))
        )
    )


def forward_log_prob_one_step(
    prev_log_prob: float,
    current_time_step: float,
    transition_log_prob: float,
    emission_loc: jnp.array,
    emission_scale: jnp.array
) -> float:
    """
    Log probability of observation at time step t + 1

    Args
    ====
    prev_log_prob: float
        Log probability of emissions at previous time step

    current_time_step: float
        Emission from current time step

    transition_log_prob: float
        Log transition probabilities

    emission_loc: jnp.array
        Array of normal distribution mean parameters used
        for calculating log probability of observation

    emission_scale: jnp.array
        Array of normal distribution variance parameters used 
        for calculating log probability of observation

    Returns
    =======
    float
        Log probability of observation at time step t + 1
    """
    print('prev')
    print(prev_log_prob)
    print('cur')
    print(current_time_step)
    print('trans')
    print(transition_log_prob)
    print('loc')
    print(emission_loc)
    print(emission_loc.shape)
    print(emission_scale)
    print(emission_scale.shape)
    log_prob_tmp = jnp.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + log_pdf(
        current_time_step, emission_loc, emission_scale
    )
    print(log_prob.shape)
    return logsumexp(log_prob, axis=0)

def forward_log_prob(
    init_log_prob: float,
    data: jnp.array,
    transition_log_prob: float,
    emission_loc: jnp.array,
    emission_scale: jnp.array,
    unroll_loop=True
) -> float:
    """
    Wrapper for `forward_log_prob_one_step` function that runs function
    for each time step in the semi-supervised HMM's unobserved data

    Args
    ====
    init_log_prob: float
        Log probability of observing first unsupervised emission

    data: jnp.array
        Array of unsupervised emissions

    transition_log_prob: float
        Log probability of transitions

    emission_loc: jnp.array
        Array of normal distribution mean parameters used
        for calculating log probability of observation

    emission_scale: jnp.array
        Array of normal distribution variance parameters used 
        for calculating log probability of observation

    unroll_loop: bool
        Whether or not to iterate through data via 
        a `for` loop or via a vectorized computation by Jax

    Returns
    =======
    float
        Log probability of unsupervised data
    """
    def scan_fn(log_prob, emission):
        return (
            forward_log_prob_one_step(
                log_prob,
                emission,
                transition_log_prob,
                emission_loc,
                emission_scale
            ),
            None
        )

    if unroll_loop:
        log_prob = init_log_prob
        
        for emission in data:
            log_prob = forward_log_prob_one_step(
                log_prob,
                emission,
                transition_log_prob,
                emission_loc,
                emission_scale
            )
            
    else:
        print(data.shape)
        log_prob, _ = lax.scan(scan_fn, init_log_prob, data)
    print(init_log_prob)
    
    print('log_prob')
    
    return log_prob


def semi_supervised_hmm(
    supervised_data: jnp.array,
    supervised_labels: jnp.array,
    unsupervised_data: jnp.array,
    num_states: int,
    unroll_loop: bool = False
):
    """
    NumPyro implementation of semi-supervised HMM

    Args
    ====
    supervised_data: jnp.array
        Array of emissions for supervised subset of training data

    supervised_labels: jnp.array
        Array of volatility regime labels for supervised subset of training data

    unsupervised_data: jnp.array
        Array of emissions for unsupervised portion of training data

    num_states: int
        Number of volatility regines

    unroll_loop: bool
        Whether or not to iterate through data via 
        a `for` loop or via a vectorized computation by Jax
    """
    # Dirichlet prior for transition probabilities
    probs_x = numpyro.sample(
        'probs_x',
        dist.Dirichlet(
            1 / num_states * jnp.ones((num_states, num_states))
        ).to_event(1),
    )

    # normal prior for means of emissions distributions
    probs_mu = numpyro.sample(
        'probs_mu',
        dist.Normal(
            jnp.zeros(num_states), jnp.ones(num_states)
        ).to_event(1)
    )

    # half cauchy prior for variances of emissions distributions,
    # which constrains parameters to be positive
    probs_sigma = numpyro.sample(
        'probs_sigma',
        dist.HalfCauchy(jnp.ones(num_states)).to_event(1)
    )

    # categorical distribution governing transition probabilities 
    # and conditioned on volatility regime labels
    numpyro.sample(
        'supervised_states',
        dist.Categorical(probs_x[supervised_labels[:-1]]),
        obs=supervised_labels[1:],
    )

    # normal distribution governing emissions distributions
    # and conditioned on observed emissions
    numpyro.sample(
        'supervised_emissions',
        dist.Normal(
            probs_mu[supervised_labels], probs_sigma[supervised_labels]
        ),
        obs=supervised_data
    )

    # log probability of transition probabilities
    transition_log_prob = jnp.log(probs_x)
    init_log_prob = log_pdf(unsupervised_data[0], probs_mu, probs_sigma)
    print(init_log_prob)
    # computing log probability of observing unsupervised data given
    # the conditioned parameters of the transition and emissions distributions
    log_prob = forward_log_prob(
        init_log_prob,
        unsupervised_data[1:],
        transition_log_prob,
        probs_mu,
        probs_sigma,
        unroll_loop
    )
    log_prob = logsumexp(log_prob, axis=0, keepdims=True)
    numpyro.factor('forward_log_prob', log_prob)
    