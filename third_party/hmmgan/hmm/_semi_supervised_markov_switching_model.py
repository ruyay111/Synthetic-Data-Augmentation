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


def forward_log_prob_one_step(prev_log_prob, current_emission, previous_emission, transition_log_prob, emission_loc, emission_scale, emission_beta):
    
    # print('prev')
    # print(prev_log_prob)
    # print('cur')
    # print(current_emission)
    # print('prev_emi')
    # print(previous_emission)
    # print('trans')
    # print(transition_log_prob)
    # print('loc')
    # print(emission_loc)
    # print(emission_loc.shape)
    # print(emission_scale)
    # print(emission_scale.shape)
    # print(emission_beta)
    # print(emission_beta.shape)
    log_prob_tmp = jnp.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    
    # Calculate the mean of the current emission based on the Markov Switching Model
    adjusted_emission_loc = emission_loc + emission_beta * previous_emission
    
    # Calculate log probability of the observation
    log_prob = log_prob_tmp + log_pdf(current_emission,adjusted_emission_loc,emission_scale)
    #print(log_prob.shape)
    return logsumexp(log_prob, axis=0)

def forward_log_prob(
  init_log_prob:float , 
  data : jnp.array, 
  transition_log_prob:float, 
  emission_loc: jnp.array, 
  emission_scale: jnp.array, 
  emission_beta: jnp.array, 
  unroll_loop=True):
    def scan_fn(carry, emissions_pair):
        prev_log_prob, prev_emission = carry
        current_emission, previous_emission = emissions_pair
        current_log_prob = forward_log_prob_one_step(prev_log_prob, current_emission, previous_emission, transition_log_prob, emission_loc, emission_scale, emission_beta)
        return (current_log_prob, current_emission), None

    # Create a stacked JAX array from the emissions with lag
    emissions_with_lag = jnp.stack([data[1:], data[:-1]], axis=-1)
    
    if unroll_loop:
        log_prob = init_log_prob
        prev_emission = data[0]
        # You must loop over the axis=0 (the first axis) of emissions_with_lag
        for i in range(emissions_with_lag.shape[0]):
            current_emission, previous_emission = emissions_with_lag[i]
            log_prob = forward_log_prob_one_step(log_prob, current_emission, previous_emission, transition_log_prob, emission_loc, emission_scale, emission_beta)
    else:
        # Pass the stacked array to scan, so it has a shape attribute
        log_prob, _ = lax.scan(scan_fn, (init_log_prob, data[0]), emissions_with_lag)
    tmp, _ = log_prob
    return tmp

def semi_supervised_markov_switching_model(
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
    #print(6)
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

    probs_beta = numpyro.sample(
        'probs_beta',
        dist.Normal(
            jnp.zeros(num_states), jnp.ones(num_states)
        ).to_event(1),
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
    #print(0)
    # log probability of transition probabilities
    transition_log_prob = jnp.log(probs_x)
    #print(1)
    init_log_prob = log_pdf(unsupervised_data[0], probs_mu, probs_sigma)
    #print(init_log_prob)
    #print(8)
    #print(2)
    # computing log probability of observing unsupervised data given
    # the conditioned parameters of the transition and emissions distributions
    log_prob = forward_log_prob(
        init_log_prob,
        unsupervised_data[1:],
        transition_log_prob,
        probs_mu,
        probs_sigma,
        probs_beta,
        unroll_loop
    )
    #print(3)
    log_prob = logsumexp(log_prob, axis=0, keepdims=True)
    #print(4)
    numpyro.factor('forward_log_prob', log_prob)