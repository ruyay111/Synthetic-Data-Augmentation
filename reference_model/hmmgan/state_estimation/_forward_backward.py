import numpy as np
import numpy.typing as npt

from tqdm import tqdm
from typing import Union, Tuple
from ._likelihood import gaussian_likelihood


def forward_one_step(
    alpha_0: npt.NDArray,
    transition_matrix: npt.NDArray,
    observation_likelihood: npt.NDArray
) -> npt.NDArray:
    """
    Single forward step in the Forward Algorithm. Multiplies prior time step's
    state distribution by transition matrix to produce intermediate distribution.
    Multiplies intermediate distribution by diagonal matrix of observation 
    likelihoods for emission at current time step and normalizes result.

    Args
    ====
    alpha_0: npt.NDArray
        Probability distribution over volatility regimes at time step t - 1

    transition_matrix: npt.NDArray
        Transition matrix with rows representing starting volatility regime states
        and columns representing ending volatility regime states

    observation_likelihood: npt.NDArray
        Array of observation likelihoods in the various 
        volatility regimes for emission at current time step t

    Returns
    =======
    npt.NDArray
        Array of volatility regime probabilities for current time step t
    """
    alpha_prime = transition_matrix @ alpha_0
    alpha = np.diag(observation_likelihood) @ alpha_prime
    alpha /= alpha.sum()
    
    return alpha


def backward_one_step(
    m: npt.NDArray,
    transition_matrix: npt.NDArray,
    observation_likelihood: npt.NDArray
) -> npt.NDArray:
  """
  Single backward step in the Backward Algorithm. Multiplies volatility
  regime distribution at time t + 1 by diagonal matrix of observation 
  likelihoods for emission at current time step t. Multiplies resultant 
  distribution by transpose of transition matrix to produce volatility 
  regime probabilities at current time step t after normalization. Derived
  probabilities take into account information available after time t and are
  intended to help smooth initial, forward probability estimates at time t.
  
  Args
  ====
  m: npt.NDArray
    Array of volatility regime probabilities at time t + 1

  transition_matrix: npt.NDArray
    Transition matrix with rows representing starting volatility regime states
    and columns representing ending volatility regime states

  observation_likelihood: npt.NDArray
    Array of observation likelihoods in the various 
    volatility regimes for emission at current time step t

  Returns
  =======
  npt.NDArray
    Array of volatility regime probabilities for current time step t
  """
  m_prime = np.diag(observation_likelihood) @ m
  m_0 = transition_matrix.T @ m_prime
  m_0 /= m_0.sum()
  
  return m_0


def forward(
    init_dist: npt.NDArray,
    observations: npt.NDArray,
    transition_matrix: npt.NDArray,
    mu: npt.NDArray,
    sigma: npt.NDArray,
    num_states: int, 
    return_states: bool = False
) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
  """Forward algorithm implementation that wraps `forward_one_step` function

  Args
  ====
  init_dist: npt.NDArray
    Initial distribution over volatility regimes

  observations: npt.NDArray
    Array of observed log returns

  transition_matrix: npt.NDArray
    Transition matrix for volatility regimes

  mu: npt.NDArray
    Array of mean parameters for normal distributions 
    of different volatility regime emissions

  sigma: npt.NDArray
    Array of variance parameters for normal distributions
    of different volatility regime emissions

  num_states: int
    Number of volatility regimes

  return_states: bool
    Whether or not to return volatility regime estimates for each time step
    in addition to the state distributions

  Returns
  =======
  Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]
    Either a single NumPy array containing volatility regime probabilities 
    for each time step or a tuple of NumPy arrays containing the probabilities
    and their state estimates, i.e., the argmax of the distribution at each
    time step
  """
  alpha = init_dist
  forward_probabilities = []

  for i in tqdm(range(observations.shape[0])):
    # compute observation likelihood of emission at time t
    # given parameters of each volatility regime's emission distribution
    observation_likelihood = np.array([
        gaussian_likelihood(
            emission=observations[i],
            p_mu=mu[state],
            p_sigma=sigma[state]
        )
        for state in range(num_states)
    ])

    # calculate probability distribution over volatility regimes for 
    # current time step t using observation likelihoods, transition
    # matrix, and probability distribution over volatility regimes at 
    # time t - 1
    alpha = forward_one_step(
        alpha_0=alpha,
        transition_matrix=transition_matrix,
        observation_likelihood=observation_likelihood
    )

    forward_probabilities.append(alpha)

  forward_probabilities = np.array(forward_probabilities)

  if return_states:
    states = np.array([x.argmax().item() for x in forward_probabilities])
    return forward_probabilities, states

  else:
    return forward_probabilities


def backward(
    observations: npt.NDArray,
    transition_matrix: npt.NDArray,
    mu: npt.NDArray,
    sigma: npt.NDArray,
    num_states: int,
    return_states: bool = False
):
  """Backward algorithm implementation that wraps `backward_one_step` function

  Args
  ====
  observations: npt.NDArray
    Array of observed log returns

  transition_matrix: npt.NDArray
    Transition matrix for volatility regimes

  mu: npt.NDArray
    Array of mean parameters for normal distributions 
    of different volatility regime emissions

  sigma: npt.NDArray
    Array of variance parameters for normal distributions
    of different volatility regime emissions

  num_states: int
    Number of volatility regimes

  return_states: bool
    Whether or not to return volatility regime estimates for each time step
    in addition to the state distributions

  Returns
  =======
  Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]
    Either a single NumPy array containing volatility regime probabilities 
    for each time step or a tuple of NumPy arrays containing the probabilities
    and their state estimates, i.e., the argmax of the distribution at each
    time step
  """
  beta = np.ones(num_states)
  backward_probabilities = []

  for i in tqdm(reversed(range(observations.shape[0]))):
    # compute observation likelihood of emission at time t
    # given parameters of each volatility regime's emission distribution
    observation_likelihood = np.array([
      gaussian_likelihood(
          emission=observations[i],
          p_mu=mu[state],
          p_sigma=sigma[state]
      )
      for state in range(num_states)
    ])

    # calculate probability distribution over volatility regimes for 
    # current time step t using observation likelihoods, transition
    # matrix, and probability distribution over volatility regimes at 
    # time t + 1
    beta = backward_one_step(
        m=beta,
        transition_matrix=transition_matrix,
        observation_likelihood=observation_likelihood
    )

    backward_probabilities.append(beta)

  # The above loop computes probabilities backwards in time,
  # so the resultant list must be reversed. The distribution over the 
  # final time step is just an array of ones because no observation is
  # made after the final time step (i.e., there's no information that could
  # be used to update this probability). The initial probability distribution
  # is clipped from the array since it covers the time step before the first
  # observation was made.
  backward_probabilities = list(reversed(backward_probabilities))
  backward_probabilities.append(np.ones(num_states))
  backward_probabilities = np.array(backward_probabilities[1:])

  if return_states:
    states = np.array([x.argmax().item() for x in backward_probabilities])
    return backward_probabilities, states

  else:
    return backward_probabilities


def forward_backward(
    init_dist: npt.NDArray, 
    observations: npt.NDArray, 
    transition_matrix: npt.NDArray, 
    mu: npt.NDArray, 
    sigma: npt.NDArray, 
    num_states: int,
):
  """Forward algorithm implementation that wraps `forward_one_step` function

  Args
  ====
  init_dist: npt.NDArray
    Initial distribution over volatility regimes

  observations: npt.NDArray
    Array of observed log returns

  transition_matrix: npt.NDArray
    Transition matrix for volatility regimes

  mu: npt.NDArray
    Array of mean parameters for normal distributions 
    of different volatility regime emissions

  sigma: npt.NDArray
    Array of variance parameters for normal distributions
    of different volatility regime emissions

  num_states: int
    Number of volatility regimes

  Returns
  =======
  Tuple[npt.NDArray, npt.NDArray]
    Tuple of NumPy arrays containing smoothed volatility regime 
    probabilities and volatility regime estimates for each time step
  """
  # first run forward algorithm to compute initial forward state probabilities
  forward_probabilities = forward(
      init_dist=init_dist, 
      observations=observations, 
      transition_matrix=transition_matrix,
      mu=mu,
      sigma=sigma,
      num_states=num_states,
      return_states=False
  )

  # next run backward algorithm to compute backward state
  # proabilities and enable smoothing
  backward_probabilities = backward(
      observations=observations, 
      transition_matrix=transition_matrix,
      mu=mu,
      sigma=sigma,
      num_states=num_states,
      return_states=False
  )

  # smooth initial forward probabilities via backward probabilities
  forward_backward_probs = (forward_probabilities * backward_probabilities)
  forward_backward_probs = np.array([
      x / x.sum() for x in forward_backward_probs
  ])
  forward_backward_estimates = np.array([
      x.argmax() for x in forward_backward_probs
  ])

  return forward_backward_probs, forward_backward_estimates