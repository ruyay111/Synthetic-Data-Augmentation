import numpy as np


def gaussian_likelihood(emission: float, p_mu: float, p_sigma: float) -> float:
  """
  Computes likelihood of an emission given mean 
  and variance of normal distribution. 

  Args
  ====
  emission: float
    Individual log return

  p_mu: float
    Mean of the normal distribution

  p_sigma: float
    Variance of the normal distribution

  Returns
  =======
  float
    Likelihood of emission given parameters of normal distribution
  """
  first_term = 1 / (np.sqrt(2 * np.pi * p_sigma**2))
  second_term = np.exp(
      -(emission - p_mu)**2 
      / (2 * p_sigma**2)
  )

  return first_term * second_term