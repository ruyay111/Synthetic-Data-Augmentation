import pyro
import pyro.distributions as dist

from pyro import poutine

import torch
import torch.nn as nn



class Emitter(nn.Module):
    def __init__(
        self,
        z_dim: int, 
        hidden_dim: int, 
        emission_dim: int,
    ):
        """Neural network used to learn emission mean and variance parameters
        
        Args
        ====
        z_dim: int
            Dimensionality of the HMM's hidden state

        hidden_dim: int
            Dimensionality of the neural network's hidden layers

        emission_dim: int
            Dimensionality of the observed data
        """
        super().__init__()
        self.z_to_hidden_loc = nn.Linear(z_dim, hidden_dim)
        self.z_to_hidden_scale = nn.Linear(z_dim, hidden_dim)
        self.hidden_loc_to_hidden_loc = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_scale_to_hidden_scale = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_loc_to_loc = nn.Linear(hidden_dim, emission_dim)
        self.hidden_scale_to_scale = nn.Linear(hidden_dim, emission_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        h1_loc = self.relu(self.z_to_hidden_loc(z_t))
        h2_loc = self.relu(self.hidden_loc_to_hidden_loc(h1_loc))
        loc = self.hidden_loc_to_loc(h2_loc)

        h1_scale = self.relu(self.z_to_hidden_scale(z_t))
        h2_scale = self.relu(self.hidden_scale_to_hidden_scale(h1_scale))
        scale = self.softplus(self.hidden_scale_to_scale(h2_scale))
    
        return loc, scale
  

class Transitioner(nn.Module):
    def __init__(self, z_dim, hidden_dim, probs_dim):
        """Neural network used to learn transition probabilities
        
        Args
        ====
        z_dim: int
            Dimensionality of the HMM's hidden state

        hidden_dim: int
            Dimensionality of the neural network's hidden layers

        probs_dim: int
            Number of hidden states in HMM
        """
        super().__init__()
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_to_probs = nn.Linear(hidden_dim, probs_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, z_t_1):
        h1 = self.relu(self.z_to_hidden(z_t_1))
        h2 = self.relu(self.hidden_to_hidden(h1))
        probs = self.softmax(self.hidden_to_probs(h2))
        return probs
  

def semi_supervised_neural_hmm(
    supervised_data: torch.Tensor,
    supervised_labels: torch.Tensor,
    unsupervised_data: torch.Tensor,
    num_states: int,
    emitter: Emitter,
    transitioner: Transitioner = None,
    neural_transition: bool = False
):
    pyro.module('emitter', emitter)
    if neural_transition:
        pyro.module('transitioner', transitioner)
        
    probs_z = pyro.sample(
        "probs_z",
        dist.Dirichlet(1 / num_states * torch.ones((num_states, num_states))).to_event(1)
    )

    # Handling Supervised Data
    for t in pyro.markov(range(supervised_data.size(0))):
        if neural_transition:
            # Neural network-based transition probabilities
            probs_t = transitioner(torch.tensor(supervised_labels[t-1], dtype=torch.float32).unsqueeze(0)) if t > 0 else torch.ones(num_states) / num_states
        else:
            # Categorical distribution for transitions
            probs_t = probs_z[supervised_labels[t-1]] if t > 0 else torch.ones(num_states) / num_states

        z = pyro.sample(
            f'supervised_z_{t}',
            dist.Categorical(probs_t.squeeze()),
            obs=supervised_labels[t]
        )

        loc, scale = emitter(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
        pyro.sample(
            f'supervised_x_{t}',
            dist.Normal(loc, scale),
            obs=supervised_data[t]
        )

    # Handling Unsupervised Data
    for t in pyro.markov(range(unsupervised_data.size(0))):
        if neural_transition:
            # Transition probabilities from the neural network
            probs_t = transitioner(torch.tensor(z, dtype=torch.float32).unsqueeze(0)) if t > 0 else torch.ones(num_states) / num_states
        else:
            # Categorical distribution for transitions
            probs_t = probs_z[z] if t > 0 else torch.ones(num_states) / num_states

        z = pyro.sample(f'unsupervised_z_{t}', dist.Categorical(probs_t.squeeze()))

        loc, scale = emitter(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
        pyro.sample(
            f'unsupervised_x_{t}',
            dist.Normal(loc, scale),
            obs=unsupervised_data[t]
        )
                    