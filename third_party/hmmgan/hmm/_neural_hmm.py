import pyro
import pyro.distributions as dist

from pyro import poutine

import torch
import torch.nn as nn



class Emitter(nn.Module):
    def __init__(self, z_dim, hidden_dim, emission_dim, lstm_layers=1, transformer_nhead=1):
        super().__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=z_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Transformer layer
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=transformer_nhead)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)

        # Linear layers for mean and variance
        self.fc_loc = nn.Linear(hidden_dim, emission_dim)
        self.fc_scale = nn.Linear(hidden_dim, emission_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        # LSTM forward pass
        lstm_out, _ = self.lstm(z_t.unsqueeze(0))  # LSTM expects 3D tensor [batch, seq, feature]

        # Transformer forward pass
        transformer_out = self.transformer(lstm_out)

        # Final output from transformed features
        loc = self.fc_loc(self.relu(transformer_out.squeeze(0)))
        scale = self.softplus(self.fc_scale(self.relu(transformer_out.squeeze(0))))

        return loc, scale  
        
    def init_lstm_weights(self, weights, biases=None):
        """
        Initialize the weights of the LSTM layer.

        Args:
        - weights (torch.Tensor): The weight tensor to initialize the LSTM's weights.
        - biases (Optional[torch.Tensor]): The bias tensor to initialize the LSTM's biases.
        """
        self.lstm.weight_ih_l0 = nn.Parameter(weights)
        if biases is not None:
            self.lstm.bias_ih_l0 = nn.Parameter(biases)



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
        
    def init_first_layer_weights(self, weights, biases=None):
        """
        Initialize the weights of the first linear layer.

        Args:
        - weights (torch.Tensor): The weight tensor to initialize the layer's weights.
        - biases (Optional[torch.Tensor]): The bias tensor to initialize the layer's biases.
        """
        self.z_to_hidden.weight = nn.Parameter(weights)
        if biases is not None:
            self.z_to_hidden.bias = nn.Parameter(biases)
  

def neural_hmm(
    data: torch.tensor,
    emitter: Emitter, 
    num_states: int,
    states: torch.tensor = None,
    transitioner: Transitioner = None,
    neural_transition: bool = False
):
    """
    Pyro implementation of neural HMM. Emission parameters learned 
    by neural network. Transition probabilities can be learned by 
    standard categorical distributions or by additional neural network.

    Args
    ====
    data: torch.tensor
        Tensor containing observed emissions

    emitter: Emitter
        Neural network used to learn emission distribution parameters

    num_states: int
        Number of volatility regimes

    states: Option[torch.tensor]
        Tensor containing volatility regime labels

    transitioner: Option[Transitioner]
        Neural network used to learn transition probabilities

    neural_transition: bool
        Whether or not to use a neural network to model transition probabilities
    """
    pyro.module('emitter', emitter)
    if neural_transition:
        pyro.module('transitioner', transitioner)
        
        # generate transition probabilities using neural network with
        # previous time step's volatility regime label as input
        z = dist.Categorical(torch.ones(4) / 4).sample()
        for t in pyro.markov(range(data.size(0))):
            with poutine.mask(mask=True):
                probs_t = transitioner(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                z = pyro.sample(
                    f'z_{t}',
                    dist.Categorical(probs_t),
                    infer={'enumerate': 'paallel'},
                    obs=states[t]
                )

                # generate mean and variance parameters for emissions distribution 
                # using neural network with hidden state as input
                loc, scale = emitter(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                x = pyro.sample(
                    f'x_{t}',
                    dist.Normal(loc, scale),
                    obs=data[t]
                )
    else:
        with poutine.mask(mask=True):
            # define Dirichlet prior for transition probabilities
            probs_z = pyro.sample(
                'probs_z',
                dist.Dirichlet(
                    1 / num_states * torch.ones((num_states, num_states))
                ).to_event(1),
            )
  
        # loop through observed data and condition transition probabilities
        # on volatility regime labels
        z = 0  
        for t in pyro.markov(range(data.size(0))):
            with poutine.mask(mask=True):
                z = pyro.sample(
                    f'z_{t}',
                    dist.Categorical(probs_z[z]),
                    infer={'enumerate': 'paallel'},
                    obs=states[t]
                )

                # generate mean and variance parameters for emissions distribution 
                # using neural network with hidden state as input
                loc, scale = emitter(torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                with pyro.plate(f'emission_plate_{t}', 1, dim=-1):
                    x = pyro.sample(
                        f'x_{t}',
                        dist.Normal(loc, scale),
                        obs=data[t]
                    )
                    