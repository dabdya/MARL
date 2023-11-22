import torch.nn as nn
from .policy_factory import Policy


class SimpleFullyConnected(Policy):
    def __init__(self, state_shape, n_actions, hidden_size):
        super(SimpleFullyConnected, self).__init__(state_shape, n_actions, hidden_size)

        self.network = nn.Sequential(
            nn.Linear(state_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions) # softmax
        )

    def forward(self, states):
        qvalues = self.network(states)
        return qvalues
