import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, state_size, n_actions, hidden_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size


class PolicyFactory:
    def __init__(self, state_size, n_actions, hidden_size):
        self.state_size = state_size
        self.n_actions = n_actions
        self.hidden_size = hidden_size

    def get_policy(self, policy_type: Policy):
        return policy_type(self.state_size, self.n_actions, self.hidden_size)
