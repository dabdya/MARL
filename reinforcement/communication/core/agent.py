import copy
import numpy as np
import torch.nn as nn


class Agent:
    def __init__(self, index: int, policy: nn.Module, explore_rate: float = 1.0):
        self.index = index
        self._explore_rate = explore_rate
        self._policy, self._target_policy = policy, copy.deepcopy(policy)

    @property
    def policy(self):
        return self._policy

    @property
    def target(self):
        return self._target_policy

    @property
    def explore_rate(self):
        return self._explore_rate

    @explore_rate.setter
    def explore_rate(self, value: int):
        if value < 0 or value > 1:
            raise ValueError("Explore rate should be in [0,1] range")
        self._explore_rate = value

    def update_target_policy(self):
        self._target_policy.load_state_dict(self._policy.state_dict())

    def get_qvalues(self, states):
        qvalues = self._policy(states)
        return qvalues.data.cpu().numpy()

    def get_state_values(self, states):
        qvalues = self.get_qvalues(states)
        return np.max(qvalues, axis=-1)

    def sample_actions(self, qvalues, greedy: bool = True):
        epsilon = self._explore_rate
        batch_size, n_actions = qvalues.shape

        best_actions = qvalues.argmax(axis=-1)
        if not greedy:
            return best_actions

        random_actions = np.random.choice(n_actions, size=batch_size)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])

        return np.where(should_explore, random_actions, best_actions)[0]
