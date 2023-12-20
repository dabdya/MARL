import numpy as np
from typing import List
from .base import PolicyAggregation
from ..core import Agent


class UniformAggregation(PolicyAggregation):
    def get_weights(self, squad: List[Agent], agent_state: np.array, neighbors_indexes: List[int]):
        n_neighbors = len(neighbors_indexes)
        return np.array([1 / n_neighbors for _ in range(n_neighbors)])


class RandomAggregation(PolicyAggregation):
    def get_weights(self, squad: List[Agent], agent_state: np.array, neighbors_indexes: List[int]):
        n_neighbors = len(neighbors_indexes)
        weights = np.random.uniform(size=n_neighbors)
        return weights / np.sum(weights)


class ValueBasedAggregation(PolicyAggregation):
    def get_weights(self, squad: List[Agent], agent_state: np.array, neighbors_indexes: List[int]):
        state_values = np.array([
            squad[index].get_state_values(agent_state)
            for index in neighbors_indexes
        ])
        return state_values / np.sum(state_values)
