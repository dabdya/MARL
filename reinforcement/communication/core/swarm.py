import abc
import torch
import numpy as np

from typing import List
from .agent import Agent


class Swarm(abc.ABC):
    """
    An abstract class through which a new swarm type can be defined by inheritance.
    """
    def __init__(self, squad: List[Agent], communication_matrix: List[List[int]] = None):
        self._squad = squad

        self.adjacency_list = []
        if communication_matrix is not None:
            for internal_index, agent in enumerate(squad):
                neighbors_indexes = [
                    index
                    for index, link in enumerate(communication_matrix[internal_index])
                    if link and internal_index != index
                ]
                self.adjacency_list.append(neighbors_indexes)

    @property
    def squad(self) -> List[Agent]:
        return self._squad

    @property
    def size(self) -> int:
        return len(self.squad)

    @property
    def indexes(self) -> List[int]:
        return [agent.index for agent in self.squad]

    @abc.abstractmethod
    def get_action(self, swarm_state: np.array) -> np.array:
        """
        Abstract method, should be overridden for each swarm type independently.

        Parameters:
            swarm_state (np.array):
                The state of the swarm agents obtained from the environment.
                Should be size [swarm_size, state_size].

        Returns:
            action (np.array):
                The array of swarm size that defines the action for each agent.
        """
        raise NotImplementedError

    def transform_states(self, swarm_states: np.array) -> torch.Tensor:
        """
        Transforms the state of the environment into a state convenient for swarm training.

        Parameters:
            swarm_states (np.array):
                Batch of swarm states [swarm_size, state_size].

        Returns:
            transformed_state (np.array):
                By default, flatten state into a one-dim vector for each agent across all batches.
        """
        return torch.tensor(swarm_states, dtype=torch.float32).flatten(start_dim=2)
