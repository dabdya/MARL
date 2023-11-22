import torch, abc
import numpy as np
from typing import List

from ..training import Agent


class Swarm(abc.ABC):
    def __init__(self, squad: List[Agent]):
        self._squad = squad

    @property
    def size(self) -> int:
        return len(self.squad)
    
    @property
    def squad(self) -> List[Agent]:
        return self._squad
    
    @property
    def agent_indexes(self) -> List[int]:
        return [agent.index for agent in self.squad]
    
    @abc.abstractmethod
    def get_action(self, swarm_state: np.array) -> np.array:
        raise NotImplementedError

    def transfrom_states(self, swarm_states: np.array) -> torch.Tensor:
        return torch.tensor(
            swarm_states, dtype = torch.float32).flatten(start_dim = 2)
