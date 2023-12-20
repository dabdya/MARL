import abc
import numpy as np
from typing import List

from ..core import Agent


class PolicyAggregation(abc.ABC):
    @abc.abstractmethod
    def get_weights(self, squad: List[Agent], agent_state: np.array, neighbors_indexes: List[int]):
        raise NotImplementedError
