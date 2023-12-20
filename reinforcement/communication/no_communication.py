from .core import Swarm
import numpy as np


class NoCommunication(Swarm):
    """
    Agents within this swarm do not share information
    and make decisions only based on their policies and observations.
    """
    def get_action(self, swarm_state: np.array) -> np.array:
        swarm_states = super().transform_states(np.array([swarm_state]))
        actions = np.zeros(shape=self.size)
        for i, agent in enumerate(self.squad):
            qvalues = agent.get_qvalues(swarm_states[:, i])
            actions[i] = agent.sample_actions(qvalues)
            
        return actions
