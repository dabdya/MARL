from typing import List, Tuple
import numpy as np

from reinforcement.training import Agent
from collections import Counter
from .swarm import Swarm


class SharePolicy(Swarm):
    def __init__(self, squad: List[Agent], best_neighbours: int = 1):
        super(SharePolicy, self).__init__(squad)
        self.best_neighbours = best_neighbours

    def get_action(self, swarm_state: np.array) -> np.array:
        swarm_states = super().transfrom_states(np.array([swarm_state]))
        actions = np.zeros(shape = self.size)

        for i in range(self.size):
            state_values = {
                agent: agent.get_state_values(swarm_states[:,i])
                for agent in self.squad
            }

            best_agents = sorted(state_values.items(), key = lambda x: x[1])
            best_agents = best_agents[-self.best_neighbours:]

            best_actions = Counter([
                agent.sample_actions(agent.get_qvalues(swarm_states[:,i]))
                for agent, state_value in best_agents
            ])

            actions[i] = best_actions.most_common()[0][0]

        return actions
    