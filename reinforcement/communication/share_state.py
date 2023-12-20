import numpy as np

from .core import Swarm
from collections import Counter


class ShareState(Swarm):
    """
    Allows to use the predictions of other agents according to the communication matrix.
    In this case, it is possible to select several best agents in the swarm
    based on the expected reward in a given state and vote for action among them.
    """
    def __init__(self, best_neighbors: int = 1, *args, **kwargs):
        super(ShareState, self).__init__(*args, **kwargs)
        self.best_neighbors = best_neighbors

    def get_action(self, swarm_state: np.array) -> np.array:
        swarm_states = super().transform_states(np.array([swarm_state]))
        actions = np.zeros(self.size)

        for i in range(self.size):
            state_values = {
                agent: agent.get_state_values(swarm_states[:, i])
                for index, agent in enumerate(self.squad) if index in self.adjacency_list[i] + [i]
            }

            best_agents = sorted(state_values.items(), key=lambda x: x[1])
            best_agents = best_agents[-self.best_neighbors:]

            best_actions = Counter([
                agent.sample_actions(agent.get_qvalues(swarm_states[:, i]))
                for agent, state_value in best_agents
            ])

            actions[i] = best_actions.most_common()[0][0]

        return actions
    