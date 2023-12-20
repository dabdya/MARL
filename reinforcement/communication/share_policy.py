import numpy as np
from copy import deepcopy
from .core import Swarm
from .aggregation import PolicyAggregation

from typing import List


class SharePolicy(Swarm):
    """
    In a certain way it aggregates the weights of policies across all agents
    within the swarm according to the communication matrix.
    Makes sense only if all agents inside swarm have the same policies.
    """
    def __init__(self, neighbors_depth: int, aggregation: PolicyAggregation, *args, **kwargs):
        super(SharePolicy, self).__init__(*args, **kwargs)
        self._policy_validation()

        self.neighbors_depth = neighbors_depth
        self.aggregation = aggregation

    def _policy_validation(self) -> None:
        policy_type = type(self.squad[0].policy)
        ok = all(type(agent.policy) is policy_type for agent in self.squad)
        if not ok:
            raise TypeError("All agents inside this swarm should have the same policies")

    def _get_common_policy(self, agent_state: np.array, neighbors_indexes: List[int]):
        common_policy_dict = dict()

        weights = self.aggregation.get_weights(self.squad, agent_state, neighbors_indexes)
        print(weights)
        neighbors_policy_states = [
            self.squad[index].policy.state_dict() for index in neighbors_indexes
        ]

        for key in self.squad[0].policy.state_dict().keys():
            new_policy_state = sum([
                weights[i] * policy_state[key]
                for i, policy_state in enumerate(neighbors_policy_states)
            ])
            common_policy_dict[key] = new_policy_state

        common_policy = deepcopy(self.squad[0].policy)
        common_policy.load_state_dict(common_policy_dict)
        return common_policy

    def _collect_neighbors(self, agent_internal_index: int, depth: int, visited: set) -> List[int]:
        if depth == 0:
            return []

        visited.add(agent_internal_index)

        direct_neighbors = [
            index for index in self.adjacency_list[agent_internal_index]
            if index not in visited
        ]
        all_neighbors = deepcopy(direct_neighbors)

        for neighbour_index in direct_neighbors:
            if neighbour_index not in visited:
                all_neighbors.extend(self._collect_neighbors(neighbour_index, depth-1, visited))

        return all_neighbors

    def get_action(self, swarm_state: np.array) -> np.array:
        actions = np.zeros(self.size)
        swarm_states = super().transform_states(np.array([swarm_state]))

        mock_agent = deepcopy(self.squad[0])

        for internal_index in range(self.size):
            agent_state = swarm_states[:, internal_index]
            neighbors_indexes = self._collect_neighbors(internal_index, self.neighbors_depth, set())
            common_policy = self._get_common_policy(agent_state, neighbors_indexes + [internal_index])

            mock_agent._policy = common_policy
            qvalues = mock_agent.get_qvalues(agent_state)
            actions[internal_index] = mock_agent.sample_actions(qvalues)

        return actions
