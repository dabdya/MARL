from typing import List
from .agent import Agent
from ..communication.ic3net import IC3Network

import torch
import numpy as np
from abc import abstractmethod, ABC as Abstract


class Swarm(Abstract):
    def __init__(self, squad: List[Agent]):
        self._squad = squad

    @property
    def size(self):
        return len(self.squad)
    
    @property
    def squad(self) -> List[Agent]:
        return self._squad
    
    @property
    def agent_indexes(self) -> List[int]:
        return [agent.index for agent in self.squad]
    
    def transfrom_states(self, swarm_states: np.array) -> torch.Tensor:
        return torch.tensor(
            swarm_states, dtype = torch.float32).flatten(start_dim = 2)
    
    @abstractmethod
    def get_action(self, swarm_state: np.array):
        raise NotImplementedError


class ShareObservation(Swarm):

    def get_action(self, swarm_state: np.array):
        swarm_states = super(ShareObservation, self).transfrom_states(np.array([swarm_state]))
        actions = np.zeros(shape = self.size)
        for i, agent in enumerate(self.squad):
            state_values = {
                other_agent: other_agent.get_state_values(swarm_states[:,i])
                for other_agent in self.squad
            }

            best_agent = sorted(state_values.items(), key = lambda x: x[1])[-1][0]
            qvalues = best_agent.get_qvalues(swarm_states[:,i])
            actions[i] = best_agent.sample_actions(qvalues)

        return actions


class NoCommunication(Swarm):

    def get_action(self, swarm_state: np.array):
        swarm_states = super(NoCommunication, self).transfrom_states(np.array([swarm_state]))
        actions = np.zeros(shape = self.size)
        for i, agent in enumerate(self.squad):
            qvalues = agent.get_qvalues(swarm_states[:,i])
            actions[i] = agent.sample_actions(qvalues)
        
        return actions
    

class IC3Net(Swarm):
    def __init__(self, squad: List[Agent], state_size: int, hidden_size: int = 128):
        super(IC3Net, self).__init__(squad)
        self.ic3net = IC3Network(
            state_size, n_agents = self.size, hidden_size = hidden_size)

        self.communication_vectors = torch.zeros(1, self.size, hidden_size)
        self.hidden_state = torch.randn(1, self.size, hidden_size)
        self.cell_state = torch.randn(1, self.size, hidden_size)

    def transfrom_states(self, swarm_states: np.array) -> torch.Tensor:
        swarm_states = super(IC3Net, self).transfrom_states(swarm_states)

        _, (hidden_states, _) = self.ic3net(
            swarm_states, self.communication_vectors, self.hidden_state, self.cell_state)
        
        return hidden_states
    
    def get_action(self, swarm_state: np.array):
        swarm_states = super(IC3Net, self).transfrom_states(np.array([swarm_state]))

        communication_vectors, (hidden_state, cell_state) = self.ic3net(
            swarm_states, self.communication_vectors, self.hidden_state, self.cell_state)

        self.communication_vectors = communication_vectors.detach()
        self.hidden_state, self.cell_state = hidden_state.detach(), cell_state.detach()

        return NoCommunication(self.squad).get_action(self.hidden_state.numpy()[0])
