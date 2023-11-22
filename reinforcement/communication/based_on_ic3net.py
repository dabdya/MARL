from .swarm import Swarm
from .models.ic3net import IC3Network
from ..training.agent import Agent

from .no_communication import NoCommunication

import numpy as np
from typing import List
import torch

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