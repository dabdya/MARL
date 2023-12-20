from reinforcement.communication.core.swarm import Swarm
from .models import IC3Network
from reinforcement.communication.core.agent import Agent

from .no_communication import NoCommunication

import numpy as np
from typing import List
import torch


class IC3NetBased(Swarm):
    def __init__(self, state_size: int, hidden_size: int, *args, **kwargs):
        super(IC3NetBased, self).__init__(*args, **kwargs)
        self.ic3net = IC3Network(state_size, self.size, hidden_size)
        self.communication_tensor = torch.zeros(self.size, hidden_size)
        self.hidden_state = torch.randn(self.size, hidden_size)
        self.cell_state = torch.randn(self.size, hidden_size)

    def transform_states(self, swarm_states: np.array) -> torch.Tensor:
        swarm_states = super().transform_states(swarm_states)
        transfer_data = self.hidden_state, self.cell_state, self.communication_tensor
        hidden_states, *_ = self.ic3net(swarm_states, *transfer_data)
        return hidden_states

    def get_action(self, swarm_state: np.array):
        swarm_states = super().transform_states(np.array([swarm_state]))
        transfer_data = self.hidden_state, self.cell_state, self.communication_tensor
        hidden_state, cell_state, communication_tensor = self.ic3net(swarm_states, *transfer_data)

        self.communication_tensor = communication_tensor.detach()
        self.hidden_state, self.cell_state = hidden_state.detach(), cell_state.detach()

        return NoCommunication(
            squad=self.squad).get_action(self.hidden_state.numpy())
