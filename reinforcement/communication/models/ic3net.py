"""
Implementation of Individualized Controlled Continuous Communication Model
Source: https://arxiv.org/pdf/1812.09755.pdf, 2019
"""

import torch
import torch.nn as nn


class CommunicationGate(nn.Module):
    """
    Simple network containing a softmax layer for two actions (communicate or not)
    on top of a linear layer with non-linear activation function.

    In original paper suggests using binary communication,
    but this implementation offers an option with continuous communication,
    when information is distributed in proportion to the weights from the softmax layer.
    """

    def __init__(self, input_size: int, binary_communication: bool = True):
        super(CommunicationGate, self).__init__()
        self.binary_communication = binary_communication

        self.communicate_or_not = nn.Sequential(
            nn.Linear(input_size, out_features=2), nn.Sigmoid(), nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        communicate_or_not = self.communicate_or_not(x)
        communication_indicator = torch.argmax(communicate_or_not, dim=-1, keepdim=True)

        if self.binary_communication:
            return communication_indicator

        mask = communication_indicator == 0
        communication_prob, _ = torch.max(communicate_or_not, dim=-1, keepdim=True)
        communication_prob[mask] = 1 - communication_prob[mask]

        return communication_prob


class IC3Network(nn.Module):
    """

    """

    def __init__(
            self, state_size: int, n_agents: int, hidden_size: int = 128,
            binary_communication: bool = False
    ):
        super(IC3Network, self).__init__()

        self.encoder = nn.Linear(state_size, hidden_size)
        self.communication_transformation = nn.Linear(hidden_size, hidden_size)

        self.rnn = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, batch_first=True)
            for _ in range(n_agents)
        ])

        self.communication_gate = nn.ModuleList([
            CommunicationGate(hidden_size, binary_communication)
            for _ in range(n_agents)
        ])

    def forward(
        self,
        state: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
        communication_tensor: torch.Tensor
    ):

        batch_size, n_agents, state_size = state.shape
        encoded_state = self.encoder(state) + communication_tensor

        def repeat(x):
            return x.repeat(1, batch_size, 1)

        rnn_output = []
        for i, rnn in enumerate(self.rnn):
            rnn_state = repeat(hidden_state[i]), repeat(cell_state[i])
            _, (new_rnn_state) = rnn(encoded_state[:, [i]], rnn_state)
            rnn_output.append([rnn_state.squeeze(dim=0) for rnn_state in new_rnn_state])

        # n_agents x [batch_size, hidden_size]
        new_hidden_state, new_cell_state = zip(*rnn_output)

        # n_agents x [batch_size, hidden_size]
        new_communication_vector = [
            comm(hidden_state[i]) * new_hidden_state[i]
            for i, comm in enumerate(self.communication_gate)
        ]

        # [batch_size, n_agents, hidden_size]
        new_communication_vector = torch.stack(new_communication_vector, dim=1)

        alive_normalization = 1
        # alive_normalization = 1 / (alive_agents - 1) if alive_agents else 1

        # [batch_size, hidden_size]
        communication_sum = torch.sum(new_communication_vector, dim=1)

        for i in range(n_agents):
            new_communication_vector[:, i] = \
                self.communication_transformation(communication_sum - new_communication_vector[:, i])
            new_communication_vector[:, i] *= alive_normalization

        new_hidden_state = torch.stack(new_hidden_state, dim=1)
        new_cell_state = torch.stack(new_cell_state, dim=1)

        return (
            new_hidden_state.squeeze(dim=0),
            new_cell_state.squeeze(dim=0),
            new_communication_vector.squeeze(dim=0)
        )
