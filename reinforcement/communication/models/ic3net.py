import torch
import torch.nn as nn


class CommunicationAction(nn.Module):
    """
    Simple network containing a softmax layer for 2 binary/continuous actions
    (communicate or not) on top of a linear layer with non-linearity
    """
    def __init__(
        self, input_size: int,
        n_actions: int = 2, binary_communication: bool = True
    ):
        super(CommunicationAction, self).__init__()

        self.binary_communication = binary_communication

        self.communicate_or_not = nn.Sequential(
            nn.Linear(input_size, n_actions),
            nn.Softmax(dim = -1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        communication_action = self.communicate_or_not(x).flatten()
        communication_index = torch.argmax(communication_action)

        if self.binary_communication:
            return x * communication_index

        communication_prob = communication_action[communication_index]
        if communication_index == 0:
            communication_prob = 1 - communication_prob

        return x * communication_prob
    

class IC3Network(nn.Module):
    """
        Individualized Controlled Continuous Communication Model
        https://arxiv.org/pdf/1812.09755.pdf
    """
    def __init__(
        self, state_size, n_agents: int, hidden_size: int = 128):
        super(IC3Network, self).__init__()

        self.encoder = nn.Linear(state_size, hidden_size)

        self.communication_transformation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(dim = -1)
        )

        self.rnn = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, batch_first = True)
            for _ in range(n_agents)
        ])

        self.communication_gate = nn.ModuleList([
            CommunicationAction(hidden_size, binary_communication = True)
            for _ in range(n_agents)
        ])

    def forward(
        self, 
        observations: torch.Tensor,                 # [batch_size, n_agents, state_size]
        communication_vectors: torch.Tensor,        # [batch_size, n_agents, hidden_size]
        hidden_states: torch.Tensor,                # ...
        cell_states: torch.Tensor,                  # ...
        alive_agents: int = None):
        
        batch_size, n_agents, state_size = observations.shape
        encoded_observations = self.encoder(observations) + communication_vectors

        rnn_output = []
        for i, rnn in enumerate(self.rnn):
            rnn_states = (
                hidden_states[:,i][None].repeat(1, batch_size, 1), 
                cell_states[:,i][None].repeat(1, batch_size, 1)
            )
            _, new_rnn_states = rnn(encoded_observations[:,[i]], rnn_states)
            rnn_output.append(new_rnn_states)

        # 2 x (n_agents x [1, batch_size, hidden_size])
        hidden_states, cell_states = zip(*rnn_output)

        # n_agents x [1, batch_size, hidden_size]
        communication_vectors = [
            comm(hidden_states[i])
            for i, comm in enumerate(self.communication_gate)
        ]

        alive_normalization = 1 / (alive_agents - 1) if alive_agents else 1

        # 1, batch_size, hidden_size
        communication_sum =  alive_normalization * torch.sum(
            torch.stack(communication_vectors, dim = -2), axis = -2)

        for i, communication_vector in enumerate(communication_vectors):
            communication_vectors[i] = self.communication_transformation(
                communication_sum - communication_vector
            )
        
        return (
            torch.stack(communication_vectors, dim = -2)[0], 
            (
                torch.stack(hidden_states, dim = -2)[0], 
                torch.stack(cell_states, dim = -2)[0]
            )
        )
