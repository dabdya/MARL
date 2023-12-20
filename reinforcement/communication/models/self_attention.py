import torch
from typing import Union


class SelfAttention(torch.nn.Module):
    def __init__(self, n_features: int) -> None:
        super(SelfAttention, self).__init__()

        self.query, self.key, self.value = [
            torch.nn.Linear(n_features, n_features)
            for _ in range(3)
        ]

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context_size, batch_size, n_features = x.shape

        Q, K, V = self.query(x), self.key(x), self.value(x)
        output = torch.Tensor(*x.shape)

        for i in range(batch_size):
            output[:, i, :] = self.softmax((Q[:, i, :] @ K[:, i, :].T) / n_features ** 0.5) @ V[:, i, :]

        return output


class AttentionRecurrentNetwork(torch.nn.Module):
    def __init__(self, context_size: Union[int, torch.Tensor], hidden_size: int) -> None:
        super(AttentionRecurrentNetwork, self).__init__()

        if isinstance(context_size, torch.Tensor):
            context_size = context_size.shape[2]

        self.model = torch.nn.Sequential(
            SelfAttention(context_size),
            torch.nn.LSTM(context_size, hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_output = self.model(x)
        output = lstm_output[1][0]
        output = output.reshape(output.shape[1], output.shape[2])
        return output
