"""Small feed-forward policy network."""

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """One-hidden-layer policy whose dimensions derive from board size."""

    def __init__(self, board_size: int = 3, hidden_size: int = 16) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        cell_count = board_size * board_size
        self.layers = nn.Sequential(
            nn.Linear(cell_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, cell_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate unnormalized move scores."""
        return self.layers(inputs)

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters())
