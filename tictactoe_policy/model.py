"""Small feed-forward policy network."""

import torch
from torch import nn

from .game import GameConfig


class PolicyNetwork(nn.Module):
    """One-hidden-layer policy whose dimensions derive from board size.

    :param board_size: Board width and height, from 3 through 8.
    :param hidden_size: Number of neurons in the hidden layer.
    :raises ValueError: If the board or hidden-layer size is invalid.
    :ivar layers: Feed-forward network producing one score per board cell.
    """

    def __init__(self, board_size: int = 3, hidden_size: int = 16) -> None:
        super().__init__()
        GameConfig(board_size)
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        cell_count = board_size * board_size
        self.layers = nn.Sequential(
            nn.Linear(cell_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, cell_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calculate unnormalized move scores.

        :param inputs: Tensor whose final dimension contains normalized board
            cells.
        :returns: Tensor with one unnormalized score per board cell.
        """
        return self.layers(inputs)

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters.

        :returns: Total scalar parameter count across all layers.
        """
        return sum(parameter.numel() for parameter in self.parameters())
