"""Embeddable trained policy API."""

from pathlib import Path
from typing import Sequence

import torch

from .game import GameConfig, TicTacToeGame
from .model import PolicyNetwork


class TicTacToePolicy:
    """Wrap a policy network with validation, masking, and persistence."""

    FORMAT_VERSION = 1

    def __init__(self, config: GameConfig, hidden_size: int = 16) -> None:
        self.config = config
        self.hidden_size = hidden_size
        self.game = TicTacToeGame(config)
        self.network = PolicyNetwork(config.board_size, hidden_size)

    @property
    def board_size(self) -> int:
        """Return the configured board width."""
        return self.config.board_size

    @property
    def win_length(self) -> int:
        """Return the configured winning line length."""
        return self.config.win_length

    @property
    def parameter_count(self) -> int:
        """Return the network parameter count."""
        return self.network.parameter_count

    def move_scores(self, board: Sequence[int]) -> list[float]:
        """Return scores with occupied cells masked to negative infinity."""
        valid = self.game.validate_board(board)
        if self.game.is_terminal(valid):
            raise ValueError("cannot score moves on a terminal board")
        player = self.game.infer_player(valid)
        inputs = torch.tensor(
            [cell * player for cell in valid], dtype=torch.float32
        ).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            scores = self.network(inputs).squeeze(0)
        scores = scores.masked_fill(torch.tensor(valid) != 0, -torch.inf)
        return scores.tolist()

    def choose_move(self, board: Sequence[int]) -> int:
        """Return the highest-scoring legal move."""
        scores = self.move_scores(board)
        return int(max(range(len(scores)), key=scores.__getitem__))

    def save(self, path: str | Path) -> None:
        """Save weights and the metadata needed to reconstruct this policy."""
        torch.save(
            {
                "format_version": self.FORMAT_VERSION,
                "state_dict": self.network.state_dict(),
                "board_size": self.board_size,
                "win_length": self.win_length,
                "hidden_size": self.hidden_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TicTacToePolicy":
        """Load a policy from a metadata-bearing model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        required = {"state_dict", "board_size", "win_length", "hidden_size"}
        if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
            raise ValueError("model checkpoint is missing required metadata")
        policy = cls(
            GameConfig(checkpoint["board_size"], checkpoint["win_length"]),
            checkpoint["hidden_size"],
        )
        policy.network.load_state_dict(checkpoint["state_dict"])
        policy.network.eval()
        return policy
