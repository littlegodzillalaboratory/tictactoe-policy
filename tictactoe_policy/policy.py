"""Embeddable trained policy API."""

from pathlib import Path
from typing import Sequence

import torch

from .game import GameConfig, TicTacToeGame
from .model import PolicyNetwork


class TicTacToePolicy:
    """Wrap a policy network with validation, masking, and persistence.

    :param config: Game dimensions expected by the policy.
    :param hidden_size: Number of neurons in the network's hidden layer.
    :ivar config: Game dimensions expected by the policy.
    :ivar hidden_size: Number of neurons in the hidden layer.
    :ivar game: Rules used to validate boards and moves.
    :ivar network: Neural network used to score moves.
    """

    #: Version of the metadata-bearing checkpoint representation.
    FORMAT_VERSION = 1

    def __init__(self, config: GameConfig, hidden_size: int = 16) -> None:
        self.config = config
        self.hidden_size = hidden_size
        self.game = TicTacToeGame(config)
        self.network = PolicyNetwork(config.board_size, hidden_size)

    @property
    def board_size(self) -> int:
        """Return the configured board width.

        :returns: Number of rows and columns.
        """
        return self.config.board_size

    @property
    def win_length(self) -> int:
        """Return the configured winning line length.

        :returns: Consecutive marks required to win.
        """
        return self.config.win_length

    @property
    def parameter_count(self) -> int:
        """Return the network parameter count.

        :returns: Number of trainable scalar parameters.
        """
        return self.network.parameter_count

    def move_scores(self, board: Sequence[int]) -> list[float]:
        """Score every move from the current player's perspective.

        Occupied cells are assigned negative infinity so they cannot be chosen.

        :param board: Reachable, non-terminal board in row-major order.
        :returns: One score per board cell.
        :raises ValueError: If the board is invalid or terminal.
        """
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
        """Choose the highest-scoring legal move.

        :param board: Reachable, non-terminal board in row-major order.
        :returns: Row-major index of the selected move.
        :raises ValueError: If the board is invalid or terminal.
        """
        scores = self.move_scores(board)
        return int(max(range(len(scores)), key=scores.__getitem__))

    def save(self, path: str | Path) -> None:
        """Save the policy as a metadata-bearing PyTorch checkpoint.

        :param path: Destination ``.pt`` file.
        """
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
        """Load a policy from a metadata-bearing PyTorch checkpoint.

        :param path: Source ``.pt`` file.
        :returns: Reconstructed policy in evaluation mode.
        :raises ValueError: If required checkpoint metadata is missing or
            invalid.
        """
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
