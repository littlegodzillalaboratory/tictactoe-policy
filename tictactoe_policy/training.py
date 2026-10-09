"""Supervised policy training."""

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import Example, exact_examples, sampled_examples
from .game import GameConfig, TicTacToeGame
from .policy import TicTacToePolicy


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run.

    :ivar policy: Trained policy in evaluation mode.
    :ivar example_count: Number of distinct examples used for training.
    :ivar final_loss: Loss recorded after the final epoch.
    """

    #: Trained policy in evaluation mode.
    policy: TicTacToePolicy
    #: Number of distinct examples used for training.
    example_count: int
    #: Loss recorded after the final epoch.
    final_loss: float


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch random generators.

    :param seed: Seed applied to both generators.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def train_policy(
    config: GameConfig,
    hidden_size: int = 16,
    samples: int = 10_000,
    search_depth: int = 3,
    epochs: int = 50,
    learning_rate: float = 0.01,
    seed: int = 42,
    symmetry: bool = True,
    output: str | Path | None = None,
) -> (
    TrainingResult
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Generate teacher data, train a policy, and optionally save it.

    Exact minimax examples are used for standard 3x3 games. Other game
    configurations use sampled positions labelled by bounded search.

    :param config: Dimensions and winning condition for the policy.
    :param hidden_size: Number of neurons in the network's hidden layer.
    :param samples: Number of sampled examples for non-standard games.
    :param search_depth: Teacher depth for non-standard games.
    :param epochs: Number of full optimization passes.
    :param learning_rate: Adam optimizer learning rate.
    :param seed: Seed controlling sampling and weight initialization.
    :param symmetry: Augment examples with rotations and reflections.
    :param output: Optional destination for the trained ``.pt`` checkpoint.
    :returns: Trained policy and summary metrics.
    :raises ValueError: If ``epochs`` or another training setting is invalid.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    seed_everything(seed)
    game = TicTacToeGame(config)
    examples = (
        exact_examples(use_symmetry=symmetry)
        if config.board_size == 3 and config.win_length == 3
        else sampled_examples(game, samples, search_depth, seed, symmetry)
    )
    policy = TicTacToePolicy(config, hidden_size)
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=learning_rate)
    inputs, targets = _tensors(examples, config.cell_count)
    final_loss = 0.0
    policy.network.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = policy.network(inputs)
        loss = -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    policy.network.eval()
    if output is not None:
        policy.save(output)
    return TrainingResult(policy, len(examples), final_loss)


def _tensors(
    examples: list[Example], cell_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([board for board, _ in examples], dtype=torch.float32)
    targets = torch.zeros((len(examples), cell_count), dtype=torch.float32)
    for row, (_, moves) in enumerate(examples):
        probability = 1.0 / len(moves)
        for move in moves:
            targets[row, move] = probability
    return inputs, targets
