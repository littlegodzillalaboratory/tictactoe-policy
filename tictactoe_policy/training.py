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
    """Summary of a completed training run."""

    policy: TicTacToePolicy
    example_count: int
    final_loss: float


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch generators."""
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
    """Generate teacher data, train a policy, and optionally save it."""
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
