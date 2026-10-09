"""Generated supervised training positions."""

import random
from collections.abc import Callable
from typing import Sequence, TypeAlias

from .game import TicTacToeGame
from .minimax import ExactMinimax, reachable_positions
from .search import BoundedSearch
from .symmetry import augment

Example = tuple[tuple[float, ...], set[int]]
MoveTeacher: TypeAlias = ExactMinimax | BoundedSearch


def immediate_moves(game: TicTacToeGame, board: Sequence[int], player: int) -> set[int]:
    """Find moves that immediately win for a player.

    :param game: Game rules used to evaluate candidate moves.
    :param board: Board cells in row-major order.
    :param player: Player mark, either ``1`` or ``-1``.
    :returns: Indices of all immediately winning legal moves.
    """
    return {
        move
        for move in game.legal_moves(board)
        if game.winner(game.apply_move(board, move, player)) == player
    }


def _target_moves(
    game: TicTacToeGame,
    board: Sequence[int],
    player: int,
    teacher: MoveTeacher,
) -> set[int]:
    wins = immediate_moves(game, board, player)
    if wins:
        return wins
    blocks = immediate_moves(game, board, -player)
    if blocks:
        return blocks
    return teacher.best_moves(board, player)


def exact_examples(use_symmetry: bool = False) -> list[Example]:
    """Generate exact training examples for standard 3x3 Tic-Tac-Toe.

    :param use_symmetry: Include rotated and reflected equivalents when true.
    :returns: Every distinct reachable non-terminal position paired with its
        mathematically optimal moves.
    """
    teacher = ExactMinimax()
    examples: list[Example] = []
    for board, player in reachable_positions():
        targets = teacher.best_moves(board, player)
        variants = augment(board, targets, 3) if use_symmetry else [(board, targets)]
        examples.extend(
            (tuple(float(cell * player) for cell in variant), moves)
            for variant, moves in variants
        )
    return _deduplicate(examples)


def sampled_examples(
    game: TicTacToeGame,
    sample_count: int,
    search_depth: int = 3,
    seed: int = 42,
    use_symmetry: bool = True,
) -> list[Example]:
    """Generate sampled positions and bounded-search targets.

    :param game: Game whose positions should be sampled.
    :param sample_count: Maximum number of examples to collect before
        deduplication.
    :param search_depth: Maximum teacher search depth.
    :param seed: Seed controlling reproducible game sampling.
    :param use_symmetry: Include rotated and reflected equivalents when true.
    :returns: Distinct normalized boards paired with preferred move indices.
    :raises ValueError: If ``sample_count`` is less than one.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    randomizer = random.Random(seed)
    teacher = BoundedSearch(game, search_depth)
    examples: list[Example] = []
    while len(examples) < sample_count:
        board = (0,) * game.config.cell_count
        player = 1
        while not game.is_terminal(board) and len(examples) < sample_count:
            targets = _target_moves(game, board, player, teacher)
            variants = (
                augment(board, targets, game.config.board_size)
                if use_symmetry
                else [(board, targets)]
            )
            for variant, moves in variants:
                examples.append(
                    (tuple(float(cell * player) for cell in variant), moves)
                )
                if len(examples) == sample_count:
                    break
            move = (
                randomizer.choice(sorted(targets))
                if randomizer.random() < 0.5
                else randomizer.choice(game.legal_moves(board))
            )
            board = game.apply_move(board, move, player)
            player = -player
    return _deduplicate(examples)


def _deduplicate(examples: list[Example]) -> list[Example]:
    merged: dict[tuple[float, ...], set[int]] = {}
    for board, moves in examples:
        merged.setdefault(board, set()).update(moves)
    return list(merged.items())


def tactical_accuracy(
    game: TicTacToeGame,
    examples: Sequence[Example],
    chooser: Callable[[Sequence[float]], int],
) -> tuple[float, float]:
    """Calculate tactical move accuracy over training examples.

    :param game: Game rules used to identify tactical positions.
    :param examples: Normalized boards and their target moves.
    :param chooser: Callable returning a move index for a normalized board.
    :returns: A pair containing immediate-win accuracy and required-block
        accuracy. A category with no examples has accuracy ``1.0``.
    """
    wins = blocks = win_correct = block_correct = 0
    for normalized, _ in examples:
        board = tuple(int(cell) for cell in normalized)
        winning = immediate_moves(game, board, 1)
        blocking = immediate_moves(game, board, -1) if not winning else set()
        chosen = chooser(normalized)
        if winning:
            wins += 1
            win_correct += chosen in winning
        elif blocking:
            blocks += 1
            block_correct += chosen in blocking
    return (
        win_correct / wins if wins else 1.0,
        block_correct / blocks if blocks else 1.0,
    )
