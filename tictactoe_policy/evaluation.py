"""Exact and sampled policy evaluation."""

import random
from dataclasses import asdict, dataclass

import torch

from .dataset import exact_examples, sampled_examples, tactical_accuracy
from .minimax import ExactMinimax
from .policy import TicTacToePolicy
from .search import BoundedSearch


# A report intentionally keeps its stable metrics together for serialization.
@dataclass(frozen=True)
class EvaluationReport:  # pylint: disable=too-many-instance-attributes
    """Serializable policy quality metrics."""

    board_size: int
    positions_evaluated: int
    agreement_metric: str
    move_agreement: float
    illegal_move_rate: float
    immediate_win_accuracy: float
    required_block_accuracy: float
    parameter_count: int
    games_vs_random_as_x: dict[str, int]
    games_vs_random_as_o: dict[str, int]
    games_vs_teacher_as_x: dict[str, int]
    games_vs_teacher_as_o: dict[str, int]
    opponent: str
    seed: int
    exhaustive: bool

    def to_dict(self) -> dict:
        """Return a JSON-ready representation."""
        return asdict(self)


def evaluate_policy(  # pylint: disable=too-many-locals
    policy: TicTacToePolicy,
    samples: int = 1_000,
    search_depth: int = 3,
    games: int = 20,
    seed: int = 42,
) -> EvaluationReport:
    """Evaluate exactly for 3x3 or on reproducible samples otherwise."""
    game = policy.game
    exhaustive = policy.board_size == 3 and policy.win_length == 3
    examples = (
        exact_examples()
        if exhaustive
        else sampled_examples(game, samples, search_depth, seed, False)
    )
    correct = illegal = 0
    for normalized, targets in examples:
        board = tuple(int(cell) for cell in normalized)
        move = _choose_normalized(policy, board)
        correct += move in targets
        illegal += board[move] != 0
    win_accuracy, block_accuracy = tactical_accuracy(
        game, examples, lambda board: _choose_normalized(policy, board)
    )
    if exhaustive:
        teacher = ExactMinimax()
        opponent = "perfect minimax"
    else:
        teacher = BoundedSearch(game, search_depth)
        opponent = "bounded-search teacher"
    randomizer = random.Random(seed)
    return EvaluationReport(
        board_size=policy.board_size,
        positions_evaluated=len(examples),
        agreement_metric=(
            "optimal-move accuracy" if exhaustive else "teacher-move agreement"
        ),
        move_agreement=correct / len(examples),
        illegal_move_rate=illegal / len(examples),
        immediate_win_accuracy=win_accuracy,
        required_block_accuracy=block_accuracy,
        parameter_count=policy.parameter_count,
        games_vs_random_as_x=_play_games(policy, None, 1, games, randomizer),
        games_vs_random_as_o=_play_games(policy, None, -1, games, randomizer),
        games_vs_teacher_as_x=_play_games(policy, teacher, 1, games, randomizer),
        games_vs_teacher_as_o=_play_games(policy, teacher, -1, games, randomizer),
        opponent=opponent,
        seed=seed,
        exhaustive=exhaustive,
    )


def _choose_normalized(policy: TicTacToePolicy, board: tuple[int, ...]) -> int:
    scores = policy.network(torch.tensor(board, dtype=torch.float32)).detach()
    scores[torch.tensor(board) != 0] = -torch.inf
    return int(scores.argmax())


def _play_games(policy, teacher, policy_player, count, randomizer):
    results = {"wins": 0, "draws": 0, "losses": 0}
    game = policy.game
    for _ in range(count):
        board = (0,) * game.config.cell_count
        player = 1
        while not game.is_terminal(board):
            if player == policy_player:
                move = policy.choose_move(board)
            elif teacher is None:
                move = randomizer.choice(game.legal_moves(board))
            else:
                move = randomizer.choice(sorted(teacher.best_moves(board, player)))
            board = game.apply_move(board, move, player)
            player = -player
        winner = game.winner(board)
        key = (
            "draws"
            if winner == 0
            else ("wins" if winner == policy_player else "losses")
        )
        results[key] += 1
    return results
