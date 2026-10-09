"""Exact and sampled policy evaluation."""

import random
from dataclasses import asdict, dataclass
from typing import TypeAlias

import torch

from .dataset import exact_examples, sampled_examples, tactical_accuracy
from .minimax import ExactMinimax
from .policy import TicTacToePolicy
from .search import BoundedSearch

GameResults: TypeAlias = dict[str, int]
Teacher: TypeAlias = ExactMinimax | BoundedSearch


# A report intentionally keeps its stable metrics together for serialization.
@dataclass(frozen=True)
class EvaluationReport:  # pylint: disable=too-many-instance-attributes
    """Serializable policy quality metrics.

    :ivar board_size: Width and height of the evaluated board.
    :ivar positions_evaluated: Number of positions used for move metrics.
    :ivar agreement_metric: Description of the move-agreement measurement.
    :ivar move_agreement: Fraction of moves agreeing with the teacher.
    :ivar illegal_move_rate: Fraction of selected moves that were occupied.
    :ivar immediate_win_accuracy: Fraction of immediate wins selected.
    :ivar required_block_accuracy: Fraction of required blocks selected.
    :ivar parameter_count: Number of trainable policy parameters.
    :ivar games_vs_random_as_x: Outcomes when playing X against random moves.
    :ivar games_vs_random_as_o: Outcomes when playing O against random moves.
    :ivar games_vs_teacher_as_x: Outcomes when playing X against the teacher.
    :ivar games_vs_teacher_as_o: Outcomes when playing O against the teacher.
    :ivar opponent: Name of the teacher used for evaluation.
    :ivar seed: Seed used for reproducible sampling and opponents.
    :ivar exhaustive: Whether every reachable position was evaluated.
    """

    #: Width and height of the evaluated board.
    board_size: int
    #: Number of positions used for move metrics.
    positions_evaluated: int
    #: Description of the move-agreement measurement.
    agreement_metric: str
    #: Fraction of moves agreeing with the teacher.
    move_agreement: float
    #: Fraction of selected moves that were occupied.
    illegal_move_rate: float
    #: Fraction of immediate wins selected.
    immediate_win_accuracy: float
    #: Fraction of required blocks selected.
    required_block_accuracy: float
    #: Number of trainable policy parameters.
    parameter_count: int
    #: Outcomes when playing X against random moves.
    games_vs_random_as_x: dict[str, int]
    #: Outcomes when playing O against random moves.
    games_vs_random_as_o: dict[str, int]
    #: Outcomes when playing X against the teacher.
    games_vs_teacher_as_x: dict[str, int]
    #: Outcomes when playing O against the teacher.
    games_vs_teacher_as_o: dict[str, int]
    #: Name of the teacher used for evaluation.
    opponent: str
    #: Seed used for reproducible sampling and opponents.
    seed: int
    #: Whether every reachable position was evaluated.
    exhaustive: bool

    def to_dict(self) -> dict:
        """Convert the report to a JSON-serializable dictionary.

        :returns: Report fields and their values.
        """
        return asdict(self)


def evaluate_policy(  # pylint: disable=too-many-locals
    policy: TicTacToePolicy,
    samples: int = 1_000,
    search_depth: int = 3,
    games: int = 20,
    seed: int = 42,
) -> EvaluationReport:
    """Evaluate a policy against teacher moves and simulated opponents.

    Standard 3x3 policies are evaluated over every reachable position against
    exact minimax. Larger variants use reproducible samples and bounded search.

    :param policy: Trained policy to evaluate.
    :param samples: Number of sampled positions for non-standard games.
    :param search_depth: Bounded-search depth for non-standard games.
    :param games: Games to play for each opponent and player assignment.
    :param seed: Seed controlling sampling and random opponent moves.
    :returns: Complete evaluation metrics.
    """
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


def _play_games(
    policy: TicTacToePolicy,
    teacher: Teacher | None,
    policy_player: int,
    count: int,
    randomizer: random.Random,
) -> GameResults:
    """Play games against a random or teacher opponent and tally outcomes."""
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
