"""Tests for exact and bounded teachers."""

import pytest

from tictactoe_policy import GameConfig, TicTacToeGame
from tictactoe_policy.dataset import (
    _target_moves,
    exact_examples,
    immediate_moves,
    sampled_examples,
    tactical_accuracy,
)
from tictactoe_policy.minimax import ExactMinimax, reachable_positions
from tictactoe_policy.search import BoundedSearch


def test_exact_minimax_forced_win_and_reachable_positions():
    """Exact minimax selects a forced win and enumerates the state space."""
    board = (1, -1, 0, 0, 1, 0, -1, 0, 0)
    assert ExactMinimax().best_moves(board) == {8}
    assert len(reachable_positions()) == 4520


def test_exact_minimax_preserves_multiple_best_moves():
    """Equivalent best moves are all retained."""
    assert ExactMinimax().best_moves((0,) * 9) == set(range(9))


@pytest.mark.parametrize("size", [4, 8])
def test_immediate_win_and_block_on_larger_boards(size):
    """Tactics are dimension-independent."""
    game = TicTacToeGame(GameConfig(size))
    board = [0] * (size * size)
    for column in range(size - 1):
        board[column] = 1
        board[size + column] = -1
    assert immediate_moves(game, board, 1) == {size - 1}
    assert immediate_moves(game, board, -1) == {2 * size - 1}
    assert BoundedSearch(game, 1).best_moves(board, 1) == {size - 1}


def test_bounded_search_blocks_immediate_loss():
    """A shallow teacher sees and blocks an opponent's next win."""
    game = TicTacToeGame(GameConfig(4))
    board = (-1, -1, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    assert BoundedSearch(game, 2).best_moves(board, 1) == {3}


def test_bounded_search_rejects_invalid_depth():
    """Bounded search requires a positive depth."""
    with pytest.raises(ValueError, match="max_depth"):
        BoundedSearch(TicTacToeGame(), 0)


def test_sampled_examples_reject_invalid_count():
    """Sample generation requires at least one requested position."""
    with pytest.raises(ValueError, match="sample_count"):
        sampled_examples(TicTacToeGame(GameConfig(4)), 0)


def test_target_moves_prioritize_wins_then_blocks():
    """Dataset targets prefer immediate tactics over teacher scoring."""
    game = TicTacToeGame()
    teacher = ExactMinimax()
    winning = (1, 1, 0, -1, -1, 0, 0, 0, 0)
    blocking = (-1, -1, 0, 1, 0, 0, 1, 0, 0)
    assert _target_moves(game, winning, 1, teacher) == {2}
    assert _target_moves(game, blocking, 1, teacher) == {2}


def test_exact_examples_with_symmetry_are_deduplicated():
    """Exact generation exercises augmentation and merges duplicate boards."""
    examples = exact_examples(use_symmetry=True)
    boards = [board for board, _ in examples]
    assert len(boards) == len(set(boards))
    assert len(examples) == 4520


def test_tactical_accuracy_counts_wins_blocks_and_mistakes():
    """Tactical metrics independently count immediate wins and blocks."""
    game = TicTacToeGame()
    examples = [
        ((1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0), {2}),
        ((-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0), {2}),
    ]
    choices = iter([2, 8])
    assert tactical_accuracy(game, examples, lambda _board: next(choices)) == (
        1.0,
        0.0,
    )


def test_minimax_terminal_moves_and_position_value():
    """Exact minimax handles terminal boards and exposes position values."""
    teacher = ExactMinimax()
    won = (1, 1, 1, -1, -1, 0, 0, 0, 0)
    assert teacher.best_moves(won) == set()
    assert teacher.value((1, -1, 0, 0, 1, 0, -1, 0, 0), 1) > 0


def test_bounded_search_terminal_and_draw_scores():
    """Bounded search handles terminal and filled positions."""
    search = BoundedSearch(TicTacToeGame(), 2)
    won = (1, 1, 1, -1, -1, 0, 0, 0, 0)
    draw = (1, -1, 1, 1, -1, -1, -1, 1, 1)
    assert search.best_moves(won) == set()
    assert search.score(won, 1) > 10_000
    assert search.score(won, -1) < -10_000
    assert search.score(draw, 1) == 0


def test_bounded_search_applies_alpha_beta_cutoff():
    """A closed search window stops evaluating sibling moves."""
    search = BoundedSearch(TicTacToeGame(), 1)
    result = search._search(  # pylint: disable=protected-access
        (0,) * 9, 1, 1, 0.0, 0.0
    )
    assert isinstance(result, float)
