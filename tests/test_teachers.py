"""Tests for exact and bounded teachers."""

import pytest

from tictactoe_policy import GameConfig, TicTacToeGame
from tictactoe_policy.dataset import immediate_moves, sampled_examples
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
