"""Tests for configurable game rules."""

import pytest

from tictactoe_policy import GameConfig, TicTacToeGame


@pytest.mark.parametrize("size", [3, 4, 8])
def test_valid_empty_boards(size):
    """Empty supported boards validate."""
    game = TicTacToeGame(GameConfig(size))
    assert game.validate_board([0] * (size * size)) == (0,) * (size * size)


@pytest.mark.parametrize("size", [2, 9])
def test_invalid_board_size(size):
    """Unsupported board sizes fail."""
    with pytest.raises(ValueError):
        GameConfig(size)


def test_invalid_board_length_and_cell():
    """Dimensions and cell values are checked."""
    game = TicTacToeGame()
    with pytest.raises(ValueError, match="9 cells"):
        game.validate_board([0] * 8)
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        game.validate_board([0] * 8 + [2])


@pytest.mark.parametrize("size", range(3, 9))
@pytest.mark.parametrize("kind", ["horizontal", "vertical", "main", "anti"])
def test_winner_detection_for_every_size_and_direction(size, kind):
    """All supported dimensions use generated lines."""
    board = [0] * (size * size)
    indices = {
        "horizontal": range(size),
        "vertical": range(0, size * size, size),
        "main": range(0, size * size, size + 1),
        "anti": range(size - 1, size * size - 1, size - 1),
    }[kind]
    for index in indices:
        board[index] = 1
    game = TicTacToeGame(GameConfig(size))
    assert game.winner(board) == 1
    assert game.is_terminal(board)


def test_legal_moves_and_turn_inference():
    """Moves and current player derive from board contents."""
    game = TicTacToeGame()
    board = [1, -1, 0, 0, 1, 0, 0, 0, 0]
    assert game.infer_player(board) == -1
    assert game.legal_moves(board) == [2, 3, 5, 6, 7, 8]


def test_configurable_shorter_win_length():
    """The line engine supports future shorter winning conditions."""
    game = TicTacToeGame(GameConfig(4, 3))
    assert game.winner([1, 1, 1, 0] + [0] * 12) == 1
