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


@pytest.mark.parametrize("win_length", [0, 4])
def test_invalid_win_length(win_length):
    """Winning length must fit within the configured board."""
    with pytest.raises(ValueError, match="win_length"):
        GameConfig(3, win_length)


@pytest.mark.parametrize(
    ("board", "message"),
    [
        ([1, 1, 0, 0, 0, 0, 0, 0, 0], "invalid number"),
        ([1, 1, 1, -1, -1, -1, 0, 0, 0], "both players"),
        ([1, 1, 1, -1, -1, 0, -1, 0, 0], "X win"),
        ([1, 1, 0, -1, -1, -1, 1, 0, 1], "O win"),
    ],
)
def test_unreachable_boards_are_rejected(board, message):
    """Turn counts and winning positions must describe reachable play."""
    with pytest.raises(ValueError, match=message):
        TicTacToeGame().validate_board(board)


def test_validation_can_skip_reachability_checks():
    """Callers can validate representation without validating game history."""
    board = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert TicTacToeGame().validate_board(board, check_reachable=False) == tuple(board)


def test_full_board_is_terminal_without_a_winner():
    """A filled draw is terminal and has no winning player."""
    board = (1, -1, 1, 1, -1, -1, -1, 1, 1)
    game = TicTacToeGame()
    assert game.winner(board) == 0
    assert game.is_terminal(board)
    with pytest.raises(ValueError, match="terminal"):
        game.infer_player(board)


@pytest.mark.parametrize(
    ("move", "player"),
    [(-1, 1), (9, 1), (0, 0), (0, 2)],
)
def test_apply_move_rejects_invalid_move_or_player(move, player):
    """Moves require a valid player and an empty in-range cell."""
    with pytest.raises(ValueError):
        TicTacToeGame.apply_move((0,) * 9, move, player)


def test_apply_move_rejects_occupied_cell():
    """A player cannot overwrite an occupied cell."""
    with pytest.raises(ValueError, match="empty cell"):
        TicTacToeGame.apply_move((1,) + (0,) * 8, 0, -1)
