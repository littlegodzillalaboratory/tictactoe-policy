"""Tests for configurable game rules."""

import unittest

from tictactoe_policy import GameConfig, TicTacToeGame


class TestGameConfig(unittest.TestCase):
    """Tests for game dimension configuration."""

    def test_valid_empty_boards(self):
        """Empty supported boards validate."""
        for size in (3, 4, 8):
            with self.subTest(size=size):
                game = TicTacToeGame(GameConfig(size))
                self.assertEqual(
                    game.validate_board([0] * (size * size)),
                    (0,) * (size * size),
                )

    def test_invalid_board_size(self):
        """Unsupported board sizes fail."""
        for size in (2, 9):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    GameConfig(size)

    def test_invalid_win_length(self):
        """Winning length must fit within the configured board."""
        for win_length in (0, 4):
            with self.subTest(win_length=win_length):
                with self.assertRaisesRegex(ValueError, "win_length"):
                    GameConfig(3, win_length)

    def test_configurable_shorter_win_length(self):
        """The line engine supports future shorter winning conditions."""
        game = TicTacToeGame(GameConfig(4, 3))
        self.assertEqual(game.winner([1, 1, 1, 0] + [0] * 12), 1)


class TestBoardValidation(unittest.TestCase):
    """Tests for board validation and turn inference."""

    def test_invalid_board_length_and_cell(self):
        """Dimensions and cell values are checked."""
        game = TicTacToeGame()
        with self.assertRaisesRegex(ValueError, "9 cells"):
            game.validate_board([0] * 8)
        with self.assertRaisesRegex(ValueError, "-1, 0, or 1"):
            game.validate_board([0] * 8 + [2])

    def test_unreachable_boards_are_rejected(self):
        """Turn counts and winning positions must describe reachable play."""
        cases = (
            ([1, 1, 0, 0, 0, 0, 0, 0, 0], "invalid number"),
            ([1, 1, 1, -1, -1, -1, 0, 0, 0], "both players"),
            ([1, 1, 1, -1, -1, 0, -1, 0, 0], "X win"),
            ([1, 1, 0, -1, -1, -1, 1, 0, 1], "O win"),
        )
        for board, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TicTacToeGame().validate_board(board)

    def test_validation_can_skip_reachability_checks(self):
        """Callers can validate representation without validating game history."""
        board = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(
            TicTacToeGame().validate_board(board, check_reachable=False),
            tuple(board),
        )

    def test_legal_moves_and_turn_inference(self):
        """Moves and current player derive from board contents."""
        game = TicTacToeGame()
        board = [1, -1, 0, 0, 1, 0, 0, 0, 0]
        self.assertEqual(game.infer_player(board), -1)
        self.assertEqual(game.legal_moves(board), [2, 3, 5, 6, 7, 8])

    def test_full_board_is_terminal_without_a_winner(self):
        """A filled draw is terminal and has no winning player."""
        board = (1, -1, 1, 1, -1, -1, -1, 1, 1)
        game = TicTacToeGame()
        self.assertEqual(game.winner(board), 0)
        self.assertTrue(game.is_terminal(board))
        with self.assertRaisesRegex(ValueError, "terminal"):
            game.infer_player(board)


class TestGameRules(unittest.TestCase):
    """Tests for winning lines and move application."""

    def test_winner_detection_for_every_size_and_direction(self):
        """All supported dimensions use generated lines."""
        for size in range(3, 9):
            indices_by_kind = {
                "horizontal": range(size),
                "vertical": range(0, size * size, size),
                "main": range(0, size * size, size + 1),
                "anti": range(size - 1, size * size - 1, size - 1),
            }
            for kind, indices in indices_by_kind.items():
                with self.subTest(size=size, kind=kind):
                    board = [0] * (size * size)
                    for index in indices:
                        board[index] = 1
                    game = TicTacToeGame(GameConfig(size))
                    self.assertEqual(game.winner(board), 1)
                    self.assertTrue(game.is_terminal(board))

    def test_apply_move_rejects_invalid_move_or_player(self):
        """Moves require a valid player and an empty in-range cell."""
        for move, player in ((-1, 1), (9, 1), (0, 0), (0, 2)):
            with self.subTest(move=move, player=player):
                with self.assertRaises(ValueError):
                    TicTacToeGame.apply_move((0,) * 9, move, player)

    def test_apply_move_rejects_occupied_cell(self):
        """A player cannot overwrite an occupied cell."""
        with self.assertRaisesRegex(ValueError, "empty cell"):
            TicTacToeGame.apply_move((1,) + (0,) * 8, 0, -1)


if __name__ == "__main__":
    unittest.main()
