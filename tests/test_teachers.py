"""Tests for exact and bounded teachers."""

import unittest

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


class TestExactMinimax(unittest.TestCase):
    """Tests for the exact standard-board teacher."""

    def test_forced_win_and_reachable_positions(self):
        """Exact minimax selects a forced win and enumerates the state space."""
        board = (1, -1, 0, 0, 1, 0, -1, 0, 0)
        self.assertEqual(ExactMinimax().best_moves(board), {8})
        self.assertEqual(len(reachable_positions()), 4520)

    def test_preserves_multiple_best_moves(self):
        """Equivalent best moves are all retained."""
        self.assertEqual(ExactMinimax().best_moves((0,) * 9), set(range(9)))

    def test_terminal_moves_and_position_value(self):
        """Exact minimax handles terminal boards and exposes position values."""
        teacher = ExactMinimax()
        won = (1, 1, 1, -1, -1, 0, 0, 0, 0)
        self.assertEqual(teacher.best_moves(won), set())
        self.assertGreater(teacher.value((1, -1, 0, 0, 1, 0, -1, 0, 0), 1), 0)


class TestBoundedSearch(unittest.TestCase):
    """Tests for the bounded larger-board teacher."""

    def test_immediate_win_and_block_on_larger_boards(self):
        """Tactics are dimension-independent."""
        for size in (4, 8):
            with self.subTest(size=size):
                game = TicTacToeGame(GameConfig(size))
                board = [0] * (size * size)
                for column in range(size - 1):
                    board[column] = 1
                    board[size + column] = -1
                self.assertEqual(immediate_moves(game, board, 1), {size - 1})
                self.assertEqual(immediate_moves(game, board, -1), {2 * size - 1})
                self.assertEqual(
                    BoundedSearch(game, 1).best_moves(board, 1), {size - 1}
                )

    def test_blocks_immediate_loss(self):
        """A shallow teacher sees and blocks an opponent's next win."""
        game = TicTacToeGame(GameConfig(4))
        board = (-1, -1, -1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
        self.assertEqual(BoundedSearch(game, 2).best_moves(board, 1), {3})

    def test_rejects_invalid_depth(self):
        """Bounded search requires a positive depth."""
        with self.assertRaisesRegex(ValueError, "max_depth"):
            BoundedSearch(TicTacToeGame(), 0)

    def test_terminal_and_draw_scores(self):
        """Bounded search handles terminal and filled positions."""
        search = BoundedSearch(TicTacToeGame(), 2)
        won = (1, 1, 1, -1, -1, 0, 0, 0, 0)
        draw = (1, -1, 1, 1, -1, -1, -1, 1, 1)
        self.assertEqual(search.best_moves(won), set())
        self.assertGreater(search.score(won, 1), 10_000)
        self.assertLess(search.score(won, -1), -10_000)
        self.assertEqual(search.score(draw, 1), 0)

    def test_applies_alpha_beta_cutoff(self):
        """A closed search window stops evaluating sibling moves."""
        search = BoundedSearch(TicTacToeGame(), 1)
        result = search._search(  # pylint: disable=protected-access
            (0,) * 9, 1, 1, 0.0, 0.0
        )
        self.assertIsInstance(result, float)


class TestTrainingExamples(unittest.TestCase):
    """Tests for teacher-labelled training data."""

    def test_sampled_examples_reject_invalid_count(self):
        """Sample generation requires at least one requested position."""
        with self.assertRaisesRegex(ValueError, "sample_count"):
            sampled_examples(TicTacToeGame(GameConfig(4)), 0)

    def test_target_moves_prioritize_wins_then_blocks(self):
        """Dataset targets prefer immediate tactics over teacher scoring."""
        game = TicTacToeGame()
        teacher = ExactMinimax()
        winning = (1, 1, 0, -1, -1, 0, 0, 0, 0)
        blocking = (-1, -1, 0, 1, 0, 0, 1, 0, 0)
        self.assertEqual(_target_moves(game, winning, 1, teacher), {2})
        self.assertEqual(_target_moves(game, blocking, 1, teacher), {2})

    def test_exact_examples_with_symmetry_are_deduplicated(self):
        """Exact generation exercises augmentation and merges duplicate boards."""
        examples = exact_examples(use_symmetry=True)
        boards = [board for board, _ in examples]
        self.assertEqual(len(boards), len(set(boards)))
        self.assertEqual(len(examples), 4520)

    def test_tactical_accuracy_counts_wins_blocks_and_mistakes(self):
        """Tactical metrics independently count immediate wins and blocks."""
        game = TicTacToeGame()
        examples = [
            ((1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0), {2}),
            ((-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0), {2}),
        ]
        choices = iter([2, 8])
        self.assertEqual(
            tactical_accuracy(game, examples, lambda _board: next(choices)),
            (1.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
