"""Tests for training and evaluation interfaces."""

import unittest
from unittest.mock import patch

from tictactoe_policy import GameConfig, TicTacToePolicy
from tictactoe_policy.evaluation import evaluate_policy
from tictactoe_policy.training import train_policy


class TestTraining(unittest.TestCase):
    """Tests for sampled and exact training flows."""

    def test_small_sampled_training_run(self):
        """A sampled run trains a policy with the requested metadata."""
        result = train_policy(
            GameConfig(4), hidden_size=4, samples=8, search_depth=1, epochs=1
        )
        self.assertGreater(result.example_count, 0)
        self.assertEqual(result.policy.board_size, 4)
        self.assertEqual(result.policy.hidden_size, 4)

    def test_training_rejects_invalid_epoch_count(self):
        """Training requires at least one epoch."""
        with self.assertRaisesRegex(ValueError, "epochs"):
            train_policy(GameConfig(4), epochs=0)

    @patch("tictactoe_policy.training.TicTacToePolicy.save")
    def test_exact_training_can_save_output(self, mock_save):
        """Standard 3x3 training uses exact data and saves requested output."""
        result = train_policy(
            GameConfig(), hidden_size=2, epochs=1, symmetry=False, output="policy.pt"
        )
        self.assertEqual(result.example_count, 4520)
        self.assertGreater(result.final_loss, 0)
        mock_save.assert_called_once_with("policy.pt")


class TestEvaluation(unittest.TestCase):
    """Tests for sampled and exhaustive policy evaluation."""

    def test_sampled_evaluator_report(self):
        """Larger-board reports are explicitly sampled and reproducible."""
        report = evaluate_policy(
            TicTacToePolicy(GameConfig(4), 2),
            samples=5,
            search_depth=1,
            games=1,
        )
        self.assertGreater(report.positions_evaluated, 0)
        self.assertFalse(report.exhaustive)
        self.assertEqual(report.opponent, "bounded-search teacher")
        self.assertEqual(report.agreement_metric, "teacher-move agreement")
        self.assertEqual(sum(report.games_vs_random_as_x.values()), 1)
        self.assertEqual(sum(report.games_vs_teacher_as_o.values()), 1)
        self.assertEqual(report.seed, 42)

    def test_exact_evaluator_report_covers_all_positions(self):
        """Standard 3x3 evaluation reports exhaustive minimax metrics."""
        report = evaluate_policy(TicTacToePolicy(GameConfig(), 2), games=1)
        self.assertEqual(report.positions_evaluated, 4520)
        self.assertTrue(report.exhaustive)
        self.assertEqual(report.opponent, "perfect minimax")
        self.assertEqual(report.agreement_metric, "optimal-move accuracy")
        self.assertEqual(sum(report.games_vs_random_as_o.values()), 1)
        self.assertEqual(sum(report.games_vs_teacher_as_x.values()), 1)
        self.assertEqual(report.to_dict()["board_size"], 3)


if __name__ == "__main__":
    unittest.main()
