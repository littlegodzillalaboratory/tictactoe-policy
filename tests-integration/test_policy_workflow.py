"""End-to-end tests for the embeddable policy workflow."""

from pathlib import Path
import tempfile
import unittest

from tictactoe_policy import GameConfig, TicTacToePolicy
from tictactoe_policy.evaluation import evaluate_policy
from tictactoe_policy.training import train_policy


class TestPolicyWorkflow(unittest.TestCase):
    """Integration tests for the persisted policy API."""

    def test_train_save_load_choose_and_evaluate(self):
        """Train a small policy and exercise its persisted runtime API."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "tictactoe-4x4.pt"

            result = train_policy(
                config=GameConfig(4),
                hidden_size=4,
                samples=8,
                search_depth=1,
                epochs=1,
                seed=42,
                output=model_path,
            )

            self.assertTrue(model_path.is_file())
            self.assertGreater(result.example_count, 0)

            policy = TicTacToePolicy.load(model_path)
            board = [1, -1] + [0] * 14
            move = policy.choose_move(board)

            self.assertEqual(policy.board_size, 4)
            self.assertEqual(policy.win_length, 4)
            self.assertEqual(policy.hidden_size, 4)
            self.assertIn(move, range(2, 16))

            report = evaluate_policy(
                policy,
                samples=4,
                search_depth=1,
                games=1,
                seed=42,
            )

            self.assertEqual(report.board_size, 4)
            self.assertGreater(report.positions_evaluated, 0)
            self.assertEqual(report.illegal_move_rate, 0.0)
            self.assertFalse(report.exhaustive)


if __name__ == "__main__":
    unittest.main()
