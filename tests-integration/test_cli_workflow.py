"""End-to-end tests for the installed command-line interface."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the package entry point in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-m", "tictactoe_policy", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


class TestCliWorkflow(unittest.TestCase):
    """Integration tests for training and evaluation through the CLI."""

    def test_cli_train_and_evaluate(self):
        """Train and evaluate a persisted model through the real CLI boundary."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "cli-4x4.pt"

            trained = _run_cli(
                "train",
                "--board-size",
                "4",
                "--hidden-size",
                "4",
                "--samples",
                "8",
                "--search-depth",
                "1",
                "--epochs",
                "1",
                "--seed",
                "42",
                "--output",
                str(model_path),
            )
            training_summary = json.loads(trained.stdout)

            self.assertTrue(model_path.is_file())
            self.assertGreater(training_summary["examples"], 0)
            self.assertEqual(training_summary["parameters"], 148)

            evaluated = _run_cli(
                "evaluate",
                "--model",
                str(model_path),
                "--samples",
                "4",
                "--search-depth",
                "1",
                "--games",
                "1",
                "--seed",
                "42",
            )
            report = json.loads(evaluated.stdout)

            self.assertEqual(report["board_size"], 4)
            self.assertGreater(report["positions_evaluated"], 0)
            self.assertEqual(report["illegal_move_rate"], 0.0)
            self.assertEqual(report["agreement_metric"], "teacher-move agreement")
            self.assertEqual(report["seed"], 42)


if __name__ == "__main__":
    unittest.main()
