"""Tests for command-line argument validation."""

import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tictactoe_policy import cli
from tictactoe_policy.cli import build_parser
from tictactoe_policy.logger import init


class TestBuildParser(unittest.TestCase):
    """Tests for command-line argument parsing."""

    def test_train_rejects_unsupported_board_size(self):
        """CLI parsing rejects dimensions outside the supported range."""
        parser = build_parser()
        for size in (2, 9):
            with self.subTest(size=size):
                with self.assertRaises(SystemExit):
                    parser.parse_args(
                        ["train", "--board-size", str(size), "--output", "policy.pt"]
                    )

    def test_train_accepts_supported_board_size(self):
        """CLI parsing accepts every supported board dimension."""
        parser = build_parser()
        for size in range(3, 9):
            with self.subTest(size=size):
                args = parser.parse_args(
                    ["train", "--board-size", str(size), "--output", "policy.pt"]
                )
                self.assertEqual(args.board_size, size)


class TestMain(unittest.TestCase):
    """Tests for command dispatch."""

    @patch("tictactoe_policy.cli.logger")
    @patch("tictactoe_policy.cli.train_policy")
    @patch("tictactoe_policy.cli.build_parser")
    def test_main_dispatches_train(self, mock_parser, mock_train, mock_logger):
        """CLI dispatches parsed training arguments and logs the JSON result."""
        mock_parser.return_value.parse_args.return_value = SimpleNamespace(
            command="train",
            board_size=4,
            win_length=None,
            hidden_size=8,
            samples=20,
            search_depth=2,
            epochs=3,
            learning_rate=0.02,
            seed=7,
            no_symmetry=False,
            output="policy.pt",
        )
        mock_train.return_value = SimpleNamespace(
            example_count=20,
            final_loss=0.5,
            policy=SimpleNamespace(parameter_count=280),
        )

        cli.main()

        mock_train.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch("tictactoe_policy.cli.logger")
    @patch("tictactoe_policy.cli.evaluate_policy")
    @patch("tictactoe_policy.cli.TicTacToePolicy.load")
    @patch("tictactoe_policy.cli.build_parser")
    def test_main_dispatches_evaluate(
        self, mock_parser, mock_load, mock_evaluate, mock_logger
    ):
        """CLI loads and evaluates a model, then logs the JSON report."""
        mock_parser.return_value.parse_args.return_value = SimpleNamespace(
            command="evaluate",
            model="policy.pt",
            samples=20,
            search_depth=2,
            games=3,
            seed=7,
        )
        mock_evaluate.return_value.to_dict.return_value = {"board_size": 4}

        cli.main()

        mock_load.assert_called_once_with("policy.pt")
        mock_evaluate.assert_called_once()
        mock_logger.info.assert_called_once()


class TestEntryPoint(unittest.TestCase):
    """Tests for package module execution."""

    @patch("tictactoe_policy.cli.main")
    def test_module_entry_point_calls_main(self, mock_main):
        """Executing the package module invokes the command-line entry point."""
        runpy.run_module("tictactoe_policy.__main__", run_name="__main__")
        mock_main.assert_called_once_with()

    @patch("tictactoe_policy.cli.main")
    def test_module_entry_point_does_not_run_when_imported(self, mock_main):
        """Importing the entry-point module does not execute the CLI."""
        runpy.run_module("tictactoe_policy.__main__", run_name="imported_entry_point")
        mock_main.assert_not_called()


class TestLogger(unittest.TestCase):
    """Tests for command logger configuration."""

    @patch("tictactoe_policy.logger.Conflog")
    def test_logger_can_keep_default_output_stream(self, mock_conflog):
        """Logger initialization leaves handlers unchanged by default."""
        adapter = init("test.logger")
        self.assertEqual(adapter, mock_conflog.return_value.get_logger.return_value)
        mock_conflog.return_value.handlers.__iter__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
