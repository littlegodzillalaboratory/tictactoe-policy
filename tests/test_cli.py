"""Tests for command-line argument validation."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tictactoe_policy import cli
from tictactoe_policy.cli import build_parser


@pytest.mark.parametrize("size", [2, 9])
def test_train_rejects_unsupported_board_size(size):
    """CLI parsing rejects dimensions outside the supported range."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--board-size", str(size), "--output", "policy.pt"])


@pytest.mark.parametrize("size", range(3, 9))
def test_train_accepts_supported_board_size(size):
    """CLI parsing accepts every supported board dimension."""
    args = build_parser().parse_args(
        ["train", "--board-size", str(size), "--output", "policy.pt"]
    )
    assert args.board_size == size


@patch("tictactoe_policy.cli.logger")
@patch("tictactoe_policy.cli.train_policy")
@patch("tictactoe_policy.cli.build_parser")
def test_main_dispatches_train(mock_parser, mock_train, mock_logger):
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
def test_main_dispatches_evaluate(mock_parser, mock_load, mock_evaluate, mock_logger):
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
