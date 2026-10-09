"""Tests for training and evaluation interfaces."""

from unittest.mock import patch

import pytest

from tictactoe_policy import GameConfig, TicTacToePolicy
from tictactoe_policy.evaluation import evaluate_policy
from tictactoe_policy.training import train_policy


def test_small_sampled_training_run():
    """A sampled run trains a policy with the requested metadata."""
    result = train_policy(
        GameConfig(4), hidden_size=4, samples=8, search_depth=1, epochs=1
    )
    assert result.example_count > 0
    assert result.policy.board_size == 4
    assert result.policy.hidden_size == 4


def test_sampled_evaluator_report():
    """Larger-board reports are explicitly sampled and reproducible."""
    report = evaluate_policy(
        TicTacToePolicy(GameConfig(4), 2), samples=5, search_depth=1, games=1
    )
    assert report.positions_evaluated > 0
    assert report.exhaustive is False
    assert report.opponent == "bounded-search teacher"
    assert report.agreement_metric == "teacher-move agreement"
    assert sum(report.games_vs_random_as_x.values()) == 1
    assert sum(report.games_vs_teacher_as_o.values()) == 1
    assert report.seed == 42


def test_training_rejects_invalid_epoch_count():
    """Training requires at least one epoch."""
    with pytest.raises(ValueError, match="epochs"):
        train_policy(GameConfig(4), epochs=0)


@patch("tictactoe_policy.training.TicTacToePolicy.save")
def test_exact_training_can_save_output(mock_save):
    """Standard 3x3 training uses exact data and saves requested output."""
    result = train_policy(
        GameConfig(), hidden_size=2, epochs=1, symmetry=False, output="policy.pt"
    )
    assert result.example_count == 4520
    assert result.final_loss > 0
    mock_save.assert_called_once_with("policy.pt")


def test_exact_evaluator_report_covers_all_positions():
    """Standard 3x3 evaluation reports exhaustive minimax metrics."""
    report = evaluate_policy(TicTacToePolicy(GameConfig(), 2), games=1)
    assert report.positions_evaluated == 4520
    assert report.exhaustive is True
    assert report.opponent == "perfect minimax"
    assert report.agreement_metric == "optimal-move accuracy"
    assert sum(report.games_vs_random_as_o.values()) == 1
    assert sum(report.games_vs_teacher_as_x.values()) == 1
    assert report.to_dict()["board_size"] == 3
