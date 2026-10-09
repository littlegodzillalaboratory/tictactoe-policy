"""Tests for training and evaluation interfaces."""

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
