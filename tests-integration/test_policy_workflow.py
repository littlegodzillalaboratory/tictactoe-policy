"""End-to-end tests for the embeddable policy workflow."""

from tictactoe_policy import GameConfig, TicTacToePolicy
from tictactoe_policy.evaluation import evaluate_policy
from tictactoe_policy.training import train_policy


def test_train_save_load_choose_and_evaluate(tmp_path):
    """Train a small policy and exercise its persisted runtime API."""
    model_path = tmp_path / "tictactoe-4x4.pt"

    result = train_policy(
        config=GameConfig(4),
        hidden_size=4,
        samples=8,
        search_depth=1,
        epochs=1,
        seed=42,
        output=model_path,
    )

    assert model_path.is_file()
    assert result.example_count > 0

    policy = TicTacToePolicy.load(model_path)
    board = [1, -1] + [0] * 14
    move = policy.choose_move(board)

    assert policy.board_size == 4
    assert policy.win_length == 4
    assert policy.hidden_size == 4
    assert move in range(2, 16)

    report = evaluate_policy(
        policy,
        samples=4,
        search_depth=1,
        games=1,
        seed=42,
    )

    assert report.board_size == 4
    assert report.positions_evaluated > 0
    assert report.illegal_move_rate == 0.0
    assert report.exhaustive is False
