"""Train, save, load, and use a policy through the Python API."""

from pathlib import Path
from tempfile import TemporaryDirectory

from tictactoe_policy import GameConfig, TicTacToePolicy
from tictactoe_policy.evaluation import evaluate_policy
from tictactoe_policy.training import train_policy


config = GameConfig(board_size=4)

with TemporaryDirectory() as temp_directory:
    model_path = Path(temp_directory) / "tictactoe-4x4.pt"

    result = train_policy(
        config=config,
        hidden_size=4,
        samples=8,
        search_depth=1,
        epochs=1,
        seed=42,
        output=model_path,
    )

    policy = TicTacToePolicy.load(model_path)
    board = [
        1,
        -1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    move = policy.choose_move(board)
    report = evaluate_policy(
        policy,
        samples=4,
        search_depth=1,
        games=1,
        seed=42,
    )

    assert board[move] == 0
    print("board size:", policy.board_size)
    print("win length:", policy.win_length)
    print("parameters:", policy.parameter_count)
    print("training examples:", result.example_count)
    print("chosen legal move:", move)
    print("teacher-move agreement:", report.move_agreement)
