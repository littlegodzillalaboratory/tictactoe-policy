"""Train, save, load, and use a policy through the Python API."""

# The filename follows the repository's executable-example convention, and the
# workflow intentionally mirrors integration coverage for a copyable example.
# pylint: disable=invalid-name,duplicate-code

from pathlib import Path
from tempfile import TemporaryDirectory

from tictactoe_policy import GameConfig, TicTacToePolicy
from tictactoe_policy.evaluation import evaluate_policy
from tictactoe_policy.logger import init
from tictactoe_policy.training import train_policy

logger = init(__name__)


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
    logger.info("Board size: %s", policy.board_size)
    logger.info("Win length: %s", policy.win_length)
    logger.info("Parameters: %s", policy.parameter_count)
    logger.info("Training examples: %s", result.example_count)
    logger.info("Chosen legal move: %s", move)
    logger.info("Teacher-move agreement: %s", report.move_agreement)
