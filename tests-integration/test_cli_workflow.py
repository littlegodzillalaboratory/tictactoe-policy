"""End-to-end tests for the installed command-line interface."""

import json
import subprocess
import sys


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the package entry point in a fresh Python process."""
    return subprocess.run(
        [sys.executable, "-m", "tictactoe_policy", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_train_and_evaluate(tmp_path):
    """Train and evaluate a persisted model through the real CLI boundary."""
    model_path = tmp_path / "cli-4x4.pt"

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

    assert model_path.is_file()
    assert training_summary["examples"] > 0
    assert training_summary["parameters"] == 148

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

    assert report["board_size"] == 4
    assert report["positions_evaluated"] > 0
    assert report["illegal_move_rate"] == 0.0
    assert report["agreement_metric"] == "teacher-move agreement"
    assert report["seed"] == 42
