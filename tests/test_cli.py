"""Tests for command-line argument validation."""

import pytest

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
