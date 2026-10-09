"""Tests for board symmetry transforms."""

import pytest

from tictactoe_policy.symmetry import augment, transform_board, transform_move


def test_rotation_transforms_board_and_move_together():
    """A top-left mark rotates to top-right."""
    board = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert transform_move(0, 3, 1) == 2
    assert transform_board(board, 3, 1)[2] == 1


def test_augmentation_avoids_duplicates():
    """Symmetric examples are not duplicated."""
    assert len(augment((0,) * 9, {4}, 3)) == 1


def test_transform_rejects_invalid_identifier():
    """Only the eight square symmetries are accepted."""
    with pytest.raises(ValueError, match="between 0 and 7"):
        transform_move(0, 3, 8)
