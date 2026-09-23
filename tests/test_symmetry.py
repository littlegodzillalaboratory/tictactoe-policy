"""Tests for board symmetry transforms."""

from tictactoe_policy.symmetry import augment, transform_board, transform_move


def test_rotation_transforms_board_and_move_together():
    """A top-left mark rotates to top-right."""
    board = (1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert transform_move(0, 3, 1) == 2
    assert transform_board(board, 3, 1)[2] == 1


def test_augmentation_avoids_duplicates():
    """Symmetric examples are not duplicated."""
    assert len(augment((0,) * 9, {4}, 3)) == 1
