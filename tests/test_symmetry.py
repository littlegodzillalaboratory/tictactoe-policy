"""Tests for board symmetry transforms."""

import unittest

from tictactoe_policy.symmetry import augment, transform_board, transform_move


class TestSymmetry(unittest.TestCase):
    """Tests for square-board rotations and reflections."""

    def test_rotation_transforms_board_and_move_together(self):
        """A top-left mark rotates to top-right."""
        board = (1, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(transform_move(0, 3, 1), 2)
        self.assertEqual(transform_board(board, 3, 1)[2], 1)

    def test_augmentation_avoids_duplicates(self):
        """Symmetric examples are not duplicated."""
        self.assertEqual(len(augment((0,) * 9, {4}, 3)), 1)

    def test_transform_rejects_invalid_identifier(self):
        """Only the eight square symmetries are accepted."""
        with self.assertRaisesRegex(ValueError, "between 0 and 7"):
            transform_move(0, 3, 8)


if __name__ == "__main__":
    unittest.main()
