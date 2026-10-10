"""Tests for network and runtime policy."""

import unittest
from unittest.mock import patch

import torch

from tictactoe_policy import GameConfig, PolicyNetwork, TicTacToePolicy


class TestPolicyNetwork(unittest.TestCase):
    """Tests for policy network construction and inference."""

    def test_architecture_dimensions_and_parameter_count(self):
        """Network input/output sizes derive from the board."""
        network = PolicyNetwork(4, 8)
        self.assertEqual(network(torch.zeros(2, 16)).shape, (2, 16))
        self.assertEqual(network.parameter_count, 16 * 8 + 8 + 8 * 16 + 16)

    def test_architecture_rejects_unsupported_board_size(self):
        """The low-level network enforces the supported board range."""
        for size in (2, 9):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "between 3 and 8"):
                    PolicyNetwork(size)

    def test_architecture_rejects_invalid_hidden_size(self):
        """The network requires at least one hidden unit."""
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            PolicyNetwork(3, 0)


class TestTicTacToePolicy(unittest.TestCase):
    """Tests for move scoring and checkpoint persistence."""

    def test_illegal_move_masking(self):
        """Occupied moves can never be selected."""
        policy = TicTacToePolicy(GameConfig(3), 2)
        for parameter in policy.network.parameters():
            torch.nn.init.constant_(parameter, 0)
        board = [1, -1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(policy.move_scores(board)[:2], [-torch.inf, -torch.inf])
        self.assertEqual(policy.choose_move(board), 2)

    def test_policy_metadata_and_incompatible_board(self):
        """Policy metadata exposes its architecture and enforces board size."""
        policy = TicTacToePolicy(GameConfig(4), 7)
        self.assertEqual(
            (policy.board_size, policy.win_length, policy.hidden_size),
            (4, 4, 7),
        )
        with self.assertRaisesRegex(ValueError, "16 cells"):
            policy.choose_move([0] * 9)

    def test_choose_move_forced_win_with_configured_scores(self):
        """Runtime API returns the winning legal square when scored highest."""
        policy = TicTacToePolicy(GameConfig(3), 1)
        for parameter in policy.network.parameters():
            torch.nn.init.constant_(parameter, 0)
        policy.network.layers[2].bias.data[8] = 2
        self.assertEqual(policy.choose_move([1, -1, 0, 0, 1, 0, -1, 0, 0]), 8)

    def test_terminal_board_rejected(self):
        """A policy does not move after the game ends."""
        policy = TicTacToePolicy(GameConfig(3))
        with self.assertRaisesRegex(ValueError, "terminal"):
            policy.choose_move([1, 1, 1, -1, -1, 0, 0, 0, 0])

    @patch("tictactoe_policy.policy.torch.save")
    def test_policy_save_includes_reconstruction_metadata(self, mock_save):
        """Saving delegates a complete metadata-bearing checkpoint to PyTorch."""
        policy = TicTacToePolicy(GameConfig(4, 3), 7)
        policy.save("policy.pt")
        checkpoint, path = mock_save.call_args.args
        self.assertEqual(path, "policy.pt")
        self.assertEqual(checkpoint["format_version"], policy.FORMAT_VERSION)
        self.assertEqual(checkpoint["board_size"], 4)
        self.assertEqual(checkpoint["win_length"], 3)
        self.assertEqual(checkpoint["hidden_size"], 7)
        expected_state = policy.network.state_dict()
        self.assertEqual(checkpoint["state_dict"].keys(), expected_state.keys())
        for name, parameter in checkpoint["state_dict"].items():
            self.assertTrue(torch.equal(parameter, expected_state[name]))

    @patch("tictactoe_policy.policy.torch.load")
    def test_policy_load_reconstructs_checkpoint(self, mock_load):
        """Loading reconstructs the architecture and restores its parameters."""
        original = TicTacToePolicy(GameConfig(4, 3), 7)
        mock_load.return_value = {
            "state_dict": original.network.state_dict(),
            "board_size": 4,
            "win_length": 3,
            "hidden_size": 7,
        }
        loaded = TicTacToePolicy.load("policy.pt")
        mock_load.assert_called_once_with(
            "policy.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(
            (loaded.board_size, loaded.win_length, loaded.hidden_size),
            (4, 3, 7),
        )
        self.assertFalse(loaded.network.training)

    @patch("tictactoe_policy.policy.torch.load")
    def test_policy_load_rejects_incomplete_checkpoint(self, mock_load):
        """Checkpoint loading rejects non-dictionaries and missing metadata."""
        for checkpoint in (None, {}, {"state_dict": {}}):
            with self.subTest(checkpoint=checkpoint):
                mock_load.return_value = checkpoint
                with self.assertRaisesRegex(ValueError, "required metadata"):
                    TicTacToePolicy.load("policy.pt")


if __name__ == "__main__":
    unittest.main()
