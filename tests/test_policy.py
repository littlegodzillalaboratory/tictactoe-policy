"""Tests for network and runtime policy."""

from unittest.mock import patch

import pytest
import torch

from tictactoe_policy import GameConfig, PolicyNetwork, TicTacToePolicy


def test_architecture_dimensions_and_parameter_count():
    """Network input/output sizes derive from the board."""
    network = PolicyNetwork(4, 8)
    assert network(torch.zeros(2, 16)).shape == (2, 16)
    assert network.parameter_count == 16 * 8 + 8 + 8 * 16 + 16


@pytest.mark.parametrize("size", [2, 9])
def test_architecture_rejects_unsupported_board_size(size):
    """The low-level network enforces the supported board range."""
    with pytest.raises(ValueError, match="between 3 and 8"):
        PolicyNetwork(size)


def test_architecture_rejects_invalid_hidden_size():
    """The network requires at least one hidden unit."""
    with pytest.raises(ValueError, match="hidden_size"):
        PolicyNetwork(3, 0)


def test_illegal_move_masking():
    """Occupied moves can never be selected."""
    policy = TicTacToePolicy(GameConfig(3), 2)
    for parameter in policy.network.parameters():
        torch.nn.init.constant_(parameter, 0)
    board = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    assert policy.move_scores(board)[:2] == [-torch.inf, -torch.inf]
    assert policy.choose_move(board) == 2


def test_policy_metadata_and_incompatible_board():
    """Policy metadata exposes its architecture and enforces board size."""
    policy = TicTacToePolicy(GameConfig(4), 7)
    assert (policy.board_size, policy.win_length, policy.hidden_size) == (4, 4, 7)
    with pytest.raises(ValueError, match="16 cells"):
        policy.choose_move([0] * 9)


def test_choose_move_forced_win_with_configured_scores():
    """Runtime API returns the winning legal square when scored highest."""
    policy = TicTacToePolicy(GameConfig(3), 1)
    for parameter in policy.network.parameters():
        torch.nn.init.constant_(parameter, 0)
    policy.network.layers[2].bias.data[8] = 2
    assert policy.choose_move([1, -1, 0, 0, 1, 0, -1, 0, 0]) == 8


def test_terminal_board_rejected():
    """A policy does not move after the game ends."""
    policy = TicTacToePolicy(GameConfig(3))
    with pytest.raises(ValueError, match="terminal"):
        policy.choose_move([1, 1, 1, -1, -1, 0, 0, 0, 0])


@patch("tictactoe_policy.policy.torch.save")
def test_policy_save_includes_reconstruction_metadata(mock_save):
    """Saving delegates a complete metadata-bearing checkpoint to PyTorch."""
    policy = TicTacToePolicy(GameConfig(4, 3), 7)
    policy.save("policy.pt")
    checkpoint, path = mock_save.call_args.args
    assert path == "policy.pt"
    assert checkpoint["format_version"] == policy.FORMAT_VERSION
    assert checkpoint["board_size"] == 4
    assert checkpoint["win_length"] == 3
    assert checkpoint["hidden_size"] == 7
    expected_state = policy.network.state_dict()
    assert checkpoint["state_dict"].keys() == expected_state.keys()
    for name, parameter in checkpoint["state_dict"].items():
        assert torch.equal(parameter, expected_state[name])


@patch("tictactoe_policy.policy.torch.load")
def test_policy_load_reconstructs_checkpoint(mock_load):
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
    assert (loaded.board_size, loaded.win_length, loaded.hidden_size) == (4, 3, 7)
    assert loaded.network.training is False


@pytest.mark.parametrize("checkpoint", [None, {}, {"state_dict": {}}])
@patch("tictactoe_policy.policy.torch.load")
def test_policy_load_rejects_incomplete_checkpoint(mock_load, checkpoint):
    """Checkpoint loading rejects non-dictionaries and missing metadata."""
    mock_load.return_value = checkpoint
    with pytest.raises(ValueError, match="required metadata"):
        TicTacToePolicy.load("policy.pt")
