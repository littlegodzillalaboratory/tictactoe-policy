"""Tests for network and runtime policy."""

import torch
import pytest

from tictactoe_policy import GameConfig, PolicyNetwork, TicTacToePolicy


def test_architecture_dimensions_and_parameter_count():
    """Network input/output sizes derive from the board."""
    network = PolicyNetwork(4, 8)
    assert network(torch.zeros(2, 16)).shape == (2, 16)
    assert network.parameter_count == 16 * 8 + 8 + 8 * 16 + 16


def test_illegal_move_masking():
    """Occupied moves can never be selected."""
    policy = TicTacToePolicy(GameConfig(3), 2)
    for parameter in policy.network.parameters():
        torch.nn.init.constant_(parameter, 0)
    board = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    assert policy.move_scores(board)[:2] == [-torch.inf, -torch.inf]
    assert policy.choose_move(board) == 2


def test_save_load_metadata_and_incompatible_board(tmp_path):
    """Checkpoint metadata reconstructs the architecture and enforces size."""
    path = tmp_path / "policy.pt"
    original = TicTacToePolicy(GameConfig(4), 7)
    original.save(path)
    loaded = TicTacToePolicy.load(path)
    assert (loaded.board_size, loaded.win_length, loaded.hidden_size) == (4, 4, 7)
    with pytest.raises(ValueError, match="16 cells"):
        loaded.choose_move([0] * 9)


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
