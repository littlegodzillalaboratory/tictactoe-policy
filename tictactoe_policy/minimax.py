"""Exact minimax teacher for standard 3x3 Tic-Tac-Toe."""

from functools import lru_cache
from typing import Sequence

from .game import GameConfig, TicTacToeGame


class ExactMinimax:
    """Solve standard 3x3 positions exactly."""

    def __init__(self) -> None:
        self.game = TicTacToeGame(GameConfig(3, 3))

    @lru_cache(maxsize=10_000)
    def _value(self, board: tuple[int, ...], player: int) -> int:
        winner = self.game.winner(board)
        if winner:
            magnitude = 100 + board.count(0)
            return magnitude if winner == player else -magnitude
        if not self.game.legal_moves(board):
            return 0
        return max(
            -self._value(self.game.apply_move(board, move, player), -player)
            for move in self.game.legal_moves(board)
        )

    def best_moves(self, board: Sequence[int], player: int | None = None) -> set[int]:
        """Return all mathematically optimal legal moves."""
        valid = self.game.validate_board(board)
        if self.game.is_terminal(valid):
            return set()
        current = player or self.game.infer_player(valid)
        scores = {
            move: -self._value(self.game.apply_move(valid, move, current), -current)
            for move in self.game.legal_moves(valid)
        }
        best = max(scores.values())
        return {move for move, score in scores.items() if score == best}

    def value(self, board: Sequence[int], player: int | None = None) -> int:
        """Return the exact outcome value, preferring quicker wins."""
        valid = self.game.validate_board(board)
        current = player or self.game.infer_player(valid)
        return self._value(valid, current)


def reachable_positions() -> list[tuple[tuple[int, ...], int]]:
    """Enumerate all distinct reachable, non-terminal standard positions."""
    game = TicTacToeGame()
    found: dict[tuple[int, ...], int] = {}

    def visit(board: tuple[int, ...], player: int) -> None:
        if game.is_terminal(board) or board in found:
            return
        found[board] = player
        for move in game.legal_moves(board):
            visit(game.apply_move(board, move, player), -player)

    visit((0,) * 9, 1)
    return list(found.items())
