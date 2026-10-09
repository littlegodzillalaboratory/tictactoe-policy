"""Bounded alpha-beta teacher for larger boards."""

from math import inf
from typing import Sequence

from .game import TicTacToeGame


class BoundedSearch:
    """Deterministic depth-limited negamax search with an open-line heuristic."""

    def __init__(self, game: TicTacToeGame, max_depth: int = 3) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.game = game
        self.max_depth = max_depth

    def _heuristic(self, board: Sequence[int], player: int) -> float:
        score = 0.0
        win_length = self.game.config.win_length
        for line in self.game.lines:
            values = [board[index] * player for index in line]
            own = values.count(1)
            opponent = values.count(-1)
            if not opponent:
                score += (own + 1) ** 2 / (win_length * win_length)
            if not own:
                score -= (opponent + 1) ** 2 / (win_length * win_length)
        return score

    def _search(
        self,
        board: tuple[int, ...],
        player: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        winner = self.game.winner(board)
        if winner:
            return 10_000.0 + depth if winner == player else -10_000.0 - depth
        moves = self.game.legal_moves(board)
        if not moves:
            return 0.0
        if depth == 0:
            return self._heuristic(board, player)
        value = -inf
        for move in moves:
            child = self.game.apply_move(board, move, player)
            value = max(value, -self._search(child, -player, depth - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def best_moves(self, board: Sequence[int], player: int | None = None) -> set[int]:
        """Return all moves tied for the best bounded-search score."""
        valid = self.game.validate_board(board)
        if self.game.is_terminal(valid):
            return set()
        current = player or self.game.infer_player(valid)
        scores = {}
        for move in self.game.legal_moves(valid):
            child = self.game.apply_move(valid, move, current)
            scores[move] = -self._search(child, -current, self.max_depth - 1, -inf, inf)
        best = max(scores.values())
        return {move for move, score in scores.items() if score == best}

    def score(self, board: Sequence[int], player: int | None = None) -> float:
        """Return the bounded-search value of a position."""
        valid = self.game.validate_board(board)
        current = player or self.game.infer_player(valid)
        return self._search(valid, current, self.max_depth, -inf, inf)
