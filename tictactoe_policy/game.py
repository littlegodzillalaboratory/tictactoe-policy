"""Board configuration and rules for configurable Tic-Tac-Toe."""

from dataclasses import dataclass
from typing import Iterable, Sequence

#: Smallest supported board width and height.
MIN_BOARD_SIZE = 3
#: Largest supported board width and height.
MAX_BOARD_SIZE = 8


@dataclass(frozen=True)
class GameConfig:
    """Immutable game dimensions.

    :param board_size: Board width and height, from 3 through 8.
    :param win_length: Consecutive marks required to win. Defaults to the board
        size when omitted.
    :raises ValueError: If either dimension is outside its valid range.
    :ivar board_size: Configured board width and height.
    :ivar win_length: Configured number of consecutive marks needed to win.
    """

    #: Board width and height.
    board_size: int = 3
    #: Consecutive marks required to win.
    win_length: int | None = None

    def __post_init__(self) -> None:
        if not MIN_BOARD_SIZE <= self.board_size <= MAX_BOARD_SIZE:
            raise ValueError(
                f"board_size must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}"
            )
        if self.win_length is None:
            object.__setattr__(self, "win_length", self.board_size)
        if not 1 <= self.win_length <= self.board_size:
            raise ValueError("win_length must be between 1 and board_size")

    @property
    def cell_count(self) -> int:
        """Return the total number of board cells.

        :returns: ``board_size`` squared.
        """
        return self.board_size * self.board_size


class TicTacToeGame:
    """Validate boards and implement configurable Tic-Tac-Toe rules.

    :param config: Game dimensions. Standard 3x3 rules are used when omitted.
    :ivar config: Dimensions used by this game.
    :ivar lines: Every winning line expressed as row-major cell indices.
    """

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.lines = tuple(self._winning_lines())

    def _winning_lines(self) -> Iterable[tuple[int, ...]]:
        size = self.config.board_size
        length = self.config.win_length
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for row in range(size):
            for column in range(size):
                for delta_row, delta_column in directions:
                    end_row = row + (length - 1) * delta_row
                    end_column = column + (length - 1) * delta_column
                    if 0 <= end_row < size and 0 <= end_column < size:
                        yield tuple(
                            (row + step * delta_row) * size
                            + column
                            + step * delta_column
                            for step in range(length)
                        )

    def validate_board(
        self, board: Sequence[int], check_reachable: bool = True
    ) -> tuple[int, ...]:
        """Validate a board and return an immutable representation.

        :param board: Board cells in row-major order, using ``1`` for X,
            ``-1`` for O, and ``0`` for an empty cell.
        :param check_reachable: Validate turn counts and winner consistency.
        :returns: The validated board as a tuple.
        :raises ValueError: If the dimensions, cells, turn counts, or winners
            are invalid.
        """
        if len(board) != self.config.cell_count:
            raise ValueError(
                f"board must contain {self.config.cell_count} cells, got {len(board)}"
            )
        result = tuple(board)
        if any(cell not in (-1, 0, 1) for cell in result):
            raise ValueError("board cells must be -1, 0, or 1")
        if check_reachable:
            x_count = result.count(1)
            o_count = result.count(-1)
            if x_count not in (o_count, o_count + 1):
                raise ValueError("board has an invalid number of X and O marks")
            winners = self.winners(result)
            if len(winners) > 1:
                raise ValueError("board cannot contain wins for both players")
            if winners == {1} and x_count != o_count + 1:
                raise ValueError("X win has an invalid turn count")
            if winners == {-1} and x_count != o_count:
                raise ValueError("O win has an invalid turn count")
        return result

    def winners(self, board: Sequence[int]) -> set[int]:
        """Find every player with a completed winning line.

        :param board: Board cells in row-major order.
        :returns: Set containing ``1`` and/or ``-1`` for winning players.
        """
        return {
            board[line[0]]
            for line in self.lines
            if board[line[0]] != 0
            and all(board[index] == board[line[0]] for index in line)
        }

    def winner(self, board: Sequence[int]) -> int:
        """Return the sole winner of a board.

        :param board: Board cells in row-major order.
        :returns: ``1`` for X, ``-1`` for O, or ``0`` when there is no sole
            winner.
        """
        winners = self.winners(board)
        return next(iter(winners)) if len(winners) == 1 else 0

    def legal_moves(self, board: Sequence[int]) -> list[int]:
        """List the unoccupied cells on a board.

        :param board: Board cells in row-major order.
        :returns: Legal move indices in ascending order.
        """
        return [index for index, cell in enumerate(board) if cell == 0]

    def is_terminal(self, board: Sequence[int]) -> bool:
        """Determine whether a board is won or full.

        :param board: Board cells in row-major order.
        :returns: True when play cannot continue.
        """
        return bool(self.winners(board)) or 0 not in board

    def infer_player(self, board: Sequence[int]) -> int:
        """Infer whose turn follows an X-first reachable board.

        :param board: Non-terminal board cells in row-major order.
        :returns: ``1`` when X moves next or ``-1`` when O moves next.
        :raises ValueError: If the board is invalid or terminal.
        """
        valid = self.validate_board(board)
        if self.is_terminal(valid):
            raise ValueError("cannot infer a player for a terminal board")
        return 1 if valid.count(1) == valid.count(-1) else -1

    @staticmethod
    def apply_move(board: Sequence[int], move: int, player: int) -> tuple[int, ...]:
        """Apply one legal move without modifying the input board.

        :param board: Board cells in row-major order.
        :param move: Index of the empty cell to occupy.
        :param player: Player mark, either ``1`` or ``-1``.
        :returns: A new immutable board containing the move.
        :raises ValueError: If the player or move is invalid.
        """
        if player not in (-1, 1):
            raise ValueError("player must be -1 or 1")
        if move < 0 or move >= len(board) or board[move] != 0:
            raise ValueError("move must identify an empty cell")
        result = list(board)
        result[move] = player
        return tuple(result)
