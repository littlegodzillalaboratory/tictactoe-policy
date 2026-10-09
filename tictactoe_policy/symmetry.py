"""Dihedral symmetries for square boards and move indices."""

from typing import Sequence


def transform_move(move: int, size: int, transform: int) -> int:
    """Transform a move using a square-board rotation or reflection.

    :param move: Row-major move index.
    :param size: Board width and height.
    :param transform: Dihedral transform index from ``0`` through ``7``.
    :returns: Transformed row-major move index.
    :raises ValueError: If ``transform`` is outside the supported range.
    """
    if transform not in range(8):
        raise ValueError("transform must be between 0 and 7")
    row, column = divmod(move, size)
    if transform >= 4:
        column = size - 1 - column
    for _ in range(transform % 4):
        row, column = column, size - 1 - row
    return row * size + column


def transform_board(board: Sequence[int], size: int, transform: int) -> tuple[int, ...]:
    """Transform a board using the move-index mapping.

    :param board: Board cells in row-major order.
    :param size: Board width and height.
    :param transform: Dihedral transform index from ``0`` through ``7``.
    :returns: Transformed immutable board.
    :raises ValueError: If ``transform`` is outside the supported range.
    """
    result = [0] * len(board)
    for move, cell in enumerate(board):
        result[transform_move(move, size, transform)] = cell
    return tuple(result)


def augment(
    board: Sequence[int], moves: set[int], size: int
) -> list[tuple[tuple[int, ...], set[int]]]:
    """Generate unique symmetry-equivalent board and target pairs.

    :param board: Board cells in row-major order.
    :param moves: Target move indices associated with the board.
    :param size: Board width and height.
    :returns: Unique rotated and reflected board/move pairs.
    """
    unique: dict[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], set[int]]
    ] = {}
    for transform in range(8):
        new_board = transform_board(board, size, transform)
        new_moves = {transform_move(move, size, transform) for move in moves}
        unique[(new_board, tuple(sorted(new_moves)))] = (new_board, new_moves)
    return list(unique.values())
