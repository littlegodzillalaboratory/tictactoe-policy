# TicTacToe Policy

`tictactoe-policy` is a deliberately small PyTorch policy for square
Tic-Tac-Toe boards from 3×3 through 8×8. By default an N×N game requires N
marks in a horizontal, vertical, or diagonal line. The internal `GameConfig`
also supports a shorter `win_length` for experiments.

## Board representation

Boards are flat sequences containing `1` for X, `-1` for O, and `0` for an
empty cell. Moves are zero-based offsets. A 3×3 board is indexed as follows:

```text
0 1 2
3 4 5
6 7 8
```

`choose_move()` validates reachable X-first play, infers the current player,
normalizes the input by that player, rejects terminal or wrongly sized boards,
and masks every occupied cell.

## Install and train

```shell
poetry install
poetry run tictactoe-policy train \
  --board-size 3 --hidden-size 16 --output tictactoe-3x3.pt
```

Standard 3×3 data contains every reachable non-terminal position and uses
exact minimax targets. It is therefore exactly solvable. Boards from 4×4 to
8×8 are not claimed to be solved: they use reproducible sampled games, tactical
targets, and a depth-limited alpha-beta teacher.

```shell
poetry run tictactoe-policy train \
  --board-size 6 --hidden-size 128 --samples 100000 \
  --search-depth 3 --seed 42 --output tictactoe-6x6.pt
```

Data augmentation applies the eight square rotations/reflections and transforms
target moves with their boards. Pass `--no-symmetry` to disable it. If several
moves share the best teacher score, all share the target probability.

## Embed a policy

```python
from tictactoe_policy import TicTacToePolicy

policy = TicTacToePolicy.load("tictactoe-3x3.pt")
move = policy.choose_move([
    1, -1, 0,
    0,  1, 0,
   -1,  0, 0,
])
assert move == 8  # assuming the trained policy learned this forced win
```

The same API applies to larger boards:

```python
policy = TicTacToePolicy.load("tictactoe-5x5.pt")
move = policy.choose_move([0] * 25)
```

`policy.move_scores(board)`, `policy.parameter_count`, `policy.board_size`, and
`policy.win_length` are also available. A model file is a plain dictionary with
`state_dict`, `board_size`, `win_length`, `hidden_size`, and `format_version`;
it does not pickle a policy object.

## Evaluate

```shell
poetry run tictactoe-policy evaluate --model tictactoe-3x3.pt
poetry run tictactoe-policy evaluate \
  --model tictactoe-6x6.pt --samples 2000 --search-depth 3 --seed 42
```

For 3×3, evaluation covers all legal non-terminal positions, reports exact
optimal-move agreement and tactical accuracy, and plays as X and O against
perfect minimax. For larger boards it reports sampled teacher-move agreement,
immediate-win/block accuracy, illegal-move rate, and games against the bounded
teacher. Larger-board results measure agreement with that limited teacher, not
mathematical optimality.

## Model sizes

With `C = board_size²` cells and `H = hidden_size`, the one-hidden-layer model
has `C×H + H + H×C + C` trainable parameters. Examples:

| Board | Hidden | Parameters |
|---|---:|---:|
| 3×3 | 16 | 313 |
| 5×5 | 64 | 3,289 |
| 8×8 | 128 | 16,576 |

Hidden sizes and sample counts are CLI parameters so later benchmarking can
compare sizes such as 2, 4, 8, 16, 32, 64, 128, and 256 without architecture
changes.
