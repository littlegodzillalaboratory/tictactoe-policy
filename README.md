<!-- BEGIN:AVATAR -->
![Avatar](avatar.jpg)
<!-- END:AVATAR -->

<!-- BEGIN:BADGES -->
[![Build Status](https://github.com/littlegodzillalaboratory/tictactoe-policy/workflows/CI/badge.svg)](https://github.com/littlegodzillalaboratory/tictactoe-policy/actions?query=workflow%3ACI)
[![Code Scanning Status](https://github.com/littlegodzillalaboratory/tictactoe-policy/workflows/CodeQL/badge.svg)](https://github.com/littlegodzillalaboratory/tictactoe-policy/actions?query=workflow%3ACodeQL)
[![Dependencies Status](https://img.shields.io/librariesio/release/pypi/tictactoe-policy)](https://libraries.io/github/littlegodzillalaboratory/tictactoe-policy)
[![Security Status](https://snyk.io/test/github/littlegodzillalaboratory/tictactoe-policy/badge.svg)](https://snyk.io/test/github/littlegodzillalaboratory/tictactoe-policy)
[![Published Version](https://img.shields.io/pypi/v/tictactoe-policy.svg)](https://pypi.python.org/pypi/tictactoe-policy)
<!-- END:BADGES -->

# TicTacToe Policy

TicTacToe Policy is a deliberately small PyTorch policy for square Tic-Tac-Toe
boards from 3×3 through 8×8. It can generate its own teacher data, train and
evaluate a neural policy, persist model metadata, and choose legal moves when
embedded in another Python application.

For an N×N board, the default winning condition is N identical marks in a
horizontal, vertical, or diagonal line. A shorter `win_length` can also be
configured for experiments.

## Installation

```shell
pip3 install tictactoe-policy
```

For development from a checkout:

```shell
poetry install
```

## Usage

Load a trained policy and choose a move:

```python
from tictactoe_policy import TicTacToePolicy

policy = TicTacToePolicy.load("tictactoe-3x3.pt")
move = policy.choose_move([
    1, -1, 0,
    0,  1, 0,
   -1,  0, 0,
])
```

`choose_move()` validates the board, infers whose turn it is, normalizes the
position to the current player's perspective, rejects terminal positions, and
masks occupied cells so they can never be selected.

The same API applies to larger boards:

```python
policy = TicTacToePolicy.load("tictactoe-5x5.pt")
move = policy.choose_move([0] * 25)
```

The policy also exposes `move_scores(board)`, `parameter_count`, `board_size`,
and `win_length`.

### Board Representation

Boards are flat sequences containing:

| Value | Meaning |
| ----- | ------- |
| `1` | X |
| `-1` | O |
| `0` | Empty cell |

Moves are zero-based offsets. A 3×3 board is indexed as follows:

```text
0 1 2
3 4 5
6 7 8
```

An N×N board contains `N * N` cells, with moves from `0` through
`N * N - 1`.

### Train a Policy

Train an exact 3×3 policy:

```shell
poetry run tictactoe-policy train \
  --board-size 3 \
  --hidden-size 16 \
  --seed 42 \
  --output tictactoe-3x3.pt
```

Train a sampled larger-board policy:

```shell
poetry run tictactoe-policy train \
  --board-size 6 \
  --hidden-size 128 \
  --samples 100000 \
  --search-depth 3 \
  --seed 42 \
  --output tictactoe-6x6.pt
```

Data augmentation applies the eight rotations and reflections of a square and
transforms target moves with their boards. Use `--no-symmetry` to disable it.
When multiple moves share the best teacher score, all share the target
probability.

### Evaluate a Policy

```shell
poetry run tictactoe-policy evaluate --model tictactoe-3x3.pt
```

For a larger model, evaluation parameters are configurable:

```shell
poetry run tictactoe-policy evaluate \
  --model tictactoe-6x6.pt \
  --samples 2000 \
  --search-depth 3 \
  --games 20 \
  --seed 42
```

## Configuration

Game dimensions are represented by `GameConfig`:

```python
from tictactoe_policy import GameConfig

standard_game = GameConfig(board_size=5)
experimental_game = GameConfig(board_size=5, win_length=4)
```

| Property | Description | Default | Valid values |
| -------- | ----------- | ------- | ------------ |
| `board_size` | Width and height of the square board | `3` | `3` through `8` |
| `win_length` | Identical marks required in a winning line | `board_size` | `1` through `board_size` |

Important training options include:

| Option | Description | Default |
| ------ | ----------- | ------- |
| `--hidden-size` | Hidden units in the feed-forward network | `16` |
| `--samples` | Requested sampled training positions for larger boards | `10000` |
| `--search-depth` | Bounded teacher search depth | `3` |
| `--epochs` | Supervised training epochs | `50` |
| `--learning-rate` | Adam optimizer learning rate | `0.01` |
| `--seed` | Python and PyTorch random seed | `42` |

## Training Strategies

### Exact 3×3 Training

Standard 3×3 Tic-Tac-Toe is solved exactly. Training data contains every
reachable non-terminal position, and exact minimax supplies every equally good
move as a target.

Evaluation covers every legal non-terminal position, reports exact
optimal-move and tactical accuracy, and plays as both X and O against perfect
minimax.

### Sampled 4×4–8×8 Training

Boards from 4×4 through 8×8 are too large for exhaustive enumeration. Their
positions are generated from reproducible sampled games and labelled by a
bounded alpha-beta teacher with immediate-win and required-block handling.

Evaluation uses a reproducible sampled set and reports teacher-move agreement,
immediate-win accuracy, required-block accuracy, illegal-move rate, and games
against random and bounded-search opponents.

These larger games are not claimed to be mathematically solved. Their metrics
measure agreement with the configured bounded teacher.

## Model Architecture

The policy is a one-hidden-layer feed-forward network:

```text
board_size² inputs → hidden_size → ReLU → board_size² outputs
```

With `C = board_size²` cells and `H = hidden_size`, the model contains
`C×H + H + H×C + C` trainable parameters.

| Board | Hidden units | Parameters |
| ----- | -----------: | ---------: |
| 3×3 | 16 | 313 |
| 5×5 | 64 | 3,289 |
| 8×8 | 128 | 16,576 |

Board size, hidden size, sample count, search depth, and evaluation settings
are configurable so model-size experiments can compare hidden sizes such as
2, 4, 8, 16, 32, 64, 128, and 256.

## Model Files

Model files are PyTorch dictionaries rather than pickled policy objects. Each
file contains:

```python
{
    "format_version": 1,
    "state_dict": ...,
    "board_size": 3,
    "win_length": 3,
    "hidden_size": 16,
}
```

This metadata allows `TicTacToePolicy.load()` to reconstruct the correct
network automatically. A model rejects boards whose dimensions do not match
the dimensions used during training.

## Examples

Runnable examples are available in the [`examples`](examples) directory:

```shell
bash examples/tictactoe-policy-lib.sh
bash examples/tictactoe-policy-cli.sh
```

The library example demonstrates training, saving, loading, choosing a move,
and evaluation. The CLI example demonstrates the `train` and `evaluate`
commands with a temporary model file.

## Colophon

<!-- BEGIN:DEVELOPERS_GUIDE -->
[Developer's Guide](https://cliffano.github.io/developers-guide-python.html)
<!-- END:DEVELOPERS_GUIDE -->

<!-- BEGIN:BUILD_REPORTS -->
Build reports:

* [Lint report](https://littlegodzillalaboratory.github.io/tictactoe-policy/lint/pylint/index.html)
* [Code complexity report](https://littlegodzillalaboratory.github.io/tictactoe-policy/complexity/radon/index.html)
* [Unit tests report](https://littlegodzillalaboratory.github.io/tictactoe-policy/test/pytest/index.html)
* [Test coverage report](https://littlegodzillalaboratory.github.io/tictactoe-policy/coverage/coverage/index.html)
* [Integration tests report](https://littlegodzillalaboratory.github.io/tictactoe-policy/test-integration/pytest/index.html)
* [API Documentation](https://littlegodzillalaboratory.github.io/tictactoe-policy/doc/sphinx/index.html)

<!-- END:BUILD_REPORTS -->
