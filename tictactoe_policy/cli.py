"""Command-line interface for training and evaluation."""

import argparse
import json

from .evaluation import evaluate_policy
from .game import MAX_BOARD_SIZE, MIN_BOARD_SIZE, GameConfig
from .policy import TicTacToePolicy
from .training import train_policy


def build_parser() -> argparse.ArgumentParser:
    """Build the discoverable command parser."""
    parser = argparse.ArgumentParser(prog="tictactoe-policy")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="generate data and train a policy")
    train.add_argument(
        "--board-size",
        type=int,
        choices=range(MIN_BOARD_SIZE, MAX_BOARD_SIZE + 1),
        required=True,
    )
    train.add_argument("--win-length", type=int)
    train.add_argument("--hidden-size", type=int, default=16)
    train.add_argument("--samples", type=int, default=10_000)
    train.add_argument("--search-depth", type=int, default=3)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-symmetry", action="store_true")
    train.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="evaluate a saved policy")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--samples", type=int, default=1_000)
    evaluate.add_argument("--search-depth", type=int, default=3)
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run the selected command."""
    args = build_parser().parse_args()
    if args.command == "train":
        config = GameConfig(args.board_size, args.win_length)
        result = train_policy(
            config=config,
            hidden_size=args.hidden_size,
            samples=args.samples,
            search_depth=args.search_depth,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            symmetry=not args.no_symmetry,
            output=args.output,
        )
        print(
            json.dumps(
                {
                    "examples": result.example_count,
                    "loss": result.final_loss,
                    "parameters": result.policy.parameter_count,
                },
                indent=2,
            )
        )
    else:
        report = evaluate_policy(
            TicTacToePolicy.load(args.model),
            samples=args.samples,
            search_depth=args.search_depth,
            games=args.games,
            seed=args.seed,
        )
        print(json.dumps(report.to_dict(), indent=2))
