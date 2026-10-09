#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${EXAMPLES_DIR}/.." && pwd)"
MODEL_DIR="$(mktemp -d)"
MODEL_PATH="${MODEL_DIR}/tictactoe-4x4.pt"

cleanup() {
  rm -rf "${MODEL_DIR}"
}
trap cleanup EXIT

cd "${PROJECT_DIR}"
. ./.venv/bin/activate

tictactoe-policy train \
  --board-size 4 \
  --hidden-size 4 \
  --samples 8 \
  --search-depth 1 \
  --epochs 1 \
  --seed 42 \
  --output "${MODEL_PATH}"

tictactoe-policy evaluate \
  --model "${MODEL_PATH}" \
  --samples 4 \
  --search-depth 1 \
  --games 1 \
  --seed 42
