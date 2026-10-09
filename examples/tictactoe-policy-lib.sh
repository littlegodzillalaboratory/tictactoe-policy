#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${EXAMPLES_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
. ./.venv/bin/activate
cd "${EXAMPLES_DIR}"

python3 _tictactoe-policy.py
