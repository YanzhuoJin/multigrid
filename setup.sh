#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Use Python 3.10-3.12. Keep the environment outside the source tree.
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-../.venv-socialrl}"
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10 to 3.12 via PYTHON_BIN=..."'
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" -m pytest -q
