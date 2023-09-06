#!/bin/bash
# This script sets the pythonpath appropriately
# Usage:
#   Replace: `python arg1 arg2 ...`
#   with   : `bash run-python.sh arg1 arg2 ...`

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DIR")"
LIB_DIR="$ROOT_DIR/pento_parser_poc"

export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

source "$ROOT_DIR/venv/bin/activate"

python -m build
