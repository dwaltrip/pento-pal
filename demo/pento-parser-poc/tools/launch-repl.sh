#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DIR")"
CODE_DIR="$ROOT_DIR/pento_parser_poc"

export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

source "$ROOT_DIR/venv/bin/activate"

ptpython --vi
