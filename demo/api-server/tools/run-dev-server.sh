#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DIR")"

source "$ROOT_DIR/venv/bin/activate"

cd "$ROOT_DIR/app"

flask run
