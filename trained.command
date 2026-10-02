#!/bin/zsh
cd "$(dirname "$0")" || exit 1
.venv/bin/dmotion run --mode trained
echo "Press Return to close."
read
