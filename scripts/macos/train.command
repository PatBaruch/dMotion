#!/bin/zsh
cd "$(dirname "$0")/../.." || exit 1
.venv/bin/dmotion train
echo "Press Return to close."
read
