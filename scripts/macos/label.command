#!/bin/zsh
cd "$(dirname "$0")/../.." || exit 1
.venv/bin/dmotion label
