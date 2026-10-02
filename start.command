#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/dmotion ]; then
    printf 'Run make setup in this folder first.\n'
    exit 1
fi
exec .venv/bin/dmotion run "$@"
