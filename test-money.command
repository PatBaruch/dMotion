#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/dmotion ]; then
    printf 'Run make setup in this folder first.\n'
    exit 1
fi
printf 'Sensitive money test: paper money at confidence 0.10. False alarms are possible.\n'
printf 'Hold a spread in view for 2 seconds. T tests sound, S saves a photo, Q quits.\n'
exec .venv/bin/dmotion run --prompt "paper money" --confidence 0.1 --device cpu "$@"
