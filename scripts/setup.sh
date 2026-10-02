#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

if [ -n "${DMOTION_PYTHON:-}" ]; then
    task_python="$DMOTION_PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
    task_python="$(command -v python3.11)"
else
    task_python="$(command -v python3)"
fi

"$task_python" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Use Python 3.11, 3.12, or 3.13; set DMOTION_PYTHON to its path."'
if [ ! -x .tools/bin/uv ]; then
    "$task_python" -m venv .tools
    .tools/bin/python -m pip install --disable-pip-version-check uv==0.12.22
fi
.tools/bin/uv sync --locked --extra vision --python "$task_python" --cache-dir "$PWD/.cache/uv"
printf '\nSetup complete. Run make prepare, then make run.\n'
