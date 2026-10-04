#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
workflow_hooks=$(git config --local --get core.hooksPath || true)
if [ -n "$workflow_hooks" ] && [ "$workflow_hooks" != ".githooks" ]; then
    echo "Existing core.hooksPath=$workflow_hooks; integrate the hook instead of replacing it." >&2
    exit 1
fi
if [ -z "$workflow_hooks" ] && [ -f "$(git rev-parse --git-path hooks/pre-push)" ]; then
    echo "An existing pre-push hook needs integration; it was left intact." >&2
    exit 1
fi
git config --local core.hooksPath .githooks
echo "Installed dMotion push checks for this repository and its worktrees."
echo "Future agents follow AGENTS.md and finish features without a separate reminder."
