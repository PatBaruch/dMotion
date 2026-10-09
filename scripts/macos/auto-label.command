#!/bin/zsh
cd "$(dirname "$0")/../.." || exit 1
make auto-label
echo "Press Return to close."
read
