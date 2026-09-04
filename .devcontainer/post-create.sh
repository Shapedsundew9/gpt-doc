#!/usr/bin/env bash

set -euo pipefail

echo "--- Preparing workspace ---"
python3 -m venv --upgrade-deps .venv

if [[ -f requirements.txt ]]; then
    .venv/bin/python -m pip install -r requirements.txt
fi

if [[ -f python/pyproject.toml ]]; then
    .venv/bin/python -m pip install --editable python
fi

if ! command -v antigravity >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -fsSL https://antigravity.google/cli/install.sh | bash
fi