#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
venv="${SWIGLU_VENV:-$HOME/.venvs/swiglu-fold}"
exec "$venv/bin/python" -u -m swiglu_fold.run "$@"
