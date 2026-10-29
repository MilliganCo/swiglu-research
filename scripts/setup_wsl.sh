#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/swiglu-uv-install.sh
    UV_NO_MODIFY_PATH=1 sh /tmp/swiglu-uv-install.sh
fi
uv_bin="$(command -v uv || printf '%s' "$HOME/.local/bin/uv")"
venv="${SWIGLU_VENV:-$HOME/.venvs/swiglu-fold}"
"$uv_bin" venv --python 3.12 "$venv" --allow-existing
"$uv_bin" pip sync --python "$venv/bin/python" requirements-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
"$uv_bin" pip install --python "$venv/bin/python" -e . --no-deps
"$venv/bin/python" -m pytest -q
