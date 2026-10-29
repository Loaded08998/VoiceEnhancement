#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-./.uv-cache}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/models/hf}"
export TORCH_HOME="${TORCH_HOME:-$SCRIPT_DIR/models/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRIPT_DIR/models/cache}"
export ASTEROID_CACHE="${ASTEROID_CACHE:-$SCRIPT_DIR/models/asteroid}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

exec "$SCRIPT_DIR/envs/eval/bin/python" "$SCRIPT_DIR/src/bench/pipeline.py" "$@"
