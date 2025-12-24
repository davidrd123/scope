#!/usr/bin/env bash
set -euo pipefail

# Create (or re-sync) an isolated B300/cu130 env without touching the shared `.venv`.
#
# This intentionally does:
#   1) `uv sync` (installs the project's pinned deps, including cu128 torch)
#   2) a cu130 override (reinstalls torch/torchvision + flash-attn)
#
# Usage:
#   scripts/setup_b300_cu130_env.sh [ENV_DIR]
#
# Default env: .venv-b300-cu130-decode (historical name from early experiments).

ENV_DIR="${1:-.venv-b300-cu130-decode}"

echo "==> Creating env (if missing): $ENV_DIR"
uv venv "$ENV_DIR" --python 3.12

echo "==> Syncing project deps into $ENV_DIR (safe: does NOT touch .venv)"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --frozen --no-dev

echo "==> Applying cu130 override (torch/triton/flash-attn)"
./scripts/b300_env_fix_cu130.sh "$ENV_DIR"

