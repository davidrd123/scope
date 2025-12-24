#!/usr/bin/env bash
set -euo pipefail

# Fix a B300/cu130 virtualenv after it gets "clobbered" by `uv sync` (cu128 pins).
#
# Default env: .venv-b300-cu130-decode (historical name from early experiments).
#
# Usage:
#   scripts/b300_env_fix_cu130.sh [ENV_DIR]
#
# Notes:
# - This script does NOT touch the shared `.venv`.
# - It pins torch/torchvision to cu130, and reinstalls flash-attn without deps so
#   we don't accidentally downgrade torch to a cu12 wheel.

ENV_DIR="${1:-.venv-b300-cu130-decode}"
PY="$ENV_DIR/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected Python at: $PY" >&2
  echo "If you haven't created the env yet, do:" >&2
  echo "  uv venv $ENV_DIR --python 3.12" >&2
  echo "  UV_PROJECT_ENVIRONMENT=$ENV_DIR uv sync --frozen --no-dev" >&2
  exit 1
fi

echo "==> Using env: $ENV_DIR"
echo "==> Before:"
"$PY" - <<'PY'
import flash_attn
import torch

try:
    import triton
except Exception:
    triton = None

print("torch:", torch.__version__, "(cuda=", torch.version.cuda, ")", sep="")
print("triton:", getattr(triton, "__version__", None))
print("flash_attn:", getattr(flash_attn, "__version__", None))
PY

echo "==> Restoring torch/torchvision to cu130..."
uv pip install -p "$PY" --upgrade --index-url https://download.pytorch.org/whl/cu130 \
  --force-reinstall \
  torch==2.9.0+cu130 torchvision==0.24.0+cu130

echo "==> Ensuring build tools for flash-attn..."
uv pip install -p "$PY" wheel ninja packaging

echo "==> Reinstalling flash-attn (no deps; force source build)..."
uv pip install -p "$PY" --force-reinstall \
  --no-deps \
  --no-build-isolation \
  --no-binary flash-attn \
  flash-attn==2.8.3

echo "==> After:"
"$PY" - <<'PY'
import torch
import triton
import flash_attn

print("torch:", torch.__version__, "(cuda=", torch.version.cuda, ")", sep="")
print("triton:", triton.__version__)
print("flash_attn:", getattr(flash_attn, "__version__", None))
PY

echo ""
echo "Next (B300 run):"
echo "  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \\"
echo "  DISABLE_FLEX_ATTENTION_COMPILE=1 \\"
echo "  WANVAE_STREAM_DECODE_MODE=chunk \\"
echo "  $ENV_DIR/bin/daydream-scope"

