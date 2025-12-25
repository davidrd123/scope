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
# - The repo pins `torchao==0.13.0` (built against torch 2.8). For torch 2.9/cu130
#   we optionally upgrade torchao so its C++ extensions can load.

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
import torch

try:
    import triton
except Exception:
    triton = None

try:
    import flash_attn  # type: ignore
except Exception:
    flash_attn = None

try:
    import torchao  # type: ignore
except Exception:
    torchao = None

print("torch:", torch.__version__, "(cuda=", torch.version.cuda, ")", sep="")
print("triton:", getattr(triton, "__version__", None))
print("flash_attn:", getattr(flash_attn, "__version__", None))
print("torchao:", getattr(torchao, "__version__", None))
PY

echo "==> Restoring torch/torchvision to cu130..."
uv pip install -p "$PY" --upgrade --index-url https://download.pytorch.org/whl/cu130 \
  --force-reinstall \
  torch==2.9.0+cu130 torchvision==0.24.0+cu130

echo "==> Aligning torchao with torch (best-effort; override with TORCHAO_VERSION=... )..."
# Prefer the PyTorch cu130 index so we get a build aligned with torch==2.9.0+cu130.
# If that fails (e.g. no matching wheel), fall back to PyPI wheels.
TORCHAO_VERSION="${TORCHAO_VERSION:-0.15.0+cu130}"
if [[ "$TORCHAO_VERSION" != "skip" ]]; then
  set +e
  uv pip install -p "$PY" --upgrade --index-url https://download.pytorch.org/whl/cu130 \
    --force-reinstall \
    --no-deps \
    "torchao==${TORCHAO_VERSION}"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "WARN: torchao install from cu130 index failed; trying PyPI wheels..."
    set +e
    uv pip install -p "$PY" --upgrade \
      --force-reinstall \
      --no-deps \
      --only-binary=:all: \
      "torchao==${TORCHAO_VERSION%%+*}"
    status=$?
    set -e
    if [[ $status -ne 0 ]]; then
      echo "WARN: torchao wheel install failed (keeping existing torchao)."
      echo "      Set TORCHAO_VERSION=skip to suppress this step."
    fi
  fi
fi

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
try:
    import triton
except Exception:
    triton = None

try:
    import flash_attn  # type: ignore
except Exception:
    flash_attn = None

try:
    import torchao  # type: ignore
except Exception:
    torchao = None

print("torch:", torch.__version__, "(cuda=", torch.version.cuda, ")", sep="")
print("triton:", getattr(triton, "__version__", None))
print("flash_attn:", getattr(flash_attn, "__version__", None))
print("torchao:", getattr(torchao, "__version__", None))
PY

echo ""
echo "Next (B300 run):"
echo "  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \\"
echo "  DISABLE_FLEX_ATTENTION_COMPILE=1 \\"
echo "  WANVAE_STREAM_DECODE_MODE=chunk \\"
echo "  $ENV_DIR/bin/daydream-scope"
