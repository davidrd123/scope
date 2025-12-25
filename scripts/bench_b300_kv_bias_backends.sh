#!/usr/bin/env bash
set -euo pipefail

# Quick A/B benchmark for B300 KV-bias backends (flash vs triton vs fa4).
#
# This script is intended to be safe for shared repos:
# - Uses an isolated env by default (does not touch `.venv`)
# - Writes artifacts under `outputs/`
#
# Usage:
#   scripts/bench_b300_kv_bias_backends.sh
#
# Env overrides:
#   B300_ENV_DIR=...                (defaults to .venv-b300-cu130-decode)
#   BACKENDS="flash fa4 triton"     (space-separated; defaults to "flash fa4")
#   HEIGHT=... WIDTH=...            (defaults to 320x576)
#   ITERS=... SKIP=...              (defaults to 6 iters, skip 2)
#   QUANTIZATION=fp8_e4m3fn|none    (defaults to none; Daydream GUI typically runs unquantized on B300)
#   KV_CACHE_ATTENTION_BIAS=...     (defaults to 0.3)
#   PROFILE_ATTENTION=0|1           (defaults to 0; set 1 to print profiler report)
#
# Notes:
# - `SCOPE_KV_BIAS_BACKEND` is read at import time, so we launch a new python process per backend.
# - B300 needs a ptxas that knows sm_103 for Triton/Inductor. Prefer CUDA 12.9+.

ENV_DIR="${B300_ENV_DIR:-.venv-b300-cu130-decode}"
PY="$ENV_DIR/bin/python"

BACKENDS="${BACKENDS:-flash fa4}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-576}"
ITERS="${ITERS:-6}"
SKIP="${SKIP:-2}"
QUANTIZATION="${QUANTIZATION:-none}"
KV_CACHE_ATTENTION_BIAS="${KV_CACHE_ATTENTION_BIAS:-0.3}"
PROFILE_ATTENTION="${PROFILE_ATTENTION:-0}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected $PY (is the env created?)" >&2
  echo "Create/fix the env with:" >&2
  echo "  scripts/setup_b300_cu130_env.sh" >&2
  echo "  # or:" >&2
  echo "  scripts/b300_env_fix_cu130.sh $ENV_DIR" >&2
  exit 1
fi

mkdir -p outputs

if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
  for candidate in \
    /usr/local/cuda-13.1/bin/ptxas \
    /usr/local/cuda-13.0/bin/ptxas \
    /usr/local/cuda-12.9/bin/ptxas \
    /usr/local/cuda/bin/ptxas \
    /usr/local/cuda-12.8/bin/ptxas
  do
    if [[ -x "$candidate" ]]; then
      TRITON_PTXAS_PATH="$candidate"
      export TRITON_PTXAS_PATH
      break
    fi
  done
fi

export DISABLE_FLEX_ATTENTION_COMPILE="${DISABLE_FLEX_ATTENTION_COMPILE:-1}"
export WANVAE_STREAM_DECODE_MODE="${WANVAE_STREAM_DECODE_MODE:-chunk}"

if ! PYTHONPATH=src "$PY" - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"ERROR: failed to import torch: {e}", file=sys.stderr)
    raise SystemExit(2)

try:
    ok = torch.cuda.is_available()
except Exception as e:
    print(f"ERROR: torch.cuda.is_available() raised: {e}", file=sys.stderr)
    raise SystemExit(2)

if not ok:
    print("ERROR: CUDA not available (check `nvidia-smi` and/or CUDA init errors).", file=sys.stderr)
    raise SystemExit(2)

try:
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"CUDA device: {name} (cc={cc})")
except Exception:
    pass
PY
then
  echo "Aborting benchmark: CUDA unavailable in $ENV_DIR" >&2
  exit 1
fi

for backend in $BACKENDS; do
  echo "============================================================"
  echo "B300 KV-bias backend: $backend"
  echo "============================================================"

  export SCOPE_KV_BIAS_BACKEND="$backend"
  export PROFILE_ATTENTION="$PROFILE_ATTENTION"

  out_prefix="outputs/b300_cu130_${QUANTIZATION}_bias${KV_CACHE_ATTENTION_BIAS}_kvbias_${backend}"
  PYTHONPATH=src "$PY" scripts/profile_krea_pipeline_blocks.py \
    --height "$HEIGHT" --width "$WIDTH" \
    --iters "$ITERS" --skip "$SKIP" \
    --quantization "$QUANTIZATION" \
    --kv-cache-attention-bias "$KV_CACHE_ATTENTION_BIAS" \
    --cudnn-benchmark \
    --json "${out_prefix}.json" \
    2>&1 | tee "${out_prefix}.log"
done
