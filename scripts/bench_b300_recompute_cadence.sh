#!/usr/bin/env bash
set -euo pipefail

# Benchmark the impact of skipping `recompute_kv_cache` on B300 (SM103).
#
# This script is intended to be safe for shared repos:
# - Uses an isolated env by default (does not touch `.venv`)
# - Writes artifacts under `outputs/`
#
# Usage:
#   scripts/bench_b300_recompute_cadence.sh
#
# Env overrides:
#   B300_ENV_DIR=...                   (defaults to .venv-b300-cu130-decode)
#   CADENCES="1 2 3"                   (space-separated; defaults to "1 2 3")
#   SCOPE_KV_BIAS_BACKEND=fa4|flash    (defaults to fa4)
#   HEIGHT=... WIDTH=...               (defaults to 320x576)
#   ITERS=... SKIP=...                 (defaults to 6 iters, skip 2)
#   QUANTIZATION=fp8_e4m3fn|none       (defaults to none; Daydream GUI typically runs unquantized on B300)
#   KV_CACHE_ATTENTION_BIAS=...        (defaults to 0.3)
#
# Notes:
# - This is an algorithmic trade-off; validate quality before adopting.
# - On B300, `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` was observed to visibly glitch in Daydream.
# - On B300, `SCOPE_KV_BIAS_BACKEND=triton` is unusably slow.

ENV_DIR="${B300_ENV_DIR:-.venv-b300-cu130-decode}"
PY="$ENV_DIR/bin/python"

CADENCES="${CADENCES:-1 2 3}"
SCOPE_KV_BIAS_BACKEND="${SCOPE_KV_BIAS_BACKEND:-fa4}"

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-576}"
ITERS="${ITERS:-6}"
SKIP="${SKIP:-2}"
QUANTIZATION="${QUANTIZATION:-none}"
KV_CACHE_ATTENTION_BIAS="${KV_CACHE_ATTENTION_BIAS:-0.3}"

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

export SCOPE_KV_BIAS_BACKEND

for cadence in $CADENCES; do
  echo "============================================================"
  echo "B300 recompute cadence: every=$cadence blocks (backend=$SCOPE_KV_BIAS_BACKEND)"
  echo "============================================================"

  export SCOPE_KV_CACHE_RECOMPUTE_EVERY="$cadence"

  out_prefix="outputs/b300_cu130_${QUANTIZATION}_bias${KV_CACHE_ATTENTION_BIAS}_recompute_every${cadence}"
  PYTHONPATH=src "$PY" scripts/profile_krea_pipeline_blocks.py \
    --height "$HEIGHT" --width "$WIDTH" \
    --iters "$ITERS" --skip "$SKIP" \
    --quantization "$QUANTIZATION" \
    --kv-cache-attention-bias "$KV_CACHE_ATTENTION_BIAS" \
    --cudnn-benchmark \
    --json "${out_prefix}.json" \
    2>&1 | tee "${out_prefix}.log"
done
