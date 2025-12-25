#!/usr/bin/env bash
set -euo pipefail

# One-shot capture of B300 denoise/decode profiling artifacts (JSON + log).
#
# This script is intended to be safe for shared repos:
# - Uses an isolated env by default (does not touch `.venv`)
# - Writes artifacts under `outputs/`
#
# Usage:
#   scripts/profile_b300_denoise_drilldown.sh
#
# Env overrides:
#   B300_ENV_DIR=...         (defaults to .venv-b300-cu130-decode)
#   OUT_PREFIX=...           (defaults to outputs/b300_cu130_${QUANTIZATION}_bias${KV_CACHE_ATTENTION_BIAS}_drilldown)
#   HEIGHT=... WIDTH=...     (defaults to 320x576)
#   ITERS=... SKIP=...       (defaults to 6 iters, skip 2)
#   QUANTIZATION=fp8_e4m3fn|none  (defaults to none; Daydream GUI typically runs unquantized on B300)
#   KV_CACHE_ATTENTION_BIAS=...   (defaults to 0.3)

ENV_DIR="${B300_ENV_DIR:-.venv-b300-cu130-decode}"
PY="$ENV_DIR/bin/python"

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-576}"
ITERS="${ITERS:-6}"
SKIP="${SKIP:-2}"
QUANTIZATION="${QUANTIZATION:-none}"
KV_CACHE_ATTENTION_BIAS="${KV_CACHE_ATTENTION_BIAS:-0.3}"

OUT_PREFIX_DEFAULT="outputs/b300_cu130_${QUANTIZATION}_bias${KV_CACHE_ATTENTION_BIAS}_drilldown"
OUT_PREFIX="${OUT_PREFIX:-$OUT_PREFIX_DEFAULT}"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected $PY (is the env created?)" >&2
  echo "Create/fix the env with:" >&2
  echo "  scripts/setup_b300_cu130_env.sh" >&2
  echo "  # or:" >&2
  echo "  scripts/b300_env_fix_cu130.sh $ENV_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_PREFIX")"

# B300 needs a ptxas that knows sm_103 for Triton/Inductor. Prefer CUDA 12.9+.
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

# torch 2.9 / triton 3.5 currently hard-aborts compiling flex_attention on SM103 (tcgen05 LLVM).
export DISABLE_FLEX_ATTENTION_COMPILE="${DISABLE_FLEX_ATTENTION_COMPILE:-1}"

# Faster steady-state decode mode on B300.
export WANVAE_STREAM_DECODE_MODE="${WANVAE_STREAM_DECODE_MODE:-chunk}"

export PROFILE_PIPELINE_BLOCKS=1
export PROFILE_PIPELINE_BLOCKS_JSON="${OUT_PREFIX}_blocks_profile.json"
export PROFILE_DENOISE_STEPS=1
export PROFILE_DENOISE_STEPS_JSON="${OUT_PREFIX}_denoise_steps.json"
export PROFILE_GENERATOR_STEPS=1
export PROFILE_GENERATOR_STEPS_JSON="${OUT_PREFIX}_generator_steps.json"
export PROFILE_WANVAE_DECODE=1
export PROFILE_WANVAE_DECODE_JSON="${OUT_PREFIX}_vae_decode.json"
export PROFILE_WANVAE_DECODE_INNER=1
export PROFILE_WANVAE_DECODE_INNER_JSON="${OUT_PREFIX}_vae_decode_inner.json"
export PROFILE_ATTENTION=1

PYTHONPATH=src "$PY" scripts/profile_krea_pipeline_blocks.py \
  --height "$HEIGHT" --width "$WIDTH" \
  --iters "$ITERS" --skip "$SKIP" \
  --quantization "$QUANTIZATION" \
  --kv-cache-attention-bias "$KV_CACHE_ATTENTION_BIAS" \
  --cudnn-benchmark \
  --json "${OUT_PREFIX}_perf.json" \
  2>&1 | tee "${OUT_PREFIX}_perf.log"
