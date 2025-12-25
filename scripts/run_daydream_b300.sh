#!/usr/bin/env bash
set -euo pipefail

# Run Daydream (daydream-scope) on B300 using the isolated cu130 env.
#
# Usage:
#   scripts/run_daydream_b300.sh [daydream-scope args...]
#
# Env overrides:
#   B300_ENV_DIR=...   (defaults to .venv-b300-cu130-decode)
#   SCOPE_KV_BIAS_BACKEND=...  (defaults to fa4; falls back if unavailable)

ENV_DIR="${B300_ENV_DIR:-.venv-b300-cu130-decode}"
BIN="$ENV_DIR/bin/daydream-scope"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: Expected $BIN (is the env created + synced?)" >&2
  echo "Fix/create the env with:" >&2
  echo "  scripts/b300_env_fix_cu130.sh $ENV_DIR" >&2
  exit 1
fi

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

# Enable torch.compile for the diffusion blocks on B300 (opt-out via SCOPE_COMPILE_KREA_PIPELINE=0).
export SCOPE_COMPILE_KREA_PIPELINE="${SCOPE_COMPILE_KREA_PIPELINE:-1}"

# Best-known KV-bias backend on B300 is FA4/CuTe score_mod (falls back if unavailable).
export SCOPE_KV_BIAS_BACKEND="${SCOPE_KV_BIAS_BACKEND:-fa4}"

# Faster steady-state decode mode on B300.
export WANVAE_STREAM_DECODE_MODE="${WANVAE_STREAM_DECODE_MODE:-chunk}"

exec "$BIN" "$@"
