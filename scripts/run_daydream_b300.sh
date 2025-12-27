#!/usr/bin/env bash
set -euo pipefail

# Run Daydream (daydream-scope) on B300 using the isolated cu130 env.
#
# Usage:
#   scripts/run_daydream_b300.sh [--compile-fp8] [--allow-fp8] [daydream-scope args...]
#
# Env overrides:
#   B300_ENV_DIR=...   (defaults to .venv-b300-cu130-decode)
#   SCOPE_KV_BIAS_BACKEND=...  (defaults to fa4; falls back if unavailable)
#
# Flags (handled by this script, stripped from the exec args):
#   --compile-fp8
#     Enable torch.compile even when FP8 quantization is enabled in the server
#     (sets SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION=1). On B300 this is
#     now viable because the pipeline applies a PerTensor-only TorchAO
#     `aten.as_strided.default` workaround by default (disable with
#     SCOPE_TORCHAO_PATCH_FLOAT8_AS_STRIDED=0).
#   --allow-fp8
#     Opt back into FP8 quantization on B300 (clears SCOPE_DISABLE_FP8_QUANTIZATION=1).

ENV_DIR="${B300_ENV_DIR:-.venv-b300-cu130-decode}"
BIN="$ENV_DIR/bin/daydream-scope"

SCRIPT_ARGS=()
ENABLE_COMPILE_FP8=0
ALLOW_FP8=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile-fp8)
      ENABLE_COMPILE_FP8=1
      shift
      ;;
    --allow-fp8)
      ALLOW_FP8=1
      shift
      ;;
    --help|-h)
      echo "Usage: scripts/run_daydream_b300.sh [--compile-fp8] [--allow-fp8] [daydream-scope args...]" >&2
      exit 0
      ;;
    *)
      SCRIPT_ARGS+=("$1")
      shift
      ;;
  esac
done

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

# Fused QKV projections can introduce extra view/contiguity overhead on SM103.
# Disable by default on B300; opt back in by setting SCOPE_DISABLE_FUSED_PROJECTIONS=0.
export SCOPE_DISABLE_FUSED_PROJECTIONS="${SCOPE_DISABLE_FUSED_PROJECTIONS:-1}"

# Default to highest-quality output on B300 (disable FP8 quantization unless explicitly allowed).
if [[ "$ALLOW_FP8" == "1" ]]; then
  export SCOPE_DISABLE_FP8_QUANTIZATION=0
else
  export SCOPE_DISABLE_FP8_QUANTIZATION=1
fi

if [[ "$ENABLE_COMPILE_FP8" == "1" && "$ALLOW_FP8" != "1" ]]; then
  echo "NOTE: --compile-fp8 set, but FP8 quantization is disabled by default on B300." >&2
  echo "      Use --allow-fp8 to opt back into FP8 (experimental / may be incorrect)." >&2
fi

# The server disables compile by default when quantization is enabled on non-SM100
# GPUs. This flag explicitly opts in.
if [[ "$ENABLE_COMPILE_FP8" == "1" ]]; then
  export SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION=1
fi

# Best-known KV-bias backend on B300 is FA4/CuTe score_mod (falls back if unavailable).
export SCOPE_KV_BIAS_BACKEND="${SCOPE_KV_BIAS_BACKEND:-fa4}"

# Faster steady-state decode mode on B300.
export WANVAE_STREAM_DECODE_MODE="${WANVAE_STREAM_DECODE_MODE:-chunk}"

# Small decode win: use channels-last 3D activations for Conv3d-heavy VAE decode.
export WANVAE_DECODE_CHANNELS_LAST_3D="${WANVAE_DECODE_CHANNELS_LAST_3D:-1}"

exec "$BIN" "${SCRIPT_ARGS[@]}"
