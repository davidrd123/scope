#!/usr/bin/env bash
set -euo pipefail

# Render a Scope timeline JSON export on B300 using the isolated cu130 env and
# the current "known-good" environment knobs (mirrors scripts/run_daydream_b300.sh).
#
# Usage:
#   scripts/render_timeline_b300.sh <timeline.json> <out.mp4> [render_timeline args...]
#
# Defaults appended (unless already specified in args):
#   --timebase chunk
#   --quantization none
#   --compile
#
# Env overrides:
#   B300_ENV_DIR=...                 (defaults to .venv-b300-cu130-decode)
#   TRITON_PTXAS_PATH=...            (auto-detected if unset)
#   SCOPE_KV_BIAS_BACKEND=...        (defaults to fa4)
#   SCOPE_ENABLE_FA4_VARLEN=...      (defaults to 1)
#   WANVAE_STREAM_DECODE_MODE=...    (defaults to chunk)
#   WANVAE_DECODE_CHANNELS_LAST_3D=... (defaults to 1)
#   WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=... (defaults to 1)
#   DISABLE_FLEX_ATTENTION_COMPILE=... (defaults to 1)

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/render_timeline_b300.sh <timeline.json> <out.mp4> [render_timeline args...]

Example (recommended defaults):
  scripts/render_timeline_b300.sh ~/.daydream-scope/recordings/session_*.timeline.json out.mp4

Example (override resolution + steps):
  scripts/render_timeline_b300.sh timeline.json out.mp4 --height 320 --width 576 --denoising-steps 1000,750,500,250
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

TIMELINE="$1"
OUTPUT="$2"
shift 2

if [[ ! -f "$TIMELINE" ]]; then
  echo "ERROR: timeline not found: $TIMELINE" >&2
  exit 2
fi

EXTRA_ARGS=("$@")
DEFAULT_ARGS=()

has_arg_prefix() {
  local prefix="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$prefix" ]]; then
      return 0
    fi
  done
  return 1
}

has_any_of() {
  local arg
  for arg in "$@"; do
    if has_arg_prefix "$arg" "${EXTRA_ARGS[@]}"; then
      return 0
    fi
  done
  return 1
}

# Prefer chunk scheduling since the server recorder exports startChunk/endChunk.
if ! has_arg_prefix "--timebase" "${EXTRA_ARGS[@]}"; then
  DEFAULT_ARGS+=("--timebase" "chunk")
fi

# B300 quality gate: default to BF16 (quantization none).
if ! has_arg_prefix "--quantization" "${EXTRA_ARGS[@]}"; then
  DEFAULT_ARGS+=("--quantization" "none")
fi

# B300: compile is a throughput win; render_timeline won't auto-enable it on SM103.
if ! has_any_of "--compile" "--no-compile"; then
  DEFAULT_ARGS+=("--compile")
fi

ENV_DIR="${B300_ENV_DIR:-.venv-b300-cu130-decode}"
PY="$ENV_DIR/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Expected $PY (is the env created + synced?)" >&2
  echo "Fix/create the env with:" >&2
  echo "  scripts/b300_env_fix_cu130.sh $ENV_DIR" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Ensure local sources are importable even if the venv doesn't have an editable install.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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

# Best-known KV-bias backend on B300 is FA4/CuTe score_mod (falls back if unavailable).
export SCOPE_KV_BIAS_BACKEND="${SCOPE_KV_BIAS_BACKEND:-fa4}"

# Small denoise win: opt into FA4/CuTe varlen for the plain (non-bias) attention path.
export SCOPE_ENABLE_FA4_VARLEN="${SCOPE_ENABLE_FA4_VARLEN:-1}"

# Faster steady-state decode mode on B300.
export WANVAE_STREAM_DECODE_MODE="${WANVAE_STREAM_DECODE_MODE:-chunk}"

# Small decode win: use channels-last 3D activations for Conv3d-heavy VAE decode.
export WANVAE_DECODE_CHANNELS_LAST_3D="${WANVAE_DECODE_CHANNELS_LAST_3D:-1}"

# Big decode win: keep streaming Resample outputs contiguous so Conv3d stays on cuDNN/CUTLASS.
export WANVAE_RESAMPLE_ENSURE_CONTIGUOUS="${WANVAE_RESAMPLE_ENSURE_CONTIGUOUS:-1}"

exec "$PY" -m scope.cli.render_timeline "$TIMELINE" "$OUTPUT" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}"

