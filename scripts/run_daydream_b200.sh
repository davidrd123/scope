#!/usr/bin/env bash
set -euo pipefail

# Run Daydream (daydream-scope) on B200 using the default uv env.
#
# Usage:
#   scripts/run_daydream_b200.sh [daydream-scope args...]
#
# Env overrides:
#   B200_ENV_DIR=...   (defaults to .venv)
#   SCOPE_COMPILE_KREA_PIPELINE=...  (defaults to 1)

ENV_DIR="${B200_ENV_DIR:-.venv}"
BIN="$ENV_DIR/bin/daydream-scope"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: Expected $BIN (is the env created + synced?)" >&2
  echo "Fix/create the env with:" >&2
  echo "  uv sync" >&2
  exit 1
fi

# B200 (SM100) sees a large steady-state win from torch.compile on the diffusion blocks.
export SCOPE_COMPILE_KREA_PIPELINE="${SCOPE_COMPILE_KREA_PIPELINE:-1}"

exec "$BIN" "$@"
