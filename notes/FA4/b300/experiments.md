# B300 Experiments Log (Small, Reproducible)

**Purpose:** Capture “one-change” experiments as tiny cards so we can learn fast *without* losing the thread.

**Why a single file?** It avoids repo clutter (no explosion of one-off files). If this grows too large, we can later split into `notes/FA4/b300/experiments/YYYY-MM-DD-<slug>.md` and add an index.

## How this differs from the other docs

- `notes/FA4/b300/session-state.md`: “what to run today” (known-good commands + caveats)
- `notes/FA4/b300/investigation-runbook.md`: “how to measure” (protocol + profiler knobs)
- `notes/FA4/b300/experiments.md` (this file): “what we changed + what happened” (one change per card)

---

## Canonical benchmark settings (for comparable perf numbers)

- **Resolution:** `320x576`
- **Denoising steps:** `4`
- **KV-cache attention bias:** `0.3`
- **Quality-preserving:** avoid accuracy shortcuts (e.g. don’t use KV recompute skipping for “real” results)

Suggested measurement harness:
- `scripts/profile_krea_pipeline_blocks.py` (end-to-end FPS; optional block JSON)
- `scripts/profile_b300_denoise_drilldown.sh` (deeper split; writes JSON under `outputs/`)

---

## Experiment Card Template (copy/paste)

### YYYY-MM-DD — <short title>

**Question:**  
<What are we trying to learn?>

**Hypothesis:**  
<What do we expect to happen, and why?>

**Change (one thing):**  
<Single config tweak / code change / backend swap>

**Benchmark config:**  
- GPU: <B300>  
- Env: <cu130 / repo-default>  
- torch / cuda: <e.g. 2.9.0+cu130 / 13.0>  
- Settings: `320x576`, steps=`4`, bias=`0.3`  
- Notes: <compile on/off, quantization, backend vars>

**Command(s):**
```bash
<exact command(s) used>
```

**Baseline:**  
<FPS + any relevant breakdown you’re comparing against>

**Result:**  
<FPS delta + anything surprising in the breakdown>

**Decision:**  
<Keep / revert / follow-up / not worth it>

**Artifacts:**  
- `outputs/<...>.log`  
- `outputs/<...>.json`

**Lessons (write like you’re teaching “future you”):**  
- <What did we learn about the system / tooling / GPU behavior?>
- <What would you try next, and why?>

---

## Experiments

### 2025-12-25 — KV-bias backend: `flash` → `fa4`

**Question:**  
Does FA4/CuTe `score_mod` KV-bias outperform FlashAttention’s segment-combine KV-bias in our end-to-end benchmark?

**Hypothesis:**  
Yes. Segment-combine adds extra work/launches; `score_mod` should keep KV-bias inside the main attention kernel and reduce overhead.

**Change (one thing):**  
Switch `SCOPE_KV_BIAS_BACKEND=flash` → `SCOPE_KV_BIAS_BACKEND=fa4` (keep everything else identical).

**Benchmark config:**  
- GPU: B300 (SM103)  
- Env: cu130 decode env (`.venv-b300-cu130-decode`)  
- torch / cuda: `2.9.0+cu130` / `13.0`  
- Settings: `320x576`, steps=`4`, bias=`0.3`, quantization=`none`  

**Command(s):**
```bash
SCOPE_KV_BIAS_BACKEND=flash \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark

SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark
```

**Baseline:**  
`SCOPE_KV_BIAS_BACKEND=flash` ≈ **14.9 FPS**

**Result:**  
`SCOPE_KV_BIAS_BACKEND=fa4` ≈ **16.7 FPS**

**Decision:**  
Keep FA4 `score_mod` as the preferred KV-bias backend on B300 when the cu130 stack is available.

**Artifacts:**  
- `outputs/b300_cu130_ops_profile_flash.json`  
- `outputs/b300_cu130_ops_profile_fa4.json`  
- `outputs/b300_cu130_none_bias0.3_drilldown_perf.log`  

**Lessons (write like you’re teaching “future you”):**  
- The KV-bias microkernel can be meaningfully faster with `score_mod`, but end-to-end wins are still bounded by the remaining `self_attn` work (`other_in_self`) + GEMMs + copies.  
- Once decode is fixed (cu130), attention backend selection becomes “worth doing,” but it’s not the only lever; profiling still matters.

### 2025-12-26 — Baseline choice: `--quantization none` vs `fp8` (planned)

**Status:** Planned (no results recorded yet)

**Question:**  
Is fp8 actually a win on B300 for our canonical settings, or do conversions/scales dominate (making `quantization none` faster)?

**Hypothesis:**  
`quantization none` may be faster/more stable in some stacks; fp8 might win only when the exact fastpaths are active (and not silently skipped).

**Change (one thing):**  
Switch quantization mode (keep everything else fixed).

**Benchmark config:**  
- GPU: B300 (SM103)  
- Env: <record actual env here>  
- Settings: `320x576`, steps=`4`, bias=`0.3`  

**Command(s):**
```bash
# Use your active Python env; avoid hardcoding venv paths in the long term.

python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3 \
  --quantization none

python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3 \
  --quantization fp8
```

**Baseline / Result / Decision / Artifacts / Lessons:**  
TBD (fill in after the first run).

---

### 2025-12-26 — Fix `flash` segment-combine on SM103 (avoid FA4 `return_lse` ICE)

**Question:**  
Why did `SCOPE_KV_BIAS_BACKEND=flash` collapse to ~`2–3 FPS` (falling back to flex_attention) on B300/cu130?

**Hypothesis:**  
`_flash_attn_with_lse()` prefers FA4/CuTe `_flash_attn_fwd(..., return_lse=True)`, but that path can ICE in some cutlass-dsl builds on SM103. If we skip FA4 `return_lse` (or fall back to FA2 varlen), segment-combine should work and avoid the flex fallback.

**Change (one thing):**  
In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`, make flash segment-combine use the stable FA2 varlen op by default on SM103 (opt-in FA4 via `SCOPE_FLASH_COMBINE_USE_FA4_LSE=1`), and fall back to FA2 if FA4 `return_lse` throws.

**Benchmark config:**  
- GPU: B300 (SM103)  
- Env: cu130 decode env (`.venv-b300-cu130-decode`)  
- torch / cuda: `2.9.0+cu130` / `13.0`  
- Settings: `320x576`, steps=`4`, bias=`0.3`, quantization=`none`  
- Notes: `WANVAE_STREAM_DECODE_MODE=chunk`, `DISABLE_FLEX_ATTENTION_COMPILE=1` (measured `--compile` off/on)

**Command(s):**
```bash
SCOPE_KV_BIAS_BACKEND=flash \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark
```

**Baseline:**  
Before fix: `SCOPE_KV_BIAS_BACKEND=flash` threw a cutlass-dsl ICE and fell back to **uncompiled flex_attention**, yielding **~`2.77 FPS`**.

**Result:**  
After fix: `SCOPE_KV_BIAS_BACKEND=flash` is stable again:  
- `--compile` off: **~`15.1 FPS`** (this run: `15.06 FPS`)  
- `--compile` on: **~`18.4 FPS`** (this run: `18.40 FPS`; Dynamo graph breaks inside `_kv_bias_flash_combine`, but it still helps)

**Decision:**  
Keep: this makes the “flash fallback ladder” usable on SM103 and prevents catastrophic flex fallbacks when FA4 `return_lse` is broken.

**Lessons:**  
- FA4 score_mod can be healthy while FA4 `return_lse` is not; keep these as separate “availability” checks.  
- For SM103, segment-combine should default to the stable FA2 varlen kernel unless explicitly experimenting.
