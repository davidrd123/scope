# B200 Experiments Log (Small, Reproducible)

**Purpose:** Capture “one-change” experiments as tiny cards so we can learn fast without losing the thread.

**Why a single file?** It avoids repo clutter (no explosion of one-off files). If this grows too large, we can later split into dated files and add an index.

---

## Canonical benchmark settings (for comparable perf numbers)

- **Resolution:** `320x576`
- **Denoising steps:** `4`
- **KV-cache attention bias:** `0.3`
- **Quality-preserving:** avoid accuracy shortcuts (e.g. don’t use KV recompute skipping for “real” results)

Suggested measurement harness:
- `scripts/profile_krea_pipeline_blocks.py` (end-to-end FPS; optional block JSON)
- `PROFILE_ATTENTION=1` (attention bucket breakdown)

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
- GPU: B200 (SM100)  
- Env: <uv/.venv details>  
- torch / cuda: <e.g. 2.x.y+cu12x / 12.x>  
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

**Artifacts (optional):**  
- `outputs/<...>.log`  
- `outputs/<...>.json`

**Lessons (write like you’re teaching “future you”):**  
- <What did we learn about the system / tooling / GPU behavior?>
- <What would you try next, and why?>

---

## Experiments

### 2025-12-26 — Baseline (eager, FP8, KV-bias on)

**Question:**  
What is today’s canonical baseline on B200 at `320x576`, 4 steps, KV-bias `0.3`?

**Hypothesis:**  
N/A (recording baseline).

**Change (one thing):**  
None (baseline run).

**Benchmark config:**  
- GPU: NVIDIA B200 (SM100 / cc 10.0)  
- torch / cuda: `2.8.0+cu128` / `12.8`  
- Settings: `320x576`, steps=`4`, bias=`0.3`  
- Notes: quantization=`fp8_e4m3fn`, compile=off, `SCOPE_KV_BIAS_BACKEND` unset (defaults to `triton`)

**Command(s):**
```bash
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3

PROFILE_ATTENTION=1 \
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3
```

**Result:**  
- End-to-end: **`18.36 FPS`** avg (skip=`2`)
- Attention breakdown (profiling overhead reduces FPS; use for ms attribution):
  - `self_attn`: `1495.0ms` (58.4% of transformer time)
  - `self_attn_kv_bias`: `411.5ms` (27.5% of self_attn)
  - `other_in_self`: `1024.9ms` (68.6% of self_attn)
  - `p_bias`: `87.5%`, `p_recompute`: `12.5%`

**Decision:**  
Use this as the baseline for B200 comparisons.

**Lessons:**  
- Biggest remaining win is shrinking `other_in_self` (projections/copies/glue) rather than KV-bias itself.

---

### 2025-12-26 — torch.compile attention blocks (FP8)

**Question:**  
Is `torch.compile` a stable, big steady-state win on B200 with FP8 quantization enabled?

**Hypothesis:**  
Yes; B200 is a friendlier target than SM103 and compile should reduce `other_in_self` overhead via fusion.

**Change (one thing):**  
Enable compilation of the diffusion attention blocks (`--compile`).

**Benchmark config:**  
- GPU: NVIDIA B200 (SM100 / cc 10.0)  
- torch / cuda: `2.8.0+cu128` / `12.8`  
- Settings: `320x576`, steps=`4`, bias=`0.3`  
- Notes: quantization=`fp8_e4m3fn`, compile=on

**Command(s):**
```bash
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3 \
  --compile
```

**Baseline:**  
`18.36 FPS` (eager baseline above)

**Result:**  
**`29.98 FPS`** avg (skip=`2`) in steady-state.

**Decision:**  
Treat `--compile` / `SCOPE_COMPILE_KREA_PIPELINE=1` as the primary B200 “Phase 2” lever once correctness is validated.

**Lessons:**  
- First run pays a heavy compile warmup; subsequent runs are much faster due to compile caching.
- `PROFILE_ATTENTION=1` does not produce bucket breakdown under `torch.compile` (use `scripts/profile_krea_pipeline_ops.py --compile` if you need op-level attribution).

---

### 2025-12-26 — KV-bias backend sweep (eager)

**Question:**  
With the current environment, which KV-bias backend is fastest on B200 in eager mode?

**Hypothesis:**  
`triton` should beat `flash` and `flex`; `fa4` may be unavailable if `cutlass` isn’t installed.

**Change (one thing):**  
Set `SCOPE_KV_BIAS_BACKEND=<backend>` (one backend per run).

**Benchmark config:**  
- GPU: NVIDIA B200 (SM100 / cc 10.0)  
- torch / cuda: `2.8.0+cu128` / `12.8`  
- Settings: `320x576`, steps=`4`, bias=`0.3`  
- Notes: quantization=`fp8_e4m3fn`, compile=off

**Command(s):**
```bash
SCOPE_KV_BIAS_BACKEND=triton uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 --iters 6 --skip 2 --kv-cache-attention-bias 0.3

SCOPE_KV_BIAS_BACKEND=flash uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 --iters 6 --skip 2 --kv-cache-attention-bias 0.3

SCOPE_KV_BIAS_BACKEND=flex uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 --iters 6 --skip 2 --kv-cache-attention-bias 0.3

SCOPE_KV_BIAS_BACKEND=fa4 uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 --iters 6 --skip 2 --kv-cache-attention-bias 0.3
```

**Result:**  
- `triton`: **`18.36 FPS`**  
- `flash`: **`17.97 FPS`**  
- `flex`: **`16.20 FPS`**  
- `fa4`: unavailable (falls back; error: `No module named 'cutlass'`)

**Decision:**  
Keep default `triton` in this environment; `fa4` requires the `cutlass` Python package.
