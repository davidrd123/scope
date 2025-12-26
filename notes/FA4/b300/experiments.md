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

### 2025-12-26 — Baseline choice: `--quantization none` vs `fp8_e4m3fn` (B300 cu130)

**Status:** Done

**Question:**  
Is fp8 actually a win on B300 for our canonical settings, or do conversions/scales dominate (making `quantization none` faster)?

**Hypothesis:**  
`quantization none` may be faster/more stable in some stacks; fp8 might win only when the exact fastpaths are active (and not silently skipped).

**Change (one thing):**  
Switch quantization mode (keep everything else fixed).

**Benchmark config:**  
- GPU: B300 (SM103)  
- Env: cu130 decode env (`.venv-b300-cu130-decode`)  
- torch / cuda: `2.9.0+cu130` / `13.0`  
- Settings: `320x576`, steps=`4`, bias=`0.3`  
- Notes: `SCOPE_KV_BIAS_BACKEND=fa4`, `WANVAE_STREAM_DECODE_MODE=chunk`, `DISABLE_FLEX_ATTENTION_COMPILE=1`, `--cudnn-benchmark`

**Command(s):**
```bash
SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3 \
  --quantization none \
  --cudnn-benchmark

SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3 \
  --quantization fp8_e4m3fn \
  --cudnn-benchmark
```

**Baseline:**  
`--quantization none` ≈ **~17.0 FPS** (this run: `16.99 FPS`)

**Result:**  
`--quantization fp8_e4m3fn` ≈ **~15.2 FPS** (this run: `15.22 FPS`)

**Additional result (compile interaction):**
- Without the PerTensor-only TorchAO workaround, `--compile` + `--quantization fp8_e4m3fn` fails with:
  - `NotImplementedError: Float8Tensor dispatch ... aten.as_strided ...` (torchao quantization Float8Tensor workflow)
- With the PerTensor-only workaround (applied automatically by `KreaRealtimeVideoPipeline` unless `SCOPE_TORCHAO_PATCH_FLOAT8_AS_STRIDED=0`), `--compile + fp8_e4m3fn` runs and is **~25 FPS** on B300/cu130 with `SCOPE_KV_BIAS_BACKEND=fa4` (see `notes/FA4/b300/session-state.md` for the current command).

**Decision:**  
- If not compiling: use `--quantization none` as the canonical B300/cu130 baseline for perf work (fp8 is slower on this stack unless compile is enabled).
- If compiling: fp8 becomes viable again (still pending upstream TorchAO fix; we currently rely on a PerTensor-only monkeypatch).

**Lessons:**
- On SM103, fp8 can be *slower* than BF16 if the intended fp8 kernels aren’t actually active (and/or conversion overhead dominates).
- Treat `--compile + fp8` as a separate compatibility axis (TorchAO tensor-subclass dispatch) and record failures explicitly (don’t assume “compile just works”).

---

### 2025-12-26 — torch.compile mode `reduce-overhead` is unstable on SM103

**Status:** Done (failed)

**Question:**  
Can `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` improve B300 steady-state FPS (via CUDA graphs / lower overhead)?

**Result:**  
On B300/cu130 with `SCOPE_KV_BIAS_BACKEND=fa4`, `--compile` + `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` failed with CUDAGraph overwrite errors:
- `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`

Attempted mitigation:
- `SCOPE_CUDAGRAPH_MARK_STEP_BEGIN=1` (calls `torch.compiler.cudagraph_mark_step_begin()` in the generator wrapper)

Still failed (internal Dynamo error referencing CUDAGraph overwrite).

**Decision:**  
Treat `reduce-overhead` as **known-bad** on SM103 for now; stick to default compile mode unless/until this is resolved upstream or via a targeted workaround.

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

---

### 2025-12-26 — Reduce denoise-step “glue” allocations (`timestep`, `random_noise`)

**Status:** Unmeasured (needs A/B)

**Question:**  
Can we reduce `aten::fill_` / `aten::copy_` / allocation churn by avoiding per-step creation of tiny-but-frequent tensors in the denoise loop?

**Change (one thing):**  
Reuse denoise-loop buffers:
- preallocate `timestep` and `next_timestep_tensor` and update via `fill_()` instead of `torch.ones(...) * t`
- reuse `random_noise` buffer and refill via `normal_()` instead of allocating `torch.randn(...)` each step

**Files changed:**  
- `src/scope/core/pipelines/wan2_1/blocks/denoise.py`  
- `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` (use `torch.zeros` for `context_timestep`)

**Suggested measurement:**  
- `scripts/profile_krea_pipeline_blocks.py` for FPS (canonical settings)  
- `scripts/profile_krea_pipeline_ops.py --json ...` to check whether `aten::to`/`aten::copy_`/`aten::fill_` counts/time move

---

### 2025-12-26 — Patch embedding: use Conv2d fastpath when `patch_size[0]==1`

**Status:** Done (big win)

**Question:**  
Why do op profiles show enormous `aten::copy_` / `aten::fill_` call counts on B300 even with `--compile`?

**Finding (via stack-aware op profiler):**  
`scripts/profile_krea_pipeline_ops.py --with-stack` showed the majority of `aten::copy_` / `aten::fill_` came from the **Conv3d patch embedding** inside the diffusion model (`CausalWanModel.patch_embedding`, `kernel_size=(1,2,2)`, `stride=(1,2,2)`), which was taking a slow path that triggers a big copy/fill storm.

**Hypothesis:**  
For `patch_size=(1,2,2)`, Conv3d is mathematically equivalent to running a Conv2d per-frame. If we compute it as Conv2d, we should get a faster cuDNN path and avoid the slow Conv3d implementation on SM103.

**Change (one thing):**  
In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`, add `CausalWanModel._patch_embed()` and use it everywhere we previously did:
- `self.patch_embedding(u.unsqueeze(0))`

When `patch_size[0]==1`, `_patch_embed()` reshapes `[B,C,F,H,W] → [B*F,C,H,W]`, calls `torch.nn.functional.conv2d(...)` with `weight.squeeze(2)`, then reshapes back to `[B,C_out,F,H',W']`. Otherwise it falls back to Conv3d.

**Benchmark config:**  
- GPU: B300 (SM103)  
- Env: cu130 decode env (`.venv-b300-cu130-decode`)  
- Settings: `320x576`, steps=`4`, bias=`0.3`, `SCOPE_KV_BIAS_BACKEND=fa4`

**Result (FPS):**
- No compile:
  - `--quantization none`: **~19.7 FPS**
  - `--quantization fp8_e4m3fn`: **~17.3 FPS**
- With compile:
  - `--compile --quantization none`: **~22.8 FPS**
  - `--compile --quantization fp8_e4m3fn`: **~25.1 FPS** (requires TorchAO PerTensor `as_strided` monkeypatch; applied automatically unless `SCOPE_TORCHAO_PATCH_FLOAT8_AS_STRIDED=0`)

**Result (op profile evidence):**
- Before: `aten::copy_` / `aten::fill_` were ~`35k` calls each (`outputs/b300_cu130_ops_profile_fa4_qnone_compile.json`)
- After: `aten::copy_` / `aten::fill_` dropped to ~`9.6k` calls each (`outputs/b300_cu130_ops_profile_fa4_qnone_compile_post_patch_embed.json`)

**Decision:**  
Keep. This is a large end-to-end win and moves the “remaining glue” story from “mysterious copy/fill” to specific remaining hotspots (mostly VAE conv3d / other ops).

**Lessons:**  
- Huge `aten::copy_` / `aten::fill_` counts can be a **slow Conv3d fallback** problem, not an attention problem.  
- When a Conv3d has temporal kernel `1`, it may be worth rewriting as a per-frame Conv2d (especially on SM103).
