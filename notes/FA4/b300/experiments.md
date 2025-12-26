# B300 Experiments Log (Small, Reproducible)

**Purpose:** Capture “one-change” experiments as tiny cards so we can learn fast *without* losing the thread.

**Why a single file?** It avoids repo clutter (no explosion of one-off files). If this grows too large, we can later split into `notes/FA4/b300/experiments/YYYY-MM-DD-<slug>.md` and add an index.

**Quality gate:** FP8 quantization is currently **off-limits** on B300 (garbage output). Any FP8 numbers in this file are perf-only breadcrumbs for upstream debugging; use `--quantization none` for real output.

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
- If `--compile` is involved, also record **warmup time** (it matters for UX and can dominate “time to first frame”).

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
- Warmup: <seconds to reach steady-state (esp. `--compile`)>

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
- If compiling: fp8 can benchmark higher, but is currently **not viable for quality** on B300 (gray/noise output). Treat fp8 runs as perf-only.

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
- Stabilize KV-cache index tensors across iterations (avoid storing fresh CUDAGraph outputs in a long-lived Python dict slot)
- Compile the whole diffusion model in `reduce-overhead` mode instead of compiling each transformer block independently

Still failed (internal Dynamo error referencing CUDAGraph overwrite).

**Decision:**  
Treat `reduce-overhead` as **known-bad** on SM103 for now; stick to default compile mode unless/until this is resolved upstream or via a targeted workaround.

**Guardrail:**  
On SM103, `KreaRealtimeVideoPipeline` now ignores `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` unless you explicitly set `SCOPE_ALLOW_REDUCE_OVERHEAD_SM103=1`.

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

### 2025-12-26 — WanVAE decode: avoid BF16 → FP32 → BF16 in `Upsample`

**Status:** Done (small cleanup; not a big lever)

**Question:**  
Is VAE decode spending meaningful time doing dtype roundtrips in upsample (`x.float().type_as(x)`), and can we remove them safely?

**Hypothesis:**  
Yes: nearest-exact upsample used to require FP32 on some stacks, but if BF16/FP16 upsample works on SM103 we should prefer it to reduce `aten::to`/`aten::copy_` noise in decode.

**Change (one thing):**  
In `src/scope/core/pipelines/wan2_1/vae/modules/vae.py`, `class Upsample.forward` now:
- Tries `super().forward(x)` in BF16/FP16 on CUDA
- Falls back to FP32 only if the kernel errors due to dtype support
- Adds `WANVAE_UPSAMPLE_FORCE_FP32=1` to force legacy behavior for A/B testing

**Correctness sanity check:**  
Directly verified BF16 upsample works on CUDA (B300) and matches even-grid samples (`max_diff_even_grid == 0.0`) on a small test tensor.

**Result (stack-attributed op profile):**  
The upsample frames disappear from the `aten::copy_` grouped-by-stack output (as expected), but overall `aten::copy_` GPU self time was essentially unchanged (decode is dominated by other work, mostly conv3d).

**Artifacts:**  
- Baseline (no change): `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_nocompile_withstack.md`  
- After change: `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_nocompile_withstack_upsamplefix_s12.md`  
- Baseline (compile): `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_compile_withstack.md`  
- After change (compile): `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_compile_withstack_upsamplefix_s12.md`

**Decision:**  
Keep (it removes a suspicious dtype roundtrip and adds an escape hatch), but don’t expect a large end-to-end FPS gain from this alone.

### 2025-12-26 — WanRMSNorm: remove `x.float()` copies (use fused `rms_norm`)

**Status:** Done (nice eager-mode cleanup; small end-to-end win)

**Question:**  
Can we eliminate the `aten::copy_` / `aten::_to_copy` storm caused by `WanRMSNorm.forward` doing `x.float().type_as(x)`?

**Hypothesis:**  
Yes. PyTorch has a fused `rms_norm` implementation that should avoid explicit dtype roundtrips and collapse several elementwise ops.

**Change (one thing):**  
In `src/scope/core/pipelines/krea_realtime_video/modules/model.py`, `WanRMSNorm.forward` now prefers `torch.nn.functional.rms_norm` (fused CUDA path) and falls back to the legacy implementation when:
- `SCOPE_WAN_RMSNORM_IMPL=legacy`, or
- the torch build lacks `F.rms_norm`

**Benchmark config:**  
- GPU: B300 (SM103)  
- Env: cu130 decode env (`.venv-b300-cu130-decode`)  
- torch / cuda: `2.9.0+cu130` / `13.0`  
- Settings: `320x576`, bias=`0.3`, quantization=`none`, compile=`False`, `SCOPE_KV_BIAS_BACKEND=fa4`  

**Command(s):**
```bash
# Legacy (baseline in same code revision)
SCOPE_WAN_RMSNORM_IMPL=legacy \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_ops.py \
  --height 320 --width 576 \
  --iters 1 --pre-iters 1 \
  --kv-cache-attention-bias 0.3 \
  --kv-bias-backend fa4 \
  --quantization none \
  --with-stack --stack-n 12

# Auto (fused rms_norm)
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_ops.py \
  --height 320 --width 576 \
  --iters 1 --pre-iters 1 \
  --kv-cache-attention-bias 0.3 \
  --kv-bias-backend fa4 \
  --quantization none \
  --with-stack --stack-n 12
```

**Baseline (legacy):**  
From `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_nocompile_withstack_rmsnormlegacy_s12.md`:
- `aten::copy_`: **24.45ms**, **11,376** calls (stack groups dominated by `WanRMSNorm.forward` → `x.float()`)
- Profiled wall time: **5.451s**

**Result (auto/fused):**  
From `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_nocompile_withstack_rmsnormfix_s12.md`:
- `aten::copy_`: **19.26ms**, **10,576** calls (WanRMSNorm `x.float()` stack groups disappear)
- `aten::_fused_rms_norm`: **6.12ms**, **600** calls (new fused kernel)
- Profiled wall time: **5.418s**

**Decision:**  
Keep. It’s not “the” bottleneck, but it removes a very visible eager-mode copy hotspot and makes the op profile easier to interpret.

**Lessons:**  
- When stack-attributed `aten::copy_` points at a dtype cast in a hot per-layer op, try a fused primitive first (`rms_norm` / `layer_norm`) before chasing deeper kernel fusion.  
- Using an env-var switch (`SCOPE_WAN_RMSNORM_IMPL`) makes A/B testing safer when we care about quality.

### 2025-12-26 — WanVAE `CausalConv3d`: implicit spatial padding (avoid `F.pad` when cache is warm)

**Status:** Done (tiny win / mostly “cleanup + learning”)

**Question:**  
Does explicit spatial `F.pad(...)` inside `CausalConv3d.forward` meaningfully contribute to decode overhead/copy+fill noise?

**Hypothesis:**  
If we let cuDNN handle **spatial padding implicitly** (via Conv3d’s `padding=(0, pad_h, pad_w)`), then once KV/feature caches are warm (so time pad is effectively 0), we can often avoid `F.pad(...)` entirely.

**Change (one thing):**  
In `src/scope/core/pipelines/wan2_1/vae/modules/vae.py`, `CausalConv3d` now supports:
- `WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING=1` (default) → use implicit spatial padding and only pad time when needed
- `WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING=0` → legacy “pad everything explicitly” path  
Also added `implicit_spatial_padding=` kwarg for per-instance A/B in tests.

**Correctness sanity check:**  
Verified BF16 CUDA output matches **exactly** between explicit vs implicit padding modes (including with cache tensors of length 1 and 2).

**Result (end-to-end blocks benchmark, B300 cu130, quantization none):**
- Explicit (`WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING=0`): **20.08 FPS**
- Implicit (`WANVAE_CONV3D_IMPLICIT_SPATIAL_PADDING=1`): **20.20 FPS** (~`+0.6%`, likely near noise)

**Result (stack-aware op profiler, 1-iter):**
- `aten::copy_` self CUDA: **19.25ms → 19.15ms**
- `aten::copy_` calls: **10,572 → 10,542**

**Artifacts:**  
- Op profile explicit: `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_nocompile_withstack_conv3dpad_explicit_s12.md`  
- Op profile implicit: `outputs/observe_b300_2025-12-26_qnone/ops_profile_qnone_fa4_bias0.3_nocompile_withstack_conv3dpad_implicit_s12.md`

**Decision:**  
Keep (default-on). The win is small, but it’s correctness-preserving and makes the causal Conv3d implementation a bit more “cuDNN-native”.

**Lessons:**  
- A lot of the VAE decode cost is still in conv3d itself (not just padding glue), so bigger wins likely need **algorithmic** or **layout** changes (or cuDNN version fixes), not just removing `F.pad`.  
