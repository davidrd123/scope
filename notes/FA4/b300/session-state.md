# B300 Session State - 2025-12-27 (Updated)

## Navigation

- **How to reproduce / measure:** [`investigation-runbook.md`](investigation-runbook.md)
- **Where to record “one-change” trials:** [`experiments.md`](experiments.md)
- **Roadmap / next levers:** [`optimization-vision.md`](optimization-vision.md)
- **Development plan (what to build next):** [`development-plan.md`](development-plan.md)
- **Level 5/6 reading list:** [`level5-level6-resources.md`](level5-level6-resources.md)
- **Deep-research packets (2025-12-26):** [`claude_dr.md`](../DeepResearch/2025-12-26/B300_optim_ladder/round02/claude_dr.md) (source-backed links + spellings)
- **Paste-ready upstream issue (TorchAO as_strided):** [`torchao-as-strided-dispatch.md`](../../issues/torchao-as-strided-dispatch.md)

## Quality Gate (Read This First)

**We prioritize output quality over speed.** As of 2025-12-27 on B300/SM103, **FP8 quantization is off-limits** for “real” runs because it produces **garbage output** (gray/noise).  
Use `--quantization none` (BF16) for the best-known quality. FP8 measurements are kept only as *perf-only* debugging breadcrumbs for upstream work.

## Upstream refs (source-backed quick links)

These are the three “version/typo landmines” we keep tripping over; keep this section short and prefer upstream links.

- **TorchAO FP8 + `torch.compile` + view-ish ops (`aten.as_strided`)**
  - We hit: `NotImplementedError: Float8Tensor dispatch ... aten.as_strided.default ...` (see [`experiments.md`](experiments.md)).
  - Root cause (today): `torchao.quantization.Float8Tensor` (quantization workflow used by `quantize_`) does **not** implement `aten.as_strided.default` in torchao **v0.14.1** and **v0.15.0** (so Inductor lowering can trip it).
    - https://github.com/pytorch/ao/blob/v0.14.1/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
    - https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
  - TorchAO ↔ torch compatibility table: https://github.com/pytorch/ao/issues/2919
  - Local unblock for experiments (PerTensor-only): the realtime pipeline auto-applies a PerTensor-only monkeypatch when running `--compile + fp8` (disable with `SCOPE_TORCHAO_PATCH_FLOAT8_AS_STRIDED=0`). Patch code: `src/scope/core/compat/torchao_float8_as_strided.py`. Upstream issue text: [`torchao-as-strided-dispatch.md`](../../issues/torchao-as-strided-dispatch.md). (For ad-hoc experiments outside the pipeline: `scripts/patch_float8_as_strided.py`.)

- **Conv3d BF16/FP16 regressions (PyTorch 2.9 era)**
  - PyTorch **v2.9.1** release notes recommend: install **`nvidia-cudnn-cu12>=9.15`** if impacted by BF16 Conv3d regressions: https://github.com/pytorch/pytorch/releases/tag/v2.9.1
  - Primary issue thread: https://github.com/pytorch/pytorch/issues/166643

- **`nvidia-cutlass-dsl` top-level `cutlass` + Inductor**
  - Can trigger `AttributeError: module 'cutlass' has no attribute 'CACHE_FILE'` from `torch._inductor.codegen.cuda.cutlass_utils` (often ignored at exit), and older stacks also saw compilation failures depending on codepaths.
  - Current B300/cu130 path runs FA4 score_mod alongside regional `torch.compile` by keeping CuTe calls opaque to Dynamo and disabling flex_attention compilation; see “torch.compile Status (B300)” below.
  - Details: [`investigation.md`](investigation.md) (Issue 2) and [`setup-guide.md`](setup-guide.md).

- **CUDAGraph “output overwritten” + correct step-marker / knob names**
  - Step marker API: `torch.compiler.cudagraph_mark_step_begin()` (docs): https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html
  - CUDAGraph Trees doc (error explanation + mitigation): https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
  - Upstream issue tracker: `pytorch/pytorch#158551` (see also [`pytorch-cudagraph-output-overwritten.md`](../../issues/pytorch-cudagraph-output-overwritten.md)).
  - Inductor cudagraph master env var (v2.9.1): `TORCHINDUCTOR_CUDAGRAPHS=1` (see `torch/_inductor/config.py`): https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/config.py

- **Triton/Inductor SM103 tcgen05 LLVM abort (“tcgen05.wait.st”)**
  - This can hard-abort the process during compilation on B300/SM103 for some Triton/TensorCore codegen paths.
  - Upstream: `triton-lang/triton#8473` / `triton-lang/triton#8481` (see [`triton-sm103-tcgen05-llvm-abort.md`](../../issues/triton-sm103-tcgen05-llvm-abort.md)).
  - **Version note:** our cu130 env currently has `triton==3.5.1` (includes the SM103 fix; release notes: https://github.com/triton-lang/triton/releases/tag/v3.5.1).

## Current Status

**Baseline (repo default stack):** B300 is ~8.8 FPS at `320x576` (reference resolution) and the number is stable across iterations.

### Production-viable (quality-preserving; BF16 / `--quantization none`)

Benchmark harness (`scripts/profile_krea_pipeline_blocks.py`, cu130 env, bias=0.3):
- **Best-known (today):** `SCOPE_KV_BIAS_BACKEND=fa4` + `SCOPE_ENABLE_FA4_VARLEN=1` + `--compile` + `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1` + fused projections ON: **~34.5 FPS** (Avg FPS `skip=3`; `outputs/b300_cu130_compile_best_blocks_no_kvcache_zero_2025-12-27.log`)
- Optional (experiment): `SCOPE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`: **~34.65 FPS** (Avg FPS `skip=3`; warmup `~17.3s` vs `~15.7s` default) (`outputs/b300_cu130_compile_mode_maxautotune_nocg_best_2025-12-27.log`)
- Same settings but without the VAE resample contiguity fix (`WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=0`): **~21.45 FPS** (`outputs/b300_cu130_triton351_compile_default_blocks_perf.log`)
- Historical (needs re-measure post VAE fix): `SCOPE_KV_BIAS_BACKEND=fa4`: **~19.7 FPS**

Daydream end-to-end (cu130 env): **TBD re-measure** (historical note: pre patch-embed fastpath it was ~`14.8–15.0 FPS`; do not rely on that number now).

### Perf-only / blocked paths (not usable for quality)

- *(perf-only; quality broken)* `SCOPE_KV_BIAS_BACKEND=fa4` + `--quantization fp8_e4m3fn` (no compile): **~17.3 FPS**
- *(perf-only; quality broken)* `SCOPE_KV_BIAS_BACKEND=fa4` + `--compile` + `--quantization fp8_e4m3fn`: **~25.1 FPS** (requires the PerTensor-only TorchAO `as_strided` monkeypatch; applied automatically by the pipeline unless `SCOPE_TORCHAO_PATCH_FLOAT8_AS_STRIDED=0`).
- Note: on SM103 we default flash segment-combine to the stable FA2 varlen op; opt in to FA4 `return_lse` experiments with `SCOPE_FLASH_COMBINE_USE_FA4_LSE=1`.

`torchao` note: repo pins `torchao==0.13.0` (torch 2.8 ABI). For torch `2.9.0+cu130`, `scripts/b300_env_fix_cu130.sh` now tries `torchao==0.15.0+cu130` from the cu130 index (then PyPI as fallback). **As of 2025-12-26**, `torchao==0.15.0+cu130` still prints `Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu130 ...` (likely upstream; no FPS change observed).

This is roughly a **3.9× throughput improvement** over the repo-default baseline (~8.8 → ~34.5 FPS in the benchmark harness).

External doc brief (for RepoPrompt / web research): [`blackwell-docs.md`](blackwell-docs.md)

Repro (isolated env; does not touch shared `.venv`):

```bash
# Setup (one-time)
./scripts/setup_b300_cu130_env.sh

# Run daydream
./scripts/run_daydream_b300.sh
```

Or for benchmarking:

```bash
SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
WANVAE_DECODE_CHANNELS_LAST_3D=1 \
WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1 \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark \
  --compile
```

Or for a one-shot denoise/decode drill-down (writes JSON artifacts under `outputs/`):

```bash
scripts/profile_b300_denoise_drilldown.sh
```

Latest drill-down run (cu130, profiling enabled): `outputs/b300_cu130_none_bias0.3_drilldown_perf.log` averaged **~14.7 FPS** (expected lower than the non-profiled benchmark due to extra synchronizations).

Artifacts:
- `outputs/b300_cu130_none_bias0.3_drilldown_perf.log`
- `outputs/b300_cu130_none_bias0.3_drilldown_perf.json`
- `outputs/b300_cu130_none_bias0.3_drilldown_blocks_profile.json`

**Key update:** This no longer looks like an “invisible FPS cap” caused by WebRTC/codec pacing or CPU stalls. A block-level CUDA-event profile shows GPU time ≈ wall time, and the dominant blocks are `denoise` and `decode` (not the KV-bias attention microkernel).

See also: [`investigation-runbook.md`](investigation-runbook.md) (“Ground Truth” section).

**Operational note (intermittent):** We have seen rare CUDA/NVML wedges on this host (`nvidia-smi` → “Failed to initialize NVML: Unknown Error”, PyTorch `cudaGetDeviceCount` → Error 304). When `nvidia-smi` is healthy (normal driver table), you’re fine; when it wedges, bench scripts can fail at import-time and you’ll likely need a host-level driver restart or reboot.

**Another failure mode (looks similar): “monitor-only GPU access”**

Sometimes you can still see the GPU (e.g. `nvidia-smi -L` works) but **CUDA compute is blocked** (common on login nodes / mis-scoped containers / device-cgroup restrictions). Symptoms:
- PyTorch: `torch.cuda.is_available() == False` with `cudaGetDeviceCount` **Error 304**
- `/dev/nvidia*` can be opened **read-only** but not read-write

Quick check:
```bash
python3 - <<'PY'
import os
for p in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"):
    try:
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        print(p, "RW OK")
    except Exception as e:
        print(p, "RW FAIL:", type(e).__name__, e)
PY
```

If RW opens fail, you won’t be able to benchmark or run `daydream-scope` until you re-enter a GPU compute allocation (or re-launch the container/job with proper GPU device permissions).

### “Why did FPS regress?” quick checklist

When you see ~`8.8 FPS` again, it’s almost always one of these:

- **Wrong env / wrong torch**: `python -c "import torch; print(torch.__version__, torch.version.cuda)"` should be `2.9.0+cu130` / `13.0` for the B300 env.
- **Missing FlashAttention**: if `import flash_attn` fails, KV-bias can fall back to flex (and perf can collapse). On SM103 we now avoid Triton fallback by default because it is usually catastrophically slow.
- **Wrong backend**: `SCOPE_KV_BIAS_BACKEND=triton` is unusable on SM103 (forced scalar kernel in Triton 3.5).
- **ptxas mismatch**: Triton/Inductor need a `ptxas` that knows `sm_103` (`/usr/local/cuda-12.9+`).
- **Competing GPU work**: check `nvidia-smi` for other running processes (e.g. a background `daydream-scope` server can cut benchmark FPS in half).

**Next optimization step (denoise):** `denoise` (+ `recompute_kv_cache`) is now the dominant cost on the cu130 stack. Run with `PROFILE_ATTENTION=1` to split transformer time into `self_attn` vs `cross_attn` vs `ffn`, then decide whether to pursue attention work (FA4/FlashAttention/Triton) or GEMM/compile work.

**New tool (op-level profile):** `scripts/profile_krea_pipeline_ops.py` produces a torch-profiler-style table that helps answer “is it attention vs copies vs GEMMs?”. Artifacts from a representative run:
- `outputs/b300_cu130_ops_profile_fa4.json` (FA4 KV-bias)
- `outputs/b300_cu130_ops_profile_flash.json` (Flash segment-combine KV-bias)

High-level takeaway from those profiles: a large fraction of GPU time shows up as `aten::copy_` / `aten::to` / elementwise kernels and FP8 GEMMs (`aten::_scaled_mm`), with attention kernels still significant but not the only lever.

Collab note: profiling work is intentionally biased toward **adding measurement scripts + notes** (like `scripts/profile_krea_pipeline_ops.py`) and avoiding core pipeline changes until we have data. If you see churn in profiling scripts, that’s expected.

Latest B300 `PROFILE_ATTENTION=1` signal (cu130, `kv_cache_attention_bias=0.3`, `--iters 6 --skip 2`):
- Transformer Block Split (top-level): `self_attn` ~`56%`, `cross_attn` ~`22%`, `ffn` ~`22%` (with `SCOPE_KV_BIAS_BACKEND=flash`)
- self_attn Breakdown (nested): KV-bias is ~`38%` of `self_attn` (~`0.91ms/call`) on `flash`, and drops to ~`22%` (~`0.42ms/call`) on `fa4`

Interpretation: the biggest denoise win is still reducing `self_attn` time. FA4 score_mod materially reduces the KV-bias slice, but the remaining `other_in_self` (QKV projections + non-bias attention) is still the majority.

KV-bias backend A/B (**perf-only**; fp8 output quality broken) (B300, cu130, `320x576`, `kv_cache_attention_bias=0.3`, fp8):
- `SCOPE_KV_BIAS_BACKEND=flash`: **~15.2 FPS**
- `SCOPE_KV_BIAS_BACKEND=fa4`: **~17.3 FPS**
- `SCOPE_KV_BIAS_BACKEND=triton`: **~1 FPS** (still unusable on SM103 + triton 3.5)

Interpretation: do **not** use the Triton Kernel B backend on SM103 right now. It is forced into a slow scalar kernel on SM103+triton 3.5 to avoid a tcgen05 LLVM hard-abort, and the end-to-end result is unusable. Prefer `SCOPE_KV_BIAS_BACKEND=fa4` when it works, otherwise keep the SM103 default on the flash backend (segment-combine).

KV-bias backend A/B (B300, cu130, `320x576`, `kv_cache_attention_bias=0.3`, quantization none):
- `SCOPE_KV_BIAS_BACKEND=flash`: **~17.2 FPS**
- `SCOPE_KV_BIAS_BACKEND=fa4`: **~19.7 FPS**

### Quantization Note (B300)

> **⚠️ WARNING (2025-12-27): FP8 produces garbage output on B300.**
>
> While benchmark scripts report FPS numbers, the actual server output with FP8 quantization is **gray distorted noise** — not usable video. This affects both `--compile-fp8` and FP8-only (no compile) modes.
>
> **Root cause (likely):** TorchAO cpp extensions are being skipped:
> ```
> Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu130 for torchao version 0.15.0+cu130
> ```
> This warning appears at import time and suggests FP8 kernels may be falling back to broken codepaths. See upstream: https://github.com/pytorch/ao/issues/2919
>
> **Recommendation:** Use `--quantization none` (BF16) on B300 until the TorchAO/torch 2.9 compatibility issue is resolved.

The FPS numbers below are from benchmark scripts and **do not reflect usable output quality**:
- No compile:
  - `--quantization fp8_e4m3fn`: **~17.3 FPS** (garbage output)
  - `--quantization none` (bf16 weights): **~19.7 FPS** (works)
- With compile:
  - `--compile --quantization fp8_e4m3fn`: **~25.1 FPS** (garbage output; requires PerTensor-only TorchAO `as_strided` monkeypatch)
  - `--compile --quantization none`: **~29.36 FPS** (works; requires `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1` for the current best number)

Interpretation: FP8 is currently broken on B300/torch 2.9+cu130 due to TorchAO cpp extension incompatibility. Use BF16 (`--quantization none`) for working output.

### KV Cache Recompute Cadence (Perf Win, Quality Regression)

Implemented an experimental knob to skip `recompute_kv_cache` in steady-state:

```bash
export SCOPE_KV_CACHE_RECOMPUTE_EVERY=2  # default 1 (always recompute)
```

Measured (B300 cu130, `320x576`, fp8, `kv_cache_attention_bias=0.3`, `SCOPE_KV_BIAS_BACKEND=fa4`):
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY=1`: ~`15.0 FPS`
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2`: ~`16.8 FPS` (alternating fast/slow iters)

**Quality result (Daydream, B300):** `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` **visibly glitches**. This knob is **debug-only**; do not use it for real runs.

Shortcut: `scripts/bench_b300_recompute_cadence.sh` sweeps `SCOPE_KV_CACHE_RECOMPUTE_EVERY` values and writes logs/JSON under `outputs/`.

### Quick Sanity: Bias Disabled (kv_cache_attention_bias=1.0)

This is only a performance sanity check (quality impact TBD):

- With `SCOPE_KV_BIAS_BACKEND=fa4` (FA4 disabled for non-bias attention to avoid CuTe mixing): **~14.20 FPS**
- With `SCOPE_KV_BIAS_BACKEND=flash` (FA4 allowed for non-bias attention): **~15.18 FPS**

Takeaway: the `SCOPE_KV_BIAS_BACKEND=fa4` safety disable can cost ~1 FPS **when bias is disabled** because it forces the plain attention path onto FA2. This doesn’t necessarily apply when bias is enabled (`kv_cache_attention_bias=0.3`), because KV-bias uses FA4 score_mod directly.

### torch.compile Status (B300)

As of 2025-12-27:

- **Quantization none:** `scripts/profile_krea_pipeline_blocks.py --compile` now works on B300 and improves throughput.
  - Example (B300, cu130 env, `320x576`, bias `0.3`, `SCOPE_KV_BIAS_BACKEND=fa4`):
    - `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=0`: **~21.45 FPS** (`outputs/b300_cu130_triton351_compile_default_blocks_perf.log`)
    - `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1`: **~34.5 FPS** (`outputs/b300_cu130_compile_best_blocks_no_kvcache_zero_2025-12-27.log`)
  - Tradeoff: longer warmup due to compilation (expect ~10–30s depending on cache state).
- **FP8 (torchao):** ⚠️ **BROKEN on B300** — produces garbage output (gray noise). The `as_strided` monkeypatch is applied but TorchAO cpp extensions are skipped due to torch 2.9+cu130 incompatibility, causing FP8 kernels to malfunction. Use `--quantization none` (BF16) instead. Patch code: `src/scope/core/compat/torchao_float8_as_strided.py` (upstream issue: [`torchao-as-strided-dispatch.md`](../../issues/torchao-as-strided-dispatch.md)).

Implementation note: we keep CuTe/FA4 calls **opaque** to Dynamo during compilation to avoid FakeTensor/DLPack failures; this enables compiling the surrounding transformer regions without trying to trace into CUTLASS DSL.

Noise note: if you previously saw `torch/_dynamo` “Backend compiler exception … aten._local_scalar_dense” spam from `triton_rope_fused.py`, it should now be gone (we disable Dynamo for `_as_int3()` which does `.tolist()` scalar extraction).

Server opt-in: set `SCOPE_COMPILE_KREA_PIPELINE=1` before launching. `scripts/run_daydream_b300.sh` defaults it to `1`; the server still disables compile by default when quantization is enabled (override for experiments with `SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION=1`).
Convenience: `scripts/run_daydream_b300.sh --max-autotune` sets `SCOPE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs`.

**Compile mode experiments:** `src/scope/core/pipelines/krea_realtime_video/pipeline.py` supports `SCOPE_TORCH_COMPILE_MODE`, but on B300/SM103:
- `max-autotune*` can hard-abort with a tcgen05 LLVM intrinsic error (Triton/Inductor SM103). Guardrail: we ignore `max-autotune*` on SM103 unless `SCOPE_ALLOW_MAX_AUTOTUNE_SM103=1` (and you should have `triton>=3.5.1`).
- `reduce-overhead` (CUDAGraph Trees) is known-bad: we can hit `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`.
  - Tried: `SCOPE_CUDAGRAPH_MARK_STEP_BEGIN=1`, stabilizing KV-cache index tensors, compiling the whole model in `reduce-overhead` mode → still unstable.
  - Guardrail: on SM103 we now ignore `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` unless `SCOPE_ALLOW_REDUCE_OVERHEAD_SM103=1`.
Recommendation: leave `SCOPE_TORCH_COMPILE_MODE` unset (default) unless you’re explicitly experimenting.

**Compile strategy experiments:** by default we compile **each transformer block** (keeps graph breaks localized). Whole-model compilation is slightly worse here and increases warmup:
- Default (`blocks`): **~34.49 FPS**, warmup `~15.7s` (`outputs/b300_cu130_compile_best_blocks_no_kvcache_zero_2025-12-27.log`)
- Whole model (`model`): **~34.43 FPS**, warmup `~17.3s` (`outputs/b300_cu130_compile_strategy_model_best_2025-12-27.log`)

**FA4 varlen opt-in (non-bias attention):** when `SCOPE_KV_BIAS_BACKEND=fa4`, FA4/CuTe varlen attention remains disabled by default (stable FA2 for non-bias attention). You can opt in with `SCOPE_ENABLE_FA4_VARLEN=1`; on B300 compiled BF16 this was a ~2–3% win (≈`30.08 → 30.76` FPS) but increased warmup/JIT time (≈`12s → 19s`). `scripts/run_daydream_b300.sh` now defaults this to `1` (set `SCOPE_ENABLE_FA4_VARLEN=0` to disable).

Update: FA4 score_mod KV-bias is now working on B300 and is faster than flash segment-combine at the canonical resolution. It required:
- Removing static `import imageio` debug imports in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (cutlass-dsl AST preprocessor imports everything in the module).
- Normalizing B=1 K/V slice stride (use `[0].unsqueeze(0)` views) before calling FA4, to avoid “Can't decude the leading dimension…” when slicing from the KV cache tensor.
- Default-disabling FlashAttention 4 (CuTe) for non-bias attention when `SCOPE_KV_BIAS_BACKEND=fa4` (opt-in via `SCOPE_ENABLE_FA4_VARLEN=1`) to avoid mixing CuTe module variants.

## Block Profile (Current Best Config)

**Config:** cu130 env, BF16, `--compile`, FA4 KV-bias + FA4 varlen, `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1`

Per-call breakdown (~34.5 FPS):

| Block | Per-call | Share | Notes |
|------:|-------:|------:|-------|
| `denoise` | ~227 ms | ~65% | Dominant; next optimization target |
| `recompute_kv_cache` | ~60 ms | ~17% | Still non-trivial; runs every iteration |
| `decode` | ~60 ms | ~17% | Decode is no longer the bottleneck (was ~195ms pre resample fix) |

**Interpretation:** VAE decode is no longer the bottleneck. The mountain is now `denoise` + `recompute_kv_cache`.

### Historical: Pre-Optimization Block Profiles

<details>
<summary>Repo-default stack (~8.8 FPS)</summary>

Artifact: `outputs/b300_pipeline_blocks_profile.json`

| Block | GPU ms | Share | Per-call |
|------:|-------:|------:|---------:|
| `denoise` | ~4208 | ~46% | ~1052 ms |
| `decode` | ~3010 | ~33% | ~752 ms |
| `recompute_kv_cache` | ~1218 | ~13% | ~305 ms |
| `text_conditioning` | ~655 | ~7% | ~164 ms |

</details>

<details>
<summary>cu130 + FlashAttention, pre-VAE fix (~19-20 FPS)</summary>

Artifact: `outputs/b300_cu130_fp8_bias03_blocks_profile.json`

| Block | GPU ms | Share | Per-call |
|------:|-------:|------:|---------:|
| `denoise` | ~2113 | ~62% | ~528 ms |
| `decode` | ~833 | ~25% | ~208 ms |
| `recompute_kv_cache` | ~410 | ~12% | ~102 ms |
| `text_conditioning` | ~36 | ~1% | ~9 ms |

</details>

## New Evidence (Deeper Drill: Decode Dominates)

Using the denoise/decode CUDA-event profilers (post-warmup, `320x576`, `4` steps, `kv_cache_attention_bias=0.3`):

- `denoise` ≈ **all** `components.generator(...)` time; `randn` and `scheduler_add_noise` are negligible.
- `decode` ≈ **all** `WanVAEWrapper.decode_to_pixel(...).stream_decode(...)` time.
- `decode` is currently the single largest block on B300 in steady-state (~`740–760ms` per pipeline call for `t=3` frames).

This strengthens the hypothesis that the main B300 gap is likely in **cuDNN / conv3d performance** (and therefore may depend strongly on having an SM103-native PyTorch/CUDA runtime stack).

### Update: SM103-Native Stack Matters (Big)

A decode-only microbenchmark shows `WanVAE_.stream_decode(t=3)` is **~3.9× faster** on a cu130 stack:

- `torch 2.8.0+cu129` (CUDA 12.9, cuDNN 9.10): ~`760ms/call`
- `torch 2.9.0+cu130` (CUDA 13.0, cuDNN 9.13): ~`194ms/call`

Artifacts:
- `outputs/b300_cu129_vae_stream_decode_bench.log`
- `outputs/b300_cu130_vae_stream_decode_bench.log`

Implication: To move B300 off ~8.8 FPS, **prioritize a cu130 (or newer) runtime** over more attention microkernel work.

### Update: cu130 env needs FlashAttention installed

Without `flash_attn` installed in the cu130 env, KV-bias falls back to Triton (and flex_attention can run unfused), which regresses end-to-end to ~`1 FPS`.

Recommended: use the repo script (it keeps the shared `.venv` untouched and repairs the env end-to-end):

```bash
# Default: upgrades Triton to 3.5.1 (SM103 tcgen05 fix) and installs FA4 deps.
./scripts/b300_env_fix_cu130.sh .venv-b300-cu130-decode

# Optional knobs:
# - TRITON_VERSION=3.5.1         (override if needed)
# - INSTALL_FA4_DEPS=0           (skip cuda-python + nvidia-cutlass-dsl)
```

Manual install (in `.venv-b300-cu130-decode`; note the CUDA extension is large, ~1GB):

```bash
uv pip install -p .venv-b300-cu130-decode/bin/python pip wheel ninja packaging
FLASH_ATTENTION_SKIP_CUDA_BUILD=0 \
  .venv-b300-cu130-decode/bin/python -m pip install --force-reinstall \
    --no-deps --no-build-isolation --no-binary flash-attn \
    flash-attn==2.8.3
```

FA4 / CuTe (recommended for the B300 KV-bias path): install `cuda-python` + `nvidia-cutlass-dsl`:

```bash
uv pip install -p .venv-b300-cu130-decode/bin/python cuda-python nvidia-cutlass-dsl==4.1.0
```

On SM103, `scope.core.compat.sm103.patch_cutlass_for_sm103()` now patches the CUTLASS DSL arch checks at runtime, so you should not need to modify site-packages (the legacy script `scripts/patch_cutlass_sm103.sh` still exists if you want a persistent on-disk patch).

If the env ever gets clobbered back to cu128 (e.g. by `uv sync`), restore it with:

```bash
./scripts/b300_env_fix_cu130.sh .venv-b300-cu130-decode
```

## Implemented Knobs (B300 Experiments)

1) **Faster VAE streaming decode**

`WanVAE_.stream_decode()` supports:

- `WANVAE_STREAM_DECODE_MODE=chunk` (default; fewer decoder calls, slightly faster)
- `WANVAE_STREAM_DECODE_MODE=loop` (old per-frame loop; slower on B300)

2) **cuDNN benchmark**

`scripts/profile_krea_pipeline_blocks.py --cudnn-benchmark` improved steady-state FPS slightly on B300 (at the cost of slower warmup).

3) **Disable fused QKV projections (B300 hazard)**

On B300/SM103, fused projections (`to_qkv(...).chunk(3, dim=-1)`) can create strided Q/K views and trigger extra materialization work downstream **in eager mode**.

In the **compiled** path, we currently see a small win from enabling fused projections.

- Eager/debug: set `SCOPE_DISABLE_FUSED_PROJECTIONS=1`
- Compiled path: set `SCOPE_DISABLE_FUSED_PROJECTIONS=0` (default in `scripts/run_daydream_b300.sh` when compile is enabled)

4) **Experimental: RoPE(K) directly into KV cache**

This avoids one explicit K copy by writing the RoPE’d K directly into the KV cache window.

- Set `SCOPE_ROPE_K_TO_CACHE=1` (currently ~neutral; keep opt-in)

5) **VAE decode: channels-last 3D activations**

This uses `torch.channels_last_3d` for the Conv3d-heavy VAE decode activations (small but measurable win on B300).

- Set `WANVAE_DECODE_CHANNELS_LAST_3D=1` (default in `scripts/run_daydream_b300.sh`)

6) **VAE decode: keep `Resample` outputs contiguous (big win)**

In streaming decode, the `Resample(upsample3d)` cached branch can emit a non-contiguous 5D tensor that forces many Conv3d ops onto
`aten::slow_conv_dilated3d` (vol2col). Enabling this knob re-contiguates the 5D activation (channels-last-3d when enabled), restoring cuDNN/CUTLASS conv3d.

- Set `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1` (default in `scripts/run_daydream_b300.sh`)
- Measured (B300, cu130, `320x576`, bias `0.3`, quantization `none`, `--compile`, FA4): ~`21.45 → 29.36 FPS` and decode `~195ms → ~60ms` (see `experiments.md`)

## Key Discovery This Session

The `flash-attention` symlink was shadowing the working FA4 package. Removed it, FA4 now imports correctly:
- `FLASH_ATTN_4_AVAILABLE: True`
- FA4 benchmarks show 0.38ms for KV-cache attention (vs 1.6ms Triton)

**BUT FPS stays at exactly 8.8 regardless of:**
- Enabling FA4 for main attention
- Setting `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0` (bypasses Triton Kernel B)
- Changing codec MAX_FRAME_RATE from 8 to 30

This originally suggested a “hard limiter”, but the block profile above points to a more mundane explanation: **the slow path is dominated by non-attention work** (`denoise` / `decode` / `recompute_kv_cache`).

## Historical: Two Codex Agents (Completed)

<details>
<summary>FA4 score_mod + Segment-combine work (both complete)</summary>

### Codex1: FA4 score_mod fix (preferred approach) — ✅ DONE
- Patched `flash-attention.bak` to replace `FastDivmodDivisor` with custom `FastDivmod`
- Files modified: `fast_math.py`, `tile_scheduler.py`, `paged_kv.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `flash_fwd_combine.py`, `flash_bwd_sm100.py`
- **Status: Complete, working on B300**

### Codex2: Segment-combine fallback — ✅ DONE
- Added `_kv_bias_flash_combine()` - splits KV into 3 segments, runs FA4 on each, merges with LSE
- Auto-detects SM103 → defaults to "flash" backend
- **Status: Complete, merged into causal_model.py**

Both approaches work. FA4 score_mod (`SCOPE_KV_BIAS_BACKEND=fa4`) is now the recommended default on B300.

</details>

## Current Code State

| Component | Status |
|-----------|--------|
| `flash-attention` symlink | Missing (removed) |
| `flash-attention.bak/` | Has Codex1's FastDivmod patches (incomplete) |
| `causal_model.py` | Has Codex2's flash-combine + Codex1's path extension |
| `webrtc.py` | MAX_FRAME_RATE changed 8→30 |
| `pipeline.py` | Added SCOPE_KV_CACHE_ATTENTION_BIAS env var |

## Backend Selection on B300

```python
_KV_BIAS_BACKEND = "flash" if _is_sm103() else "triton"
```

- `fa4`: FA4 with score_mod (needs Codex1 to finish + symlink)
- `flash`: Segment-combine (Codex2, currently active on B300)
- `triton`: Triton Kernel B (slow on B300: 1.6ms)
- `flex`: flex_attention fallback

## Repo/Packaging Note (Reproducibility)

- `flash-attention.bak/` is gitignored and not available to RepoPrompt / other machines by default.
- If/when we want the FA4 score_mod path reproducible in-repo, prefer vendoring the minimal CuTe sources (already present at `vendored/flash_attn_cute_score_mod/`) rather than relying on `flash-attention.bak/`.

## To Test When Codex1 Finishes

```bash
# Recreate symlink
ln -s flash-attention.bak flash-attention

# Run with FA4 score_mod
SCOPE_KV_BIAS_BACKEND=fa4 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
```

## To Test Current State (Codex2's approach)

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope
```

## Next Questions (Now That It Looks Compute-Bound)

Priority is to drill into the *dominant* blocks:

- Within `denoise`: how much is attention vs other ops (conv/linear/norm/rope/etc.)?
- Within `decode`: is VAE decode unusually slow on SM103 (cuDNN algo selection / kernel fallback / precision path)?
- Can we reduce `recompute_kv_cache` work without hurting quality (knobs like cache lengths / frames)?

Practical note: `torch.profiler` CUDA timelines appear broken on this environment (CUPTI errors). Prefer CUDA-event based probes (`PROFILE_PIPELINE_BLOCKS`, `PROFILE_ATTENTION`) or external `nsys` if available.

## Files to Investigate

- `/root/scope/src/scope/server/frame_processor.py` - FPS calculation at line 296
- `/root/scope/src/scope/server/tracks.py` - Frame rate control
- The actual pipeline call path that generates frames

## Environment

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export DISABLE_FLEX_ATTENTION_COMPILE=1  # needed for torch 2.9 / SM103 (tcgen05 LLVM abort)
# Optional overrides:
export SCOPE_KV_BIAS_BACKEND=flash  # or fa4, triton
export SCOPE_KV_CACHE_ATTENTION_BIAS=1.0  # disable bias for plain FA4
export PROFILE_ATTENTION=1  # enable profiling
```
