# B300 Session State - 2025-12-24 (Updated)

## Navigation

- **How to reproduce / measure:** `notes/FA4/b300/investigation-runbook.md`
- **Where to record “one-change” trials:** `notes/FA4/b300/experiments.md`
- **Roadmap / next levers:** `notes/FA4/b300/optimization-vision.md`
- **Development plan (what to build next):** `notes/FA4/b300/development-plan.md`
- **Level 5/6 reading list:** `notes/FA4/b300/level5-level6-resources.md`

## Current Status

**Baseline (repo default stack): B300 is ~8.8 FPS at `320x576` (reference resolution) and the number is stable across iterations.**

**Update (cu130 + FlashAttention + FA4 KV-bias):**
- Daydream end-to-end (cu130 env): **~14.8–15.0 FPS** at `320x576` (canonical; measured before defaulting to FA4 KV-bias)
- `scripts/profile_krea_pipeline_blocks.py` benchmark (cu130 env, quantization none, bias=0.3):
  - `SCOPE_KV_BIAS_BACKEND=flash`: **~14.9 FPS**
  - `SCOPE_KV_BIAS_BACKEND=fa4`: **~16.7 FPS**
  - `SCOPE_KV_BIAS_BACKEND=fa4` + `--compile`: **~19.0 FPS**
- `torchao` note: repo pins `torchao==0.13.0` (torch 2.8 ABI). For torch `2.9.0+cu130`, `scripts/b300_env_fix_cu130.sh` now tries `torchao==0.15.0+cu130` from the cu130 index (then PyPI as fallback). **As of 2025-12-25**, `torchao==0.15.0+cu130` still prints `Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu130 ...` (likely upstream; no FPS change observed).

This is a ~70% improvement over the repo-default baseline.

External doc brief (for RepoPrompt / web research): `notes/FA4/b300/blackwell-docs.md`

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
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark
```

Or for a one-shot denoise/decode drill-down (writes JSON artifacts under `outputs/`):

```bash
scripts/profile_b300_denoise_drilldown.sh
```

Latest drill-down run (cu130, profiling enabled): `outputs/b300_cu130_none_bias0.3_drilldown_perf.log` averaged **~14.7 FPS** (expected lower than the non-profiled benchmark due to extra synchronizations).

Artifacts:
- `outputs/b300_cu130_fp8_bias03_flashattn.log`
- `outputs/b300_cu130_fp8_bias03_flashattn.json`

**Key update:** This no longer looks like an “invisible FPS cap” caused by WebRTC/codec pacing or CPU stalls. A block-level CUDA-event profile shows GPU time ≈ wall time, and the dominant blocks are `denoise` and `decode` (not the KV-bias attention microkernel).

See also: `notes/FA4/b300/investigation-runbook.md` (“Ground Truth” section).

**Operational note (intermittent):** We have seen rare CUDA/NVML wedges on this host (`nvidia-smi` → “Failed to initialize NVML: Unknown Error”, PyTorch `cudaGetDeviceCount` → Error 304). When `nvidia-smi` is healthy (normal driver table), you’re fine; when it wedges, bench scripts can fail at import-time and you’ll likely need a host-level driver restart or reboot.

### “Why did FPS regress?” quick checklist

When you see ~`8.8 FPS` again, it’s almost always one of these:

- **Wrong env / wrong torch**: `python -c "import torch; print(torch.__version__, torch.version.cuda)"` should be `2.9.0+cu130` / `13.0` for the B300 env.
- **Missing FlashAttention**: if `import flash_attn` fails, KV-bias can fall back to Triton/flex and go catastrophic on SM103.
- **Wrong backend**: `SCOPE_KV_BIAS_BACKEND=triton` is unusable on SM103 (forced scalar kernel in Triton 3.5).
- **ptxas mismatch**: Triton/Inductor need a `ptxas` that knows `sm_103` (`/usr/local/cuda-12.9+`).
- **Competing GPU work**: check `nvidia-smi` for other running processes (e.g. a background `daydream-scope` server can cut benchmark FPS in half).

**Next optimization step (denoise):** `denoise` (+ `recompute_kv_cache`) is now the dominant cost on the cu130 stack. Run with `PROFILE_ATTENTION=1` to split transformer time into `self_attn` vs `cross_attn` vs `ffn`, then decide whether to pursue attention work (FA4/FlashAttention/Triton) or GEMM/compile work.

**New tool (op-level profile):** `scripts/profile_krea_pipeline_ops.py` produces a torch-profiler-style table that helps answer “is it attention vs copies vs GEMMs?”. Artifacts from a representative run:
- `outputs/b300_cu130_ops_profile_fa4.json` (FA4 KV-bias)
- `outputs/b300_cu130_ops_profile_flash.json` (Flash segment-combine KV-bias)

High-level takeaway from those profiles: a large fraction of GPU time shows up as `aten::copy_` / `aten::to` / elementwise kernels and FP8 GEMMs (`aten::_scaled_mm`), with attention kernels still significant but not the only lever.

Latest B300 `PROFILE_ATTENTION=1` signal (cu130, `kv_cache_attention_bias=0.3`, `--iters 6 --skip 2`):
- Transformer Block Split (top-level): `self_attn` ~`56%`, `cross_attn` ~`22%`, `ffn` ~`22%` (with `SCOPE_KV_BIAS_BACKEND=flash`)
- self_attn Breakdown (nested): KV-bias is ~`38%` of `self_attn` (~`0.91ms/call`) on `flash`, and drops to ~`22%` (~`0.42ms/call`) on `fa4`

Interpretation: the biggest denoise win is still reducing `self_attn` time. FA4 score_mod materially reduces the KV-bias slice, but the remaining `other_in_self` (QKV projections + non-bias attention) is still the majority.

KV-bias backend A/B (B300, cu130, `320x576`, `kv_cache_attention_bias=0.3`, fp8):
- `SCOPE_KV_BIAS_BACKEND=flash`: **~13.47 FPS** (`outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_flash.log`)
- `SCOPE_KV_BIAS_BACKEND=fa4`: **~15.01 FPS** (`outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_fa4.log`)
- `SCOPE_KV_BIAS_BACKEND=triton`: **~1.07 FPS** (`outputs/b300_cu130_fp8_e4m3fn_bias0.3_kvbias_triton.log`)

Interpretation: do **not** use the Triton Kernel B backend on SM103 right now. It is forced into a slow scalar kernel on SM103+triton 3.5 to avoid a tcgen05 LLVM hard-abort, and the end-to-end result is unusable. Prefer `SCOPE_KV_BIAS_BACKEND=fa4` when it works, otherwise keep the SM103 default on the flash backend (segment-combine).

KV-bias backend A/B (B300, cu130, `320x576`, `kv_cache_attention_bias=0.3`, quantization none):
- `SCOPE_KV_BIAS_BACKEND=flash`: **~14.9 FPS**
- `SCOPE_KV_BIAS_BACKEND=fa4`: **~16.7 FPS**

### Quantization Note (B300)

On B300, FP8 quantization via torchao is not automatically a win. In a direct A/B run (same settings, FA4 backend):
- `--quantization fp8_e4m3fn`: **~15.0 FPS**
- `--quantization none` (bf16 weights): **~16.7 FPS**

Interpretation: FP8 introduces extra conversion/scaling overhead on this stack. If you have ample VRAM (B300), consider running unquantized for higher FPS and simpler dependencies.

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

As of 2025-12-25:

- **Quantization none:** `scripts/profile_krea_pipeline_blocks.py --compile` now works on B300 and improves throughput.
  - Example (B300, cu130 env, `320x576`, bias `0.3`, `SCOPE_KV_BIAS_BACKEND=fa4`): **~16.7 FPS → ~19.0 FPS**
  - Tradeoff: longer warmup due to compilation (expect ~10–30s depending on cache state).
- **FP8 (torchao):** still fails under `--compile` due to float8 wrapper limitations under Dynamo/Inductor.

Implementation note: we keep CuTe/FA4 calls **opaque** to Dynamo during compilation to avoid FakeTensor/DLPack failures; this enables compiling the surrounding transformer regions without trying to trace into CUTLASS DSL.

Noise note: if you previously saw `torch/_dynamo` “Backend compiler exception … aten._local_scalar_dense” spam from `triton_rope_fused.py`, it should now be gone (we disable Dynamo for `_as_int3()` which does `.tolist()` scalar extraction).

Server opt-in: set `SCOPE_COMPILE_KREA_PIPELINE=1` before launching. `scripts/run_daydream_b300.sh` defaults it to `1`; the server still disables compile by default when quantization is enabled (override for experiments with `SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION=1`).

**Compile mode experiments:** `src/scope/core/pipelines/krea_realtime_video/pipeline.py` supports `SCOPE_TORCH_COMPILE_MODE`, but on B300/SM103:
- `max-autotune-no-cudagraphs` can hard-abort with a tcgen05 LLVM intrinsic error.
- `reduce-overhead` is still failing with a CUDAGraphs “output overwritten” runtime error even after adding a `SCOPE_CUDAGRAPH_MARK_STEP_BEGIN=1` experiment (calls `torch.compiler.cudagraph_mark_step_begin()` before model invocations).
Recommendation: leave `SCOPE_TORCH_COMPILE_MODE` unset (default).

**FA4 varlen opt-in (non-bias attention):** when `SCOPE_KV_BIAS_BACKEND=fa4`, FA4/CuTe varlen attention remains disabled by default (stable FA2 for non-bias attention). You can opt in with `SCOPE_ENABLE_FA4_VARLEN=1`; on B300 this was a small (~1–2%) win but increased warmup/JIT time.

Update: FA4 score_mod KV-bias is now working on B300 and is faster than flash segment-combine at the canonical resolution. It required:
- Removing static `import imageio` debug imports in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (cutlass-dsl AST preprocessor imports everything in the module).
- Normalizing B=1 K/V slice stride (use `[0].unsqueeze(0)` views) before calling FA4, to avoid “Can't decude the leading dimension…” when slicing from the KV cache tensor.
- Default-disabling FlashAttention 4 (CuTe) for non-bias attention when `SCOPE_KV_BIAS_BACKEND=fa4` (opt-in via `SCOPE_ENABLE_FA4_VARLEN=1`) to avoid mixing CuTe module variants.

## New Evidence (Zoom-Out Block Profile)

Artifact: `outputs/b300_pipeline_blocks_profile.json` (generated via `PROFILE_PIPELINE_BLOCKS=1`).

Per-block GPU time breakdown (4 pipeline calls):

| Block | GPU ms | Share (of total GPU) | Per-call |
|------:|-------:|----------------------:|---------:|
| `denoise` | ~4208 | ~46% | ~1052 ms |
| `decode` | ~3010 | ~33% | ~752 ms |
| `recompute_kv_cache` | ~1218 | ~13% | ~305 ms |
| `text_conditioning` | ~655 | ~7% | ~164 ms |

**Interpretation:** End-to-end throughput is largely compute-bound inside `denoise` (+ `decode`), so swapping KV-bias backends (Triton Kernel B vs FA4 vs segment-combine) won’t materially move FPS unless it changes time inside `denoise` (or reduces work in `decode` / `recompute_kv_cache`).

### Update: cu130 + FlashAttention Block Profile (Much Better)

Artifact: `outputs/b300_cu130_fp8_bias03_blocks_profile.json` (generated via `--profile-blocks` in the cu130 env).

Aggregated GPU time breakdown (4 pipeline calls):

| Block | GPU ms | Share (of total GPU) | Per-call |
|------:|-------:|----------------------:|---------:|
| `denoise` | ~2113 | ~62% | ~528 ms |
| `decode` | ~833 | ~25% | ~208 ms |
| `recompute_kv_cache` | ~410 | ~12% | ~102 ms |
| `text_conditioning` | ~36 | ~1% | ~9 ms |

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

Install (in `.venv-b300-cu130-decode`; note the CUDA extension is large, ~1GB):

```bash
uv pip install -p .venv-b300-cu130-decode/bin/python wheel ninja
uv pip install -p .venv-b300-cu130-decode/bin/python --no-deps --no-build-isolation --no-binary flash-attn flash-attn==2.8.3
```

Optional (FA4 / CuTe): if you want `FLASH_ATTN_4_AVAILABLE=True` on B300 in this env, you also need `cuda-python` + `nvidia-cutlass-dsl`:

```bash
uv pip install -p .venv-b300-cu130-decode/bin/python cuda-python nvidia-cutlass-dsl==4.1.0
PATH=.venv-b300-cu130-decode/bin:$PATH ./scripts/patch_cutlass_sm103.sh .venv-b300-cu130-decode
```

This makes `flash_attn.cute` importable (CuTe). End-to-end FPS was still ~13.4 in our run (small/no gain vs FA2).

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

## Key Discovery This Session

The `flash-attention` symlink was shadowing the working FA4 package. Removed it, FA4 now imports correctly:
- `FLASH_ATTN_4_AVAILABLE: True`
- FA4 benchmarks show 0.38ms for KV-cache attention (vs 1.6ms Triton)

**BUT FPS stays at exactly 8.8 regardless of:**
- Enabling FA4 for main attention
- Setting `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0` (bypasses Triton Kernel B)
- Changing codec MAX_FRAME_RATE from 8 to 30

This originally suggested a “hard limiter”, but the block profile above points to a more mundane explanation: **the slow path is dominated by non-attention work** (`denoise` / `decode` / `recompute_kv_cache`).

## Two Codex Agents Working

### Codex1: FA4 score_mod fix (preferred approach)
- Patching `flash-attention.bak` to replace `FastDivmodDivisor` with custom `FastDivmod`
- Files modified: `fast_math.py`, `tile_scheduler.py`, `paged_kv.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `flash_fwd_combine.py`, `flash_bwd_sm100.py`
- **Status: Still working**

### Codex2: Segment-combine fallback
- Added `_kv_bias_flash_combine()` - splits KV into 3 segments, runs FA4 on each, merges with LSE
- Auto-detects SM103 → defaults to "flash" backend
- **Status: Complete, merged into causal_model.py**

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
