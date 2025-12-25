# B300 Session State - 2025-12-24 (Updated)

## Current Status

**Baseline (repo default stack): B300 is ~8.8 FPS at `320x576` (reference resolution) and the number is stable across iterations.**

**Update (cu130 + FlashAttention):**
- Daydream end-to-end: **~14.8–15.0 FPS** at `320x576` (canonical)
- `scripts/profile_krea_pipeline_blocks.py` benchmark: **~13.3–13.5 FPS** at `320x576` (see artifacts below)
- `torchao` note: repo pins `torchao==0.13.0` (torch 2.8 ABI). For torch `2.9.0+cu130`, install `torchao==0.14.1` (or try `0.15.0`) to avoid “Skipping import of cpp extensions…” warnings; `scripts/b300_env_fix_cu130.sh` now does this best-effort via `TORCHAO_VERSION=...`.

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
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization fp8_e4m3fn \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark
```

Or for a one-shot denoise/decode drill-down (writes JSON artifacts under `outputs/`):

```bash
scripts/profile_b300_denoise_drilldown.sh
```

Latest drill-down run (cu130, profiling enabled): `outputs/b300_cu130_fp8_bias03_drilldown_perf.log` averaged **~11.4 FPS** (expected lower than the non-profiled benchmark due to extra synchronizations).

Artifacts:
- `outputs/b300_cu130_fp8_bias03_flashattn.log`
- `outputs/b300_cu130_fp8_bias03_flashattn.json`

**Key update:** This no longer looks like an “invisible FPS cap” caused by WebRTC/codec pacing or CPU stalls. A block-level CUDA-event profile shows GPU time ≈ wall time, and the dominant blocks are `denoise` and `decode` (not the KV-bias attention microkernel).

See also: `notes/FA4/b300/investigation-runbook.md` (“Ground Truth” section).

**Operational note:** The box can intermittently wedge CUDA/NVML (`nvidia-smi` → “Failed to initialize NVML: Unknown Error”, PyTorch `cudaGetDeviceCount` → Error 304). When that happens, profiling/bench scripts can crash at import-time and you’ll likely need a host-level driver restart or reboot.

**Next optimization step (denoise):** `denoise` (+ `recompute_kv_cache`) is now the dominant cost on the cu130 stack. Run with `PROFILE_ATTENTION=1` to split transformer time into `self_attn` vs `cross_attn` vs `ffn`, then decide whether to pursue attention work (FA4/FlashAttention/Triton) or GEMM/compile work.

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
