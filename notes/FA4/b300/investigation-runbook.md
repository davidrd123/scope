# B300 Investigation Runbook

**Purpose:** Systematic approach to diagnose why B300 is stuck at 8.8 FPS while B200 hits ~20 FPS.

**Key insight:** Attention backend changes don't move FPS. The bottleneck is elsewhere.

---

## Ground Truth (Already Measured on B300)

These observations are meant to prevent re-running the same “is it pacing?” and “is it attention?” loops.

### Repro Settings (Reference)

- **Resolution:** `320x576` (reference resolution for comparisons)
- **Denoising steps:** `4` (current reference)
- **KV-cache attention bias:** `0.3` (unless explicitly testing bias off/on)
- **Typical steady-state:** ~`8.7–8.8 FPS` on B300, stable across iterations

Repro command (includes per-block CUDA-event timings):

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
PROFILE_PIPELINE_BLOCKS=1 \
PROFILE_PIPELINE_BLOCKS_JSON=outputs/b300_pipeline_blocks_profile.json \
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 10 --skip 3 \
  --kv-cache-attention-bias 0.3
```

### “Zoom-Out” Block Profile (Key Finding)

`outputs/b300_pipeline_blocks_profile.json` shows **GPU time ~= CPU time**, i.e. the pipeline is not obviously paced or CPU-stalled; it’s largely **compute-bound** at the block level (CUDA events + `synchronize()` per block).

Dominant blocks (4 pipeline calls):

| Block | GPU ms | Share (of total GPU) | Per-call |
|------:|-------:|----------------------:|---------:|
| `denoise` | ~4208 | ~46% | ~1052 ms |
| `decode` | ~3010 | ~33% | ~752 ms |
| `recompute_kv_cache` | ~1218 | ~13% | ~305 ms |
| `text_conditioning` | ~655 | ~7% | ~164 ms |

Implication: attention micro-optimizations (FA4 score_mod, Triton Kernel B tuning, etc.) will not move end-to-end FPS unless they materially reduce time inside `denoise` (and possibly also reduce work in `decode` / `recompute_kv_cache`).

### SM103-Native Stack Hypothesis (New, High-Value)

On B300, **WanVAE decode is conv3d-heavy and cuDNN-dominated**. If the runtime’s cuDNN is missing fast SM103 kernels, decode can become the bottleneck even when attention is fast.

We now have strong evidence that **the PyTorch + cuDNN bundle matters dramatically** on SM103:

- **Torch 2.8 + cu129 (CUDA 12.9, cuDNN 9.10)**: `stream_decode(t=3)` is ~`760ms/call`
- **Torch 2.9 + cu130 (CUDA 13.0, cuDNN 9.13)**: `stream_decode(t=3)` is ~`194ms/call` (**~3.9× faster**)

If this carries over to the full pipeline, it would largely explain why B300 is stuck near ~8.8 FPS on the default cu128/cu129 stack.

### Why This Profile Is Trustworthy

- Per-block GPU timing is implemented via CUDA events in `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` (records start/end, then `synchronize()`).
- This makes timings “real” but also disables async overlap while profiling; use it for relative breakdown, not absolute peak throughput.

### Known Profiling Constraint (CUPTI)

On this B300 environment, `torch.profiler` CUDA timelines are currently unusable (observed `CUPTI_ERROR_INVALID_DEVICE` / `device_time=0`). Prefer:
- `PROFILE_PIPELINE_BLOCKS=1` (block-level CUDA events)
- `PROFILE_ATTENTION=1` (attention-level logging where available)
- External `nsys` if accessible on the host

### Repo/Packaging Reality (Important for Repro + RepoPrompt)

- `flash-attention.bak/` exists locally but is **ignored** by git (`.gitignore`), so other machines/models won’t see it.
- Minimal CuTe sources needed for FA4 `score_mod` experimentation are already vendored in-repo at `vendored/flash_attn_cute_score_mod/` and are referenced by the path-injection logic when `SCOPE_KV_BIAS_BACKEND=fa4`.
- Small artifacts live in `outputs/` (e.g. `outputs/b300_320x576_fa4.log`). The block-profile JSON is currently a local artifact; if it’s useful long-term, consider checking it in (or keep the table above as the durable record).

## Quick Reference

```bash
# Output directory (create per-GPU subdirs)
export INVESTIGATION_DIR=/tmp/gpu-investigation
mkdir -p $INVESTIGATION_DIR/{b200,b300,h100}

# Set GPU tag for this session
export GPU_TAG=b300  # or b200, h100
export OUT_DIR=$INVESTIGATION_DIR/$GPU_TAG
```

---

## Hypotheses

| ID | Hypothesis | Symptom if true |
|----|------------|-----------------|
| H1 | Output pacing limit (not compute) | gpu_ms << wall_ms |
| H2 | CPU bound | Low GPU util, high CPU |
| H3 | Power/thermal throttling | High GPU util, low clocks |
| H4 | GPU not exclusive (MIG/vGPU) | Other processes visible |
| H5 | Non-attention op falling back | GEMMs dominate profile |

---

## Test 0: Environment Baseline

Run on each GPU before any other tests.

```bash
# Capture system info
nvidia-smi -q > $OUT_DIR/nvidia-smi-full.txt
nvidia-smi -L > $OUT_DIR/nvidia-smi-devices.txt
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,power.limit --format=csv > $OUT_DIR/gpu-info.csv

# Check for other processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > $OUT_DIR/gpu-processes.csv

# CPU info
lscpu > $OUT_DIR/cpu-info.txt
free -h > $OUT_DIR/memory-info.txt
```

**Checkpoint:** If `gpu-processes.csv` shows other processes, H4 may be in play.

---

## Test 1: GPU vs Wall Time (H1 - Pacing)

This is the most important discriminating test.

```bash
# Run the block profiler with timing
uv run python scripts/profile_krea_pipeline_blocks.py \
  --iters 20 --skip 3 \
  --profile-blocks \
  --profile-blocks-json $OUT_DIR/block-profile.json \
  2>&1 | tee $OUT_DIR/profile-output.log
```

**What to look for:**
- Compare GPU time vs wall time per block
- If `gpu_ms << wall_ms` consistently → pacing/backpressure (H1)
- If `gpu_ms ≈ wall_ms` → compute bound, continue to H2-H5

### Test 1b: Drill Into `denoise` and `decode` (B300 Priority)

Once you know the pipeline is compute-bound, go one level deeper:

- **Denoise step profiler:** splits `denoise` into `generator` vs `randn` vs `scheduler_add_noise`
- **WanVAE decode profiler:** splits decode into wrapper-level steps (`prep_permute_cast`, `stream_decode`, `postprocess`)
- **WanVAE inner decode profiler (optional):** splits `stream_decode()` into `apply_scale`, `conv2`, decoder phases

Example (writes JSON artifacts into `$OUT_DIR/`):

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
PROFILE_PIPELINE_BLOCKS=1 \
PROFILE_PIPELINE_BLOCKS_JSON=$OUT_DIR/block-profile.json \
PROFILE_DENOISE_STEPS=1 \
PROFILE_DENOISE_STEPS_JSON=$OUT_DIR/denoise-steps.json \
PROFILE_WANVAE_DECODE=1 \
PROFILE_WANVAE_DECODE_JSON=$OUT_DIR/vae-decode.json \
PROFILE_WANVAE_DECODE_INNER=1 \
PROFILE_WANVAE_DECODE_INNER_JSON=$OUT_DIR/vae-decode-inner.json \
PROFILE_ATTENTION=1 \
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 10 --skip 3 \
  --kv-cache-attention-bias 0.3
```

Notes:
- These profilers use CUDA events + `synchronize()`, so they reduce overlap and will lower absolute throughput while enabled.
- `PROFILE_ATTENTION=1` helps answer “how much of `denoise` is attention vs everything else?”

Shortcut (B300, cu130 env): `scripts/profile_b300_denoise_drilldown.sh` runs the same drill-down and writes all JSON + logs under `outputs/` with a single command.

### Interpreting Test 1b (Known B300 Findings)

On B300 at `320x576`, `4` denoise steps, `kv_cache_attention_bias=0.3`, the drill-down shows:

- `denoise` is almost entirely `components.generator(...)` time (random noise + scheduler are negligible).
- `decode` is almost entirely `WanVAEWrapper.decode_to_pixel(...).stream_decode(...)` time (conv3d-heavy).
- In steady-state (post-warmup), `decode` is currently the largest single block (~`740–760ms` per pipeline call for `t=3` frames).

Artifacts that establish this (baseline stack, torch `2.8.0+cu128`):
- Block profile: `outputs/b300_pipeline_blocks_profile_reset.json`
- Denoise step split: `outputs/b300_denoise_steps_reset.json`
- Decode wrapper split: `outputs/b300_vae_decode_reset.json`
- Decode inner split: `outputs/b300_vae_decode_inner_reset.json`

Key numbers (from `outputs/b300_denoise_steps_reset.json`, 4 timed pipeline calls):
- `generator`: ~`1857.5ms` GPU total (~`464.4ms` per pipeline call)
- `randn`: ~`0.29ms` GPU total
- `scheduler_add_noise`: ~`1.29ms` GPU total

Key numbers (from `outputs/b300_vae_decode_reset.json`, 4 timed decode calls):
- `stream_decode`: ~`3042.5ms` GPU total (~`760.6ms` per call)
- Everything else (`get_scale`, `prep_permute_cast`, `postprocess`) is noise-level.

Two low-effort knobs that measurably help on B300:

1) **VAE streaming decode mode**

`WanVAE_.stream_decode()` previously decoded per-frame in a Python loop. We added a faster chunk decode:

- Default: `WANVAE_STREAM_DECODE_MODE=chunk` (single decoder call for all `t` frames)
- Fallback: `WANVAE_STREAM_DECODE_MODE=loop` (old behavior, slower on B300)

2) **cuDNN benchmark**

Enabling cuDNN benchmark improves decode somewhat (at the cost of slower warmup due to algorithm search):

- Use `scripts/profile_krea_pipeline_blocks.py --cudnn-benchmark` for measurement.
- If it helps in your environment, consider enabling `torch.backends.cudnn.benchmark=True` in the server process.

### Test 1b.1: Drill Into `generator(...)` (New, Finer-Grained)

The denoise-step profiler tells you that `generator(...)` dominates `denoise`, but not *why*.

Enable the generator-internal profiler to split `generator(...)` into:
- `call_model_*` (CausalWanModel forward; this is where attention/MLP lives)
- `convert_flow_pred_to_x0` (post-processing)

Example:

```bash
PROFILE_GENERATOR_STEPS=1 \
PROFILE_GENERATOR_STEPS_JSON=$OUT_DIR/generator-steps.json \
uv run python scripts/profile_krea_pipeline_blocks.py --iters 10 --skip 3
```

Notes:
- Profiling uses CUDA events + synchronizes, so treat it as a breakdown tool, not a “max FPS” benchmark.
- Avoid combining with `torch.compile` runs; the profiler auto-disables during tracing.

#### Latest cu130 drill-down (profiling enabled)

Command used:
- `scripts/profile_b300_denoise_drilldown.sh`

Artifacts:
- `outputs/b300_cu130_fp8_bias03_drilldown_perf.log` / `outputs/b300_cu130_fp8_bias03_drilldown_perf.json`
- `outputs/b300_cu130_fp8_bias03_drilldown_blocks_profile.json`
- `outputs/b300_cu130_fp8_bias03_drilldown_denoise_steps.json`
- `outputs/b300_cu130_fp8_bias03_drilldown_generator_steps.json`
- `outputs/b300_cu130_fp8_bias03_drilldown_vae_decode.json`
- `outputs/b300_cu130_fp8_bias03_drilldown_vae_decode_inner.json`

Notes:
- Avg FPS in this run is ~`11.4` (lower than the non-profiled benchmark) because the profilers add synchronizations.

Key breakdown (from `outputs/b300_cu130_fp8_bias03_drilldown_blocks_profile.json`):
- `denoise`: ~`658.5ms` per pipeline call
- `decode`: ~`207.0ms` per pipeline call
- `recompute_kv_cache`: ~`140.3ms` per pipeline call

Within `denoise` (from `outputs/b300_cu130_fp8_bias03_drilldown_denoise_steps.json`):
- `generator`: ~`164.5ms` per call (4 calls per pipeline call)
- `randn` + `scheduler_add_noise`: noise-level

Within `generator(...)` (from `outputs/b300_cu130_fp8_bias03_drilldown_generator_steps.json`):
- `call_model_kv_cache`: ~`159.8ms` per call (dominates)
- `convert_flow_pred_to_x0`: ~`0.1ms` per call (negligible)

### Test 1c: “SM103-native” Runtime Check (cu130 cuDNN)

Goal: determine whether B300 is slow because the **runtime stack** (cuDNN/cuBLAS) is not optimized for SM103.

#### Create isolated envs (do NOT touch `.venv`)

These were created on the B300 box to avoid colliding with other agents using the shared `.venv`:

```bash
# Full project env but torch 2.8 + cu129 (CUDA 12.9 runtime) for A/B sanity
uv venv .venv-b300-cu129 --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-b300-cu129 uv sync --frozen --no-dev
uv pip install -p .venv-b300-cu129/bin/python --upgrade --index-url https://download.pytorch.org/whl/cu129 \
  torch==2.8.0+cu129 torchvision==0.23.0+cu129

# Minimal decode-only env for torch 2.9 + cu130 (CUDA 13 runtime)
uv venv .venv-b300-cu130-decode --python 3.12
uv pip install -p .venv-b300-cu130-decode/bin/python --index-url https://download.pytorch.org/whl/cu130 torch==2.9.0+cu130
uv pip install -p .venv-b300-cu130-decode/bin/python einops numpy
```

Disk note (B300 box at time of writing): `.venv` ~7.9G, `.venv-b300-cu129` ~8.4G, `.venv-b300-cu130-decode` ~4.5G, ~24G free.

#### Run the pipeline benchmark under cu129 (expected: still ~8.8 FPS)

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu129/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 4 --skip 1 \
  --cudnn-benchmark
```

Observed (cu129): **~8.6 FPS**, `decode` still ~`736ms/call`.

#### Run the decode-only microbenchmark under cu129 vs cu130

Use the repo code directly via `PYTHONPATH=src`:

```bash
# cu129: ~760ms/call (t=3)
PYTHONPATH=src .venv-b300-cu129/bin/python scripts/bench_wanvae_stream_decode.py --height 320 --width 576 --t 3 --cudnn-benchmark

# cu130: ~194ms/call (t=3)
PYTHONPATH=src .venv-b300-cu130-decode/bin/python scripts/bench_wanvae_stream_decode.py --height 320 --width 576 --t 3 --cudnn-benchmark
```

Note: `stream_decode` has streaming caches; if warmup is too small, you will measure a “cold start” that can be ~2× slower than steady-state.

Results recorded:
- `outputs/b300_cu129_vae_stream_decode_bench.log`
- `outputs/b300_cu130_vae_stream_decode_bench.log`

Interpretation:
- If decode is ~4× faster on cu130, **the path forward is to run B300 on a cu130 (or newer) stack**, not to keep tuning attention.

#### Update (2025-12-24): cu130 + FlashAttention makes the full pipeline fast again

The cu130 env **must** have `flash_attn` installed, otherwise KV-bias falls back to slow paths and end-to-end can drop to ~`1 FPS`.

Install FlashAttention into the cu130 env (note: builds a large CUDA extension, ~1GB):

```bash
uv pip install -p .venv-b300-cu130-decode/bin/python wheel ninja
uv pip install -p .venv-b300-cu130-decode/bin/python --no-deps --no-build-isolation --no-binary flash-attn flash-attn==2.8.3
```

If the env ever gets clobbered back to cu128 (common after `uv sync`), restore it with:

```bash
./scripts/b300_env_fix_cu130.sh .venv-b300-cu130-decode
```

Then run the full pipeline benchmark (reference settings):

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
PYTHONPATH=src \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization fp8_e4m3fn \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark \
  --json outputs/b300_cu130_fp8_bias03_flashattn.json
```

Observed:
- `scripts/profile_krea_pipeline_blocks.py`: **~13.3–13.5 FPS** at `320x576` (see `outputs/b300_cu130_fp8_bias03_flashattn.log`)
- Daydream (end-to-end): **~14.8–15.0 FPS** at `320x576` (canonical)

Optional: capture per-block profile under cu130 (helps confirm decode is no longer dominant):

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
PYTHONPATH=src \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 4 --skip 1 \
  --quantization fp8_e4m3fn \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark \
  --profile-blocks \
  --profile-blocks-json outputs/b300_cu130_fp8_bias03_blocks_profile.json
```

---

## Troubleshooting: CUDA/NVML Disappeared (Error 304 / NVML Unknown Error)

If you see:
- PyTorch: `cudaGetDeviceCount` **Error 304** (`torch.cuda.is_available() == False`)
- `nvidia-smi`: **Failed to initialize NVML: Unknown Error**

Then the GPU/driver is likely wedged and you won’t be able to run profiling.

In this environment, you may not have permission to reset the GPU from inside the container (sysfs/proc writes can be denied). Usual fixes are host-level:
- Stop any GPU users, then attempt `nvidia-smi --gpu-reset -i 0` (if supported).
- Restart the NVIDIA driver stack / services.
- Reboot the box.

## Test 2: Live GPU Monitoring (H2, H3)

Run in a separate terminal while inference is running.

```bash
# Start monitoring (run for 60+ seconds during inference)
nvidia-smi dmon -s pucvmet -d 1 > $OUT_DIR/dmon.log &
DMON_PID=$!

# Run inference
uv run python scripts/profile_krea_pipeline_blocks.py --iters 50 --skip 3

# Stop monitoring
kill $DMON_PID
```

**Columns in dmon output:**
- `pwr` - power draw (watts)
- `gtemp` - GPU temperature
- `sm` - SM utilization %
- `mem` - memory utilization %
- `enc/dec` - encoder/decoder utilization
- `mclk/pclk` - memory/graphics clock

**What to look for:**
- Low `sm` with spiky pattern → CPU bound (H2)
- High `sm` but low `pclk` → throttling (H3)
- `pwr` way below TDP → power capped (H3)

---

## Test 3: Power and Clocks Snapshot (H3)

```bash
# Detailed power/clock query during inference
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $OUT_DIR/power-clocks.txt

# Also capture in a loop during inference
for i in {1..30}; do
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate \
    --format=csv >> $OUT_DIR/clocks-timeseries.csv
  sleep 1
done &
CLOCK_PID=$!

# Run inference
uv run python scripts/profile_krea_pipeline_blocks.py --iters 30 --skip 3

kill $CLOCK_PID
```

**What to look for:**
- P-state should be P0 under load
- SM clocks should be near max boost
- Power should be near TDP

---

## Test 4: Bypass WebRTC (H1 - Pacing Isolation)

If H1 is suspected, run inference without WebRTC to isolate.

```bash
# Direct pipeline test (no streaming)
uv run python scripts/profile_krea_pipeline_blocks.py \
  --iters 50 --skip 5 \
  --profile-blocks \
  --profile-blocks-json $OUT_DIR/no-webrtc-profile.json \
  2>&1 | tee $OUT_DIR/no-webrtc-output.log
```

**What to look for:**
- If FPS jumps up without WebRTC → pacing is in streaming layer
- If FPS stays at 8.8 → bottleneck is in model/pipeline

---

## Test 5: GEMM Microbenchmark (H5)

If compute-bound but attention doesn't matter, test raw GEMM perf.

```bash
# Quick matmul benchmark
python3 -c "
import torch
import time

# Typical shapes from QKV projection
M, N, K = 4680, 2048, 2048
A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(10):
    C = torch.mm(A, B)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
    C = torch.mm(A, B)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f'GEMM {M}x{N}x{K}: {elapsed/100*1000:.3f} ms/iter')
print(f'TFLOPS: {2*M*N*K*100/elapsed/1e12:.2f}')
" 2>&1 | tee $OUT_DIR/gemm-benchmark.txt
```

---

## Test 6: Full Nsight Systems Profile (H5 - Deep Dive)

Only if other tests point to GPU compute issues.

```bash
# Requires nsys installed
nsys profile -o $OUT_DIR/nsys-profile \
  --stats=true \
  uv run python scripts/profile_krea_pipeline_blocks.py --iters 10 --skip 2

# Generate report
nsys stats $OUT_DIR/nsys-profile.nsys-rep > $OUT_DIR/nsys-stats.txt
```

---

## Comparison Checklist

After running on multiple GPUs, compare:

| Metric | B200 | B300 | H100 |
|--------|------|------|------|
| FPS (block profiler) | | | |
| gpu_ms per frame | | | |
| wall_ms per frame | | | |
| SM utilization % | | | |
| SM clock (MHz) | | | |
| Power draw (W) | | | |
| P-state | | | |
| GEMM TFLOPS | | | |

---

## Decision Tree

```
gpu_ms << wall_ms?
├── YES → H1: Pacing/backpressure
│   └── Next: Find the pacer (WebRTC, queue, sleep)
│
└── NO (gpu_ms ≈ wall_ms) → Compute bound
    │
    ├── SM util low?
    │   └── YES → H2: CPU bound
    │       └── Next: Profile CPU, find single-threaded bottleneck
    │
    ├── SM util high, clocks low?
    │   └── YES → H3: Throttling
    │       └── Next: Check power limits, thermals
    │
    ├── Other GPU processes?
    │   └── YES → H4: Shared GPU
    │       └── Next: Get exclusive access
    │
    └── None of above?
        └── H5: Kernel fallback
            └── Next: Nsight profile, check GEMM perf
```

---

## H100 as Discriminator

H100 result interpretation:

| H100 FPS | Meaning |
|----------|---------|
| ~8.8 FPS (same as B300) | Pacing/scheduling issue in code, not GPU-specific |
| ~15-20 FPS (like B200) | B300-specific issue (SM103 stack, driver, hardware) |

---

## File Checklist

After investigation, each GPU dir should have:

```
$INVESTIGATION_DIR/<gpu>/
├── nvidia-smi-full.txt
├── nvidia-smi-devices.txt
├── gpu-info.csv
├── gpu-processes.csv
├── cpu-info.txt
├── memory-info.txt
├── block-profile.json
├── profile-output.log
├── denoise-steps.json
├── vae-decode.json
├── vae-decode-inner.json
├── dmon.log
├── power-clocks.txt
├── clocks-timeseries.csv
├── gemm-benchmark.txt
└── (optional) nsys-profile.nsys-rep
```
