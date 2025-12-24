# B300 Investigation Runbook

**Purpose:** Systematic approach to diagnose why B300 is stuck at 8.8 FPS while B200 hits ~20 FPS.

**Key insight:** Attention backend changes don't move FPS. The bottleneck is elsewhere.

---

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

---

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
├── dmon.log
├── power-clocks.txt
├── clocks-timeseries.csv
├── gemm-benchmark.txt
└── (optional) nsys-profile.nsys-rep
```
