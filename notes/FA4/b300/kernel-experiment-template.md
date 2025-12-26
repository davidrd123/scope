# Kernel Experiment Template

> Status: Ready to use
> Priority: Medium — structured approach for Level 6 kernel development
> Date: 2025-12-26
> Extends: `experiments.md` card format

## Purpose

Template for developing custom kernels with proper validation gates. Prevents "fast in isolation, broken in integration" failures.

---

## Kernel Experiment Card

Copy this template for each kernel development effort:

```markdown
## Kernel: [Name]

**Date:** YYYY-MM-DD
**Status:** [ ] Phase 0 | [ ] Phase 1 | [ ] Phase 2 | [ ] Phase 3 | [ ] Phase 4
**Target:** [What ops/overhead this kernel eliminates]
**Expected gain:** [X ms / Y% of component]

### Phase 0: Baseline Measurement

**Stack-attributed profile of current path:**
```
[Paste relevant profile output]
```

**Ops to eliminate:**
| Op | Current time (ms) | Call count |
|----|-------------------|------------|
| ? | ? | ? |

**Input specification:**
| Tensor | Shape | Dtype | Layout | Alignment |
|--------|-------|-------|--------|-----------|
| ? | ? | ? | ? | ? |

**Output specification:**
| Tensor | Shape | Dtype | Layout |
|--------|-------|-------|--------|
| ? | ? | ? | ? |

**Correctness oracle:**
- [ ] Reference implementation identified
- [ ] Golden outputs saved to: `tests/golden/[kernel_name]/`
- [ ] Numerical tolerance defined: atol=?, rtol=?

---

### Phase 1: Microbench (Standalone Kernel)

**Implementation:**
- Location: `src/scope/kernels/[name].py` or `.cu`
- Approach: [CuTe/CUTLASS/Triton/handwritten]

**Synthetic input generation:**
```python
# Code to generate test inputs
```

**Microbench results:**
| Metric | Value |
|--------|-------|
| Kernel time | ? ms |
| Bandwidth utilization | ?% |
| Theoretical peak | ? GB/s |
| Achieved | ? GB/s |

**Numerical validation:**
- [ ] Matches reference within tolerance
- [ ] Edge cases tested (zero, max, boundary shapes)
- [ ] Dtype variants tested (bf16, fp16, fp32)

---

### Phase 2: Integration

**Drop-in replacement:**
- Location: `src/scope/core/pipelines/.../[module].py`
- Fallback: `SCOPE_[NAME]_IMPL=legacy` reverts to original

**Integration checklist:**
- [ ] Env var toggle added
- [ ] Fallback path preserved
- [ ] No shape/dtype assumptions violated

**End-to-end correctness:**
- [ ] Full pipeline runs without error
- [ ] Output matches legacy path (visual + numerical)

**Kernel deletion check:**
- [ ] Profile before: [X kernels, Y ms]
- [ ] Profile after: [X' kernels, Y' ms]
- [ ] Net kernels eliminated: [N]

---

### Phase 3: Validation Gates

**Numerical equivalence:**
| Test | Status | Notes |
|------|--------|-------|
| atol check | [ ] | |
| rtol check | [ ] | |
| Gradient check (if applicable) | [ ] | |

**Video quality sentinel:**
- [ ] Test clip: `tests/golden/clips/[name].mp4`
- [ ] Perceptual metric: [PSNR/SSIM/LPIPS]
- [ ] Baseline score: ?
- [ ] New score: ?
- [ ] Delta within threshold: [ ]

**Performance regression gate:**
| Metric | Baseline | New | Delta |
|--------|----------|-----|-------|
| FPS | ? | ? | ? |
| Component time | ? ms | ? ms | ? ms |
| Memory | ? GB | ? GB | ? |

---

### Phase 4: Ship

**Fallback ladder:**
```
1. New kernel (default when SCOPE_[NAME]_IMPL=auto)
2. Legacy path (SCOPE_[NAME]_IMPL=legacy)
3. Error with clear message if both fail
```

**Kernel provenance logging:**
```python
logger.info(f"[NAME] kernel: impl={impl}, backend={backend}")
```

**Nsight repro script:**
- Location: `scripts/nsight_[name].sh`
- Isolates kernel for profiling

**Documentation:**
- [ ] Added to `experiments.md`
- [ ] Added to `optimization-ladder.md`
- [ ] Updated `layout-contracts.md` if layouts changed

---

### Observations

[Free-form notes, surprises, learnings]

---

### Next Steps

- [ ] ?
```

---

## Quality Gates Checklist

Use this for any kernel before merging:

### Correctness

- [ ] Numerical equivalence to reference (atol/rtol)
- [ ] No NaN/Inf in outputs
- [ ] Edge cases handled (empty batch, max sequence, etc.)
- [ ] Deterministic (same inputs → same outputs)

### Performance

- [ ] Faster than baseline (not just "looks fast")
- [ ] Kernels actually eliminated (not just added)
- [ ] Memory usage same or lower
- [ ] Warmup time acceptable
- [ ] **"Did we slow GEMM?" check** — see below

### Robustness

- [ ] Works with torch.compile
- [ ] Works with/without cudnn.benchmark
- [ ] Fallback path tested
- [ ] Error messages are clear

### Integration

- [ ] Env var toggle documented
- [ ] No silent behavior changes
- [ ] Backward compatible API

---

## Video Quality Sentinel

For video generation, visual quality is the ultimate gate.

### Setup

```bash
# Generate reference clip
python scripts/generate_golden_clip.py \
  --prompt "test prompt" \
  --config legacy \
  --output tests/golden/clips/baseline.mp4

# Generate test clip
python scripts/generate_golden_clip.py \
  --prompt "test prompt" \
  --config new_kernel \
  --output tests/golden/clips/new.mp4

# Compare
python scripts/compare_clips.py \
  --baseline tests/golden/clips/baseline.mp4 \
  --test tests/golden/clips/new.mp4 \
  --metrics psnr,ssim,lpips
```

### Thresholds

| Metric | Threshold | Notes |
|--------|-----------|-------|
| PSNR | > 30 dB | Lower bound for "same" |
| SSIM | > 0.95 | Structural similarity |
| LPIPS | < 0.1 | Perceptual similarity |

---

## Nsight Isolation Script Template

```bash
#!/bin/bash
# scripts/nsight_[kernel_name].sh

# Minimal repro for Nsight profiling

export SCOPE_[NAME]_IMPL=new
export CUDA_VISIBLE_DEVICES=0

nsys profile \
  --trace=cuda,nvtx \
  --output=nsight_[kernel_name] \
  python -c "
import torch
from scope.kernels.[name] import kernel_func

# Minimal test case
x = torch.randn(B, S, D, device='cuda', dtype=torch.bfloat16)
y = kernel_func(x)
torch.cuda.synchronize()
"
```

---

## "Did We Slow GEMM?" Gate

> **From 5pro01.md**: If you fuse too aggressively and accidentally pull the GEMM away from a best-in-class cuBLAS kernel, your total step can get slower even if you deleted kernels.

### Why This Matters

The QKV projection (and other linear layers) use cuBLAS GEMMs that are highly optimized. If your fusion:
- Changes the output layout of a GEMM
- Inserts a custom epilogue that cuBLAS can't use
- Forces a less-optimized GEMM path

...you may *delete* layout kernels but *slow down* the GEMM, resulting in a net loss.

### Check Protocol

Before declaring a fusion "done", compare:

| Metric | Baseline | With Fusion | Delta |
|--------|----------|-------------|-------|
| GEMM kernel time (ns) | ? | ? | ? |
| GEMM kernel name | ? | ? | Same? |
| Total component time | ? | ? | ? |

**If GEMM got slower:**
1. Check if cuBLAS is using a different kernel (Nsight will show kernel name)
2. Check if output layout is forcing a slow path
3. Consider: is the layout win worth the GEMM loss?

### Quick Nsight Check

```bash
nsys profile --trace=cuda \
  python -c "your_gemm_test" 2>&1 | grep -E "gemm|cutlass"
```

Compare kernel names and durations before/after fusion.

---

## References

- Existing experiment cards: `notes/FA4/b300/experiments.md`
- Investigation runbook: `notes/FA4/b300/investigation-runbook.md`
- Layout contracts: `notes/FA4/b300/layout-contracts.md`
- External research: `notes/FA4/DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/5pro01.md`
