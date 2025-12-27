# Kernel experiment template (Level 6 work)

This is a kernel-specific extension of an experiment card, with explicit performance and quality gates.

---

## 0. Snapshot

- Kernel name:
- Target GPU(s):
- CUDA toolkit:
- PyTorch version:
- Commit hash:
- Build flags:

---

## 1. Problem statement

- What kernels are we trying to delete?
- What is the measured time today (ms and % of step)?
- What layout contract assumptions are required?

Links:
- layout-contracts.md:
- other-in-self-breakdown.md:

---

## 2. Correctness oracle

- Reference implementation:
- Tolerance (atol/rtol):
- Determinism expectations:
- Edge cases:
  - short sequences
  - odd head dims
  - bf16 vs fp16
  - causal vs non-causal
  - varlen vs padded

---

## 3. Phase ladder

### Phase A: Baseline measurement
- [ ] Stack-attributed profile of current path
- [ ] Document: shapes, dtypes, strides
- [ ] Establish oracle tests

### Phase B: Microbench
- [ ] Standalone kernel on synthetic inputs
- [ ] Validate numerics vs oracle
- [ ] Report: latency, achieved bandwidth, achieved FLOPs
- [ ] Nsight Compute capture for one representative shape

### Phase C: Integration
- [ ] Drop-in replacement behind a flag
- [ ] End-to-end correctness tests
- [ ] Measure: how many kernels were deleted?

### Phase D: Validation gates
- [ ] Numerical equivalence gate
- [ ] Video quality gate (perceptual metric or sentinel prompts)
- [ ] Performance regression gate (perf CI threshold)
- [ ] Memory regression gate (peak memory)

### Phase E: Ship
- [ ] Fallback ladder documented
- [ ] Provenance logging (kernel version, path)
- [ ] Repro script (nsys + ncu + shapes)
- [ ] Docs updated (layout contract, cheat sheet links)

---

## 4. Measurement protocol

To avoid false wins:
- Warmup iterations:
- Measure iterations:
- Fix GPU clocks if possible:
- Disable unrelated work (logging, checkpoints):
- Record environment variables:

---

## 5. Postmortem checklist (after first prototype)

- Did we accidentally move work from GPU to CPU?
- Did register pressure explode?
- Did we create a silent fallback path?
- Is the win shape-dependent (only helps one batch/seq)?

