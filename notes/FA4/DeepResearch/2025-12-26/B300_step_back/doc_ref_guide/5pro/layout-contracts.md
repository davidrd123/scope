# Layout contracts (B300, FA4)

Goal: make fusion work safe by turning implicit layout assumptions into explicit, testable contracts.

This doc should answer, unambiguously:

1) What layout comes out of QKV projection today
2) What layout RoPE reads and writes
3) What layout KV-cache writes and reads
4) What layouts FlashAttention (FA2 and FA4) accept, and which ones are fast vs. trigger internal repacking

If any cell below is unknown, treat it as a blocked item. Do not design a fused kernel until it is filled.

---

## Notation

- B: batch
- S: sequence length (tokens per sequence)
- T: total tokens for varlen (sum of per-seq lengths)
- Hq: number of query heads
- Hkv: number of KV heads (can differ with GQA/MQA)
- D: head dim
- dtype: fp16, bf16, fp8 (if used)
- "contiguous last dim": stride[-1] == 1
- "row-major": last dim contiguous, then next-to-last, etc.

Record both: logical shape and actual strides.

---

## A. QKV projection output contract

### A1. What we compute conceptually

We want three tensors:
- Q: [B, S, Hq, D]
- K: [B, S, Hkv, D]
- V: [B, S, Hkv, D]

If packed: QKV: [B, S, 3, H?, D] or [B, S, H?, 3, D] or [B, S, (3*H*D)].

### A2. What cuBLAS/cuBLASLt actually produces today

Fill this with an actual runtime printout for one representative shape.

| Item | Current producer | dtype | logical shape | stride tuple | alignment | notes |
|------|------------------|-------|---------------|--------------|----------|-------|
| QKV output | (cublas/cublasLt) |  |  |  |  |  |
| Bias (if any) |  |  |  |  |  |  |
| Output scaling (fp8?) |  |  |  |  |  |  |

**How to fill:** add an assertion helper that prints `tensor.shape`, `tensor.stride()`, `tensor.dtype`, `tensor.is_contiguous()` for Q, K, V (or packed).

---

## B. RoPE contract

### B1. What RoPE expects

RoPE typically applies to Q and K (not V). Capture the exact contract used in this repo.

| Item | Input logical shape | input stride tuple | output logical shape | output stride tuple | dtype | notes |
|------|----------------------|--------------------|----------------------|---------------------|-------|-------|
| Q (pre-RoPE) |  |  |  |  |  |  |
| K (pre-RoPE) |  |  |  |  |  |  |
| Q (post-RoPE) |  |  |  |  |  |  |
| K (post-RoPE) |  |  |  |  |  |  |

### B2. Frequency representation

Record the exact representation, since it affects what can be fused.

- sinusoid cache shape:
- dtype:
- layout:
- how positions are computed (absolute, offset, per-token):

---

## C. KV-cache write and read contract

This is the main place fusion attempts die. Be precise.

### C1. Logical KV-cache view

| Item | logical shape | index meaning | notes |
|------|---------------|--------------|-------|
| K cache |  |  |  |
| V cache |  |  |  |

Examples of index meanings to nail down:
- Is sequence dim contiguous, or blocked/paged?
- Is head dim padded?
- Is this per-layer or interleaved across layers?

### C2. Physical layout

| Item | physical shape | stride tuple | address formula sketch | alignment | notes |
|------|----------------|--------------|------------------------|----------|-------|
| K cache storage |  |  |  |  |  |
| V cache storage |  |  |  |  |  |

If paged:
- page size in tokens:
- page table format:
- how to map (batch, token) -> (page, offset):

---

## D. FlashAttention interface contract (FA2 and FA4)

### D1. Accepted layouts

Document what each interface accepts, and which cases are slow (internal repack) vs. fast path.

| Kernel | Q layout accepted | K layout accepted | V layout accepted | packed QKV accepted | varlen accepted | notes |
|--------|-------------------|-------------------|-------------------|---------------------|----------------|-------|
| FA2 |  |  |  |  |  |  |
| FA4 (CuTe path) |  |  |  |  |  |  |

### D2. Alignment and head dim rules

Fill with concrete requirements from the exact FA version vendored here:

- head dim allowed set:
- head dim multiple-of constraints:
- alignment constraints for Q/K/V base pointers:
- dtype support:

---

## E. What fusion kernel must guarantee

Once A-D are filled, write a single "kernel contract" paragraph here.

A fused "post-projection pack" kernel must:
1) Read: [describe exact producer layout]
2) Apply: RoPE with [describe exact representation]
3) Write: Q in [layout required by FA4 fast path]
4) Write: K/V into KV-cache in [exact physical layout]
5) Preserve: numerics within [atol/rtol], and obey alignment rules

---

## F. Contract validation helpers

Add lightweight runtime checks that run in debug mode:

- Shapes and strides match expected
- Contiguity and alignment checks
- A small randomized correctness test (compare fused vs. unfused)

Suggested: put helpers in a single file so all kernels can import them.

