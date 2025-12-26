# Layout Contracts: QKV → RoPE → Cache → Attention

> Status: Partial (FA constraints documented, local code audit pending)
> Priority: **High** — blocks all fusion/kernel work
> Date: 2025-12-26
> Source: `notes/FA4/DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md`

## Purpose

Document the concrete tensor shapes, dtypes, memory layouts, and alignment requirements at each boundary in the self-attention path. Without this, we can't design a "post-projection pack" kernel safely.

---

## Contract 1: QKV Projection Output

**Location:** `src/scope/core/pipelines/wan2_1/modules/attention.py` (or similar)

| Property | Value | Notes |
|----------|-------|-------|
| Shape | `[B, S, 3*H*D]` or `[B, S, 3, H, D]`? | TBD — check actual output |
| Dtype | `bfloat16` | Confirm |
| Layout | Contiguous? Channels-last? | TBD |
| Alignment | 16-byte? 128-byte? | Affects TMA viability |
| Producer | cuBLAS GEMM | Via `F.linear` |

**Key questions:**
- [ ] Is QKV fused (single GEMM) or separate Q/K/V projections?
- [ ] What's the exact output shape after reshape/view?
- [ ] Any `.contiguous()` calls immediately after?

---

## Contract 2: RoPE Inputs/Outputs

**Location:** `src/scope/core/pipelines/wan2_1/modules/rope.py` (or similar)

### Input

| Property | Value | Notes |
|----------|-------|-------|
| Q shape | `[B, H, S, D]`? | TBD |
| K shape | `[B, H, S, D]`? | TBD |
| cos/sin shape | `[S, D]` or `[1, 1, S, D]`? | TBD |
| Dtype | `bfloat16` | Confirm |

### Output

| Property | Value | Notes |
|----------|-------|-------|
| Q_rotated shape | Same as input? | TBD |
| K_rotated shape | Same as input? | TBD |
| In-place? | Yes/No | TBD |

**Key questions:**
- [ ] Is RoPE applied in-place or does it allocate?
- [ ] What's the cos/sin table format (interleaved, split, complex)?
- [ ] Any precision sensitivity (fp32 intermediates)?

---

## Contract 3: KV Cache Write Layout

**Location:** `src/scope/core/pipelines/krea_realtime_video/` (cache management)

### Write Path

| Property | Value | Notes |
|----------|-------|-------|
| K_cache shape | `[B, H, MaxS, D]`? | TBD |
| V_cache shape | `[B, H, MaxS, D]`? | TBD |
| Write pattern | Append at position? Scatter? | TBD |
| Dtype | `bfloat16` | Confirm |

### Read Path (for attention)

| Property | Value | Notes |
|----------|-------|-------|
| K shape expected by attention | ? | TBD |
| V shape expected by attention | ? | TBD |
| Contiguous requirement? | ? | TBD |

**Key questions:**
- [ ] Is cache pre-allocated or grown dynamically?
- [ ] Any packing/unpacking between write and read?
- [ ] Position indexing: absolute or relative?

---

## Contract 4: FA2/FA4 Attention Requirements

### FA2 (flash_attn)

**Reference:** https://github.com/Dao-AILab/flash-attention

#### Batch Mode (`flash_attn_func`)

| Property | Requirement | Notes |
|----------|-------------|-------|
| Q shape | `[B, S, H, D]` | Batch, Seq, Heads, HeadDim |
| K shape | `[B, S, H, D]` | |
| V shape | `[B, S, H, D]` | |
| Dtype | `fp16` or `bf16` | |
| LSE output | `[B, H, S_q]` float32 | |

#### Varlen Mode (`flash_attn_varlen_func`)

| Property | Requirement | Notes |
|----------|-------------|-------|
| Q shape | `[total_q, H, D]` | Packed sequences |
| K shape | `[total_k, H, D]` | |
| V shape | `[total_v, H, D]` | |
| cu_seqlens_q | `[B+1]` int32 | Cumsum starting 0 |
| cu_seqlens_k | `[B+1]` int32 | e.g., [3,4,3] → [0,3,7,10] |
| max_seqlen_q | int | Required |
| max_seqlen_k | int | Required |
| LSE output | `[H, total_q]` float32 | Different from batch! |

#### Critical Constraints (from doc_ref_guide)

| # | Constraint | Impact |
|---|------------|--------|
| 1 | **Last dim MUST be contiguous** | `k.stride(-1)==1`, `v.stride(-1)==1` enforced. Non-contiguous → expensive copies |
| 2 | **head_dim alignment** | FA2: multiples of 8. Some builds (vLLM): multiples of 32 |
| 3 | **MQA/GQA ratio** | Q heads must divide evenly by KV heads |
| 4 | **head_dim >192 backward** | Only A100/H100; consumer GPUs ≤192 |

### FA3 (Hopper)

| Property | Requirement | Notes |
|----------|-------------|-------|
| head_dim | Rounds to 64, 96, 128, 192, or **256 max** | Internal rounding |
| FP8 | **K-major layout required** | For WGMMA instructions |
| FP16/BF16 | MN-major or K-major | Both accepted |

### FA4 (Blackwell, with score_mod)

**Reference:** PyTorch `flex_attention`, `flash_attn/cute/interface.py`

| Property | Requirement | Notes |
|----------|-------------|-------|
| Dtype | **BF16 only** | No FP16, no FP8 yet |
| Direction | **Forward only** | Backward not implemented |
| Target | SM100 (B200) | 5-stage pipeline, CuTe DSL |
| score_mod | `def score_mod(score, batch, head, q_idx, k_idx)` | All idx args are `torch.int` scalars |
| score_mod constraint | **No trainable parameters** inside | Pure function only |

#### What Forces Internal Re-packing

| Condition | Result |
|-----------|--------|
| Non-contiguous last dim | Copy to contiguous |
| Wrong head_dim alignment | Pad internally |
| K-major when MN-major expected (or vice versa) | Transpose |

#### Accepted vs Fast Layouts

> **Key insight from 5pro01.md**: Even when multiple layouts are *accepted*, only some are *fast-path*.

| Layout | Accepted? | Fast? | Notes |
|--------|-----------|-------|-------|
| `[B, S, H, D]` contiguous | ✓ | ✓ | Preferred for batch mode |
| `[B, H, S, D]` contiguous | ✓ | ? | May trigger internal transpose |
| Non-contiguous last dim | ✓ | ✗ | Forces copy to contiguous |
| Misaligned head_dim | ✓ | ✗ | Forces internal padding |

**Implication for fusion**: Your pack kernel should target the *fast-path* layout, not just any accepted layout. Otherwise you can accidentally fuse yourself into a slower overall pipeline.

#### Causal Convention Warning (v2.1.0)

> **Critical**: FlashAttention's causal masking convention changed in v2.1.0.

- **Pre-v2.1.0**: Top-left alignment
- **v2.1.0+**: Bottom-right alignment (current)

**Action**: Hard-code the convention in tests. If upgrading FA versions, verify causal outputs match expected.

### Key Docs

- `flash_attn_func` docstring: shape/stride requirements
- `flash_attn_varlen_func` docstring: cu_seqlens format
- FA3 Paper §3.3: FP8 layout transformation rules
- GitHub Issue #880: cu_seqlens batch_size+1 explanation
- `flash_attn/cute/interface.py`: FA4/Blackwell patterns

---

## Current Glue Ops (To Eliminate)

Based on profiling, these ops appear between the contracts above:

| Op | Location | Why It Exists | Can Fuse? |
|----|----------|---------------|-----------|
| `aten::transpose` | ? | Layout mismatch | TBD |
| `aten::contiguous` | ? | Non-contiguous input | TBD |
| `aten::_to_copy` | ? | Dtype conversion | TBD |
| `aten::view` | ? | Reshape | Usually free |

---

## Action Items

1. [ ] **Profile with stack attribution** to identify exact locations of layout ops
2. [ ] **Code audit** of attention.py, rope.py, cache management
3. [ ] **Read FA2/FA4 source** for shape/stride requirements
4. [ ] **Document cos/sin table format** for RoPE

---

## References

- FlashAttention repo: https://github.com/Dao-AILab/flash-attention
- PyTorch flex_attention: https://pytorch.org/docs/stable/generated/torch.nn.attention.flex_attention.html
- Current attention impl: `src/scope/core/pipelines/wan2_1/modules/attention.py`
- Current RoPE impl: `src/scope/core/pipelines/wan2_1/modules/rope.py`
