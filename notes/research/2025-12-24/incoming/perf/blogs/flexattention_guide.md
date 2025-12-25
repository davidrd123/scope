# A User's Guide to FlexAttention in FlashAttention CuTe DSL

> **Source:** Colfax Research | Published November 14, 2025  
> **Authors:** Collaboration with Driss Guessous (Meta) and Tri Dao (Princeton; Together AI)

---

## Introduction

Many variants of attention (Vaswani et al., 2017) have become popular for performance and model quality reasons:

| Variant | Description | Complexity |
|---------|-------------|------------|
| **Causal attention** | Token only attends to prior tokens (autoregressive LM) | O(n²d) |
| **Sliding window** | Token attends to prior tokens within window size `w` | O(nwd) |
| **ALiBi** | Positional bias linear in distance, no explicit embeddings | - |
| **T5 bias / PrefixLM** | Learned additive biases; partial bidirectional attention | - |
| **Attention sink** | Fixed cached KV tokens for global context in sliding window | O(n) |

The PyTorch team at Meta unified these under **FlexAttention** (Guessous et al., 2024), providing a simple API for defining attention variants with minimal development overhead.

### FlexAttention Formula

```
FlexAttention(Q, K, V) = Softmax(mask_mod(score_mod(QK^T)))V
```

Two customization options:
- **`score_mod`**: Modifies pre-softmax attention scores
- **`mask_mod`**: Masks out pre-softmax scores (sets to -∞)

> **Note:** `mask_mod` is a special case of `score_mod` kept separate for efficiency (block sparsity optimization).

### Performance

- CuTe DSL implementation achieves **95% of FlashAttention 3 performance** in forward pass
- **~50% speedup** over Triton version in most cases
- Similar/greater gains on Blackwell (FlashAttention 4)

---

## Score Modification

### Generic Signature

```python
generic_score_mod(
    score: float,
    batch_idx: int,
    head_idx: int,
    q_idx: int,
    kv_idx: int,
    aux_tensors: Optional[list[tensor]],
) -> float
```

### Example 1: T5 (Relative Positional) Bias

```python
def rel_bias_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
    bias_tensor = aux_tensors[0]
    rel_pos = math.abs(q_idx - kv_idx)
    return score + bias_tensor[batch_idx, head_idx, rel_pos]
```

### Example 2: ALiBi

```python
def alibi_score_mod(score, batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
    slope = math.exp2(-(head_idx + 1))
    dist = math.abs(q_idx - kv_idx)
    return score - slope * dist
```

### CuTe DSL Implementation

Score mods must use the **TensorSSA abstraction**:

```python
@cute.jit
def rel_bias_score_mod_cute(
    tSrS_ssa: cute.TensorSSA,
    batch_idx: cute.TensorSSA,
    head_idx: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    aux_tensors: Optional[list]
) -> cute.TensorSSA:
    bias_tensor = aux_tensors[0]
    rel_pos = cute.TensorSSA(
        mlir_math.absi(q_idx - kv_idx),
        q_idx.shape,
        q_idx.dtype
    )
    bias = bias_tensor[batch_idx[0], head_idx[0], rel_pos[0]]
    return tSrS_ssa + bias
```

> **Note:** Vectorization of `score_mod` is not feasible when using `aux_tensors` without further assumptions.

### Usage

**Direct CuTe DSL interface:**

```python
from flash_attn.cute.interface import _flash_attn_fwd

out, _ = _flash_attn_fwd(
    q, k, v,  # torch.Tensor
    score_mod=rel_bias_score_mod_cute,
    aux_tensors=aux_tensors,  # Optional[list[torch.Tensor]]
)
```

**PyTorch integrated interface:**

```python
from torch.nn.attention.flex_attention import flex_attention

compiled_fn = torch.compile(flex_attention)
out = compiled_fn(
    q, k, v,
    score_mod=rel_bias_score_mod,
    kernel_options={"force_flash": True},  # Use CuTe DSL backend
)
```

---

## Mask Modification

### Generic Signature

```python
generic_mask_mod(
    batch_idx: cute.TensorSSA,
    head_idx: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    aux_tensors: Optional[list],
) -> cute.TensorSSA  # dtype == cutlass.Boolean
```

> **Note:** Unlike `score_mod`, we don't pass in the score—only positional information.

### Example 1: Causal Mask with Offset

```python
import flash_attn.cute.utils as utils

def create_causal_mask_with_offset(offset: int):
    @cute.jit
    def _causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
        offset_ssa = utils.scalar_to_ssa(val=offset, dtype=cutlass.Int32)
        return kv_idx <= q_idx + offset_ssa
    return _causal_mask_mod
```

> **Note:** This mask requires recompilation when `seqlen_k - seqlen_q` changes. Pass `offset` as an `aux_tensor` to avoid this.

### Example 2: Document Masking

Prevents information leakage across document boundaries when sequences are concatenated:

```python
@cute.jit
def document_mask_mod(batch_idx, head_idx, q_idx, kv_idx, aux_tensors):
    doc_ids = aux_tensors[0]  # Shape: (B, H, seqlen), Int32
    doc_id_q = doc_ids[batch_idx[0], head_idx[0], q_idx[0]]
    doc_id_kv = doc_ids[batch_idx[0], head_idx[0], kv_idx[0]]
    q_doc = utils.scalar_to_ssa(doc_id_q, cutlass.Int32)
    kv_doc = utils.scalar_to_ssa(doc_id_kv, cutlass.Int32)
    return q_doc == kv_doc
```

### Usage

```python
out, _ = _flash_attn_fwd(
    q, k, v,
    mask_mod=document_mask_mod,
    aux_tensors=[doc_ids],
)
```

---

## Block Sparsity

When large portions of the scores matrix are masked, FlexAttention implements **block sparsity** to skip unnecessary computation.

### Block Categories

For causal masking example (batch=1, head=1, seqlen_q=768, seqlen_kv=896, tile=128×128):

| Block Type | Count | Description |
|------------|-------|-------------|
| **Partially-masked** | 6 | On diagonal, split by causal mask; needs `mask_mod` |
| **Fully-computed** | 21 | Below diagonal, no masking; skip `mask_mod` |
| **Skipped entirely** | 15 | Above diagonal, completely masked; don't even load |

```
┌─────────────────────────────────────────────┐
│  Block Sparsity Visualization (Causal)      │
├─────────────────────────────────────────────┤
│  Legend:                                    │
│  ▓▓▓ = fully-computed block                 │
│  ▒▒▒ = partially-masked block               │
│  ░░░ = masked values (skipped)              │
│  ─── = unmasked values                      │
├─────────────────────────────────────────────┤
│       0   1   2   3   4   5   6             │
│   0  [▒▓][░░][░░][░░][░░][░░][░░]           │
│   1  [▓▓][▒▓][░░][░░][░░][░░][░░]           │
│   2  [▓▓][▓▓][▒▓][░░][░░][░░][░░]           │
│   3  [▓▓][▓▓][▓▓][▒▓][░░][░░][░░]           │
│   4  [▓▓][▓▓][▓▓][▓▓][▒▓][░░][░░]           │
│   5  [▓▓][▓▓][▓▓][▓▓][▓▓][▒▓][░░]           │
└─────────────────────────────────────────────┘
```

### Block Sparsity Tensors

Each work tile corresponds to one `(batch, head, q_block)` coordinate.

**Required Tensors:**

```python
class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor   # [B, H, num_q_blocks]
    mask_block_idx: cute.Tensor   # [B, H, num_q_blocks, num_kv_blocks]
    full_block_cnt: Optional[cute.Tensor]  # [B, H, num_q_blocks]
    full_block_idx: Optional[cute.Tensor]  # [B, H, num_q_blocks, num_kv_blocks]
```

Where:
- `num_q_blocks = ceil_div(seqlen_q, tile_m)`
- `num_kv_blocks = ceil_div(seqlen_kv, tile_n)`

**Invariants:**
1. `mask_block_idx[b, h, q_block, :mask_cnt]` is strictly increasing
2. `full_block_idx[b, h, q_block, :full_cnt]` is strictly increasing and **disjoint** from mask indices

### Example: Causal Masking Block Sparsity

```python
# seqlen_q=768, seqlen_kv=896, tile=128×128
mask_block_cnt = [[[1, 1, 1, 1, 1, 1]]]
mask_block_idx = [[[[1, 0, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0, 0],
                    [3, 0, 0, 0, 0, 0, 0],
                    [4, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0],
                    [6, 0, 0, 0, 0, 0, 0]]]]

full_block_cnt = [[[1, 2, 3, 4, 5, 6]]]
full_block_idx = [[[[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 2, 0, 0, 0, 0],
                    [0, 1, 2, 3, 0, 0, 0],
                    [0, 1, 2, 3, 4, 0, 0],
                    [0, 1, 2, 3, 4, 5, 0]]]]
```

### Computing Block Sparsity

```python
from torch.nn.attention.flex_attention import create_block_mask

block_mask_torch = create_block_mask(
    mask_mod_fn,  # PyTorch mask function
    B, H, seqlen_q, seqlen_kv,
    device="cuda",
    BLOCK_SIZE=(tile_m, tile_n),
)

# Convert to CuTe DSL format
_, _, mask_cnt, mask_idx, full_cnt, full_idx, *_ = block_mask_torch
block_sparse_tensors = BlockSparseTensorsTorch(
    mask_block_cnt=mask_cnt,
    mask_block_idx=mask_idx,
    full_block_cnt=full_cnt,
    full_block_idx=full_idx,
)
```

> ⚠️ **Warning:** Tile size for block sparsity computation must match kernel tile size.

---

## Complete API Call

```python
_flash_attn_fwd(
    q, k, v,                              # torch.Tensor
    score_mod=score_mod,                  # Callable
    mask_mod=mask_mod,                    # Callable
    block_sparse_tensors_torch=block_sparse_tensors,  # BlockSparseTensorsTorch
    aux_tensors=aux_tensors,              # Optional[list[torch.Tensor]]
)
```

---

## Advanced Examples

### Example 1: Document Masking with Relative Positional Bias

Combines `score_mod` and `mask_mod` using shared `aux_tensors`:

```python
# Setup: 3 documents at positions [0:230], [230:410], [410:640]
doc_ids = torch.zeros((1, 1, 640), dtype=torch.int32)
doc_ids[0, 0, :230] = 0
doc_ids[0, 0, 230:410] = 1
doc_ids[0, 0, 410:] = 2

@cute.jit
def doc_rel_bias_score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    rel_bias = aux_tensors[0]
    distance = cute.TensorSSA(
        mlir_math.absi(q_idx - kv_idx),
        q_idx.shape, q_idx.dtype
    )
    bias = rel_bias[b_idx[0], h_idx[0], distance[0]].to(cutlass.Float32)
    return tSrS_ssa + bias

@cute.jit
def document_mask_mod(b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    doc_ids = aux_tensors[1]  # Second aux tensor
    q_doc = doc_ids[b_idx[0], h_idx[0], q_idx[0]]
    kv_doc = doc_ids[b_idx[0], h_idx[0], kv_idx[0]]
    q_doc_ssa = utils.scalar_to_ssa(q_doc, cutlass.Int32)
    kv_doc_ssa = utils.scalar_to_ssa(kv_doc, cutlass.Int32)
    return q_doc_ssa == kv_doc_ssa

rel_bias = torch.randn((1, 1, 640), dtype=torch.float32)
aux_tensors = [rel_bias, doc_ids]

out, _ = _flash_attn_fwd(
    q, k, v,
    score_mod=doc_rel_bias_score_mod,
    mask_mod=document_mask_mod,
    block_sparse_tensors_torch=block_sparse_tensors,
    aux_tensors=aux_tensors,
)
```

### Example 2: PrefixLM with Per-Head Bias

All tokens attend bidirectionally to a fixed-length prefix + causal masking:

```python
def create_prefix_lm_mask(prefix: int, offset: int):
    @cute.jit
    def _prefix_lm_mask_mod(b_idx, h_idx, q_idx, kv_idx, aux_tensors):
        prefix_ssa = utils.scalar_to_ssa(prefix, cutlass.Int32)
        offset_ssa = utils.scalar_to_ssa(offset, cutlass.Int32)
        # Allow bidirectional attention in prefix OR causal after
        in_prefix = kv_idx < prefix_ssa
        causal = kv_idx <= q_idx + offset_ssa
        return in_prefix | causal
    return _prefix_lm_mask_mod

@cute.jit
def head_bias_score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    head_bias = aux_tensors[0]
    bias_val = head_bias[h_idx[0]].to(cutlass.Float32)
    return tSrS_ssa + bias_val

head_biases = torch.randn(num_heads, dtype=torch.float32)
mask_mod = create_prefix_lm_mask(prefix=204, offset=0)

out, _ = _flash_attn_fwd(
    q, k, v,
    score_mod=head_bias_score_mod,
    mask_mod=mask_mod,
    block_sparse_tensors_torch=block_sparse_tensors,
    aux_tensors=[head_biases],
)
```

---

## Quick Reference

| Feature | Type | Example |
|---------|------|---------|
| **ALiBi** | `score_mod` | `-slope * distance` |
| **Causal** | `mask_mod` | `kv_idx <= q_idx` |
| **Sliding window** | `mask_mod` | `abs(q_idx - kv_idx) <= w` |
| **T5 bias** | `score_mod` | `score + bias[rel_pos]` |
| **Document mask** | `mask_mod` | `doc[q] == doc[kv]` |
| **PrefixLM** | `mask_mod` | `kv < prefix \| kv <= q` |

---

## Getting Started

Minimal working example:

```python
# 1. Define mods
import flash_attn.cute.utils as utils

@cute.jit
def my_score_mod(score, b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    scale = utils.scalar_to_ssa(1.1, cutlass.Float32)
    return score * scale  # Example: scale scores

@cute.jit
def my_mask_mod(b_idx, h_idx, q_idx, kv_idx, aux_tensors):
    return kv_idx <= q_idx  # Example: causal w/o offset

# 2. Compute block sparsity
from torch.nn.attention.flex_attention import create_block_mask

block_mask = create_block_mask(
    my_mask_mod,
    B, H, seqlen_q, seqlen_kv,
    device="cuda",
    BLOCK_SIZE=(128, 128)
)

# 3. Run attention
from flash_attn.cute.interface import _flash_attn_fwd

out, lse = _flash_attn_fwd(
    q, k, v,
    score_mod=my_score_mod,
    mask_mod=my_mask_mod,
    block_sparse_tensors_torch=block_mask
)
```

**Key steps:**
1. Define attention modifications as callables
2. Compute block sparsity once (can be cached across layers/iterations)
3. Call forward function with modifications

---

## Appendix: CuTe DSL-native API

### Block Sparsity Computation

```python
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity

compute_block_sparsity(
    tile_m: int,
    tile_n: int,
    batch_size: int,
    num_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    mask_mod: Callable,
    aux_tensors: Optional[list[cute.Tensor]],
    device: str = "cuda",
    compute_full_blocks: bool = True,
    use_fast_sampling: bool = False,
) -> Tuple[BlockSparseTensors, Tuple[torch.Tensor, torch.Tensor]]
```

### Complete Hopper Workflow

```python
from flash_attn.cute.compute_block_sparsity import compute_block_sparsity
from flash_attn.cute.flash_fwd import FlashAttentionForwardSm90

tile_m, tile_n = 128, 128
batch_size, num_heads, seqlen_q, seqlen_k = 2, 8, 8192, 8192

# Compute block sparsity
blocksparse_tensors, blocksparse_torch_tensors = compute_block_sparsity(
    tile_m, tile_n,
    batch_size, num_heads, seqlen_q, seqlen_k,
    mask_mod, aux_tensors, device,
)

# Instantiate kernel
fa_fwd = FlashAttentionForwardSm90(
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    is_causal=False,
    is_local=False,
    pack_gqa=False,
    tile_m=tile_m,
    tile_n=tile_n,
    num_stages=2,
    num_threads=384,
    Q_in_regs=False,
    intra_wg_overlap=True,   # tunable
    mma_pv_is_rs=True,       # tunable
    mask_mod=mask_mod,
    score_mod=score_mod,
    has_aux_tensors=aux_tensors is not None,
)

# Compile kernel (cache compiled kernels in practice)
fa_fwd_compiled = cute.compile(
    fa_fwd,
    q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
    softmax_scale, current_stream,
    cu_seqlens_q_tensor, cu_seqlens_k_tensor,
    seqused_q_tensor, seqused_k_tensor,
    page_table_tensor,
    None, None,  # window size left/right
    learnable_sink_tensor,
    blocksparse_tensors,
    aux_tensors,
)

# Run with new arguments
fa_fwd_compiled(
    q_tensor_new, k_tensor_new, v_tensor_new, o_tensor_new, lse_tensor_new,
    softmax_scale_new, current_stream_new,
    # ... other tensors
)
```

> **Blackwell:** Replace `FlashAttentionForwardSm90` with `FlashAttentionForwardSm100`.

---

## References

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", 2021 — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
3. Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020
4. Xiao et al., "Efficient Streaming Language Models with Attention Sinks", 2023 — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
5. Guessous et al., "FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention", 2024 — [PyTorch Blog](https://pytorch.org/blog/flexattention/)
