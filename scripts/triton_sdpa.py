#!/usr/bin/env python3
"""
Triton SDPA Kernel - Apprentice Track M1-M4

Goal: Write an optimized attention kernel for blockwise causal attention.

Usage:
    python scripts/triton_sdpa.py

Milestones:
- M1: Basic SDPA matching PyTorch
- M2: Blockwise causal mask
- M3: KV cache (Lq != Lk) + bias
- M4: Autotune + early exit optimization
"""

import torch
import triton
import triton.language as tl


# =============================================================================
# M4: Autotune configurations
# =============================================================================
def get_autotune_configs():
    """Generate autotune configurations for different hardware."""
    configs = []
    # Sweep block sizes and parallelism
    # B200/Blackwell benefits from larger blocks and more stages
    for BLOCK_M in [64, 128, 256]:
        for BLOCK_N in [32, 64, 128, 256]:
            for num_warps in [4, 8, 16]:
                for num_stages in [2, 3, 4, 5]:
                    # Skip configs that would use too much shared memory
                    # Rough estimate: 2 * BLOCK_M * BLOCK_N * 2 bytes (bf16)
                    shared_mem_estimate = 2 * BLOCK_M * BLOCK_N * 2
                    if shared_mem_estimate > 96 * 1024:  # 96KB shared mem limit
                        continue
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=['seq_len_q', 'seq_len_k', 'head_dim', 'MASK_MODE', 'HAS_BIAS'],
)
@triton.jit
def sdpa_kernel_autotuned(
    # Pointers to input/output tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Bias pointer (can be None/0 for no bias)
    Bias_ptr,
    # Strides for Q: [B, H, Lq, D] - how many elements to skip for each dim
    stride_qb, stride_qh, stride_qm, stride_qd,
    # Strides for K: [B, H, Lk, D]
    stride_kb, stride_kh, stride_kn, stride_kd,
    # Strides for V: [B, H, Lk, D]
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Strides for O: [B, H, Lq, D]
    stride_ob, stride_oh, stride_om, stride_od,
    # Strides for Bias: [B, H, Lq, Lk] or [1, 1, Lq, Lk] for broadcast
    stride_bb, stride_bh, stride_bm, stride_bn,
    # Dimensions
    seq_len_q,    # Lq - query sequence length
    seq_len_k,    # Lk - key/value sequence length
    head_dim,     # D - head dimension
    scale,        # 1/sqrt(D) for softmax scaling
    # Blockwise causal parameters
    frame_seqlen,  # Tokens per frame (e.g., 1560)
    block_size,    # Tokens per block = frame_seqlen * num_frame_per_block
    # Mask mode
    MASK_MODE: tl.constexpr,  # 0=none, 1=causal, 2=blockwise_causal
    HAS_BIAS: tl.constexpr,   # Whether to add bias to attention scores
    # Block sizes (compile-time constants)
    BLOCK_M: tl.constexpr,  # How many query positions per block
    BLOCK_N: tl.constexpr,  # How many key positions per block
    BLOCK_D: tl.constexpr,  # Head dimension (must cover full D)
):
    """
    Compute attention for one block of queries.

    Each program instance handles:
    - One (batch, head) pair
    - BLOCK_M query positions
    - Iterates over all key positions in chunks of BLOCK_N

    The attention formula:
        O = softmax(Q @ K^T / sqrt(D)) @ V

    We compute this in tiles to fit in GPU shared memory.
    """

    # ========== Step 1: Figure out what this program instance handles ==========
    # program_id(0) = which block of queries (0 to ceil(L/BLOCK_M))
    # program_id(1) = which (batch, head) pair (0 to B*H)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Convert pid_bh to batch and head indices
    # (For simplicity, we treat B*H as a single flattened dimension)

    # ========== Step 2: Compute base pointers for this (batch, head) ==========
    # Q, K, V, O have shapes [B, H, Lq/Lk, D]
    # We need to offset to the right (batch, head) slice
    Q_block_ptr = Q_ptr + pid_bh * stride_qh  # Skip to right head
    K_block_ptr = K_ptr + pid_bh * stride_kh
    V_block_ptr = V_ptr + pid_bh * stride_vh
    O_block_ptr = O_ptr + pid_bh * stride_oh
    if HAS_BIAS:
        Bias_block_ptr = Bias_ptr + pid_bh * stride_bh

    # ========== Step 3: Create index vectors ==========
    # offs_m: which query positions this block handles [0, 1, ..., BLOCK_M-1] + pid_m * BLOCK_M
    # offs_n: which key positions we're currently loading [0, 1, ..., BLOCK_N-1]
    # offs_d: head dimension indices [0, 1, ..., BLOCK_D-1]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                      # [BLOCK_N]
    offs_d = tl.arange(0, BLOCK_D)                      # [BLOCK_D]

    # ========== Step 4: Load Q block ==========
    # Q has shape [L, D] for this (batch, head)
    # We want Q[offs_m, offs_d] -> shape [BLOCK_M, BLOCK_D]
    #
    # Pointer arithmetic: Q_block_ptr + offs_m * stride_qm + offs_d * stride_qd
    # But we need a 2D block, so we use broadcasting:
    #   offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    q_ptrs = Q_block_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd

    # Mask for out-of-bounds queries (when Lq is not divisible by BLOCK_M)
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)

    # Load Q block - shape [BLOCK_M, BLOCK_D]
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ========== Step 5: Initialize accumulators for online softmax ==========
    # We'll compute softmax in a numerically stable way:
    #   softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    #
    # For online computation, we track:
    #   m_i = max so far (per query position)
    #   l_i = sum of exp(x - m_i) so far
    #   o_i = weighted sum of V so far

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # Running sum
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)        # Running output

    # ========== Step 6: Loop over K/V blocks ==========
    # We iterate over all key positions in chunks of BLOCK_N
    #
    # M4 optimization: For causal/blockwise modes, compute the max KV tile we need
    # and use a compile-time loop bound with runtime skip.
    #
    # For blockwise causal with block_size=4680, L=9360:
    # - Query positions 0-4679 (block 0) only need KV tiles 0-73
    # - Query positions 4680-9359 (block 1) need all KV tiles 0-146
    #
    # We use the full loop but skip iterations where all keys would be masked.

    max_q_pos = pid_m * BLOCK_M + BLOCK_M - 1

    # Compute the max KV position this query tile needs
    if MASK_MODE == 0:
        max_k_needed = seq_len_k  # Full attention - need all keys
    elif MASK_MODE == 1:
        max_k_needed = max_q_pos + 1  # Causal - only need keys <= max query pos
    else:  # MASK_MODE == 2
        max_q_block = max_q_pos // block_size
        max_k_needed = (max_q_block + 1) * block_size  # Blockwise - need keys up to end of query's block
        max_k_needed = tl.minimum(max_k_needed, seq_len_k)

    # Full loop bound (compile-time friendly)
    num_k_blocks_full = tl.cdiv(seq_len_k, BLOCK_N)

    for kv_block_idx in range(num_k_blocks_full):
        # Current key positions: [k_block_idx * BLOCK_N, ..., (k_block_idx+1) * BLOCK_N - 1]
        k_offs = kv_block_idx * BLOCK_N + offs_n  # [BLOCK_N]

        # M4: Skip this KV tile if all keys are beyond what we need
        kv_tile_start = kv_block_idx * BLOCK_N
        if MASK_MODE != 0:
            # Skip if this tile starts beyond our needed range
            # Use tl.where to avoid divergent control flow
            skip_tile = kv_tile_start >= max_k_needed
        else:
            skip_tile = False

        # ----- Load K block [BLOCK_N, BLOCK_D] -----
        k_ptrs = K_block_ptr + k_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (k_offs[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # ----- Compute QK^T [BLOCK_M, BLOCK_N] -----
        # q: [BLOCK_M, BLOCK_D], k: [BLOCK_N, BLOCK_D]
        # We want q @ k^T = [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_M, BLOCK_N]

        # Apply scaling
        qk = qk * scale

        # ----- Add bias if provided -----
        if HAS_BIAS:
            # Load bias block [BLOCK_M, BLOCK_N]
            bias_ptrs = Bias_block_ptr + offs_m[:, None] * stride_bm + k_offs[None, :] * stride_bn
            bias_mask = (offs_m[:, None] < seq_len_q) & (k_offs[None, :] < seq_len_k)
            bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
            qk = qk + bias

        # Mask out-of-bounds keys and skipped tiles
        if MASK_MODE != 0:
            qk = tl.where(skip_tile, float('-inf'), qk)
        qk = tl.where(k_offs[None, :] < seq_len_k, qk, float('-inf'))

        # ----- Apply attention mask based on mode -----
        if MASK_MODE == 1:
            # Standard causal: query can only attend to keys at positions <= its own
            causal_mask = offs_m[:, None] >= k_offs[None, :]  # [BLOCK_M, BLOCK_N]
            qk = tl.where(causal_mask, qk, float('-inf'))
        elif MASK_MODE == 2:
            # Blockwise causal:
            # - Full attention within a block
            # - Causal across blocks (only attend to same or earlier blocks)
            # block_idx = position // block_size
            # Allow attention if: key_block <= query_block
            q_block_idx = offs_m[:, None] // block_size  # [BLOCK_M, 1]
            k_block_idx = k_offs[None, :] // block_size  # [1, BLOCK_N]
            blockwise_mask = q_block_idx >= k_block_idx  # [BLOCK_M, BLOCK_N]
            qk = tl.where(blockwise_mask, qk, float('-inf'))

        # ----- Online softmax update -----
        # New max for this block
        m_ij = tl.max(qk, axis=1)  # [BLOCK_M]

        # New running max
        m_new = tl.maximum(m_i, m_ij)  # [BLOCK_M]

        # Correction factors for previous and current terms
        alpha = tl.exp(m_i - m_new)      # Scale down previous terms
        beta = tl.exp(m_ij - m_new)      # Scale for current terms

        # Update running sum
        p_ij = tl.exp(qk - m_new[:, None])  # [BLOCK_M, BLOCK_N] - attention weights
        l_new = alpha * l_i + tl.sum(p_ij, axis=1)  # [BLOCK_M]

        # ----- Load V block [BLOCK_N, BLOCK_D] -----
        v_ptrs = V_block_ptr + k_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (k_offs[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        if MASK_MODE != 0:
            v_mask = v_mask & ~skip_tile
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # ----- Update output accumulator -----
        # Scale previous output by alpha, add new contribution
        # p_ij: [BLOCK_M, BLOCK_N], v: [BLOCK_N, BLOCK_D]
        o_i = alpha[:, None] * o_i + tl.dot(p_ij.to(v.dtype), v)

        # Update running stats
        m_i = m_new
        l_i = l_new

    # ========== Step 7: Finalize output ==========
    # Divide by sum to complete softmax normalization
    o_i = o_i / l_i[:, None]

    # ========== Step 8: Store output ==========
    o_ptrs = O_block_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=o_mask)


def triton_sdpa(Q, K, V, scale=None, mask_mode="causal", block_size=None, bias=None, use_autotune=True):
    """
    Wrapper to call the Triton SDPA kernel.

    Args:
        Q: [B, H, Lq, D] query tensor
        K: [B, H, Lk, D] key tensor
        V: [B, H, Lk, D] value tensor
        scale: scaling factor (default: 1/sqrt(D))
        mask_mode: "none", "causal", or "blockwise_causal"
        block_size: for blockwise_causal, the number of tokens per block
        bias: [B, H, Lq, Lk] or [1, 1, Lq, Lk] bias tensor to add to attention scores
        use_autotune: whether to use autotuned kernel (default: True)

    Returns:
        O: [B, H, Lq, D] attention output
    """
    B, H, Lq, D = Q.shape
    _, _, Lk, _ = K.shape
    assert K.shape == V.shape, "K and V must have same shape"
    assert K.shape[0] == B and K.shape[1] == H and K.shape[3] == D, "K must match Q in B, H, D"

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # Convert mask_mode to int for kernel
    mask_mode_int = {"none": 0, "causal": 1, "blockwise_causal": 2}[mask_mode]

    # Default block_size to Lk (single block = causal)
    if block_size is None:
        block_size = Lk

    # Handle bias
    has_bias = bias is not None
    if has_bias:
        assert bias.shape[-2] == Lq and bias.shape[-1] == Lk, f"Bias shape mismatch: {bias.shape} vs Lq={Lq}, Lk={Lk}"
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        # Dummy values when no bias
        bias = Q  # Won't be used
        bias_strides = (0, 0, 0, 0)

    # Allocate output
    O = torch.empty(B, H, Lq, D, device=Q.device, dtype=Q.dtype)

    # BLOCK_D must cover full head dim
    BLOCK_D = D

    if use_autotune:
        # Use autotuned kernel - grid computed inside based on autotuned BLOCK_M
        def grid(meta):
            return (triton.cdiv(Lq, meta['BLOCK_M']), B * H)

        sdpa_kernel_autotuned[grid](
            Q, K, V, O,
            bias,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            bias_strides[0], bias_strides[1], bias_strides[2], bias_strides[3],
            Lq, Lk, D, scale,
            block_size, block_size,
            MASK_MODE=mask_mode_int,
            HAS_BIAS=has_bias,
            BLOCK_D=BLOCK_D,
        )
    else:
        # Use fixed block sizes (for debugging)
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(Lq, BLOCK_M), B * H)

        sdpa_kernel_autotuned[grid](
            Q, K, V, O,
            bias,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            bias_strides[0], bias_strides[1], bias_strides[2], bias_strides[3],
            Lq, Lk, D, scale,
            block_size, block_size,
            MASK_MODE=mask_mode_int,
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

    return O


def create_blockwise_causal_mask(Lq, block_size, device, Lk=None):
    """
    Create a blockwise causal attention mask.

    Within a block: full attention (all True)
    Across blocks: causal (earlier blocks only)

    Args:
        Lq: query sequence length
        block_size: tokens per block
        device: torch device
        Lk: key sequence length (default: Lq)

    Returns: [Lq, Lk] boolean mask where True = attend, False = mask out
    """
    if Lk is None:
        Lk = Lq
    q_positions = torch.arange(Lq, device=device)
    k_positions = torch.arange(Lk, device=device)
    q_block = q_positions[:, None] // block_size  # [Lq, 1]
    k_block = k_positions[None, :] // block_size  # [1, Lk]
    # Can attend if key's block <= query's block
    mask = q_block >= k_block  # [Lq, Lk]
    return mask


def test_correctness():
    """Compare our kernel against PyTorch's SDPA."""
    print("=" * 60)
    print("M1: Triton SDPA Correctness Test (Causal)")
    print("=" * 60)

    # Start with tiny shapes for debugging
    test_cases = [
        (1, 1, 16, 16),    # Minimal
        (1, 4, 64, 64),    # Small
        (2, 8, 128, 64),   # Medium
    ]

    for B, H, L, D in test_cases:
        print(f"\nTesting shape: B={B}, H={H}, L={L}, D={D}")

        # Create inputs
        Q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

        # Reference: PyTorch SDPA with causal mask
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Our kernel
        out = triton_sdpa(Q, K, V, mask_mode="causal")

        # Compare
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()

        status = "PASS" if max_err < 0.05 else "FAIL"
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Status:     {status}")

        if status == "FAIL":
            print("  DEBUG: First few values:")
            print(f"    ref[:2,:2,0,0]: {ref[0,0,:4,0].tolist()}")
            print(f"    out[:2,:2,0,0]: {out[0,0,:4,0].tolist()}")

    # ===== M2: Blockwise causal test =====
    print("\n" + "=" * 60)
    print("M2: Triton SDPA Correctness Test (Blockwise Causal)")
    print("=" * 60)

    # Test cases: (B, H, L, D, block_size)
    blockwise_cases = [
        (1, 1, 64, 16, 16),     # 4 blocks of 16
        (1, 4, 128, 64, 32),    # 4 blocks of 32
        (1, 8, 256, 64, 64),    # 4 blocks of 64
    ]

    for B, H, L, D, block_size in blockwise_cases:
        num_blocks = L // block_size
        print(f"\nTesting: B={B}, H={H}, L={L}, D={D}, block_size={block_size} ({num_blocks} blocks)")

        # Create inputs
        Q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

        # Reference: PyTorch SDPA with explicit blockwise mask
        blockwise_mask = create_blockwise_causal_mask(L, block_size, Q.device)
        # Convert to attn_mask format: 0 for attend, -inf for mask
        attn_mask = torch.where(blockwise_mask, 0.0, float('-inf'))
        attn_mask = attn_mask.to(Q.dtype)

        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask
            )

        # Our kernel
        out = triton_sdpa(Q, K, V, mask_mode="blockwise_causal", block_size=block_size)

        # Compare
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()

        status = "PASS" if max_err < 0.05 else "FAIL"
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Status:     {status}")

        if status == "FAIL":
            print("  DEBUG: First few values:")
            print(f"    ref[0,0,:4,0]: {ref[0,0,:4,0].tolist()}")
            print(f"    out[0,0,:4,0]: {out[0,0,:4,0].tolist()}")
            # Also show mask pattern
            print(f"  Mask pattern (first 8x8):")
            print(blockwise_mask[:8, :8].int())

    # ===== M3: KV Cache + Bias tests =====
    print("\n" + "=" * 60)
    print("M3: Triton SDPA Correctness Test (KV Cache + Bias)")
    print("=" * 60)

    # Test 3a: KV cache - Lq < Lk, no mask (full attention)
    print("\n--- Test 3a: KV Cache (Lq < Lk, no mask) ---")
    kv_cache_cases = [
        (1, 1, 32, 64, 16),    # Lq=32, Lk=64
        (1, 4, 64, 128, 64),   # Lq=64, Lk=128
        (1, 8, 128, 256, 64),  # Lq=128, Lk=256
    ]

    for B, H, Lq, Lk, D in kv_cache_cases:
        print(f"\nTesting: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D}")

        Q = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)

        # Reference: PyTorch SDPA with no mask
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

        # Our kernel with no mask
        out = triton_sdpa(Q, K, V, mask_mode="none")

        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        status = "PASS" if max_err < 0.05 else "FAIL"
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Status:     {status}")

    # Test 3b: Bias addition (Lq == Lk)
    print("\n--- Test 3b: Bias Addition (Lq == Lk) ---")
    bias_cases = [
        (1, 1, 64, 64, 16),
        (1, 4, 128, 128, 64),
        (2, 8, 256, 256, 64),
    ]

    for B, H, Lq, Lk, D in bias_cases:
        print(f"\nTesting: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D} (with bias)")

        Q = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
        bias = torch.randn(B, H, Lq, Lk, device='cuda', dtype=torch.bfloat16) * 0.1

        # Reference: PyTorch SDPA with bias as attn_mask
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=bias)

        # Our kernel with bias
        out = triton_sdpa(Q, K, V, mask_mode="none", bias=bias)

        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        status = "PASS" if max_err < 0.05 else "FAIL"
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Status:     {status}")

    # Test 3c: KV cache + bias + blockwise mask (the full scenario)
    print("\n--- Test 3c: KV Cache + Bias + Blockwise Mask ---")
    full_cases = [
        (1, 1, 32, 64, 16, 16),    # Lq=32, Lk=64, block_size=16
        (1, 4, 64, 128, 64, 32),   # Lq=64, Lk=128, block_size=32
        (1, 8, 128, 256, 64, 64),  # Lq=128, Lk=256, block_size=64
    ]

    for B, H, Lq, Lk, D, block_size in full_cases:
        print(f"\nTesting: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D}, block_size={block_size}")

        Q = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
        bias = torch.randn(B, H, Lq, Lk, device='cuda', dtype=torch.bfloat16) * 0.1

        # Create blockwise mask for asymmetric Lq, Lk
        blockwise_mask = create_blockwise_causal_mask(Lq, block_size, Q.device, Lk)
        # Combined: bias + blockwise mask
        attn_mask = torch.where(blockwise_mask, bias, torch.tensor(float('-inf'), device=Q.device, dtype=Q.dtype))

        # Reference: PyTorch SDPA with combined mask
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)

        # Our kernel with bias + blockwise mask
        out = triton_sdpa(Q, K, V, mask_mode="blockwise_causal", block_size=block_size, bias=bias)

        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        status = "PASS" if max_err < 0.05 else "FAIL"
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Status:     {status}")

        if status == "FAIL":
            print("  DEBUG: First few values:")
            print(f"    ref[0,0,:4,0]: {ref[0,0,:4,0].tolist()}")
            print(f"    out[0,0,:4,0]: {out[0,0,:4,0].tolist()}")


def benchmark():
    """Benchmark our kernel vs PyTorch SDPA."""
    import time

    print("\n" + "=" * 60)
    print("M1: Triton SDPA Benchmark (Causal)")
    print("=" * 60)

    B, H, L, D = 1, 16, 1024, 128
    print(f"\nShape: B={B}, H={H}, L={L}, D={D}")

    Q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = triton_sdpa(Q, K, V)
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = triton_sdpa(Q, K, V)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 100 * 1000

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000

    print(f"\nTriton SDPA:  {triton_time:.3f} ms")
    print(f"PyTorch SDPA: {pytorch_time:.3f} ms")
    print(f"Speedup:      {pytorch_time / triton_time:.2f}x")


def benchmark_m4():
    """M4 Benchmark: Compare against flex_attention on target shapes."""
    import time

    print("\n" + "=" * 60)
    print("M4: Triton SDPA vs flex_attention (Blockwise Causal)")
    print("=" * 60)

    # Try to import flex_attention
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        has_flex = True
    except ImportError:
        print("flex_attention not available, skipping comparison")
        has_flex = False

    # Target shape from the pipeline
    B, H, L, D = 1, 16, 9360, 128
    frame_seqlen = 1560
    num_frame_per_block = 3
    block_size = frame_seqlen * num_frame_per_block  # 4680

    print(f"\nTarget shape: B={B}, H={H}, L={L}, D={D}")
    print(f"Block size: {block_size} (frame_seqlen={frame_seqlen} × {num_frame_per_block})")

    Q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

    # Create reference mask for comparison
    blockwise_mask = create_blockwise_causal_mask(L, block_size, Q.device)
    attn_mask = torch.where(blockwise_mask, 0.0, float('-inf')).to(Q.dtype)

    # ===== Triton SDPA =====
    print("\nWarming up Triton SDPA (autotune)...")
    print(f"  Autotune will try {len(get_autotune_configs())} configurations...")
    for _ in range(10):  # More warmup for autotune
        _ = triton_sdpa(Q, K, V, mask_mode="blockwise_causal", block_size=block_size)
    torch.cuda.synchronize()

    # Print best config if available
    try:
        best = sdpa_kernel_autotuned.best_config
        print(f"  Best config: BLOCK_M={best.kwargs.get('BLOCK_M')}, BLOCK_N={best.kwargs.get('BLOCK_N')}, "
              f"num_warps={best.num_warps}, num_stages={best.num_stages}")
    except Exception:
        pass

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = triton_sdpa(Q, K, V, mask_mode="blockwise_causal", block_size=block_size)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 50 * 1000

    print(f"Triton SDPA:      {triton_time:.3f} ms")

    # ===== PyTorch SDPA with mask =====
    print("\nWarming up PyTorch SDPA...")
    for _ in range(5):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / 50 * 1000

    print(f"PyTorch SDPA:     {pytorch_time:.3f} ms")

    # ===== flex_attention =====
    if has_flex:
        # Create block_mask for flex_attention
        def blockwise_causal_mask_fn(b, h, q_idx, kv_idx):
            q_block = q_idx // block_size
            kv_block = kv_idx // block_size
            return q_block >= kv_block

        block_mask = create_block_mask(
            blockwise_causal_mask_fn,
            B=B, H=H, Q_LEN=L, KV_LEN=L,
            device='cuda',
        )

        # Compile flex_attention
        flex_attn_compiled = torch.compile(flex_attention)

        print("\nWarming up flex_attention...")
        for _ in range(5):
            _ = flex_attn_compiled(Q, K, V, block_mask=block_mask)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = flex_attn_compiled(Q, K, V, block_mask=block_mask)
        torch.cuda.synchronize()
        flex_time = (time.perf_counter() - start) / 50 * 1000

        print(f"flex_attention:   {flex_time:.3f} ms")

        print(f"\n--- Results ---")
        print(f"Triton vs PyTorch:      {pytorch_time / triton_time:.2f}x")
        print(f"Triton vs flex_attention: {flex_time / triton_time:.2f}x")

        if triton_time < flex_time:
            print(f"\n🎉 M4 PASS: Triton beats flex_attention by {(flex_time - triton_time) / flex_time * 100:.1f}%")
        else:
            print(f"\n⚠️  M4 pending: flex_attention is {(triton_time - flex_time) / triton_time * 100:.1f}% faster")
    else:
        print(f"\n--- Results ---")
        print(f"Triton vs PyTorch: {pytorch_time / triton_time:.2f}x")


def analyze_sparsity():
    """Analyze the sparsity pattern and theoretical speedup."""
    print("\n" + "=" * 60)
    print("Sparsity Analysis")
    print("=" * 60)

    L = 9360
    block_size = 4680
    BLOCK_M = 64
    BLOCK_N = 64

    num_q_tiles = (L + BLOCK_M - 1) // BLOCK_M
    num_k_tiles_full = (L + BLOCK_N - 1) // BLOCK_N

    total_tiles_dense = num_q_tiles * num_k_tiles_full
    total_tiles_sparse = 0

    print(f"\nL={L}, block_size={block_size}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"Query tiles: {num_q_tiles}, KV tiles (full): {num_k_tiles_full}")
    print()

    for q_tile in range(num_q_tiles):
        q_start = q_tile * BLOCK_M
        q_end = min(q_start + BLOCK_M - 1, L - 1)
        max_q_pos = q_end

        # Compute how many KV tiles this query tile needs (our early exit logic)
        max_q_block = max_q_pos // block_size
        max_k_pos = (max_q_block + 1) * block_size
        num_k_tiles_needed = (min(max_k_pos, L) + BLOCK_N - 1) // BLOCK_N

        total_tiles_sparse += num_k_tiles_needed

        if q_tile < 3 or q_tile >= num_q_tiles - 3:
            print(f"  Q tile {q_tile}: positions {q_start}-{q_end}, "
                  f"semantic block {max_q_block}, KV tiles needed: {num_k_tiles_needed}/{num_k_tiles_full}")
        elif q_tile == 3:
            print("  ...")

    sparsity = 1 - total_tiles_sparse / total_tiles_dense
    theoretical_speedup = total_tiles_dense / total_tiles_sparse

    print(f"\nTotal tiles (dense): {total_tiles_dense}")
    print(f"Total tiles (sparse): {total_tiles_sparse}")
    print(f"Sparsity: {sparsity*100:.1f}%")
    print(f"Theoretical speedup from early exit: {theoretical_speedup:.2f}x")
    print(f"\nflex_attention time: 1.867 ms")
    print(f"Our time: 2.295 ms")
    print(f"If early exit is working, expected time: ~{2.295 / theoretical_speedup:.3f} ms")
    print(f"(But we're already measuring 2.295ms, so early exit might not be helping)")


def benchmark_no_early_exit():
    """Benchmark without early exit to measure its impact."""
    import time

    print("\n" + "=" * 60)
    print("Testing Early Exit Impact")
    print("=" * 60)

    B, H, L, D = 1, 16, 9360, 128
    block_size = 4680

    Q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

    print(f"\nShape: B={B}, H={H}, L={L}, D={D}")

    # Test with mask_mode="none" (full attention, no early exit)
    print("\nFull attention (no mask, no early exit):")
    for _ in range(5):
        _ = triton_sdpa(Q, K, V, mask_mode="none")
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        _ = triton_sdpa(Q, K, V, mask_mode="none")
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) / 20 * 1000
    print(f"  Time: {full_time:.3f} ms")

    # Test with blockwise causal (with early exit)
    print("\nBlockwise causal (with early exit):")
    for _ in range(5):
        _ = triton_sdpa(Q, K, V, mask_mode="blockwise_causal", block_size=block_size)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        _ = triton_sdpa(Q, K, V, mask_mode="blockwise_causal", block_size=block_size)
    torch.cuda.synchronize()
    sparse_time = (time.perf_counter() - start) / 20 * 1000
    print(f"  Time: {sparse_time:.3f} ms")

    speedup = full_time / sparse_time
    print(f"\nEarly exit speedup: {speedup:.2f}x")
    print(f"Expected (from sparsity): ~1.33x")


# =============================================================================
# KERNEL B: Sampling with piecewise bias (no mask)
# =============================================================================
# This kernel is for the KV-cache sampling path where we apply a negative bias
# to past-frame tokens to mitigate error accumulation.
#
# Bias rule:
#   - First frame (kv_idx < frame_seqlen): no bias
#   - Past frames (frame_seqlen <= kv_idx < current_block_start): bias = log(beta)
#   - Current block (kv_idx >= current_block_start): no bias


def get_kernel_b_configs():
    """Autotune configs for Kernel B."""
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128]:
            for num_warps in [4, 8]:
                for num_stages in [2, 3, 4]:
                    configs.append(
                        triton.Config(
                            {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(
    configs=get_kernel_b_configs(),
    key=['seq_len_q', 'seq_len_k', 'head_dim'],
)
@triton.jit
def kernel_b_bias_attention(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Strides for Q: [B, H, Lq, D]
    stride_qb, stride_qh, stride_qm, stride_qd,
    # Strides for K: [B, H, Lk, D]
    stride_kb, stride_kh, stride_kn, stride_kd,
    # Strides for V: [B, H, Lk, D]
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Strides for O: [B, H, Lq, D]
    stride_ob, stride_oh, stride_om, stride_od,
    # Dimensions
    seq_len_q,      # Lq
    seq_len_k,      # Lk
    head_dim,       # D
    scale,          # 1/sqrt(D)
    # Bias parameters
    frame_seqlen,           # Tokens per frame (e.g., 1560)
    current_block_start,    # First token of current block (past frames end here)
    log_bias,               # log(beta), typically log(0.3) ≈ -1.2
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel B: Attention with piecewise constant bias for KV-cache sampling.

    No masking - full attention with bias applied to past-frame tokens.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # Query block
    pid_bh = tl.program_id(1)  # Batch * Head

    # Base pointers for this (batch, head)
    Q_block_ptr = Q_ptr + pid_bh * stride_qh
    K_block_ptr = K_ptr + pid_bh * stride_kh
    V_block_ptr = V_ptr + pid_bh * stride_vh
    O_block_ptr = O_ptr + pid_bh * stride_oh

    # Index vectors
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q block [BLOCK_M, BLOCK_D]
    q_ptrs = Q_block_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Loop over KV blocks
    num_k_blocks = tl.cdiv(seq_len_k, BLOCK_N)

    for kv_block_idx in range(num_k_blocks):
        k_offs = kv_block_idx * BLOCK_N + offs_n

        # Load K [BLOCK_N, BLOCK_D]
        k_ptrs = K_block_ptr + k_offs[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (k_offs[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * scale

        # Apply piecewise bias
        # past_frame_mask: True for tokens in past frames (not first frame, not current block)
        past_frame_mask = (k_offs[None, :] >= frame_seqlen) & (k_offs[None, :] < current_block_start)
        qk = tl.where(past_frame_mask, qk + log_bias, qk)

        # Mask out-of-bounds keys
        qk = tl.where(k_offs[None, :] < seq_len_k, qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        p_ij = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p_ij, axis=1)

        # Load V [BLOCK_N, BLOCK_D]
        v_ptrs = V_block_ptr + k_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (k_offs[:, None] < seq_len_k) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Update output accumulator
        o_i = alpha[:, None] * o_i + tl.dot(p_ij.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    # Finalize: divide by sum
    o_i = o_i / l_i[:, None]

    # Store output
    o_ptrs = O_block_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=o_mask)


def triton_kernel_b(Q, K, V, frame_seqlen, current_block_start, log_bias, scale=None):
    """
    Wrapper for Kernel B: Attention with piecewise bias.

    Args:
        Q: [B, H, Lq, D] query tensor
        K: [B, H, Lk, D] key tensor
        V: [B, H, Lk, D] value tensor
        frame_seqlen: tokens per frame (e.g., 1560)
        current_block_start: first token index of current block
        log_bias: log(beta) to apply to past frames (e.g., log(0.3))
        scale: attention scale (default: 1/sqrt(D))

    Returns:
        O: [B, H, Lq, D] attention output
    """
    B, H, Lq, D = Q.shape
    _, _, Lk, _ = K.shape

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    O = torch.empty(B, H, Lq, D, device=Q.device, dtype=Q.dtype)

    BLOCK_D = D

    # Dynamic grid based on autotuned BLOCK_M
    def grid(meta):
        return (triton.cdiv(Lq, meta['BLOCK_M']), B * H)

    kernel_b_bias_attention[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Lq, Lk, D, scale,
        frame_seqlen, current_block_start, log_bias,
        BLOCK_D=BLOCK_D,
    )

    return O


def test_kernel_b_correctness():
    """Test Kernel B against flex_attention with score_mod."""
    import math

    print("\n" + "=" * 60)
    print("Kernel B: Correctness Test (Piecewise Bias)")
    print("=" * 60)

    try:
        from torch.nn.attention.flex_attention import flex_attention
        has_flex = True
    except ImportError:
        print("flex_attention not available, using manual reference")
        has_flex = False

    # Test cases: (B, H, Lq, Lk, D, frame_seqlen, num_frame_per_block, beta)
    test_cases = [
        (1, 1, 64, 128, 16, 32, 2, 0.3),       # Small
        (1, 4, 128, 256, 64, 64, 2, 0.3),      # Medium
        (1, 16, 4680, 9360, 128, 1560, 3, 0.3), # Krea target shape
    ]

    for B, H, Lq, Lk, D, frame_seqlen, num_frame_per_block, beta in test_cases:
        current_block_start = Lk - frame_seqlen * num_frame_per_block
        log_bias = math.log(beta)

        print(f"\nTest: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D}")
        print(f"  frame_seqlen={frame_seqlen}, current_block_start={current_block_start}")
        print(f"  beta={beta}, log_bias={log_bias:.4f}")

        Q = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.bfloat16)
        K = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
        V = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)

        # Reference implementation
        if has_flex:
            frame_seqlen_t = torch.tensor(frame_seqlen, dtype=torch.int32, device='cuda')
            current_block_start_t = torch.tensor(current_block_start, dtype=torch.int32, device='cuda')
            log_bias_t = torch.tensor(log_bias, dtype=Q.dtype, device='cuda')

            def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
                past_mask = (kv_idx >= frame_seqlen_t) & (kv_idx < current_block_start_t)
                return torch.where(past_mask, score + log_bias_t, score)

            with torch.no_grad():
                ref = flex_attention(Q, K, V, score_mod=score_mod)
        else:
            # Manual reference: compute full attention with bias
            scale = 1.0 / (D ** 0.5)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            # Apply bias
            kv_indices = torch.arange(Lk, device='cuda')
            past_mask = (kv_indices >= frame_seqlen) & (kv_indices < current_block_start)
            scores[:, :, :, past_mask] += log_bias

            attn = torch.softmax(scores, dim=-1)
            ref = torch.matmul(attn, V)

        # Our kernel
        out = triton_kernel_b(Q, K, V, frame_seqlen, current_block_start, log_bias)

        # Compare
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()

        status = "PASS" if max_err < 0.1 else "FAIL"
        print(f"  Max error:  {max_err:.6f}")
        print(f"  Mean error: {mean_err:.6f}")
        print(f"  Status:     {status}")

        if status == "FAIL":
            print("  DEBUG: Sample values:")
            print(f"    ref[0,0,:4,0]: {ref[0,0,:4,0].tolist()}")
            print(f"    out[0,0,:4,0]: {out[0,0,:4,0].tolist()}")


def benchmark_kernel_b():
    """Benchmark Kernel B against flex_attention."""
    import math
    import time

    print("\n" + "=" * 60)
    print("Kernel B: Benchmark (Piecewise Bias)")
    print("=" * 60)

    try:
        from torch.nn.attention.flex_attention import flex_attention
        flex_attn = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
        has_flex = True
    except ImportError:
        print("flex_attention not available")
        has_flex = False
        return

    # Krea target shape
    B, H, Lq, Lk, D = 1, 16, 4680, 9360, 128
    frame_seqlen = 1560
    num_frame_per_block = 3
    beta = 0.3

    current_block_start = Lk - frame_seqlen * num_frame_per_block
    log_bias = math.log(beta)

    print(f"\nShape: B={B}, H={H}, Lq={Lq}, Lk={Lk}, D={D}")
    print(f"frame_seqlen={frame_seqlen}, current_block_start={current_block_start}")
    print(f"beta={beta}, log_bias={log_bias:.4f}")

    Q = torch.randn(B, H, Lq, D, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(B, H, Lk, D, device='cuda', dtype=torch.bfloat16)

    # flex_attention reference
    frame_seqlen_t = torch.tensor(frame_seqlen, dtype=torch.int32, device='cuda')
    current_block_start_t = torch.tensor(current_block_start, dtype=torch.int32, device='cuda')
    log_bias_t = torch.tensor(log_bias, dtype=Q.dtype, device='cuda')

    def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
        past_mask = (kv_idx >= frame_seqlen_t) & (kv_idx < current_block_start_t)
        return torch.where(past_mask, score + log_bias_t, score)

    # Warmup flex
    print("\nWarming up flex_attention...")
    for _ in range(5):
        _ = flex_attn(Q, K, V, score_mod=score_mod)
    torch.cuda.synchronize()

    # Benchmark flex
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = flex_attn(Q, K, V, score_mod=score_mod)
    torch.cuda.synchronize()
    flex_time = (time.perf_counter() - start) / 50 * 1000

    print(f"flex_attention: {flex_time:.3f} ms")

    # Warmup Triton
    print("\nWarming up Triton Kernel B...")
    for _ in range(5):
        _ = triton_kernel_b(Q, K, V, frame_seqlen, current_block_start, log_bias)
    torch.cuda.synchronize()

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        _ = triton_kernel_b(Q, K, V, frame_seqlen, current_block_start, log_bias)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 50 * 1000

    print(f"Triton Kernel B: {triton_time:.3f} ms")

    print(f"\n--- Results ---")
    print(f"Triton vs flex: {flex_time / triton_time:.2f}x")

    if triton_time < flex_time:
        print(f"\n🎉 Kernel B beats flex_attention by {(flex_time - triton_time) / flex_time * 100:.1f}%")
    else:
        print(f"\n⚠️  flex_attention is {(triton_time - flex_time) / triton_time * 100:.1f}% faster")


if __name__ == "__main__":
    import sys
    if "--m4" in sys.argv:
        # Run M4 benchmark only
        benchmark_m4()
    elif "--analyze" in sys.argv:
        # Analyze sparsity and early exit
        analyze_sparsity()
        benchmark_no_early_exit()
    elif "--bench" in sys.argv:
        # Run all benchmarks
        benchmark()
        benchmark_m4()
    elif "--kernel-b" in sys.argv:
        # Test and benchmark Kernel B
        test_kernel_b_correctness()
        benchmark_kernel_b()
    else:
        # Run correctness tests + basic benchmark
        test_correctness()
        benchmark()
