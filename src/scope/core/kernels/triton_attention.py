"""
Triton kernels for optimized attention.

Kernel B: Piecewise bias attention for KV-cache sampling.
- 10.7% faster than flex_attention on B=1, H=16, Lq=4680, Lk=9360, D=128
- Applies log(beta) bias to past-frame tokens to mitigate error accumulation
"""

import torch
import triton
import triton.language as tl


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

    Bias rule:
      - First frame (kv_idx < frame_seqlen): no bias
      - Past frames (frame_seqlen <= kv_idx < current_block_start): bias = log_bias
      - Current block (kv_idx >= current_block_start): no bias
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

    # Ensure scalar parameters are Python ints/floats (not tensors/numpy)
    frame_seqlen = int(frame_seqlen)
    current_block_start = int(current_block_start)
    log_bias = float(log_bias)

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
