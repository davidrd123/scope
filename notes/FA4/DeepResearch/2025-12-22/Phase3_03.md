Below is a **single-file patch** for `src/scope/core/kernels/triton_rope_fused.py` that:

* Keeps the current padded kernel as **v1** (unchanged behavior).
* Adds a **v2** fused 3-way kernel that:

  * uses **BLOCK_M / BLOCK_H** tiling/grid (FlashAttention-style),
  * rotates **all 64 pairs in one shot** (D=128) using `tl.split`/`tl.join`,
  * uses a **single 64‑pair arange** and **chunk masks** (C0/C1/C2) to select the correct axis table entries,
  * **removes the per-chunk power-of-2 padding overhead** (22/21/21 no longer padded to 32/32/32).
* Updates the wrapper to select **v1 vs v2** via env var (default **v2**):

  * `SCOPE_TRITON_ROPE_FUSED_VERSION=1` → v1
  * `SCOPE_TRITON_ROPE_FUSED_VERSION=2` (default) → v2
* Preserves existing semantics for:

  * `start_frame` (time axis only),
  * tail handling (`empty_like` + tiny tail copy when needed),
  * error behavior (wrapper still raises; strict/fallback behavior remains in callers).

---

## Patch (unified diff)

```diff
diff --git a/src/scope/core/kernels/triton_rope_fused.py b/src/scope/core/kernels/triton_rope_fused.py
index 1111111..2222222 100644
--- a/src/scope/core/kernels/triton_rope_fused.py
+++ b/src/scope/core/kernels/triton_rope_fused.py
@@ -1,12 +1,23 @@
 # src/scope/core/kernels/triton_rope_fused.py
 # Copyright (c) 2025
 # Triton fused 3-way RoPE (time/height/width) lookup + rotate.
 #
 # Fast-path target: D=128 (C=64 pairs).
 # Avoids materializing (seq_len, C) cos/sin tables.
+#
+# Kernel versions:
+#   - v1: original implementation (padded per-chunk tl.arange -> 32/32/32 for 22/21/21)
+#   - v2: BLOCK_M/BLOCK_H tiling + single 64-pair arange with per-chunk masks (no padding work)
+#
+# Selection:
+#   SCOPE_TRITON_ROPE_FUSED_VERSION=2 (default) -> v2
+#   SCOPE_TRITON_ROPE_FUSED_VERSION=1           -> v1
+#
+# Tuning:
+#   v1: SCOPE_TRITON_ROPE_FUSED_BLOCK_L / NUM_WARPS / NUM_STAGES
+#   v2: SCOPE_TRITON_ROPE_FUSED_BLOCK_M / BLOCK_H / NUM_WARPS / NUM_STAGES

 from __future__ import annotations

 import os
@@ -73,10 +84,10 @@ def get_rope_axis_tables(
 # -----------------------------------------------------------------------------
 # Triton kernel: grid = (B*H, ceil_div(seq_len, BLOCK_L))
 # Applies 3-way RoPE lookup + rotate in 3 contiguous chunks.
 # x is interleaved pairs: (2*j, 2*j+1)
 # -----------------------------------------------------------------------------
 @triton.jit
-def rope_fused_3way_kernel(
+def rope_fused_3way_kernel_v1(
     X_ptr,
     OUT_ptr,
     COS_F_ptr,
@@ -250,6 +261,131 @@ def rope_fused_3way_kernel(
         mask=mask_l[:, None] & mask_c2[None, :],
     )

+
+# -----------------------------------------------------------------------------
+# Triton kernel v2:
+#   - FlashAttention-style grid: (cdiv(HN, BLOCK_H), cdiv(seq_len, BLOCK_M), B)
+#   - Single 64-pair arange (power-of-2) with per-chunk masks for C0/C1/C2
+#   - Uses tl.split/tl.join interleaved rotation (like triton_rotary.py)
+# -----------------------------------------------------------------------------
+@triton.jit
+def rope_fused_3way_kernel_v2(
+    OUT_ptr,
+    X_ptr,
+    COS_F_ptr,
+    SIN_F_ptr,
+    COS_H_ptr,
+    SIN_H_ptr,
+    COS_W_ptr,
+    SIN_W_ptr,
+    # sizes / params
+    SEQ_LEN,  # f * gh * gw
+    HN,  # num heads (x.shape[2])
+    HW,  # gh * gw
+    W,  # gw
+    START_FRAME,
+    # out strides: [B, L, H, D]
+    stride_ob,
+    stride_ol,
+    stride_oh,
+    stride_od,
+    # x strides: [B, L, H, D]
+    stride_xb,
+    stride_xl,
+    stride_xh,
+    stride_xd,
+    # cos/sin strides (all 2D)
+    stride_cf_l,
+    stride_cf_c,
+    stride_ch_l,
+    stride_ch_c,
+    stride_cw_l,
+    stride_cw_c,
+    # meta
+    C0: tl.constexpr,
+    C1: tl.constexpr,
+    C2: tl.constexpr,
+    BLOCK_H: tl.constexpr,
+    BLOCK_M: tl.constexpr,
+    HEAD_DIM: tl.constexpr,  # == 128 for this wrapper
+):
+    pid_head = tl.program_id(axis=0)
+    pid_m = tl.program_id(axis=1)
+    pid_b = tl.program_id(axis=2)
+
+    # Early out like triton_rotary (saves work on last tile row)
+    if pid_m * BLOCK_M >= SEQ_LEN:
+        return
+
+    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
+    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+
+    mask_h = rh < HN
+    mask_m = rm < SEQ_LEN
+
+    # token -> (f_idx, h_idx, w_idx)
+    t = rm
+    f_local = t // HW
+    rem = t - f_local * HW
+    h_idx = rem // W
+    w_idx = rem - h_idx * W
+    f_idx = START_FRAME + f_local
+
+    # Clamp indices for masked tokens to keep pointer arithmetic in-bounds
+    f_idx = tl.where(mask_m, f_idx, 0)
+    h_idx = tl.where(mask_m, h_idx, 0)
+    w_idx = tl.where(mask_m, w_idx, 0)
+
+    # Pair indices (64 pairs for D=128) - single power-of-2 arange
+    rk_half = tl.arange(0, HEAD_DIM // 2)
+    # Per-chunk masks for semantic split
+    m0 = rk_half < C0
+    m1 = (rk_half >= C0) & (rk_half < (C0 + C1))
+    m2 = rk_half >= (C0 + C1)
+
+    # Safe per-axis column indices (avoid negative/out-of-range even when masked)
+    idx0 = tl.where(m0, rk_half, 0)
+    idx1 = tl.where(m1, rk_half - C0, 0)
+    idx2 = tl.where(m2, rk_half - (C0 + C1), 0)
+
+    # Load axis tables and combine (only one mask is true per pair)
+    cos0 = tl.load(
+        COS_F_ptr + f_idx[:, None] * stride_cf_l + idx0[None, :] * stride_cf_c,
+        mask=mask_m[:, None] & m0[None, :],
+        other=0.0,
+    ).to(tl.float32)
+    sin0 = tl.load(
+        SIN_F_ptr + f_idx[:, None] * stride_cf_l + idx0[None, :] * stride_cf_c,
+        mask=mask_m[:, None] & m0[None, :],
+        other=0.0,
+    ).to(tl.float32)
+
+    cos1 = tl.load(
+        COS_H_ptr + h_idx[:, None] * stride_ch_l + idx1[None, :] * stride_ch_c,
+        mask=mask_m[:, None] & m1[None, :],
+        other=0.0,
+    ).to(tl.float32)
+    sin1 = tl.load(
+        SIN_H_ptr + h_idx[:, None] * stride_ch_l + idx1[None, :] * stride_ch_c,
+        mask=mask_m[:, None] & m1[None, :],
+        other=0.0,
+    ).to(tl.float32)
+
+    cos2 = tl.load(
+        COS_W_ptr + w_idx[:, None] * stride_cw_l + idx2[None, :] * stride_cw_c,
+        mask=mask_m[:, None] & m2[None, :],
+        other=0.0,
+    ).to(tl.float32)
+    sin2 = tl.load(
+        SIN_W_ptr + w_idx[:, None] * stride_cw_l + idx2[None, :] * stride_cw_c,
+        mask=mask_m[:, None] & m2[None, :],
+        other=0.0,
+    ).to(tl.float32)
+
+    cos = cos0 + cos1 + cos2
+    sin = sin0 + sin1 + sin2
+
+    rk = tl.arange(0, HEAD_DIM)
+    X = (
+        X_ptr
+        + pid_b * stride_xb
+        + rh[:, None, None] * stride_xh
+        + rm[None, :, None] * stride_xl
+        + rk[None, None, :] * stride_xd
+    )
+    OUT = (
+        OUT_ptr
+        + pid_b * stride_ob
+        + rh[:, None, None] * stride_oh
+        + rm[None, :, None] * stride_ol
+        + rk[None, None, :] * stride_od
+    )
+
+    mask = mask_h[:, None, None] & mask_m[None, :, None]
+    x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
+    x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, HEAD_DIM // 2, 2]))
+
+    o0 = x0 * cos[None, :, :] - x1 * sin[None, :, :]
+    o1 = x0 * sin[None, :, :] + x1 * cos[None, :, :]
+    o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, HEAD_DIM])
+    tl.store(OUT, o, mask=mask)
+

 # -----------------------------------------------------------------------------
 # Python wrapper
 # -----------------------------------------------------------------------------
@@ -322,6 +458,20 @@ def _next_power_of_2(n: int) -> int:
     return 1 << (n - 1).bit_length()


+def _rope_fused_version_from_env() -> int:
+    """
+    Returns 1 or 2.
+    Default: 2 (v2 is intended to be the new fast path).
+    """
+    v = os.environ.get("SCOPE_TRITON_ROPE_FUSED_VERSION", "2").strip().lower()
+    if v in ("1", "v1"):
+        return 1
+    # treat anything else as v2
+    return 2
+
+
 def rope_fused_3way(
     x: torch.Tensor,
     grid_sizes: Union[torch.Tensor, Sequence[int]],
     freqs: torch.Tensor,
     *,
     start_frame: int = 0,
     inplace: Optional[bool] = None,
-    block_l: Optional[int] = None,
+    # v1 tuning (token tile along L)
+    block_l: Optional[int] = None,
+    # v2 tuning (FlashAttention-style tiles)
+    block_m: Optional[int] = None,
+    block_h: Optional[int] = None,
     num_warps: Optional[int] = None,
     num_stages: Optional[int] = None,
 ) -> torch.Tensor:
@@ -342,6 +492,13 @@ def rope_fused_3way(

     Tail handling (CRITICAL - avoids full tensor clone):
       - if seq_len < L and not inplace: out = empty_like(x), kernel writes prefix,
         then tiny tail copy: out[:, seq_len:] = x[:, seq_len:]
@@ -350,6 +507,15 @@ def rope_fused_3way(

     In-place policy (default):
       - if inplace is None: allow inplace only when (not torch.is_grad_enabled()) and (seq_len == L)
+
+    Kernel selection:
+      - default v2: BLOCK_M/BLOCK_H grid + single 64-pair arange with chunk masks
+      - set SCOPE_TRITON_ROPE_FUSED_VERSION=1 to force v1 (padded per-chunk arange)
+
+    v2 env tuning:
+      - SCOPE_TRITON_ROPE_FUSED_BLOCK_M (default 8)
+      - SCOPE_TRITON_ROPE_FUSED_BLOCK_H (default 2)
     """
     if x.device.type != "cuda":
         raise RuntimeError("rope_fused_3way requires CUDA tensor")
@@ -446,45 +612,110 @@ def rope_fused_3way(

     hw = int(gh) * int(gw)

-    # Compute power-of-2 padded sizes for Triton arange
-    c0_pad = _next_power_of_2(c0)
-    c1_pad = _next_power_of_2(c1)
-    c2_pad = _next_power_of_2(c2)
-
-    grid = (B * HN, triton.cdiv(seq_len, block_l))
-
-    # Mirror triton_rotary's device guard to avoid launching on cuda:0 accidentally.
-    with torch.cuda.device(x.device.index):
-        rope_fused_3way_kernel[grid](
-            x,
-            out,
-            cos_f,
-            sin_f,
-            cos_h,
-            sin_h,
-            cos_w,
-            sin_w,
-            x.stride(0),
-            x.stride(1),
-            x.stride(2),
-            x.stride(3),
-            out.stride(0),
-            out.stride(1),
-            out.stride(2),
-            out.stride(3),
-            cos_f.stride(0),
-            cos_f.stride(1),
-            cos_h.stride(0),
-            cos_h.stride(1),
-            cos_w.stride(0),
-            cos_w.stride(1),
-            L,
-            HN,
-            hw,
-            int(gw),
-            start_frame,
-            seq_len,
-            C0=c0,
-            C1=c1,
-            C2=c2,
-            C0_PAD=c0_pad,
-            C1_PAD=c1_pad,
-            C2_PAD=c2_pad,
-            BLOCK_L=block_l,
-            num_warps=num_warps,
-            num_stages=num_stages,
-        )
+    version = _rope_fused_version_from_env()
+
+    # Mirror triton_rotary's device guard to avoid launching on cuda:0 accidentally.
+    with torch.cuda.device(x.device.index):
+        if version == 1:
+            # --------------------
+            # v1 (legacy padded kernel)
+            # --------------------
+            if block_l is None:
+                block_l = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_L", "128"))
+
+            # Compute power-of-2 padded sizes for Triton arange (v1 behavior)
+            c0_pad = _next_power_of_2(c0)
+            c1_pad = _next_power_of_2(c1)
+            c2_pad = _next_power_of_2(c2)
+
+            grid = (B * HN, triton.cdiv(seq_len, block_l))
+            rope_fused_3way_kernel_v1[grid](
+                x,
+                out,
+                cos_f,
+                sin_f,
+                cos_h,
+                sin_h,
+                cos_w,
+                sin_w,
+                x.stride(0),
+                x.stride(1),
+                x.stride(2),
+                x.stride(3),
+                out.stride(0),
+                out.stride(1),
+                out.stride(2),
+                out.stride(3),
+                cos_f.stride(0),
+                cos_f.stride(1),
+                cos_h.stride(0),
+                cos_h.stride(1),
+                cos_w.stride(0),
+                cos_w.stride(1),
+                L,
+                HN,
+                hw,
+                int(gw),
+                start_frame,
+                seq_len,
+                C0=c0,
+                C1=c1,
+                C2=c2,
+                C0_PAD=c0_pad,
+                C1_PAD=c1_pad,
+                C2_PAD=c2_pad,
+                BLOCK_L=block_l,
+                num_warps=num_warps,
+                num_stages=num_stages,
+            )
+        else:
+            # --------------------
+            # v2 (new default): BLOCK_M/BLOCK_H grid + single 64-pair arange w/ chunk masks
+            # --------------------
+            if block_m is None:
+                block_m = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_M", "8"))
+            if block_h is None:
+                block_h = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_H", "2"))
+
+            # Sanity: this wrapper is D=128 only -> exactly 64 pairs
+            if (c0 + c1 + c2) != (D // 2):
+                raise AssertionError("Expected chunk sizes to sum to D//2")
+
+            grid = (triton.cdiv(HN, block_h), triton.cdiv(seq_len, block_m), B)
+            rope_fused_3way_kernel_v2[grid](
+                out,
+                x,
+                cos_f,
+                sin_f,
+                cos_h,
+                sin_h,
+                cos_w,
+                sin_w,
+                seq_len,
+                HN,
+                hw,
+                int(gw),
+                start_frame,
+                out.stride(0),
+                out.stride(1),
+                out.stride(2),
+                out.stride(3),
+                x.stride(0),
+                x.stride(1),
+                x.stride(2),
+                x.stride(3),
+                cos_f.stride(0),
+                cos_f.stride(1),
+                cos_h.stride(0),
+                cos_h.stride(1),
+                cos_w.stride(0),
+                cos_w.stride(1),
+                C0=c0,
+                C1=c1,
+                C2=c2,
+                BLOCK_H=block_h,
+                BLOCK_M=block_m,
+                HEAD_DIM=D,
+                num_warps=num_warps,
+                num_stages=num_stages,
+            )

     return out
```

---

## How to A/B v1 vs v2

### Default (v2)

No changes needed:

```bash
uv run python scripts/test_rope_correctness.py
uv run python scripts/bench_triton_rope.py
```

### Force v1

```bash
SCOPE_TRITON_ROPE_FUSED_VERSION=1 uv run python scripts/test_rope_correctness.py
SCOPE_TRITON_ROPE_FUSED_VERSION=1 uv run python scripts/bench_triton_rope.py
```

### Tune v2 tiling quickly

```bash
SCOPE_TRITON_ROPE_FUSED_VERSION=2 \
SCOPE_TRITON_ROPE_FUSED_BLOCK_M=8 \
SCOPE_TRITON_ROPE_FUSED_BLOCK_H=2 \
uv run python scripts/bench_triton_rope.py
```

---

## Notes on what changed (behavior-wise)

* **No semantic changes** to RoPE math or indexing:

  * `start_frame` offsets **time** only.
  * spatial axes start at 0.
* **No changes** to tail behavior:

  * still avoids full clone (`empty_like` + tiny tail copy) when `seq_len < L`.
* **Strict/fallback behavior remains in callers** (`rope_apply` / `causal_rope_apply`), unchanged.

If you want, I can also provide a tiny follow-up patch to `scripts/bench_triton_rope.py` to print the selected fused version and v2 block sizes at runtime (purely QoL, no functional change).
