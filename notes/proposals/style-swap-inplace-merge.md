# Style Swap: In-Place Merge Strategy

> **Status:** Draft proposal
> **Date:** 2025-12-27
> **Builds on:** `notes/proposals/style-swap-mode.md`
> **Research:** `notes/research/comfyui-wrapper-techniques.md`

---

## Problem Statement

Current `STYLE_SWAP_MODE=1` uses `runtime_peft`, which applies LoRA deltas **every forward pass**:

```python
# Every frame, for every LoRA-enabled layer:
output = linear(x, W) + scale * (x @ A.T @ B.T)
```

This incurs ~50% FPS overhead regardless of compile stability.

---

## Proposed Solution: In-Place Weight Merge

Instead of applying LoRA every frame, merge weights **on style change only**:

```python
# On style change (once):
ΔW = (B @ A)  # A: (rank, in_features), B: (out_features, rank)
W.copy_(W_base + effective_scale * ΔW)  # effective_scale ~= user_scale * (alpha / rank)

# Every frame (fast path):
output = linear(x, W)  # Pure GEMM, no LoRA overhead
```

### Key Insight

The LoRA math `W_eff = W_base + scale * ΔW` only needs to run when:
1. The style changes, OR
2. The scale changes

Between style changes, run the fused weight at full speed.

---

## Performance Comparison

| Strategy | Per-frame cost | Style-switch cost | Steady-state FPS |
|----------|----------------|-------------------|------------------|
| `permanent_merge` | None | N/A (restart required) | 100% |
| `runtime_peft` | Extra matmul/layer | ~0ms | ~50% |
| **In-place merge** | None | **TBD (measure)**: likely 10s–100s of ms (depends on #layers, shapes, dtype, device) | ~100% |

The tradeoff: brief latency spike on style switch vs. constant overhead.

---

## Implementation Sketch

### 1. Store Base Weights

At pipeline load, snapshot the base weights before any LoRA merge:

```python
class InPlaceMergeStrategy(LoRAStrategy):
    def __init__(self, model):
        self.base_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and is_lora_target(name):
                # Clone base weight (before any LoRA)
                self.base_weights[name] = module.weight.data.clone()
```

### 2. Precompute LoRA Deltas

For each style LoRA, precompute the full delta `ΔW = B @ A`:

```python
def precompute_delta(self, adapter_name: str) -> dict[str, Tensor]:
    deltas = {}
    for name, module in self.model.named_modules():
        if has_lora(module, adapter_name):
            A = module.lora_A[adapter_name].weight  # (rank, in_features)
            B = module.lora_B[adapter_name].weight  # (out_features, rank)
            alpha = getattr(module, "lora_alpha", {}).get(adapter_name)  # optional
            rank = A.shape[0]
            scale = (alpha / rank) if alpha is not None else 1.0
            deltas[name] = ((B * scale) @ A).to(module.weight.dtype)  # (out, in)
    return deltas
```

Note: Scope’s LoRA utilities treat `lora_A` as the down/input matrix and `lora_B` as the up/output matrix (`src/scope/core/pipelines/wan2_1/lora/utils.py`).

### 3. Apply on Style Change

When style changes, rebuild weights in-place:

```python
def apply_style(self, style_name: str, scale: float = 1.0):
    delta = self.precomputed_deltas[style_name]

    for name, module in self.model.named_modules():
        if name in self.base_weights:
            # W = W_base + scale * ΔW
            module.weight.data.copy_(
                self.base_weights[name] + scale * delta.get(name, 0)
            )
```

### 4. No Per-Frame LoRA Logic

The forward pass is just normal `nn.Linear` - no hooks, no extra math.

---

## Compile Compatibility

This strategy is **inherently compile-friendly**:

- No dynamic weight patching in forward pass
- No Python-float scales in hot path
- Weights are stable tensors between style changes
- Dynamo guards are based on shapes/dtypes, so changing weight *values* should not inherently force recompiles

Caveat (needs early validation): TorchInductor may prepack/cache weights for some ops. If a compiled region uses cached packed weights derived from old values, **in-place mutation could appear “ignored”** until that cache is refreshed (or prepacking is disabled).

---

## Multi-LoRA Support

For blending multiple LoRAs:

```python
def apply_blend(self, styles: dict[str, float]):
    """Apply weighted blend of multiple styles.

    Args:
        styles: {"rat": 0.7, "tmnt": 0.3}
    """
    for name, module in self.model.named_modules():
        if name in self.base_weights:
            merged = self.base_weights[name].clone()
            for style_name, scale in styles.items():
                if style_name in self.precomputed_deltas:
                    merged += scale * self.precomputed_deltas[style_name].get(name, 0)
            module.weight.data.copy_(merged)
```

---

## Memory Overhead

| Storage | Size |
|---------|------|
| Base weight snapshots | Extra copy of targeted weights |
| Per-style deltas | `num_lora_layers × out × in × dtype` |

For a typical 14B model with LoRA on attention projections:
- ~100 layers × 4 projections × (4096 × 4096) × 2 bytes ≈ **13GB per style**

### On B300-class hardware

Even if VRAM is plentiful, the “store full ΔW per style” approach can still be expensive:
- Base snapshots + multiple full deltas can consume **tens to 100+ GB**.
- Precomputing deltas at pipeline load can add **significant startup time**.

Recommendation: start with **lazy delta computation** (compute `ΔW` on style switch), then consider caching a small number of recently used deltas if switching latency is too high.

### On Consumer Hardware (Future)

If we ever target 24GB GPUs, the mitigations become relevant:
1. **Lazy delta computation:** Compute `B @ A` on style switch, don't prestore
2. **Keep only active delta:** Evict deltas for inactive styles
3. **FP16 deltas:** Sufficient precision for additive merge

---

## Integration with Existing Code

### Changes to LoRAManager

Scope’s `LoRAManager` is currently a static dispatcher (`src/scope/core/pipelines/wan2_1/lora/manager.py`). The expected integration is:

- Add `"inplace_merge"` to `src/scope/server/schema.py` (`LoRAMergeMode`)
- Add `"inplace_merge"` to `src/scope/core/pipelines/wan2_1/lora/manager.py` (`_get_manager_class(...)` + load order)
- Implement a new strategy class under `src/scope/core/pipelines/wan2_1/lora/strategies/` (e.g. `inplace_merge_lora.py`)

### Changes to FrameProcessor

```python
def _rcp_set_style(self, style_name: str):
    # ... existing prompt recompile logic ...

    # No special casing needed if inplace_merge is implemented as a LoRA strategy:
    # FrameProcessor emits edge-triggered `lora_scales`, and the pipeline already
    # routes those updates through LoRAManager.
    self._emit_lora_scales(...)
```

### New Env Var

```bash
STYLE_SWAP_MODE=1
STYLE_SWAP_LORA_STRATEGY=inplace_merge  # proposed (avoid overloading STYLE_SWAP_MODE)
```

---

## Open Questions

1. **Async rebuild?** Should weight rebuild happen async to avoid frame drop on style switch? Risk: frames during rebuild use stale weights.

2. **Compile + caching interactions?** If Inductor prepacks weights, do we need a “refresh” mechanism or to disable prepacking for affected ops?

3. **PEFT compatibility?** Can we use PEFT's existing `merge_and_unload()` infrastructure, or do we need custom code?

4. **Gradient checkpointing?** If training is ever added, base weight storage needs to account for this.

5. **Quantization compatibility?** If the base model weights are FP8/quantized (e.g. torchao Float8Tensor), can we safely do in-place merges, or do we need to keep an FP16/BF16 weight copy and re-quantize on switches?

---

## Validation Plan

1. **Correctness:** Output of in-place merge should match `permanent_merge` output (same effective weights)

2. **FPS measurement:**
   - Steady-state FPS should match `permanent_merge` (~34 FPS on B300)
   - Style-switch latency target: TBD (measure; likely 10s–100s of ms)

3. **Compile stability:**
   - No recompilation after style switch
   - Verify with `TORCH_LOGS=recompiles`
   - Also verify *effectiveness*: outputs change after weight mutation under compiled execution (no stale packed weights)

4. **Memory:** Measure peak VRAM with N styles preloaded

---

## Comparison: Why Not Just Fix runtime_peft?

The ComfyUI research (`notes/research/comfyui-wrapper-techniques.md`) shows two paths:

| Approach | What it fixes | Steady-state overhead |
|----------|---------------|----------------------|
| Tensor-backed scales + custom_op | Compile graph breaks | Still ~50% (extra math) |
| In-place merge | Everything | ~0% |

Custom_ops are worth adopting for compile stability, but **they don't eliminate the fundamental compute cost** of runtime LoRA. In-place merge does.

---

## Related

- Current style swap: `notes/proposals/style-swap-mode.md`
- ComfyUI research: `notes/research/comfyui-wrapper-techniques.md`
- LoRA manager: `src/scope/core/pipelines/wan2_1/lora/manager.py`
- PEFT strategy: `src/scope/core/pipelines/wan2_1/lora/strategies/peft_lora.py`
