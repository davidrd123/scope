# ComfyUI-WanVideoWrapper Techniques (LoRA custom_op + offload)

> **Source:** `/root/ComfyUI-WanVideoWrapper/`
> **Focus:** Extract techniques relevant to Scope (style swap + compile stability)
> **Related:** `notes/proposals/style-swap-mode.md`

---

## Executive Summary (What’s potentially transferable)

The key idea in `/root/ComfyUI-WanVideoWrapper/custom_linear.py` is not “faster LoRA math” so much as “make runtime LoRA application *traceable* and *device-movable*”:

- **Custom ops + `register_fake`**: wrap LoRA patching (and even `linear`) in `torch.library.custom_op(...)` with `register_fake` shape stubs so FakeTensor/`torch.compile` tracing can proceed without falling back to Python-level glue.
- **LoRA tensors as module buffers**: store LoRA diffs and per-step strengths as `register_buffer(...)` so they move with the module (especially important for their CPU↔GPU block swapping / offload story).
- **Explicit “unmerged LoRA” path**: when LoRAs are *not merged* into weights, they patch the model’s `nn.Linear` layers to apply LoRA in the forward pass.

What this may help with in Scope:

- If `torch.compile` is part of the plan for style swap, **moving LoRA strength from Python floats to tensors** (buffer/parameter) can avoid Dynamo specializing/guarding on Python values and recompiling on style changes.
- If we want to keep a compiled graph stable while changing scales, we likely want **tensor-backed knobs** (like this wrapper does), not dict-of-floats knobs.

What it probably does *not* do:

- It does **not remove the fundamental compute overhead** of runtime LoRA application. It can prevent compile thrash / graph breaks, but runtime LoRA still costs extra math versus permanently merged weights.

---

## 1) How their unmerged-LoRA path is wired

Where the knobs show up:

- `/root/ComfyUI-WanVideoWrapper/nodes_model_loading.py`: exposes `allow_unmerged_lora_compile` and feeds it into `compile_args`.
- `/root/ComfyUI-WanVideoWrapper/nodes_sampler.py`: calls `_replace_linear(...)` and then `set_lora_params(...)` when it detects “patched linear layers (unmerged loras, fp8 scaled)”.

One subtlety:

- In `custom_linear.py`, `allow_unmerged_lora_compile` is plumbed into `CustomLinear(allow_compile=...)`.
- **But the code behavior is inverted from the tooltip**: when `allow_compile=False` (default), `CustomLinear` uses the *custom-op* path; when `allow_compile=True`, it uses *direct* PyTorch ops. This is worth sanity-checking quickly in their environment; the flag name/tooltip may be stale.

---

## 2) The LoRA custom-op pattern (what it buys them)

### What they define

In `/root/ComfyUI-WanVideoWrapper/custom_linear.py` they define three ops:

- `wanvideo::apply_lora`:
  - reconstructs a full weight delta via `mm(lora_diff_0, lora_diff_1)` (LoRA up/down),
  - applies alpha normalization,
  - returns `weight + patch_diff * scale`.
- `wanvideo::apply_single_lora`:
  - applies a precomputed full delta: `weight + lora_diff * scale`.
- `wanvideo::linear_forward`:
  - wraps `torch.nn.functional.linear(...)`.

Each has a `@...register_fake` function so FakeTensor tracing can infer output metadata without executing the real kernel.

Minimal example of the pattern (they do the same for `apply_lora` + `linear_forward`):

```python
@torch.library.custom_op("wanvideo::apply_single_lora", mutates_args=())
def apply_single_lora(weight: torch.Tensor, lora_diff: torch.Tensor, lora_strength: torch.Tensor) -> torch.Tensor:
    return weight + lora_diff * lora_strength

@apply_single_lora.register_fake
def _(weight, lora_diff, lora_strength):
    return weight.clone()
```

Mechanically, this is a **weight-patching** formulation:

- compute an “effective weight” (`W_eff = W + ΔW * scale`)
- run `linear(input, W_eff)`

This differs from PEFT’s typical **activation-patching** formulation (`linear(x, W) + scale * (x @ A^T @ B^T)`).

### Why it likely helps with `torch.compile`

Two practical pain points with “unmerged LoRAs” in many codebases:

1. **Dynamic loading / patch plumbing**: “load tensor from RAM / disk” or Python-heavy patcher logic causes Dynamo graph breaks.
2. **Python-valued knobs**: if “scale” lives as a Python float in a dict, Dynamo can guard on it and either:
   - recompile when it changes, or
   - incorrectly keep the old value if it isn’t guarded (depending on capture mechanics).

The wrapper addresses both by:

- pushing LoRA weights (and strengths) into module buffers, and
- making the math look like stable `torch.ops.wanvideo.*` operator calls with FakeTensor support.

### Perf notes (important for “can this avoid the ~50% hit?”)

- `wanvideo::apply_lora` reconstructs a *full* `ΔW` via `mm(lora_up, lora_down)`. If that happens per-forward for many layers, it’s expensive.
- `wanvideo::apply_single_lora` applies a precomputed `ΔW`. If we ever adopt weight-patching, we likely want to **precompute** `ΔW` at load time or style-change time (not per-frame).

### Offload alignment (their stated motivation)

Their readme explicitly calls this out:

> “Now the LoRA weights are assigned as buffers to the corresponding modules, so they are part of the blocks and obey the block swapping … allowing LoRA weights to benefit from the prefetch feature for async offloading.”

That is a *memory-management* feature: if a block is swapped/offloaded, the LoRA weights ride along.

Also note `CustomLinear._prepare_weight()` uses `self.weight.to(input)` (or GGUF dequantize) at forward time; that’s consistent with an offload-centric design, but it is not something we’d want in Scope’s realtime path unless we explicitly accept the PCIe/NVLink tax.

---

## 3) Mapping to Scope (style swap + runtime_peft)

### Scope today

- Style-swap preload + merge-mode forcing: `src/scope/server/pipeline_manager.py` (the `STYLE_SWAP_MODE=1` branch forces `lora_merge_mode="runtime_peft"` and preloads style LoRAs at `scale: 0.0`).
- Runtime scaling: `src/scope/core/pipelines/wan2_1/lora/strategies/peft_lora.py`
  - updates `peft.tuners.lora.LoraLayer.scaling[adapter_name] = new_scale` (Python float values).

### What the ComfyUI approach suggests for us

1. **Prefer tensor-backed scales for compile stability**
   - Their LoRA strengths live in tensors (`register_buffer("_lora_strength_i", tensor)`), and step selection uses tensor ops.
   - In Scope, `runtime_peft` updates Python floats in `module.scaling[...]`.
   - If we want “compiled graph stays compiled across style changes”, we likely want “scale is a tensor” rather than “scale is a Python float”.

2. **Custom ops are a proven pattern for “avoid graph breaks” glue**
   - They use the same pattern beyond LoRA (e.g. `/root/ComfyUI-WanVideoWrapper/wanvideo/modules/attention.py` wraps `sageattn` in `torch.library.custom_op` “to avoid graph breaks with torch.compile”).
   - If we introduce any Python-y per-step LoRA scheduling or adapter-selection logic, boxing the math into a custom op with `register_fake` is a plausible technique.

3. **If the goal is “avoid ~50% FPS hit”, the bigger win is likely a different strategy**
   - `runtime_peft` overhead is largely “extra math per forward”.
   - A more direct way to keep near-`permanent_merge` FPS while still switching styles is a “runtime in-place merge” strategy:
     - on style change: update weights in-place (`W <- W_base + Σ scale_i * ΔW_i`),
     - between style changes: run the plain fused GEMMs (no LoRA layers in forward).
   - This is closer to “merge_loras=True” in ComfyUI, but triggered by style changes rather than startup-only.

---

## 4) Concrete follow-up experiments (low-risk, high signal)

1. **Does `torch.compile` recompile when we change `runtime_peft` scales?**
   - If yes: tensor-backed scales (buffer/parameter) become high priority.
   - If no: the ~50% hit is likely pure compute cost, and custom ops won’t fix it.

2. **Prototype “tensor scale” in our runtime strategy**
   - Implement a minimal LoRA layer variant where `scale` is a tensor read in forward and updated via `copy_()` (no Python dict of floats in the hot path).

3. **Prototype “runtime in-place merge” for style swap**
   - Keep a base-weight snapshot and apply/unapply deltas on style changes only.
   - Measure: style-switch latency (ms) vs steady-state FPS.

---

## Appendix: other techniques spotted (not the focus of this report)

### FP8 fast matmul (`torch._scaled_mm`)

See `/root/ComfyUI-WanVideoWrapper/fp8_optimization.py`. It uses `torch._scaled_mm` plus clamping and per-layer scale factors. Interesting, but it’s an internal API and needs quality validation.

### SageAttention Blackwell wrapper + custom ops

See `/root/ComfyUI-WanVideoWrapper/wanvideo/modules/attention.py`. They wrap `sageattn` / `sageattn_varlen` / ultravico variants in `torch.library.custom_op` with `register_fake`.

### TeaCache / MagCache / EasyCache (step skipping)

See `/root/ComfyUI-WanVideoWrapper/cache_methods/`. Likely risky for distilled/realtime setups where every denoise step matters.
