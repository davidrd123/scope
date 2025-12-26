# GitHub Issue: pytorch/ao

**Status:** Ready to file (copy/paste)

**Context:** This issue blocks using `torch.compile` together with TorchAO FP8 quantization (`quantize_` → `torchao.quantization.Float8Tensor`) in our realtime video pipeline.

## Title

`torch.compile` fails with `torchao.quantization.Float8Tensor`: missing `aten.as_strided.default` dispatch

## Body

<!-- Copy from here -->

### Environment

- **torch:** `2.9.0+cu130`
- **torchao:** `0.14.1` and `0.15.0` (git tags `v0.14.1`, `v0.15.0`), both currently lack `aten.as_strided.default` in quantization Float8Tensor
- **GPU:** NVIDIA B300 (SM103). This looks like a tensor-subclass dispatch gap (likely GPU-agnostic), but we've only verified it on B300 so far.

### Repro

We hit this in a transformer-based diffusion pipeline.

We haven't yet found a tiny standalone repro; our smallest toy MLPs/stacked Linears didn't trigger it. Our current hypothesis is that `aten.as_strided` is being introduced by AOTAutograd aliasing/stride-correction logic during compilation (see **Where it might come from** below).

Pattern that triggers it:
```python
import torch
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerTensor

# Model must be complex enough to trigger the relevant compile/AOTAutograd paths
# (e.g., transformer attention blocks).
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
compiled_model = torch.compile(model)
compiled_model(x)  # fails
```

### Error signature (representative)

```
NotImplementedError: Float8Tensor dispatch: attempting to run unimplemented operator/function:
  func=<OpOverload(op='aten.as_strided', overload='default')>,
  types=(<class 'torchao.quantization.Float8Tensor'>,),
  arg_types=(<class 'torchao.quantization.Float8Tensor'>, <class 'list'>, <class 'list'>, <class 'int'>),
  kwarg_types={}
```

### Where it might come from (suspected)

This is **`torchao.quantization.Float8Tensor`** (the *quantization* workflow used by `quantize_`), not `torchao.float8.Float8TrainingTensor` (the *training* workflow).

In `torchao.quantization.Float8Tensor`, several view-like ops are registered (`aten.view`, `aten.transpose`, `aten.slice`, `aten.select`, …), but **`aten.as_strided.default` is not** (torchao `0.14.1` and `0.15.0`).

One place PyTorch can call `.as_strided(...)` internally during compilation is AOTAutograd's alias-from-base helper:

```python
# torch/_functorch/_aot_autograd/functional_utils.py (PyTorch v2.9.0)
aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
```

This suggests users may hit the dispatch gap without any explicit `.as_strided(...)` in their model code. It may also explain why a minimal standalone repro is non-trivial: you need a graph that triggers the relevant aliasing/stride-correction path on a Float8Tensor.

### Why this seems fixable for per-tensor scale

Our FP8 path uses `granularity=PerTensor()` (single scale). With a scalar scale, `as_strided` semantics seem relatively straightforward: views preserve the meaning of the scale factor.

The concern that `as_strided` semantics are ill-defined for float8 is strongest with per-row/per-block scaling, where a view can intermix elements that share a scale.

### Suggested fixes (any one would resolve this)

1. Add `aten.as_strided.default` support for `torchao.quantization.Float8Tensor` at least for per-tensor scale (optionally: guardrails / explicit error for non-preserving views under per-row/per-block scaling), or
2. Provide an officially supported `torch.compile` pattern for `Float8DynamicActivationFloat8WeightConfig` that avoids emitting `as_strided` on the subclass, or
3. Provide a first-class, documented compile boundary pattern (unwrap/re-wrap) so `torch.compile` can operate on plain tensors while still using float8 kernels internally.

Precedent in torchao:
- The training float8 tensor path already registers `aten.as_strided.default` and asserts tensorwise (per-tensor) scale via `_assert_tensorwise_scale(...)`.
- `NF4Tensor` registers `aten.as_strided.default` but restricts it to a narrow, “view-like” subset (contiguous strides, same storage_offset, etc.).

### Related but separate failure family (tensor subclass + inference_mode)

There is also a separate PyTorch core issue where doing view-like ops on a tensor subclass under `torch.inference_mode()` can throw:
- `RuntimeError: Cannot set version_counter for inference tensor`

Upstream pointers:
- https://github.com/pytorch/pytorch/issues/170419
- https://github.com/pytorch/ao/pull/3488

### Related evidence

- Quantization Float8Tensor lacks `aten.as_strided.default`:
  - https://github.com/pytorch/ao/blob/v0.14.1/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
  - https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
- Training float8 op table includes `aten.as_strided.default` (contrast case):
  - https://github.com/pytorch/ao/blob/v0.15.0/torchao/float8/float8_ops.py
- NF4Tensor implements `aten.as_strided.default` with strict constraints (precedent for a “safe subset” implementation):
  - https://docs.pytorch.org/ao/stable/_modules/torchao/dtypes/nf4tensor.html
- Related torchao issues:
  - https://github.com/pytorch/ao/issues/1463
  - https://github.com/pytorch/ao/issues/1418
- Compile can surface `aten.as_strided` for view-like ops (supporting example, DTensor):
  - https://github.com/pytorch/pytorch/issues/167074

### Current workaround

We're using BF16 (no quantization) as our baseline when `torch.compile` is enabled, and only enabling FP8 quantization without compile.
