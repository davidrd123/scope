# GitHub Issue: pytorch/ao

**Status:** Ready to file (copy/paste)

**Context:** This issue blocks using `torch.compile` together with TorchAO FP8 quantization (`quantize_` → `torchao.quantization.Float8Tensor`) in our realtime video pipeline.

## Title

`torch.compile` fails with `torchao.quantization.Float8Tensor`: missing `aten.as_strided.default` dispatch

## Body

### Environment

- **GPU:** NVIDIA B300 (SM103), but the failure is a tensor-subclass dispatch gap and should repro on any CUDA GPU (and possibly CPU depending on codepath).
- **torch:** `2.9.0+cu130` (primary repro); we also have a separate baseline stack (`2.8.0+cu129`) used for A/B.
- **torchao:** observed on `v0.14.1` and `v0.15.0` (the quantization-path Float8Tensor implementation lacks `aten.as_strided` in both tags).

### Repro (repo script; canonical)

```bash
# This is the highest-signal repro we currently have (it runs our pipeline benchmark).
# It quantizes via Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
# and then attempts torch.compile.

SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 2 --skip 0 \
  --kv-cache-attention-bias 0.3 \
  --quantization fp8_e4m3fn \
  --compile
```

### Repro (minimal standalone)

We do not have a minimal standalone repro because the `as_strided` call comes from AOTAutograd's aliasing logic (`gen_alias_from_base`), not from model-level view ops. Simple models with `view`/`transpose`/`reshape` compile fine — you need a graph structure that triggers AOTAutograd's alias handling on a Float8Tensor output.

The pattern that triggers it:
- `quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))`
- `torch.compile(model)` on transformer blocks with attention
- Forward pass where AOTAutograd needs to produce aliased outputs

Representative code:
```python
import torch
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerTensor

# Simple models like this do NOT trigger the issue:
# model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))

# The issue requires complex graph patterns that hit AOTAutograd's gen_alias_from_base
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
compiled_model = torch.compile(model)
```

### Observed error

```
NotImplementedError: Float8Tensor dispatch: attempting to run unimplemented operator/function:
  func=<OpOverload(op='aten.as_strided', overload='default')>,
  types=(<class 'torchao.quantization.Float8Tensor'>,),
  arg_types=(<class 'torchao.quantization.Float8Tensor'>, <class 'list'>, <class 'list'>, <class 'int'>),
  kwarg_types={}
```

### Where it comes from

This is **`torchao.quantization.Float8Tensor`** (the *quantization* workflow used by `quantize_`), not `torchao.float8.Float8TrainingTensor` (the *training* workflow).

In `torchao.quantization.Float8Tensor`, several view-like ops are registered (`aten.view`, `aten.transpose`, `aten.slice`, `aten.select`, …), but **`aten.as_strided.default` is not** (v0.14.1 and v0.15.0).

**Crucially:** the `as_strided` call does not come from explicit view ops in model code. It comes from **AOTAutograd's aliasing logic** at graph boundaries:

```
torch/_functorch/_aot_autograd/functional_utils.py:304, in gen_alias_from_base
    aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
```

This means users can hit the dispatch gap without any `.as_strided(...)` in their model. It also means a minimal standalone repro is non-trivial — you need a graph that triggers AOTAutograd's aliasing/stride-correction path on a Float8Tensor.

### Why this seems fixable for per-tensor scale

Our FP8 path uses `granularity=PerTensor()` (single scale). With a scalar scale, `as_strided` semantics are relatively straightforward: any view preserves the meaning of the scale factor.

The "`as_strided` is ill-defined" concern is strongest for per-row/per-block scaling where a view can intermix elements that are supposed to share a scale.

### Concrete asks (any one unblocks us)

1. Add `aten.as_strided.default` support for `torchao.quantization.Float8Tensor` at least for per-tensor scale (optionally: guardrails / explicit error for non-preserving views under per-row/per-block scaling), or
2. Provide an officially supported `torch.compile` pattern for `Float8DynamicActivationFloat8WeightConfig` that avoids emitting `as_strided` on the subclass, or
3. Provide a first-class, documented compile boundary pattern (unwrap/re-wrap) so `torch.compile` can operate on plain tensors while still using float8 kernels internally.

### Related but separate failure family (tensor subclass + inference_mode)

There is also a separate PyTorch core issue where doing view-like ops on a tensor subclass under `torch.inference_mode()` can throw:
- `RuntimeError: Cannot set version_counter for inference tensor`

Upstream pointers:
- https://github.com/pytorch/pytorch/issues/170419
- https://github.com/pytorch/ao/pull/3488

### Related evidence

- Quantization Float8Tensor lacks `aten.as_strided`:
  - https://github.com/pytorch/ao/blob/v0.14.1/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
  - https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
- Training float8 op table includes `aten.as_strided.default` (contrast case):
  - https://github.com/pytorch/ao/blob/v0.15.0/torchao/float8/float8_ops.py
- Compile can surface `aten.as_strided` for view-like ops (supporting example, DTensor):
  - https://github.com/pytorch/pytorch/issues/167074

### Current workaround

We treat `--compile + fp8` as a blocked axis and benchmark with BF16 as the canonical baseline until float8+compile is stable.
