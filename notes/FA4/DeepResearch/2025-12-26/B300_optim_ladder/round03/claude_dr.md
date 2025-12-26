# TorchAO Float8Tensor gaps for torch.compile on PyTorch 2.9

TorchAO's Float8 quantization path has **no explicit tracking for `aten.as_strided` support**, requires `torch.compile` for competitive performance (2-3x faster than eager), and faces fundamental blockers around `torch.inference_mode()` due to PyTorch's version counter architecture. For Blackwell B300 optimization, TorchAO v0.12.0+ includes prototype NVFP4/MXFP support with up to **61% e2e speedup** in vLLM on Qwen3 models.

## Quick corrections (verified)

- PyTorch issue `pytorch/pytorch#170419` (“Cannot set version_counter for inference tensor”) exists and is public (milestone: 2.10.0).
- TorchAO PR `pytorch/ao#3488` (“[float8] fix the transpose error”) exists and is public (currently open).
- PyTorch issue `pytorch/pytorch#134798` is **closed** (not open).

---

## No open issue tracks as_strided for quantization Float8Tensor

Extensive searching of pytorch/ao GitHub reveals **no open issues or PRs explicitly requesting `aten.as_strided.default` support** for the quantization-path Float8Tensor. This appears to be an undocumented gap rather than an actively tracked feature.

TorchAO maintains **two distinct Float8Tensor implementations**:
- **Training path**: `torchao.float8.Float8TrainingTensor` / float8 training workflow (has its own op table; `aten.as_strided.default` is registered in `torchao/float8/float8_ops.py`)
- **Quantization path**: `torchao.quantization.Float8Tensor` (used by `quantize_` via `Float8DynamicActivationFloat8WeightConfig` / `Float8WeightOnlyConfig`)

The training path's `float8_ops.py` implements various aten ops, but the quantization path's Float8Tensor has a more limited op set. The design rationale is **inferred from code patterns**: per-row/block scaling makes `as_strided` semantically ill-defined because arbitrary views can intermix rows and break scale alignment.

The `NF4Tensor` in `torchao/dtypes/nf4tensor.py` demonstrates the constraints required for `as_strided` on quantized tensors:
```python
f"aten.as_strided(NF4Tensor) only support continuous stride={make_contiguous_strides_for(size)} but got stride={stride}"
f"aten.as_strided(NF4Tensor) only support original storage offset {nf4tensor.storage_offset()} but got {storage_offset}"
```

| Scaling Granularity | as_strided Feasibility | Rationale |
|---------------------|------------------------|-----------|
| Per-tensor | Safe | Single scale applies regardless of view |
| Per-row | Ill-defined | Striding can intermix rows, breaking scale alignment |
| Per-block | Ill-defined | Arbitrary views don't preserve block boundaries |

**Relevant resources**: RFC Float8 Inference ([Issue #574](https://github.com/pytorch/ao/issues/574)), NF4Tensor implementation ([nf4tensor.py](https://github.com/pytorch/ao/blob/main/torchao/dtypes/nf4tensor.py))

---

## unwrap_tensor_subclass works with torch.compile conditionally

The helper `torchao.utils.unwrap_tensor_subclass(model)` **works with regular `torch.compile`**, but whether you need it depends on your PyTorch version. From the TorchAO quantization README:

> "If you are using pytorch 2.6 or before, you need to call unwrap_tensor_subclass before torch.export.export and aot_compile... **If you are using pytorch 2.4 or before, you'll also need unwrap_tensor_subclass before calling torch.compile as well.** Note that the workaround is also required for **torch.compile with freezing** (`torch._inductor.config.freezing=True`) until pytorch/pytorch#136265 is fixed."

| PyTorch Version | torch.compile | torch.compile + freezing | torch.export |
|-----------------|---------------|--------------------------|--------------|
| **2.7+** | ✅ Works natively | ⚠️ Needs unwrap | ✅ Works natively |
| **2.5-2.6** | ✅ Works natively | ⚠️ Needs unwrap | ⚠️ Needs unwrap |
| **2.4 and below** | ⚠️ Needs unwrap | ⚠️ Needs unwrap | ⚠️ Needs unwrap |

Usage pattern for older PyTorch:
```python
from torchao.utils import unwrap_tensor_subclass

# Required for PyTorch 2.4 or before with torch.compile
model = unwrap_tensor_subclass(model)
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

**Key resource**: [TorchAO Contributor Guide (Issue #391)](https://github.com/pytorch/ao/issues/391), Quantization README
Quantization README pointer: https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/README.md

---

## inference_mode + tensor subclass version_counter blocker status

Relevant tracked issues include:

- **PyTorch Issue #170419**: "Cannot set version_counter for inference tensor" — **OPEN** (milestoned for 2.10.0)
- **PyTorch Issue #112024**: "torch.inference_mode and tensor subclass: RuntimeError: Cannot set version_counter for inference tensor" — **CLOSED**
- **TorchAO PR #3488**: "[float8] fix the transpose error" — **OPEN** (draft/ongoing)

The issue was opened by @ezyang (Edward Z. Yang) with this analysis:

> "The proximal cause of the problem is that when we construct T inside the torch dispatch function, it is created as an inference mode tensor, even though the semantics of detach() on a non-inference mode tensor when inference mode is enabled is to create a non-inference tensor. Not entirely sure what the correct fix is."

**Related issues** (status varies; verify before relying on them):
- [Issue #134798](https://github.com/pytorch/pytorch/issues/134798): "Cannot set version_counter for inference tensor when get_data_attr is called" (**CLOSED**, milestone: 2.5.0)
- [Issue #124111](https://github.com/pytorch/pytorch/issues/124111): "torch.compile-ing triton kernels with inputs created under inference mode" (**CLOSED**)
- [Issue #134249](https://github.com/pytorch/pytorch/issues/134249): "Footgun: Dynamo x tensors created inside or outside of inference_mode" (**OPEN**)

**Workarounds documented**:

1. **Wrap tensor construction in `inference_mode(False)`**:
```python
@classmethod
def __torch_dispatch__(cls, func, types, args, kwargs=None):
    if func is torch.ops.aten.detach.default:
        (self,) = args
        with torch.inference_mode(False):  # WORKAROUND
            return cls(torch.ops.aten.detach.default(self.elem))
```

2. **Use `torch.no_grad()` instead of `torch.inference_mode()`** — less restrictive, still tracks version counters

3. **Clone tensors before operations** — per PyTorch docs: "make a clone outside InferenceMode to get a normal tensor before mutating"

---

## Float8 + torch.compile best practices from maintainers

TorchAO maintainers explicitly require torch.compile for Float8 performance. From the Float8 Inference RFC ([Issue #574](https://github.com/pytorch/ao/issues/574)):

> "As with the rest of this project, we heavily rely on the compile stack to generate efficient and fused casting code. We do actually see some performance gains on heavily compute-bound models, but in general, **we require torch.compile for competitive performance.**"

Performance without compile shows **2-3x degradation**:

| Configuration | FP8 (compile=False) | FP8 (compile=True) |
|--------------|---------------------|-------------------|
| bs=32, seq=512, dim=1024 | 7.17ms | **2.61ms** |
| bs=32, seq=512, dim=4096 | 49.25ms | **26.20ms** |

**Recommended pattern** from documentation:
```python
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow

# Use PerRow for better accuracy (bfloat16 only)
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))

# torch.compile is REQUIRED for competitive performance
model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

**Known limitations**:
- Hardware requires **CUDA compute capability 8.9+** (H100, L4, A6000 ADA)
- PerRow scaling **only supports bfloat16** weight and activation
- Float8 benefits increase with larger GEMM dimensions (M, K, N)
- `torch.export` + AOTI not supported with public APIs
- **Freezing requires unwrap**: `torch._inductor.config.freezing=True` needs `unwrap_tensor_subclass` until [pytorch/pytorch#136265](https://github.com/pytorch/pytorch/issues/136265) is fixed

**Inductor configuration**: The `quantize_` API automatically sets recommended inductor configs. For manual control:
```python
from torchao.quantization.utils import recommended_inductor_config_setter
recommended_inductor_config_setter()  # Call before torch.compile
```

**Blackwell B300 support**: TorchAO v0.12.0+ includes prototype NVFP4 (NVIDIA 4-bit FP) and MXFP8 support, achieving **1.2x speedup vs bf16** on LLaMa 3 8B pretraining on B200, and up to **61% e2e improvement** in vLLM on Qwen3 models.

---

## Summary of authoritative sources

| Item | Key Finding | Source |
|------|-------------|--------|
| as_strided gap | No open issue; gap exists between training/quantization paths | Code inspection, [Issue #574](https://github.com/pytorch/ao/issues/574) |
| unwrap_tensor_subclass | Works with compile for PyTorch 2.5+; required for 2.4- | [Issue #391](https://github.com/pytorch/ao/issues/391), Quantization README |
| inference_mode blocker | Ongoing tracker + related issues | [Issue #170419](https://github.com/pytorch/pytorch/issues/170419), [PR #3488](https://github.com/pytorch/ao/pull/3488), [Issue #112024](https://github.com/pytorch/pytorch/issues/112024) |
| Float8 + compile | torch.compile REQUIRED for performance (2-3x faster) | [Issue #574](https://github.com/pytorch/ao/issues/574), [Issue #685](https://github.com/pytorch/ao/issues/685) |
| Blackwell support | v0.12.0+ has NVFP4/MXFP8 prototype with 61% vLLM speedup | [Releases](https://github.com/pytorch/ao/releases) |

Correction: **PyTorch #170419** and **TorchAO PR #3488** are public and linked above; this draft’s earlier “not found” note was incorrect.
