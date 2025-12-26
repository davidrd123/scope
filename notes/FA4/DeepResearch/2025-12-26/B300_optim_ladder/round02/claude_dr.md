# PyTorch 2.9 / Blackwell B300 Optimization: Technical Reference

This is a “forwardable” technical packet that focuses on three blockers we’ve hit in practice (B300/SM103 + PyTorch 2.9): `torch.compile` + cudagraphs footguns, TorchAO float8 tensor subclass gaps under `torch.compile`, and Conv3d stack sensitivity (cuDNN).

## Repo repro context (what we actually saw)

- Hardware: NVIDIA B300 (SM103 / compute capability 10.3)
- “Fast” runtime stack: `torch 2.9.0+cu130` (`cuda=13.0`, `cudnn=91300`) — VAE decode is ~4× faster than cu129 on our shape.
- Canonical bench scripts:
  - `scripts/profile_krea_pipeline_blocks.py` (end-to-end pipeline blocks)
  - `outputs/b300_cu129_vae_stream_decode_bench.log` vs `outputs/b300_cu130_vae_stream_decode_bench.log` (Conv3d-heavy VAE decode microbench)

The two concrete failure signatures we care about:
- FP8 + compile: `NotImplementedError: Float8Tensor dispatch ... aten.as_strided.default ...` (TorchAO tensor subclass missing op)
- `reduce-overhead` compile mode: `RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.` (CUDAGraph Trees aliasing/liveness)

---

## TorchAO float8 + `torch.compile`: the `aten.as_strided` hole (our blocker)

### What we mean by “Float8Tensor”

TorchAO has multiple float8 “tensor subclass” types; they are easy to conflate:
- `torchao.float8.Float8TrainingTensor` (float8 *training* workflow) — has its own op table (`FLOAT8_OPS_TABLE`) that *does* include `aten.as_strided.default`.
- `torchao.quantization.Float8Tensor` (float8 *quantization workflow*, used by `quantize_`) — this is the one we hit in this repo (see `src/scope/core/pipelines/krea_realtime_video/pipeline.py` where `quantize_` is called with `Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())`).

The observed failure is specifically the second one: **`torchao.quantization.Float8Tensor`**.

### What’s missing (as of torchao v0.14.1 and v0.15.0)

In torchao v0.14.1 and v0.15.0, `torchao/quantization/quantize_/workflows/float8/float8_tensor.py` registers a handful of view-y ops (`aten.view`, `aten.transpose`, `aten.slice`, `aten.select`, …), but **does not register `aten.as_strided`**.

Primary code pointers:
- v0.14.1: `torchao.quantization.Float8Tensor` implementation (no `aten.as_strided`): https://github.com/pytorch/ao/blob/v0.14.1/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
- v0.15.0: same: https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
- The dispatch error message comes from `TorchAOBaseTensor` (`_dispatch__torch_dispatch__`): https://github.com/pytorch/ao/blob/v0.15.0/torchao/utils.py
- Contrast: float8 *training* op table includes `aten.as_strided.default`: https://github.com/pytorch/ao/blob/v0.15.0/torchao/float8/float8_ops.py

Because `torchao.quantization.Float8Tensor` inherits `TorchAOBaseTensor`, missing ops fall through to `torchao/utils.py:_dispatch__torch_dispatch__`, which raises:

```python
raise NotImplementedError(
    f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: "
    f"{func=}, {types=}, {arg_types=}, {kwarg_types=}"
)
```

So the error signature we see is of the form:
- `NotImplementedError: Float8Tensor dispatch: attempting to run unimplemented operator/function: func=<OpOverload: aten.as_strided.default> ...`

### Why `as_strided` shows up even if you never call it

`aten.as_strided` is a low-level “view primitive” that can appear after:
- decompositions / lowering of higher-level view ops (`view`, `reshape`, etc.)
- Inductor stride-correction logic at graph boundaries (Inductor needs compiled results to match eager stride semantics)

So you can hit this without any explicit `.as_strided(...)` in user code.

Supporting example (not TorchAO-specific, but demonstrates the mechanism): DTensor + `torch.compile` can fail with `aten.as_strided.default` because view-like ops get lowered to `as_strided`.
- https://github.com/pytorch/pytorch/issues/167074

### Why it might be fixable (esp. for our exact config)

The “float8 scale metadata makes `as_strided` ill-defined” argument is strongest for **row-wise / block-wise scaling**, where reinterpreting strides can change which elements share a scale.

In our repo’s default FP8 config we use `granularity=PerTensor()` (scalar scale), which makes `as_strided` semantics much simpler:
- the scale is a scalar (or single-element tensor), so any view preserves scale meaning
- a conservative implementation could still restrict to common safe cases (contiguous, same storage offset, etc.)

### Minimal “ask” to unblock us

One of these would unblock `--compile + fp8_e4m3fn` for us:
1) Add `aten.as_strided.default` support to `torchao.quantization.Float8Tensor` (at least for per-tensor scale; possibly with guardrails for other granularities), or
2) Provide an officially supported pattern for `torch.compile` with `Float8DynamicActivationFloat8WeightConfig` that guarantees Inductor won’t emit `as_strided` on the tensor subclass, or
3) Provide a documented helper to “unwrap” the float8 tensor subclass at compile boundaries while still using float8 kernels internally.

### Paste-ready upstream issue payload (TorchAO / PyTorch compile stack)

If you want to file this upstream (or send it to maintainers), here’s a copy/paste starter that is consistent with what we’ve observed in this repo.

**Title**
- `torch.compile` fails with `torchao.quantization.Float8Tensor`: missing `aten.as_strided.default` dispatch

**Environment**
- GPU: NVIDIA B300 (SM103), but the failure is a tensor-subclass dispatch gap and should repro on any CUDA GPU (and possibly even CPU, depending on the codepath).
- `torch`: `2.9.0+cu130` (our “fast” stack); we also have a separate baseline stack (`2.8.0+cu129`) used for A/B.
- `torchao`: observed on `v0.14.1` and `v0.15.0` (the quantization-path Float8Tensor implementation lacks `aten.as_strided` in both tags).

**Repro (this repo; no minimal standalone yet)**
```bash
# Canonical reproduction harness in this repo:
# - quantizes via Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
# - then attempts torch.compile
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

**Observed error (signature)**
- `NotImplementedError: Float8Tensor dispatch: attempting to run unimplemented operator/function: func=<OpOverload: aten.as_strided.default> ...`

**Where it comes from**
- This is **`torchao.quantization.Float8Tensor`** (the *quantization* workflow used by `quantize_`), not `torchao.float8.Float8TrainingTensor` (the *training* workflow).
- In `torchao.quantization.Float8Tensor`, several view-like ops are registered, but **`aten.as_strided.default` is not** (v0.14.1 and v0.15.0).
- In the compile stack (Dynamo → AOTAutograd → Inductor), Inductor can lower higher-level view/layout ops into `aten.as_strided` and/or use `as_strided` in stride-correction logic at graph boundaries. So users can hit this even with no explicit `.as_strided(...)` in model code.

**Why this seems fixable for our exact config**
- Our repo’s FP8 path uses `granularity=PerTensor()` (single scale). With a scalar scale, `as_strided` semantics are relatively straightforward: any view preserves the meaning of the scale factor.
- The “`as_strided` is ill-defined” concern is strongest for per-row/per-block scaling where a view can intermix elements that are supposed to share a scale.

**Concrete “asks” (any one unblocks us)**
1) Add `aten.as_strided.default` support for `torchao.quantization.Float8Tensor` at least for per-tensor scale (optionally: guardrails / explicit error for non-preserving views under per-row/per-block scaling), or
2) Provide an officially supported `torch.compile` pattern for `Float8DynamicActivationFloat8WeightConfig` that avoids emitting `as_strided` on the subclass, or
3) Provide a first-class, documented compile boundary pattern (unwrap/re-wrap) so `torch.compile` can operate on plain tensors while still using float8 kernels internally.

**Related evidence**
- Quantization Float8Tensor lacks `aten.as_strided`:
  - https://github.com/pytorch/ao/blob/v0.14.1/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
  - https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py
- Training float8 op table includes `aten.as_strided.default` (contrast case):
  - https://github.com/pytorch/ao/blob/v0.15.0/torchao/float8/float8_ops.py
- Compile can surface `aten.as_strided` for view-like ops (supporting example, DTensor):
  - https://github.com/pytorch/pytorch/issues/167074

### Current workaround (what we’re doing today)

We treat `--compile + fp8_e4m3fn` as a blocked axis and benchmark with `--quantization none` (BF16) as the canonical baseline until float8+compile is stable.

### Additional “subclass + view” failure family worth tracking (not our `as_strided` error, but adjacent)

Separately from `aten.as_strided` coverage, there is a PyTorch core issue where doing view-like ops on a tensor subclass inside `torch.inference_mode()` can throw:
- `RuntimeError: Cannot set version_counter for inference tensor`

Upstream pointers:
- PyTorch issue (milestoned for 2.10.0): https://github.com/pytorch/pytorch/issues/170419
- TorchAO PR (draft, transpose-specific): https://github.com/pytorch/ao/pull/3488

If our pipeline ever ends up performing transpose/view ops on quantized tensor subclasses under `inference_mode` (or if compilation moves those ops), this could show up as an additional, independent blocker.

### Related gotcha: torchao C++ extensions often get skipped on torch nightly / cu130 wheels

In our B300/cu130 environment we have seen TorchAO print:

`Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu130 for torchao version 0.15.0 ...`

This suggests the environment is *not* loading TorchAO’s compiled extensions (fastpaths), which can affect performance and makes it harder to reason about “is FP8 actually active”.

Even if that warning is fixed, the `aten.as_strided` dispatch hole is still a separate blocker for `--compile + fp8` (it’s a Python tensor-subclass op coverage gap).

### Existing helper to try (may or may not fix `as_strided`, but it’s a real, documented knob)

TorchAO documents `unwrap_tensor_subclass(model)` as a workaround to make tensor-subclass models work with `torch.export.export` / `torch.aot_compile`:
- Helper: `torchao.utils.unwrap_tensor_subclass` (see source in torchao `utils.py`)
- Docs: https://docs.pytorch.org/ao/stable/quick_start.html (search for `unwrap_tensor_subclass`)
- Tracking issue: https://github.com/pytorch/ao/issues/345

This is relevant to our “minimal ask” options (2)/(3): there *is* an existing “unwrap” pattern in TorchAO, and it’s worth checking whether it also helps `torch.compile` with `Float8DynamicActivationFloat8WeightConfig` (or if it only helps export/aot paths).

TorchAO also documents more specific version guidance in its quantization README:
- PyTorch <= 2.6: call `unwrap_tensor_subclass` before `torch.export.export` / `aot_compile`
- PyTorch <= 2.4: also call it before `torch.compile`
- Still required for `torch.compile` with `torch._inductor.config.freezing=True` until https://github.com/pytorch/pytorch/pull/136265 is fixed
Source: https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/README.md

### TorchAO “recommended Inductor config” knob (good to know, not a fix for `as_strided`)

TorchAO’s quantization README states that `quantize_` / `autoquant` now automatically apply “recommended Inductor configuration settings”, and you can:
- replicate the settings with `torchao.quantization.utils.recommended_inductor_config_setter()`
- disable the auto-setting via `set_inductor_config=False` to `quantize_` / `autoquant`
Source: https://github.com/pytorch/ao/blob/v0.15.0/torchao/quantization/README.md

---

## cuDNN version matrix for BF16/FP16 Conv3d regression

Conv3d performance is extremely sensitive to cuDNN version / heuristics in recent stacks. The public reports include catastrophic slowdowns and memory regressions for BF16/FP16 Conv3d in some PyTorch+cuDNN combinations.

### What we measured on B300 (our VAE decode shape)

From our own logs:
- `torch 2.8.0+cu129` (`cudnn=91002`): `channels_first: 761.499 ms/call (t=3)`; `channels_last_3d: 757.137 ms/call (t=3)`
  - Source: `outputs/b300_cu129_vae_stream_decode_bench.log`
- `torch 2.9.0+cu130` (`cudnn=91300`): `channels_first: 194.879 ms/call (t=3)`; `channels_last_3d: 189.796 ms/call (t=3)`
  - Source: `outputs/b300_cu130_vae_stream_decode_bench.log`

So for our workload, moving to CUDA 13 / cuDNN 9.13 already delivered a ~3.9× speedup; `channels_last_3d` was a small additional win on this particular shape.

### The official “if you’re impacted” guidance (PyTorch 2.9.1)

PyTorch 2.9.1 release notes explicitly recommend upgrading cuDNN via the PyPI package if you hit the conv3d BF16 issue:

> “conv3d with bfloat16 Inputs in PyTorch 2.9.0 (#166643) This release provides work around this issue. If you are impacted please install nvidia-cudnn package version 9.15+ from pypi.”

Source: PyTorch `v2.9.1` release notes (GitHub release body) + linked issue/PRs.

### Affected GPUs

**H100 and H200** (SM90 Hopper) are the primary affected architectures per GitHub issues. **RTX 4090** (SM89) is also confirmed affected. 

**Blackwell** note: there is also a cuDNN thread reporting that some Blackwell devices only select `sm80` conv kernels (instead of an expected Blackwell-native kernel family) — likely a separate but related “stack maturity” issue: https://forums.developer.nvidia.com/t/cudnn-only-selects-sm80-kernels-on-blackwell-devices/338923

### Fix command (when you need it)

```bash
pip install nvidia-cudnn-cu12>=9.15
```

### Quick “bad vs good” summary (source-backed)

| Category | cuDNN version(s) | Source |
|----------|-------------------|--------|
| **Bad / implicated** | `9.10.2` (incl. `nvidia-cudnn-cu12==9.10.2.21`) | PyTorch issues #168167, #166790, #166122 |
| **Recommended workaround** | `9.15+` (via `nvidia-cudnn-cu12>=9.15`) | PyTorch v2.9.1 release notes + PyTorch issue #166643 maintainer comment |
| **Local B300 VAE decode datapoint** | `9.10.2` → `9.13.0` improved ~`4×` on our shape | `outputs/b300_cu129_vae_stream_decode_bench.log` vs `outputs/b300_cu130_vae_stream_decode_bench.log` |

### Key source issues

- [PyTorch #168167](https://github.com/pytorch/pytorch/issues/168167) — 16,000x performance regression
- [PyTorch #166643](https://github.com/pytorch/pytorch/issues/166643) — 7x memory regression
- [PyTorch #166122](https://github.com/pytorch/pytorch/issues/166122) — 4x slowdown with AMP
- [PyTorch #166790](https://github.com/pytorch/pytorch/issues/166790) — OOM on Conv3D

### (Future) TorchAO prototype MXFP/NVFP on Blackwell

TorchAO’s v0.12.0 release notes mention prototype support for NVFP4 and microscaling (MX) formats on Blackwell GPUs, claiming “up to 61% end-to-end performance improvement in vLLM on Qwen3 models” (and “near 2x” for diffusion workloads). This is a separate path from TorchAO’s `Float8DynamicActivationFloat8WeightConfig` and is not validated in this repo yet.
- Release notes: https://github.com/pytorch/ao/releases/tag/v0.12.0

---

## Verified API spellings and exact paths

### CUDAGraph step-marker function

**Correct path:** `torch.compiler.cudagraph_mark_step_begin()`

This is the public API exported in `torch/compiler/__init__.py`. The implementation lives in `torch/_inductor/cudagraph_trees.py`.

```python
def cudagraph_mark_step_begin():
    """
    Indicates that a new iteration of inference or training is about to begin.
    
    CUDA Graphs will free tensors of a prior iteration. A new iteration is started 
    on each invocation of torch.compile, so long as there is not a pending backward 
    that has not been called.
    """
```

**Source:** [torch/compiler/__init__.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/compiler/__init__.py)

**Official docs:** https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html

Note: the underlying implementation called by this API is `torch._inductor.cudagraph_trees.mark_step_begin()`. For docs/examples, prefer the public `torch.compiler.cudagraph_mark_step_begin()`.

### Inductor config flags for CUDAGraph

| Config Flag | Exact Path | Description |
|-------------|-----------|-------------|
| Enable cudagraphs | `torch._inductor.config.triton.cudagraphs` | Master switch for CUDAGraphs in Inductor |
| Input mutation support | `torch._inductor.config.triton.cudagraph_support_input_mutation` | Allows mutating inputs from prior CUDAGraph outputs |
| Skip dynamic graphs | `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs` | Only capture static-shape functions |
| Rerecord limit | `torch._inductor.config.triton.cudagraph_unexpected_rerecord_limit` | Max re-records before fallback |

**Source:** [torch/_inductor/config.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/config.py)

Note: internal config spellings can move between point releases. If you want to avoid baking internal names into scripts/docs, you can introspect the exact build you’re running:

```python
import torch._inductor.config as cfg

print([k for k in dir(cfg.triton) if "cudagraph" in k])
```

The input mutation flag declaration from `cudagraph_trees.py`:

```python
self.rerecord_if_static_inputs_change = (
    torch._dynamo.config.inline_inbuilt_nn_modules
    or torch._inductor.config.triton.cudagraph_support_input_mutation
)
```

**Environment variable override:** `TORCHINDUCTOR_CUDAGRAPHS=1` enables the master switch.

### Correct usage example

```python
import torch

@torch.compile(mode="reduce-overhead")
def foo(x):
    return x + 1

# Enable input mutation support
torch._inductor.config.triton.cudagraph_support_input_mutation = True

for i in range(3):
    torch.compiler.cudagraph_mark_step_begin()  # Correct API
    inp = torch.rand([4], device="cuda")
    result = foo(inp)
```

---

## Conclusion

For Float8 quantization workflows, `torch.compile` can currently trip over missing `aten.as_strided` support in `torchao.quantization.Float8Tensor`. For Conv3d-heavy pipelines, cuDNN version is a first-order lever (we saw ~4× decode speedup moving from cu129/cudnn=91002 to cu130/cudnn=91300 on B300). For CUDAGraph integration, use the public `torch.compiler.cudagraph_mark_step_begin()` API and the `torch._inductor.config.triton.*` namespace for cudagraph settings.
