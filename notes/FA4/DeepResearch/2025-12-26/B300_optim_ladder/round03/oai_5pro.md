## TorchAO FP8 quantization + `torch.compile`: the `aten.as_strided` hole (and what it implies)

### What we can say with high confidence (code-backed)

1. **Your failure mode is real and expected given current TorchAO FP8 quantization tensor-subclass coverage**
   The FP8 quantization workflow (`torchao.quantization.Float8Tensor`, produced by `quantize_` with `Float8DynamicActivationFloat8WeightConfig` / `Float8WeightOnlyConfig`) implements a bunch of view-ish ops (for example `aten.view`, `aten.transpose`, `aten.t`, `aten.detach`, etc.) but **does not implement `aten.as_strided.default`** in torchao versions like `v0.15.0`. ([GitHub][1])

So when Inductor (or decompositions) lowers a view sequence into `as_strided`, TorchAO’s dispatch falls through to the “unimplemented operator” error path, producing exactly what you saw.

2. **TorchAO already encodes “as_strided is only safe in the tensorwise-scale case” in its float8 training workflow**
   In TorchAO’s float8 training op desugaring, `aten.as_strided.default` is explicitly registered, but guarded by `_assert_tensorwise_scale(...)` alongside other view-like ops. That is a strong signal of intended semantics: tensorwise scaling makes some stride transforms safe, while more granular scaling can make them ill-defined. ([GitHub][2])

3. **There is no obvious public canonical TorchAO issue or PR that explicitly tracks “quantization Float8Tensor needs `aten.as_strided`”**
   I could not find a public, dedicated tracker that literally says “please implement `aten.as_strided` for `torchao.quantization.Float8Tensor`.” (This matches your observation.) The practical consequence is that you should treat this as an undocumented gap: you either work around it, or you open a focused upstream issue with a minimal repro.

4. **There is a clear precedent in TorchAO for implementing `as_strided` with strict guardrails**
   For example, NF4’s `as_strided` support is constrained to specific contiguous stride and storage-offset conditions, and errors otherwise. That pattern is exactly what you would mimic for FP8 quantization if you implement a “safe subset” of `as_strided`. ([PyTorch Documentation][3])

### Why it is probably omitted for FP8 quantization Float8Tensor (inference, not training)

You do not need hand-wavy speculation here because TorchAO’s own code already implies the design:

* Training float8 supports `as_strided` only under a tensorwise-scale assertion. ([GitHub][2])
* Quantization float8 supports both tensorwise and rowwise flows, and arbitrary `as_strided` can scramble the mapping from elements to scale blocks for non-tensorwise granularities. That makes correctness ambiguous unless you impose constraints.

So the most likely “real” boundary is not “as_strided is impossible,” but “as_strided must be either (a) forbidden for non-tensorwise scale or (b) implemented only for a safe subset.”

### What TorchAO officially recommends for tensor-subclass + compile compatibility

TorchAO’s quantization README contains two key operational rules:

* If you are on **PyTorch 2.6 or earlier**, call `unwrap_tensor_subclass` before `torch.export.export` and `aot_compile`.
* If you are on **PyTorch 2.4 or earlier**, you also need `unwrap_tensor_subclass` before `torch.compile`.
* Even on newer versions, `torch.compile` with Inductor freezing (`torch._inductor.config.freezing=True`) still needs the workaround until a linked PyTorch PR is fixed. ([GitHub][4])

It also documents that `quantize_` / `autoquant` automatically apply recommended Inductor settings, and that you can replicate them via `torchao.quantization.utils.recommended_inductor_config_setter()` (or disable with `set_inductor_config=False`). ([GitHub][4])

### Related subclass pitfalls worth linking in your ladder

These are not the `as_strided` bug, but they explain why “tensor subclass + compile” can break in multiple, independent ways:

* **TorchAO issue: internal tensor subclasses can trigger `torch.compile` errors due to aliasing of graph outputs** (this is a core limitation that can show up even when all ops are implemented). ([GitHub][5])
* **TorchAO issue: Float8Tensor is not meant to leak as a user-facing object** (this supports the “not everything is implemented” reality). ([GitHub][6])
* **Inference-mode + tensor subclass version-counter failures** are a known class of issues in PyTorch, with ongoing tracking (your notes mention PyTorch #170419 and TorchAO PR #3488; those are public). ([GitHub][7])

### Practical options for your B300 ladder (ordered by “drop-in today”)

#### Option A: Avoid subclass exposure at compile boundaries (lowest risk, might reduce benefit)

Use TorchAO’s documented `unwrap_tensor_subclass` pattern around the point where Inductor captures graphs. This is officially described as a workaround for several compile/export paths. ([GitHub][4])

Caveat: depending on how the FP8 workflow is implemented for your modules, unwrapping can reduce the “subclass-driven” behavior you want. Treat it as a stability lever.

#### Option B: Constrain your FP8 config to **PerTensor** and implement a “safe subset” `as_strided`

Given your use case explicitly uses `PerTensor()` scaling, you are in the “most plausible” zone for adding an `as_strided` implementation (analogous to the training path’s tensorwise-scale guard). ([GitHub][2])

A robust upstreamable approach is:

* Only allow `as_strided` when granularity is per-tensor (single scale).
* Only allow “layout-preserving” cases (for example contiguous stride patterns, same storage offset) similar to NF4Tensor’s restrictions. ([PyTorch Documentation][3])
* Otherwise raise a clear error.

This is both correctness-first and likely sufficient to prevent Inductor view-lowering from blowing up in common cases.

#### Option C: Make Inductor less likely to generate `as_strided` on the subclass

Sometimes you can reduce view-lowering by:

* inserting explicit `.contiguous()` or `.reshape()` at strategic boundaries, or
* cloning outputs you plan to reuse across steps.

This is workload-specific and not guaranteed, but it is often a small tactical fix when you cannot patch TorchAO immediately.

---

## `torch.compile` + CUDAGraph Trees: “output overwritten” and the correct mitigation knobs

### Correct step-marker spelling

Use the public API:
`torch.compiler.cudagraph_mark_step_begin()` ([PyTorch Documentation][8])

### Canonical explanation and mitigations

The PyTorch docs for CUDAGraph Trees are the best “source of truth” for the overwrite error and recommended mitigation patterns. ([PyTorch Documentation][9])

In ladder form, the mitigations usually boil down to:

1. **Mark step boundaries** with `torch.compiler.cudagraph_mark_step_begin()` every iteration (especially for generation loops, KV-cache loops, streaming pipelines). ([PyTorch Documentation][8])
2. **Do not hold onto graph-managed outputs across iterations** unless you materialize them (for example `out = out.clone()` before the next replay). ([PyTorch Documentation][9])
3. If you need a “compile mode escape hatch,” use a mode that disables cudagraphing, or explicitly disable cudagraphs via Inductor config. (See below.) ([PyTorch Documentation][10])

### The most reliable “do not typo” references for flags and modes

#### Compile modes are introspectable

Instead of hardcoding assumptions about what each mode toggles, PyTorch explicitly recommends querying mode settings via:

* `torch._inductor.list_mode_options()` ([PyTorch Documentation][10])

That gives you an API-correct escape hatch for ladder docs: “run this in your exact wheel to see the current mode-to-config mapping.”

#### Documented config path for disabling cudagraphs

PyTorch’s compiler FAQ explicitly references:

* `torch._inductor.config.triton.cudagraphs = False` ([PyTorch Documentation][11])

#### Environment variable spelling for cudagraph enablement

In Inductor’s Triton config, cudagraph enablement is keyed off:

* `TORCHINDUCTOR_CUDAGRAPHS` ([GitHub][12])

#### Additional cudagraph-trees knobs that show up in official docs

The CUDAGraph Trees documentation references flags like:

* `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True` ([PyTorch Documentation][13])

That is especially relevant for FA4 and KV-cache style workloads where shapes can vary unless you pad.

---

## Conv3d BF16/FP16 on Blackwell-class GPUs: “bad vs good cuDNN” with an actually source-backed statement

### The only fully source-backed “good vs bad” statement you can safely bake into ladder docs

PyTorch 2.9.1 release notes explicitly call out a **significant memory regression in `F.conv3d` with bfloat16 inputs in PyTorch 2.9.0**, and provide the workaround:

* If impacted: **install `nvidia-cudnn` package version `9.15+` from PyPI**. ([GitHub][14])

This is strong enough to justify a ladder rule like:

* “If you see Conv3d BF16 regressions on 2.9.0-era stacks, treat cuDNN < 9.15 as suspect and validate by upgrading to `nvidia-cudnn-cu12>=9.15`.”

### How to reconcile this with your local “cuDNN 9.13 is great on our VAE decode”

Both can be true:

* Your specific shape and convolution set can speed up massively on cuDNN 9.13 (local measurement).
* PyTorch upstream still recommends 9.15+ as a workaround for a known regression in the 2.9.0 timeframe. ([GitHub][14])

So for future-proofing the ladder, the clean framing is:

* “9.13 can be fast for our workload, but 9.15+ is the upstream workaround if you hit the 2.9.0 BF16 Conv3d regression.”

---

## Drop-in markdown you can paste into your ladder doc

```md
## TorchAO FP8 quantization + torch.compile: `aten.as_strided` gap

### Symptom
`NotImplementedError: Float8Tensor dispatch ... aten.as_strided.default ...`

### Root cause (code-backed)
`torchao.quantization.Float8Tensor` (produced by `quantize_` FP8 workflows) implements many view-ish ops but does not implement `aten.as_strided.default` in torchao v0.15.0, so Inductor view-lowering can trip it.

Contrast: TorchAO float8 *training* supports `aten.as_strided.default`, but gates it behind a tensorwise-scale assertion, implying `as_strided` is only considered safe under per-tensor (tensorwise) scale semantics.

### Official workaround patterns
TorchAO docs recommend `unwrap_tensor_subclass` for export/AOTI on PyTorch <= 2.6, and for `torch.compile` on PyTorch <= 2.4. They also note the workaround is required for `torch.compile` with Inductor freezing until a PyTorch fix lands.

`quantize_` and `autoquant` can also set recommended Inductor configs automatically; you can replicate via `recommended_inductor_config_setter()` or disable with `set_inductor_config=False`.

### Suggested B300-specific direction
Because our FP8 config uses PerTensor scaling, implementing a “safe subset” `as_strided` for `torchao.quantization.Float8Tensor` may be feasible (mirroring TorchAO’s tensorwise-scale gating and NF4’s strict stride/offset checks).
```

(Every paragraph above corresponds to upstream citations in this message: TorchAO quantization README, float8 training ops, NF4Tensor precedent, and PyTorch 2.9.1 release notes.)

---

If you want, I can also produce a minimal repro script template (two variants: eager graph then compile, and compile first) that isolates whether your `as_strided` originates from Inductor boundary stride-correction versus an upstream decomposition. That makes it much easier to file an actionable TorchAO issue or validate a local monkeypatch.

[1]: https://raw.githubusercontent.com/pytorch/ao/v0.15.0/torchao/quantization/quantize_/workflows/float8/float8_tensor.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/pytorch/ao/v0.15.0/torchao/float8/float8_ops.py "raw.githubusercontent.com"
[3]: https://docs.pytorch.org/ao/stable/_modules/torchao/dtypes/nf4tensor.html?utm_source=chatgpt.com "Source code for torchao.dtypes.nf4tensor"
[4]: https://raw.githubusercontent.com/pytorch/ao/main/torchao/quantization/README.md "raw.githubusercontent.com"
[5]: https://github.com/pytorch/ao/issues/1463?utm_source=chatgpt.com "The internal torchao tensor subclasses cause errors with ..."
[6]: https://github.com/pytorch/ao/issues/1418?utm_source=chatgpt.com "FLoat8 Autocast Issue #1418 - pytorch/ao"
[7]: https://github.com/pytorch/ao/issues/3488/linked_closing_reference?reference_location=REPO_ISSUES_INDEX&utm_source=chatgpt.com "Cannot set version_counter for inference tensor #3487"
[8]: https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html "torch.compiler.cudagraph_mark_step_begin — PyTorch 2.9 documentation"
[9]: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html "CUDAGraph Trees — PyTorch 2.9 documentation"
[10]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html?utm_source=chatgpt.com "torch.compile — PyTorch 2.9 documentation"
[11]: https://docs.pytorch.org/docs/stable/torch.compiler_faq.html?utm_source=chatgpt.com "Frequently Asked Questions — PyTorch 2.9 documentation"
[12]: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py?utm_source=chatgpt.com "pytorch/torch/_inductor/config.py at main"
[13]: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html?utm_source=chatgpt.com "CUDAGraph Trees"
[14]: https://github.com/pytorch/pytorch/releases/tag/v2.9.1 "Release PyTorch 2.9.1 Release, bug fix release · pytorch/pytorch · GitHub"
