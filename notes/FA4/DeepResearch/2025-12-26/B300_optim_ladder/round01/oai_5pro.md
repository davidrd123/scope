## PyTorch 2.9 `torch.compile` modes (Inductor) and the “CUDAGraphs output overwritten” error

### What the built-in modes actually do (PyTorch 2.9 docs)

PyTorch documents these `mode=` strings for `torch.compile`:

* **`"default"`**: a balance between compile overhead and runtime performance
* **`"reduce-overhead"`**: reduces Python overhead (often by using CUDA graphs when possible)
* **`"max-autotune"`**: enables autotuning (for example, Triton matmul choices) and can enable CUDA graphs
* **`"max-autotune-no-cudagraphs"`**: like `max-autotune` but **explicitly avoids CUDA graphs**
  PyTorch also points out you can inspect current mode defaults via `torch._inductor.list_mode_options()`. ([PyTorch Documentation][1])

**Practical mapping for your ladder on B300 (SM103):**

* If you are chasing top throughput and are OK dealing with graph-capture constraints: start at `max-autotune`.
* If you hit CUDA graph hazards (like the overwritten-output issue), move to `max-autotune-no-cudagraphs` first (fastest “no graph” fallback).
* Use `reduce-overhead` when kernel time is already good but Python overhead is showing up in profiles.

### The “CUDAGraphs output overwritten” error: what it means

This comes up when CUDA graphs are in play and the runtime detects that an output buffer from a prior replay got reused or overwritten in a way that breaks correctness expectations.

The PyTorch forums thread with the exact error message (and the “recommended fixes” text) is here: ([PyTorch Forums][2])

PyTorch’s “CUDA Graph Trees” documentation explains the underlying constraint: CUDA graphs freeze memory addresses, so if you keep outputs alive across iterations (or alias outputs in tricky ways), you can get overwritten outputs unless you delineate steps or clone outputs. ([PyTorch Documentation][3])

### Recommended fixes (in priority order)

#### Fix A (most direct): stop using CUDA graphs for that region

Use the compile mode that disables them:

```python
compiled = torch.compile(fn, mode="max-autotune-no-cudagraphs")
```

That mode exists specifically to keep autotuning benefits while avoiding CUDA-graph constraints. ([PyTorch Documentation][1])

#### Fix B: mark step boundaries (when using CUDA graphs in a loop)

If you are doing iterative decoding or a loop where outputs persist across iterations, call the step marker before each invocation:

```python
for t in range(T):
    torch.compiler.cudagraph_mark_step_begin()  # step boundary for cudagraph trees
    y = compiled(x_t)
```

PyTorch’s CUDA Graph Trees doc uses the name `torch.compiler.mark_step_begin()` in a paragraph, but the exported symbol in `torch.compiler` is `cudagraph_mark_step_begin()` (see `torch/compiler/__init__.py`). Prefer `torch.compiler.cudagraph_mark_step_begin()` in code. ([PyTorch Documentation][3])

#### Fix C: clone outputs outside `torch.compile` before storing them

If you store outputs (for example, logits, hidden states, cached intermediates), clone them outside the compiled function so the stored tensor is not tied to a reused graph output buffer:

```python
y = compiled(x)
y_saved = y.clone()
```

This “clone outside” workaround is explicitly called out as a fix path in both the forum error context and the CUDA graph trees guidance. ([PyTorch Forums][2])

#### Fix D: if you only need compilation for part of the pipeline, isolate it

Sometimes the overwritten-output issue happens because the compiled region returns outputs that later get mutated, views taken, or cached. Returning a “fresh” tensor (clone) at the boundary or keeping the cached tensors outside the captured region often resolves it.

---

## TorchAO Float8 + PyTorch 2.9 status (and the “tensor subclass dispatch / `aten.as_strided` unimplemented” class of failures)

### First: version compatibility you should treat as “known good”

TorchAO maintains an explicit compatibility matrix:

* **torchao 0.14.1** is built for **torch 2.9.0** and its Python API supports torch **2.9.0 / 2.8.0 / 2.7.1**
* **torchao 0.15.0dev (nightly)** is built for **torch 2.10.0dev**, and its Python API supports **2.10 dev, 2.9.0, 2.8.0**
  This is the cleanest anchor when you want “torch 2.9.0+cu130 on Blackwell” and minimal surprise from binary mismatches. ([GitHub][4])

If you rely on `fbgemm_gpu` kernels (common for float8 paths), the same matrix warns that `fbgemm_gpu` versions are typically tied to a single torch version, so keep those aligned. ([GitHub][4])

### How Float8Tensor is used (relevant to compile + dispatch issues)

TorchAO’s docs describe float8 dynamic activation + weight quantization roughly like this:

* weights become a **`Float8Tensor`** (a tensor subclass that carries `qdata` and `scale`)
* during execution, `F.linear` dispatch goes through the subclass implementation and calls an optimized kernel (example shown uses an `fbgemm` float8 kernel and reshaping `qdata`) ([PyTorch Documentation][5])

This is important because tensor subclasses are still a sharp edge for `torch.compile` in a few scenarios.

### The failure mode you’re seeing: “tensor subclass dispatch” and unimplemented ops

When you see errors like:

* `NotImplementedError: <TensorSubclass> dispatch: attempting to run unimplemented operator/function: aten.<something>`

that is the subclass saying “I do not implement this op in `__torch_dispatch__` (or equivalent)”.

TorchAO has multiple public issues of this general form (example: `aten.permute.default` not implemented for a quantized tensor subclass). ([GitHub][6])
There are also float8-specific rough edges reported (example: a float8 linear path failing in `torch.inference_mode` due to an unimplemented reshape). ([GitHub][7])

Your specific report mentions `aten.as_strided` being unimplemented under compile. I did not find a public TorchAO issue titled exactly “Float8Tensor dispatch: aten.as_strided unimplemented” in the sources I pulled, but this fits the same pattern: `as_strided` is a low-level view primitive that shows up when compilers or decompositions rewrite view/reshape/indexing. If the subclass does not implement it, compilation (or runtime inside the compiled graph) can fail.

### A second, separate TorchAO + `torch.compile` pain point: subclass graph-output aliasing

There is an open TorchAO issue where `torch.compile` fails when tensor subclass inputs are involved and graph outputs alias each other. The thrown error text is explicit that this is unsupported “in the subclass use case.” ([GitHub][8])

That issue is not float8-only, but it matters because float8 quant workflows also introduce subclasses.

### Workarounds that are most useful for inference on 2.9

#### Workaround 1: ensure tensor subclasses are not graph inputs to the compiled region

If your compiled function takes (directly or indirectly) a `Float8Tensor` as an *input* (not just as a module weight used inside), you are much more likely to hit missing-op dispatch.

A strong pattern is:

* keep the compiled region’s signature “plain tensors in, plain tensors out”
* keep subclass weights inside modules where the subclass implementation handles the ops it intends to support

#### Workaround 2: use `unwrap_tensor_subclass` at the compile boundary (or adjust compile order)

TorchAO’s own quick start includes `unwrap_tensor_subclass` in its workflow and shows `torch.compile(..., mode="max-autotune")` as the intended way to speed up inference with quantized models. ([PyTorch Documentation][9])

This gives you two practical orderings to try (agents can test both quickly):

1. **Compile then quantize**: compile the BF16 model, then apply `quantize_` to introduce float8 weights
2. **Quantize then unwrap then compile**: quantize, then unwrap (to remove subclass objects crossing boundaries), then compile

Which works best can depend on whether your compiled region “sees” the subclass tensors as graph inputs/outputs.

#### Workaround 3: if you must ship now, isolate float8 and compile separate blocks

Given you are compiling attention blocks (Inductor/AOTAutograd) and not doing TensorRT, you can often get most value by:

* compiling the attention and MLP compute blocks that stay in BF16
* keeping the float8 quantized layers outside compile until the subclass coverage is stable for your exact model graph

#### Workaround 4 (diagnostic): prove it is a subclass dispatch hole

* Temporarily remove compile and run eager to confirm functionality.
* Then re-enable compile with `mode="max-autotune-no-cudagraphs"` just to reduce variables (no CUDA graphs).
* If it still fails and the stack shows `Float8Tensor dispatch: ... aten.as_strided`, you have a missing op mapping, not a cudagraph hazard.

#### What versions “fix it”?

The only hard, source-backed statement I can make from the compatibility matrix is: for PyTorch **2.9.0**, TorchAO’s “matched” stable release is **0.14.1**. ([GitHub][4])
For a specific missing-op dispatch (like `aten.as_strided`), fixes typically land in TorchAO (or sometimes PyTorch compiler lowering) and then appear in a later TorchAO release or nightly. If you can reproduce on 0.14.1, the fastest way to confirm whether it is already fixed upstream is to try torchao nightly that still supports torch 2.9.0 at the Python API level. ([GitHub][4])

---

## cuDNN / Conv3d performance on Blackwell (B300 SM103), inference BF16 (3D VAE decode focus)

### Symptom: Conv3D can be “weirdly slow” in FP16/BF16 even when attention is fast

A recent PyTorch forums report describes exactly this pattern: later attention layers are efficient, but the visual embedding Conv3D is exceptionally slow, using a straightforward `nn.Conv3d(...).` ([PyTorch Forums][10])

Treat that thread as a “canonical repro shape style” reference for agents investigating patch-embed or VAE decode Conv3D bottlenecks. ([PyTorch Forums][10])

### cuDNN best practices you can directly apply (layout, channels, alignment, env vars)

NVIDIA’s cuDNN backend docs have a dedicated “Best Practices for 3D Convolutions” section. Key points that map well to PyTorch:

* **Prefer NDHWC** data layout for 3D conv (PyTorch: `channels_last_3d`) ([NVIDIA Docs][11])
* Channel count constraints that correlate with Tensor Core friendly paths:

  * `C` multiple of 8 (and for INT8, multiple of 16)
  * `K` multiple of 8 (and for INT8, multiple of 16) ([NVIDIA Docs][11])
* It explicitly warns performance can drop when channel counts are **lower than 32**, and “gets worse the lower it is.” This is extremely relevant for 3D VAE decode blocks where channels can be small. ([NVIDIA Docs][11])
* It lists official cuDNN environment variables for troubleshooting and logging (for example `CUDNN_LOGDEST_DBG`, `CUDNN_LOGLEVEL_DBG`). ([NVIDIA Docs][11])

### Drop-in PyTorch actions for VAE decode Conv3D

#### 1) Force `channels_last_3d` end-to-end for Conv3D-heavy modules

```python
# Model weights
vae = vae.to(device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last_3d)

# Inputs
x = x.to(device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
```

Why: cuDNN explicitly calls out NDHWC (NDHWC corresponds to PyTorch channels_last_3d for 5D) as the preferred 3D layout. ([NVIDIA Docs][11])

#### 2) Make channel counts Tensor Core friendly where you can

If your 3D VAE decode has low channel counts (like 4, 8, 16), consider padding or widening internal channels to multiples of 8 and preferably >= 32 in the hottest layers, if the model architecture allows it. cuDNN’s guidance directly states low channel counts hurt performance. ([NVIDIA Docs][11])

This is a model change, but it is sometimes the single biggest lever for Conv3D throughput.

#### 3) Confirm you are actually on cuDNN fast paths (and not a fallback)

Agents should verify with:

* `torch.profiler` (look for cuDNN conv kernels vs `aten::slow_*` ops)
* cuDNN logging (official env vars below)

Minimal logging setup (cuDNN):

```bash
export CUDNN_LOGDEST_DBG=stdout
export CUDNN_LOGLEVEL_DBG=3
```

Those environment variables are explicitly listed as supported in cuDNN docs. ([NVIDIA Docs][11])

If you see PyTorch running `slow_conv_dilated3d` (or similarly named `aten::slow_*` Conv3D ops) in profiles, treat it as “cuDNN is not being used for this configuration,” and then investigate: layout, contiguity/strides, dilation, groups, dtype, and determinism settings.

#### 4) Keep determinism knobs from silently forcing slow algorithms

If you have any global flags like deterministic algorithms enabled, they can restrict algorithm choice and cause big slowdowns. (This is general PyTorch behavior; profile to confirm actual effect in your stack.)

### Blackwell-specific risk: cuDNN kernel selection bugs or forward-compat paths

There is an NVIDIA Developer Forums thread reporting that on Blackwell devices, cuDNN (and frameworks calling into it) may select **sm80** conv kernels instead of Blackwell-appropriate kernels, leading to unexpectedly poor conv performance. The thread mentions the issue showing up in PyTorch and also being reproducible on a 5090 for Conv3D. ([NVIDIA Developer Forums][12])

For B300/SM103, the actionable takeaway for your ladder is:

* If Conv3D looks “stuck” at performance levels similar to much older architectures, treat it as possibly kernel-selection related.
* Prefer the newest stable stack you can validate (driver + CUDA + cuDNN as shipped with your torch wheel or container).
* Use cuDNN logging to confirm which engines/kernels are being picked. ([NVIDIA Developer Forums][12])

### Channels_last_3d maturity in PyTorch

There is a long-running PyTorch effort to improve channels_last_3d (NDHWC) support across common modules on CUDA. This is a useful “status tracker” link for agents trying to understand which ops are expected to be fast in channels_last_3d and which are still catching up. ([GitHub][13])

---

## Quick “agent checklist” you can paste into your repo

### If you hit “CUDAGraphs output overwritten”

1. Switch to `mode="max-autotune-no-cudagraphs"` ([PyTorch Documentation][1])
2. If you need CUDA graphs, add `torch.compiler.cudagraph_mark_step_begin()` per iteration ([PyTorch Documentation][3])
3. Clone outputs you persist across steps ([PyTorch Forums][2])

### If TorchAO float8 + compile fails with tensor subclass dispatch errors

1. Verify torchao version alignment: torch 2.9.0 pairs with torchao 0.14.1 ([GitHub][4])
2. Assume missing-op dispatch is possible (common pattern) ([GitHub][6])
3. Keep subclass tensors from being inputs/outputs of compiled regions, or unwrap at boundaries ([PyTorch Documentation][9])

### If Conv3D BF16 is slow on Blackwell

1. Move tensors and modules to `channels_last_3d` and enforce contiguity ([NVIDIA Docs][11])
2. Check channel counts (multiples of 8, ideally >= 32 in hot layers) ([NVIDIA Docs][11])
3. Turn on cuDNN logging to confirm kernel selection ([NVIDIA Docs][11])
4. Be aware of Blackwell kernel selection issues reported publicly ([NVIDIA Developer Forums][12])

If you want, I can format this into a single `B300_optimization_ladder.md` file layout (sections + copy-paste code blocks + a “known issues and mitigations” appendix) using exactly the citations above so your agents can treat it as an internal reference.

[1]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html "torch.compile — PyTorch 2.9 documentation"
[2]: https://discuss.pytorch.org/t/error-accessing-tensor-output-of-cudagraphs-that-has-been-overwritten-by-a-subsequent-run/218415 "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run - torch.compile - PyTorch Forums"
[3]: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html "CUDAGraph Trees — PyTorch 2.9 documentation"
[4]: https://github.com/pytorch/ao/issues/2919 "torchao release compatibility table · Issue #2919 · pytorch/ao · GitHub"
[5]: https://docs.pytorch.org/ao/stable/quantization_overview.html "Quantization Overview — torchao 0.15 documentation"
[6]: https://github.com/pytorch/ao/issues/890?utm_source=chatgpt.com "[torchao]NotImplementedError: AffineQuantizedTensor ..."
[7]: https://github.com/pytorch/ao/issues/643?utm_source=chatgpt.com "[BUG] Float8Linear does not work with torch.inference_mode"
[8]: https://github.com/pytorch/ao/issues/1463 "The internal torchao tensor subclasses cause errors with torch.compile · Issue #1463 · pytorch/ao · GitHub"
[9]: https://docs.pytorch.org/ao/stable/quick_start.html "Quick Start Guide — torchao 0.15 documentation"
[10]: https://discuss.pytorch.org/t/conv3d-runs-very-slow-in-fp16-and-bf16/223940 "Conv3D runs very slow in fp16 and bf16 - PyTorch Forums"
[11]: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.5.1/developer/misc.html "Odds and Ends — NVIDIA cuDNN"
[12]: https://forums.developer.nvidia.com/t/cudnn-only-selects-sm80-kernels-on-blackwell-devices/338923 "cuDNN only selects sm80 kernels on blackwell devices - cuDNN - NVIDIA Developer Forums"
[13]: https://github.com/pytorch/pytorch/issues/59168?utm_source=chatgpt.com "[CUDA] Add channels_last_3d support for commonly used ..."
