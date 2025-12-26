Below is a doc-ready set of references and “what to do” notes for the three gaps you called out: (1) TorchAO Float8Tensor view or `aten.as_strided` failures under `torch.compile`, (2) cuDNN Conv3d BF16/FP16 “bad vs good” versions and combinations (with sources), and (3) an API correctness checklist for CUDA graph step markers and compile modes so you do not bake in misspellings.

---

## 1) TorchAO Float8Tensor + `aten.as_strided` / view failures under `torch.compile`

### What I can source precisely right now

There are two closely-related (and likely interacting) “failure families” that show up when you combine tensor subclasses (like Float8Tensor) with either (a) view ops or (b) compilation and CUDA graphs:

#### A. InferenceMode + view op on a tensor subclass: `Cannot set version_counter for inference tensor`

This is a PyTorch core issue (not just TorchAO). The minimal repro is: tensor subclass created outside `torch.inference_mode()`, then a view op (example `t.t()`) happens inside `torch.inference_mode()`, which triggers `RuntimeError: Cannot set version_counter for inference tensor`. ([GitHub][1])

Key tracking items:

* PyTorch issue: **pytorch/pytorch#170419** (milestoned for **2.10.0**) ([GitHub][1])
* TorchAO PR: **pytorch/ao#3488** “fix the transpose error” (currently **draft** in the public page view) ([GitHub][2])

What this means for your ladder docs:

* If your Float8 workflow does a transpose/view on weights inside inference-mode (or inside compiled regions that are executed under inference-mode), you can hit this bug today, and the “real” fix is expected in PyTorch 2.10.x (or a backport). ([GitHub][1])

#### B. `torch.compile` causing view-like ops to appear as `aten.as_strided`

Even when your model code does not call `as_strided` explicitly, compilation paths can lower reshape, expand, slicing, and other view ops into `aten.as_strided`. A very clear example is DTensor with compile failing because `aten.as_strided.default` lacks a strategy, and the issue explicitly lists “reshape”, indexing, broadcast_to/expand, etc. ([GitHub][3])

That DTensor issue is not TorchAO, but it establishes the key point: “compile-time transformations frequently surface `aten.as_strided` in places you did not write.” ([GitHub][3])

### The “canonical TorchAO Float8Tensor + `aten.as_strided` unimplemented” pointer you asked for

I did not find a public TorchAO issue that matches the exact error string “Float8Tensor dispatch … unimplemented operator/function: `aten.as_strided`” during this browsing session. What I can point to with high confidence is the nearest canonical root-cause thread for Float8Tensor view/transpose behavior:

* **pytorch/pytorch#170419** (core) ([GitHub][1])
* **pytorch/ao#3488** (TorchAO draft PR aimed at the transpose path) ([GitHub][2])

Given your report that the failure shows as `aten.as_strided` under `torch.compile`, the likely mechanism is:

* Dynamo/Inductor decomposes a view or layout op into `aten.as_strided`, then TorchAO’s Float8Tensor subclass dispatch path does not implement it (so you see “unimplemented”), or it implements a related view op that triggers the inference-mode version-counter path above.

### Version guidance you can safely put in your ladder docs (source-backed)

* **PyTorch fix trajectory:** the inference-mode + view-on-subclass bug is tracked and milestoned for **PyTorch 2.10.0**. So for **PyTorch 2.9.x**, assume you may still hit it. ([GitHub][1])
* **TorchAO fix status:** the public PR **pytorch/ao#3488** exists but appears **draft** (so you should not claim it is released). ([GitHub][2])

### Practical workarounds to document (safe, generally applicable)

These are “drop-in” mitigations you can hand to agents, even if you later replace them with a proper fix or backport:

1. **Avoid view ops on Float8Tensor inside inference-mode**

   * If a Float8Tensor weight is created outside `torch.inference_mode()`, then avoid doing `.t()`, `.view()`, `.reshape()`, slicing, etc inside inference-mode.
   * If you must do them, force the transformation at model setup time (outside the compiled/inference critical path) and store the result.

2. **If compile surfaces `aten.as_strided`, try making the operation contiguous**

   * Replace view-only transforms with a materialization step (`contiguous()` or clone) at the boundary. This is a brute-force escape hatch that trades memory/bandwidth for correctness.

3. **If you need a clean “compile boundary,” unwrap or de-subclass at the boundary**

   * If TorchAO provides an “unwrap tensor subclass” helper in your stack, apply it *before* the compiled region, then run the hot region on plain tensors.
   * This can reduce the chance that Inductor hits subclass-only dispatch for view ops.

If you want a ladder entry that is both accurate and honest: treat these as stopgaps until either (a) you backport the upstream fix, or (b) Float8Tensor implements the missing view op(s) in a way compatible with Inductor and inference-mode.

---

## 2) cuDNN “bad vs good” versions for BF16/FP16 Conv3d (with sources)

You asked for a **source-backed** mapping beyond “it regressed somewhere,” focused on inference Conv3d (3D VAE decode) and ideally relevant to Blackwell-class.

### What the public threads say (concrete combos)

A PyTorch forum thread (“Conv3D runs very slow in fp16 and bf16”, Nov 12 2025) reports testing combinations and finding:

* **Good:**

  * `torch 2.8.0 + cuDNN 9.10`
  * `torch 2.9.1 + cuDNN 9.15`
* **Bad:**

  * `torch 2.9.0` (reported problematic and “doesn’t work with either cuDNN version” in their tests) ([PyTorch Forums][4])

That is already a clean “bad vs good” statement you can cite directly. ([PyTorch Forums][4])

A PyTorch GitHub issue (“Severe Performance Regression in Conv3D / bf16 …”, Nov 19 2025) includes the actionable workaround:

* “upgrade to **2.9.1 / nightlies** and manually install a newer version of **cuDNN (9.15+)**.” ([GitHub][5])

A downstream project (sglang) added a runtime guard explicitly checking for:

* **PyTorch 2.9.1 + cuDNN < 9.15** as a problematic combination for Conv3d perf, implying **cuDNN 9.15 is the minimum “good” baseline** for that stack. ([Safety CLI][6])

### NVIDIA-side signal (Blackwell mentioned, but not Conv3d-specific)

NVIDIA cuDNN backend release notes include a note that:

* A crash fix for Hopper and Blackwell GPUs “might cause some performance regressions,” with a fix planned for the next release. ([NVIDIA Docs][7])

This does not name Conv3d specifically, but it is a legitimate, official statement that some cuDNN releases can regress performance on Hopper/Blackwell and then recover in later releases. ([NVIDIA Docs][7])

### NVIDIA forum: Conv3d BF16/FP16 regression discussed on H100

There is also an NVIDIA developer forum thread explicitly titled as a cuDNN bug report for Conv3d BF16/FP16 regression on H100. ([NVIDIA Developer Forums][8])
This is not B300/SM103, but it confirms the issue class is recognized at the cuDNN layer for modern architectures.

### What you can safely encode in your “B300 optimization ladder” doc

Because you are on **PyTorch 2.9 (cu130)** and targeting Blackwell-class GPUs:

**Recommended known-good baselines (source-backed):**

* Prefer **PyTorch 2.9.1** over 2.9.0 for Conv3d BF16/FP16 stability/perf. ([PyTorch Forums][4])
* Pair **PyTorch 2.9.1** with **cuDNN 9.15 or newer** for Conv3d BF16/FP16. ([PyTorch Forums][4])
* If you must stay older, the forum reports **torch 2.8.0 + cuDNN 9.10** as working. ([PyTorch Forums][4])

**Explicit “bad vs good” statement you can quote in the doc:**

* “In user reports, torch 2.9.0 is the problematic release for fp16/bf16 Conv3d; torch 2.9.1 + cuDNN 9.15 resolves it, and torch 2.8.0 + cuDNN 9.10 resolves it.” ([PyTorch Forums][4])

**Practical inference for B300 (label it as inference, not a proven fact):**

* The strongest public evidence is on H100/H200; however, since the recommended remedy is “cuDNN 9.15+” and cuDNN release notes explicitly call out Hopper/Blackwell sensitivity, it is reasonable to treat “cuDNN version selection” as a first-class knob on Blackwell too. ([GitHub][5])

---

## 3) API correctness check for step markers + compile mode spellings

You asked for “exact step-marker function name(s)” and reliable references so you do not bake wrong spellings into the ladder docs.

### Step-marker functions (documented names)

The safest “don’t bake typos into docs” guidance is:

- **Use:** `torch.compiler.cudagraph_mark_step_begin()`

It has its own generated API reference page. ([PyTorch Documentation][9])

Note: the CUDA Graph Trees doc contains a `torch.compiler.mark_step_begin()` mention in one paragraph, but in PyTorch v2.9.1 the exported symbol in `torch.compiler` is `cudagraph_mark_step_begin()` (see `torch/compiler/__init__.py`). Treat `mark_step_begin()` there as doc shorthand/typo and stick to `torch.compiler.cudagraph_mark_step_begin()`. ([PyTorch Documentation][10])

### The “CUDAGraph output overwritten” error and the recommended fix

The PyTorch forum thread about the exact runtime error (“Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run”) states the two standard mitigations:

* Clone the tensor output **outside** of `torch.compile()`, or
* Call **`torch.compiler.cudagraph_mark_step_begin()`** before each model invocation. ([PyTorch Forums][11])

This lines up with the official CUDA Graph Trees doc which recommends marking step boundaries (via the step-begin marker API) or cloning outputs when heuristics are wrong. ([PyTorch Documentation][10])

### `torch.compile` mode spelling (source-backed)

PyTorch’s `torch.compile` doc explicitly describes:

* `mode="reduce-overhead"` as a CUDA-graphs-based mode intended to reduce Python overhead, with caveats (graph must not mutate inputs, may increase memory usage). ([PyTorch Documentation][12])

For your ladder docs, the “spelling correctness anchor” here is: **`reduce-overhead`** exactly as shown. ([PyTorch Documentation][12])

### Minimal doc-ready snippet (safe and correctly named)

```python
import torch

# If you use a cudagraph-heavy torch.compile mode and hit:
# RuntimeError: Accessing tensor output of CUDAGraphs that has been overwritten ...
#
# Recommended fixes:
# 1) Clone outputs outside the compiled function, OR
# 2) Mark step boundaries explicitly.

@torch.compile(mode="reduce-overhead")  # spelling anchored in docs
def compiled_step(x):
    return model(x)

for batch in batches:
    torch.compiler.cudagraph_mark_step_begin()  # documented API
    y = compiled_step(batch)
    y_safe = y.clone()  # optional safety if you need to persist outputs
```

Citations for the two function names and the error fix guidance: ([PyTorch Documentation][12])

### Inductor config flag names

You asked for exact Inductor config flag spellings “used in examples.” For **PyTorch v2.9.1**, the cudagraph-related Inductor flags are defined under `torch._inductor.config.triton.*`:

- Source: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/config.py (see `class triton`)
- Master env var: `TORCHINDUCTOR_CUDAGRAPHS=1` (sets `torch._inductor.config.triton.cudagraphs`)

Key flags (v2.9.1):

- `torch._inductor.config.triton.cudagraphs`
- `torch._inductor.config.triton.cudagraph_trees`
- `torch._inductor.config.triton.cudagraph_support_input_mutation`
- `torch._inductor.config.triton.cudagraph_skip_dynamic_graphs`
- `torch._inductor.config.triton.cudagraph_unexpected_rerecord_limit`

To keep docs resilient across point releases, you can still include a small introspection helper so people can confirm what exists in the build they’re running:

```python
import torch._inductor.config as cfg

print([k for k in dir(cfg.triton) if "cudagraph" in k])
```

This avoids freezing internal names into docs when you move between 2.9.0, 2.9.1, nightlies, or custom wheels.

---

## Quick “drop-in” ladder entries you can add immediately

### Ladder entry: Conv3d BF16/FP16 is slow on 2.9.0

* Symptom: Conv3d BF16/FP16 becomes extremely slow.
* Recommended baseline: **torch 2.9.1 + cuDNN 9.15+** (known-good in reports). ([PyTorch Forums][4])
* “Bad” version: **torch 2.9.0** (reported problematic even when varying cuDNN). ([PyTorch Forums][4])

### Ladder entry: CUDAGraph overwritten output

* Fix: `torch.compiler.cudagraph_mark_step_begin()` before each invocation, or clone outputs outside compile. ([PyTorch Forums][11])

### Ladder entry: TorchAO Float8Tensor view op failures (root cause tracking)

* Root-cause tracker: **pytorch/pytorch#170419** (milestone 2.10.0) and TorchAO PR **pytorch/ao#3488** (draft). ([GitHub][1])
* Until fixed: avoid view ops on subclass tensors inside inference-mode or compiled hot paths; force materialization or precompute.

---

If you paste the exact exception text (the full line that includes the operator name and the tensor subclass name) from your Float8Tensor + `aten.as_strided` failure, I can map it to either an existing tracker or a minimal reproduction template that matches the style of #170419 and will be actionable for an upstream backport.

[1]: https://github.com/pytorch/pytorch/issues/170419 "Cannot set version_counter for inference tensor · Issue #170419 · pytorch/pytorch · GitHub"
[2]: https://github.com/pytorch/ao/pull/3488 "[float8] fix the transpose error by Stonepia · Pull Request #3488 · pytorch/ao · GitHub"
[3]: https://github.com/pytorch/pytorch/issues/167074?utm_source=chatgpt.com "DTensor with `torch.compile` fails with `aten.as_strided. ..."
[4]: https://discuss.pytorch.org/t/conv3d-runs-very-slow-in-fp16-and-bf16/223940?utm_source=chatgpt.com "Conv3D runs very slow in fp16 and bf16"
[5]: https://github.com/pytorch/pytorch/issues/168167?utm_source=chatgpt.com "Severe Performance Regression in Conv3D / bf16 ..."
[6]: https://data.safetycli.com/packages/pypi/sglang/changelog?utm_source=chatgpt.com "sglang Changelog"
[7]: https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html?utm_source=chatgpt.com "Release Notes — NVIDIA cuDNN Backend"
[8]: https://forums.developer.nvidia.com/t/cudnn-bug-report-conv3d-performance-regression-with-bfloat16-float16-on-h100/355210?utm_source=chatgpt.com "Conv3d Performance Regression with bfloat16/float16 on ..."
[9]: https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html?utm_source=chatgpt.com "torch.compiler.cudagraph_mark_step_begin"
[10]: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html?utm_source=chatgpt.com "CUDAGraph Trees"
[11]: https://discuss.pytorch.org/t/error-accessing-tensor-output-of-cudagraphs-that-has-been-overwritten-by-a-subsequent-run/218415?utm_source=chatgpt.com "Error: accessing tensor output of CUDAGraphs that has ..."
[12]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html?utm_source=chatgpt.com "torch.compile — PyTorch 2.9 documentation"
