Below is a cleaned, “external docs excerpt pack” you can drop into the B300/SM103 thread. I kept each excerpt short and only pulled the minimum lines needed to justify the repo knobs and version choices.

## 1) PyTorch 2.9 TF32 control change: new `fp32_precision` knobs

**Primary external doc (PyTorch CUDA notes, TF32 section)**

**Excerpt (new API + what it controls):**

> “Starting in PyTorch 2.7, there is a new way of controlling the precision of float32 matrix multiplications and convolutions…
> `torch.backends.cuda.matmul.fp32_precision = <value>`
> `torch.backends.cudnn.conv.fp32_precision = <value>`” ([PyTorch][1])

**Excerpt (deprecation direction):**

> “`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` will be deprecated in a future PyTorch release…” ([PyTorch][1])

**Repo implication (why this matters for B300):**

* Your log warning about “use the new API settings… `conv.fp32_precision` / `matmul.fp32_precision`” is consistent with PyTorch’s documented shift away from the old `allow_tf32` toggles toward the `fp32_precision` knobs. ([PyTorch][1])
* This directly justifies repo logic that treats TF32 behavior as “precision policy” rather than a single boolean, and explains why you see new warning text starting in the 2.9 era (the direction is documented even if the exact warning text is from runtime logs).

---

## 2) TorchInductor / Triton / FlexAttention hazards on Blackwell (SM103): tcgen05 and hard-aborts

### 2a) FlexAttention is coupled to `torch.compile` in real usage

**External context (PyTorch forum thread, FlexAttention performance issue)**

**Excerpt (FlexAttention path involves `torch.compile`):**

> “`create_block_mask` defaults to `_compile=True`, which uses `torch.compile`…” ([PyTorch Forums][2])

**Repo implication:**

* This supports the idea that “FlexAttention code paths often compile something” and that a repo-level kill switch like `DISABLE_FLEX_ATTENTION_COMPILE=1` is a reasonable operational control: it is not random, it is turning off a compilation-coupled path that exists in the ecosystem. ([PyTorch Forums][2])

### 2b) Triton + Blackwell can fail during compilation with tcgen05 intrinsics, and can abort the process

**Primary external evidence (Triton issue on GB300 / Blackwell-class):**

**Excerpt (tcgen05 intrinsic selection failure):**

> “LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st” ([GitHub][3])

**Excerpt (failure mode is a hard abort):**

> “Fatal Python error: Aborted” ([GitHub][3])

**Repo implication (why `DISABLE_FLEX_ATTENTION_COMPILE=1` exists):**

* If a FlexAttention path triggers `torch.compile`, and that path generates Triton kernels, then a compiler-stack failure like the above can become a hard-stop for the whole run (not just “kernel slower”, but “process aborted”). ([PyTorch Forums][2])
* Operationally, that makes a “disable compile for FlexAttention” escape hatch a pragmatic reliability control on SM103, even if it forces a fallback implementation.

### 2c) Why the repo pins or overrides `ptxas` for SM103 (and why it is version-sensitive)

**Primary external evidence (Triton issue about SM103 support regression):**

**Excerpt (SM103 support depends on `ptxas` new enough):**

> “release 3.4 did support sm103 as long as ptxas shipped with CUDA 12.9 or newer is used.” ([GitHub][4])

**Repo implication (why `TRITON_PTXAS_PATH=...` is justified):**

* Your explicit `TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas (or newer)` is directly aligned with upstream Triton maintainer guidance that SM103 requires a `ptxas` at least as new as the one shipped with CUDA 12.9. ([GitHub][4])
* This is also consistent with your observed “stack sensitivity” (same model, different torch+CUDA stack, huge decode delta): if the compiler/assembler component changes, you can land in a different codegen regime.

---

## 3) cuDNN 9.13 notes relevant to Blackwell and convolution-heavy workloads

**Primary external doc (cuDNN v9.13.0 release notes)**

**Excerpt (Blackwell-specific convolution performance improvement):**

> “Performance of grouped convolution has been improved on Blackwell-architecture data center GPUs.” ([NVIDIA Docs][5])

**Repo implication (why this is relevant to WanVAE decode):**

* Even though the bullet is not written as “conv3d”, it is explicitly “convolution performance improved on Blackwell”. That is the kind of release-note signal that justifies expecting major movement in conv-heavy stages like VAE decode when you shift stacks (cu129 to cu130, or cuDNN revision changes). ([NVIDIA Docs][5])
* This supports the repo narrative that decode performance is not “mysteriously volatile”, it is plausibly dominated by low-level library and codegen changes in the convolution backend.

---

## 4) CUDA 13.x notes and compatibility guidance: `ptxas sm_103` and PTX rules

I did not manage to pull NVIDIA’s CUDA Toolkit release notes / CUDA Compatibility Guide excerpts into this run (tool call limit hit before I could open those pages). So I cannot quote NVIDIA’s text here.

What I *can* anchor with a primary excerpt (from upstream Triton) is the practical requirement that matters to your repo:

**Excerpt (assembler version requirement for SM103):**

> “sm103… as long as ptxas shipped with CUDA 12.9 or newer is used.” ([GitHub][4])

**Repo implication (what this supports):**

* Your repo’s behavior of selecting a known-new-enough `ptxas` is justified by upstream statements that SM103 support is not guaranteed with older assemblers. ([GitHub][4])

**Non-quoted but operationally important (general CUDA rules of thumb):**

* If your toolchain cannot assemble SASS for `sm_103`, you either need a newer toolkit (`ptxas`) or you need to rely on PTX JIT paths that the driver can compile for the target GPU.
* “Forward compatibility” is generally about embedding PTX, not shipping only cubins for an older SM.

If you want this section to be fully “primary-source quoted” (as your brief requests), the next thing to do is: open NVIDIA’s CUDA Compatibility Guide and copy the short paragraphs defining binary compatibility vs PTX (and any mention of Blackwell / SM10x family builds), plus the CUDA 13.0–13.2 release-note lines listing supported SM targets and/or `ptxas` support.

---

## 5) torchao ↔ torch 2.9 compatibility and C++ extension mismatch

I also did not successfully extract the compatibility-table excerpt from torchao issue #2919 (same tool-limit problem). I only managed to open the repo landing page, which confirms the project identity but not the version matrix. ([GitHub][6])

**What your repo log message strongly implies (practical interpretation):**

* “Skipping import of cpp extensions due to incompatible torch version …” almost always means the torchao package detected that its compiled extension was built against a different PyTorch version (or ABI), so it intentionally disables the binary fastpath and falls back to Python code.

**Repo implication (why the repo skips torchao extensions):**

* This is a safety behavior: loading mismatched extensions can crash or produce silent wrong results, so packages commonly refuse to load them when version checks fail.

**What to quote next to make this section “done” per your brief:**

* From torchao issue #2919: the release compatibility table (torchao version ↔ supported torch versions).
* From the torchao codebase: the exact guard that prints “Skipping import…” (so the brief can cite the authoritative source of that behavior, not just an issue discussion).

---

## Punchline: how these excerpts justify the repo knobs you already have

* **`fp32_precision` knobs** are the documented direction for TF32 control in modern PyTorch, aligning with your runtime warnings and making your TF32 policy explicit rather than relying on deprecated toggles. ([PyTorch][1])
* **`DISABLE_FLEX_ATTENTION_COMPILE=1`** is justified as a reliability escape hatch because FlexAttention commonly couples to `torch.compile`, and on Blackwell-class systems Triton compilation can hit tcgen05 intrinsic selection failures that abort the process. ([PyTorch Forums][2])
* **`TRITON_PTXAS_PATH=...cuda-12.9...`** is directly supported by upstream Triton guidance that SM103 support depends on having a new-enough `ptxas`. ([GitHub][4])
* **cuDNN 9.13** contains explicit Blackwell convolution performance bullets, consistent with your measured “stack-sensitive” VAE decode timings. ([NVIDIA Docs][5])

If you paste the NVIDIA CUDA Compatibility Guide + CUDA 13.x release-note excerpts and torchao #2919 + “Skipping import…” code excerpt into this same structure, the brief will be fully closed out exactly to your TODO queue.

[1]: https://pytorch.org/docs/main/notes/cuda.html "CUDA semantics — PyTorch main documentation"
[2]: https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581 "Training with flex attention is extremely slow due to torch.compile settings - torch.compile - PyTorch Forums"
[3]: https://github.com/triton-lang/triton/issues/8481 "Fail to run triton on GB300 · Issue #8481 · triton-lang/triton · GitHub"
[4]: https://github.com/triton-lang/triton/issues/8473 "Release 3.5 broke sm103 (GB300) support - a minor release desired · Issue #8473 · triton-lang/triton · GitHub"
[5]: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/release-notes.html "Release Notes — NVIDIA cuDNN Backend"
[6]: https://github.com/pytorch/ao "GitHub - pytorch/ao: PyTorch native quantization and sparsity for training and inference"
