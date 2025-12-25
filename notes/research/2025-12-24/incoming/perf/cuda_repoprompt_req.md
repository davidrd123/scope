Below is a ready-to-drop-in draft for `notes/FA4/b300/blackwell-docs.md` (copy/paste into that path). It’s written as a “web-search brief” for GPT‑5 Pro, with short excerpts + direct URLs (no vendoring).

```md
# Blackwell / SM103 external docs brief (B300)

Date: 2025-12-24
Owner: FA4/B300 investigation thread
Goal: Collect *external* documentation + short excerpts that explain/justify the repo’s B300/SM103 decisions:
- TF32 API change in PyTorch 2.9 (new `fp32_precision` knobs)
- TorchInductor/Triton issues on Blackwell (tcgen05) and why `DISABLE_FLEX_ATTENTION_COMPILE=1` is sometimes required
- cuDNN 9.13 notes relevant to Blackwell + conv3d-heavy workloads (WanVAE decode)
- CUDA 13.x notes + compatibility guidance (ptxas `sm_103`, PTX rules)
- torchao ↔ torch 2.9 compatibility / C++ extension mismatch (why `torchao` extensions are being skipped)

> Note: Keep excerpts short. Do NOT vendor full docs.

---

## Repo context (why we care)

**Empirical symptom (from repo artifacts):**
- WanVAE `stream_decode(t=3)` is **~760ms/call** on torch+cu129, but **~195ms/call** on torch 2.9 + cu130.
  - See: `outputs/b300_cu129_vae_stream_decode_bench.log` vs `outputs/b300_cu130_vae_stream_decode_bench.log`
- In baseline cu128/cu129 stacks, `decode` dominates the pipeline block profile; on cu130, decode becomes much smaller.

**Operational knobs already in-repo:**
- `TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas` (or newer)
- `DISABLE_FLEX_ATTENTION_COMPILE=1` (avoid SM103 torch.compile hard-aborts in flex_attention path)
- `WANVAE_STREAM_DECODE_MODE=chunk` (decode full chunk in one call)

---

## Quick link index (external)

### PyTorch TF32 / CUDA notes
- PyTorch CUDA notes (TF32 section):
  https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices

### FlexAttention / torch.compile guidance (PyTorch forum)
- “Training with flex attention is extremely slow due to torch.compile settings” (thread):
  https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581

### Blackwell tcgen05 background (helps explain Triton/LLVM failures)
- Modular blog: “Matrix Multiplication on Blackwell: Part 2 — Using Hardware Features to Optimize Matmul”
  https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul

### cuDNN release notes index (find 9.13 entries + Blackwell/conv3d notes)
- cuDNN release notes (index):
  https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html

### CUDA Toolkit release notes + compatibility guide (Blackwell / sm_103 / ptxas)
- CUDA Toolkit release notes (pick 13.0 / 13.1 / 13.2+ as applicable):
  https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA Compatibility Guide:
  https://docs.nvidia.com/deploy/cuda-compatibility/index.html

### torchao compatibility / version mismatch
- torchao repo:
  https://github.com/pytorch/ao
- Issue referenced by our logs (extensions disabled due to mismatch):
  https://github.com/pytorch/ao/issues/2919

---

## 1) PyTorch 2.9 TF32 API change (fp32_precision)

### What we observed in our logs (actionable excerpt)
From `outputs/b300_cu130_fp8_bias03_drilldown_perf.log` (PyTorch warning):

> “Please use the new API settings to control TF32 behavior, such as
> `torch.backends.cudnn.conv.fp32_precision = 'tf32'` or
> `torch.backends.cuda.matmul.fp32_precision = 'ieee'`.
> Old settings … will be deprecated after Pytorch 2.9.”

This is the concrete “why” for migrating any TF32 toggles in code/config and for not relying on:
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`

### External doc link
- PyTorch CUDA notes TF32 section:
  https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices

### Suggested follow-up (for GPT‑5 Pro to extract)
Search within that PyTorch doc page for:
- `fp32_precision`
- `allow_tf32`
- any mention of “Blackwell”, “SM100”, “SM103”, “Hopper” (even if minimal)

---

## 2) TorchInductor / Triton / flex_attention on Blackwell (SM103): tcgen05 + compile hazards

### Repo-grounded reason `DISABLE_FLEX_ATTENTION_COMPILE=1` exists
In-repo comment (not external doc, but the key symptom):
- `src/scope/core/kernels/triton_attention.py`:
  - “Triton 3.5 … fails on SM103 when lowering tensor-core dot() to **tcgen05** intrinsics (LLVM error: `tcgen05.wait.st`).”

This explains “hard aborts compiling flex_attention on SM103” (and why we sometimes force scalar kernels or disable compile).

### External context: what tcgen05 is (short excerpt)
From the Modular Blackwell matmul article:

> “5th generation tensor cores … come with a new set of instructions (`tcgen05` instructions) …”

Source:
https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul

This is useful background: tcgen05 is a *new* Blackwell tensor-core ISA pathway; compiler stacks (LLVM/Triton/Inductor) can lag or have bugs.

### flex_attention compile guidance (external, but not Blackwell-specific)
From the PyTorch forum thread:

> “If you need to change the sequence length, it is recommended to set `torch.compile` to `dynamic=True` or `dynamic=None` … Otherwise, the speed will be very slow.”

Source:
https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581

This is not about SM103 specifically, but it is relevant to understanding why compile settings can dramatically change flex_attention behavior.

### What to extract next (web follow-ups)
Search terms to find *direct* SM103 / tcgen05 / Triton issues:
- `tcgen05.wait.st triton sm_103`
- `torch.compile flex_attention sm_103`
- `inductor triton blackwell tcgen05`
- `triton 3.5 blackwell llvm tcgen05`

When you find authoritative issues (PyTorch GitHub / Triton GitHub / NVIDIA dev forums), add:
- exact error strings
- recommended workarounds (env vars, version pins, flags)
- whether fixed in newer torch/triton (post-2.9 / Triton 3.5)

---

## 3) cuDNN 9.13 release notes: Blackwell + conv3d-heavy improvements

### Why this matters (repo evidence)
WanVAE decode is **conv3d-heavy** and dominates end-to-end latency on older stacks.

Measured delta (repo artifacts):
- torch 2.8 + cu129 (CUDA 12.9, cuDNN ~9.10): ~760ms/call
- torch 2.9 + cu130 (CUDA 13.0, cuDNN ~9.13): ~195ms/call

Artifacts:
- `outputs/b300_cu129_vae_stream_decode_bench.log`
- `outputs/b300_cu130_vae_stream_decode_bench.log`

### External doc link (start here)
cuDNN release notes index:
https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html

### What to extract from cuDNN notes (targeted)
Within cuDNN release notes, search for:
- “9.13”
- “Blackwell”
- “SM10x / SM100 / SM103”
- “3D convolution / conv3d”
- “bfloat16”
- “benchmark / heuristics / engine selection”
- anything about “performance improvements” for conv/3d conv on Blackwell

Goal: identify *which* changes plausibly explain the ~4× decode improvement:
- new kernels for SM103
- better algorithm selection / heuristics
- fixed regressions for bf16 conv3d
- changes in default engine behavior

---

## 4) CUDA 13.x release notes + compatibility guide: sm_103 and ptxas rules

### Why this matters (repo behavior)
Our scripts explicitly try to pick a `ptxas` that “knows sm_103”:
- `scripts/run_daydream_b300.sh`
- `scripts/profile_b300_denoise_drilldown.sh`
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (ptxas search)

This is necessary because Triton/Inductor compilation for SM103 depends on a compatible toolchain.

### External docs to use
CUDA Toolkit release notes index:
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

CUDA Compatibility Guide:
https://docs.nvidia.com/deploy/cuda-compatibility/index.html

### What to extract (targeted)
From CUDA 13.0/13.1/13.2+ notes + compatibility docs:
- explicit mention of Blackwell / compute capability 10.3 / `sm_103`
- what minimum toolkit version supports `sm_103` in ptxas
- PTX forward-compat rules and what happens when:
  - driver is newer than toolkit
  - toolkit is older than GPU arch
- guidance on using newer `ptxas` with older runtime (if documented)

Also useful: any note about `compute_103a` / “a” variants (if they exist in CUDA docs for Blackwell) and how that interacts with compilation.

---

## 5) torchao compatibility (torch 2.9)

### What our logs say (actionable excerpt)
We repeatedly see messages like:

> “Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu130 for torchao version …”

This means torchao’s compiled extensions are not loading, so performance/behavior may differ vs a fully matched build.

### External links
- torchao repo: https://github.com/pytorch/ao
- Issue referenced in our logs: https://github.com/pytorch/ao/issues/2919

### What to extract next
Find the torchao compatibility policy/matrix for:
- which torch versions are supported by which torchao versions
- how to install a matching build (wheels vs source)
- whether torchao extensions matter for our codepath:
  - FP8 quantization (`Float8DynamicActivationFloat8WeightConfig`)
  - any fused kernels we’d otherwise get

---

## “Drop-in” guidance for our repo (what GPT‑5 Pro should remember)

### Environment / tooling
- Always set a ptxas that supports SM103:
  - prefer `/usr/local/cuda-13.1/bin/ptxas` or `/usr/local/cuda-13.0/bin/ptxas` or `/usr/local/cuda-12.9/bin/ptxas`
- Keep `DISABLE_FLEX_ATTENTION_COMPILE=1` as a safe default on SM103 if compilation is unstable.
- Ensure `flash_attn` is installed in the cu130 env; otherwise KV-bias backend can fall back and tank FPS.

### Why runtime matters more than microkernels (B300)
- cuDNN / CUDA runtime stack can dominate overall performance for conv3d-heavy decode.
- Attention backend changes alone won’t fix a slow decode stack.

---

## TODO queue (web search gaps to fill)

1) A *primary-source* report (GitHub issue / release note) that explicitly ties:
   - Triton + Blackwell tcgen05 + LLVM failure
   - recommended fix / version where it’s fixed

2) cuDNN 9.13 “Blackwell + 3D conv” specific bullet(s) from release notes

3) CUDA toolkit version that first introduced:
   - `ptxas` support for `sm_103`
   - any “103a” compatibility mode (if applicable)

4) torchao official compatibility statement for torch 2.9 (not just issues)

Suggested search terms:
- `cudnn 9.13 blackwell conv3d release notes`
- `cuda 13 ptxas sm_103`
- `triton tcgen05 sm_103 wait.st`
- `torchao torch 2.9 compatibility`
```

If you want, I can also format this into a tighter “one-screen TL;DR + links” version at the top (keeping the full detail below), but the content above is already structured for GPT‑5 Pro to continue the doc-gathering work efficiently.
