# Blackwell / SM103 external docs brief (B300)

Date: 2025-12-24
Owner: FA4/B300 investigation thread
Goal: Collect *external* documentation + short excerpts that explain/justify the repo’s B300/SM103 decisions:
- TF32 API change in PyTorch 2.9 (new `fp32_precision` knobs)
- TorchInductor/Triton issues on Blackwell (tcgen05) and why `DISABLE_FLEX_ATTENTION_COMPILE=1` is sometimes required
- cuDNN 9.13 notes relevant to Blackwell + conv3d-heavy workloads (WanVAE decode)
- PyTorch 2.9 Conv3d BF16/FP16 regressions + the “install cuDNN 9.15+” workaround
- CUDAGraph Trees footgun (“output overwritten”) + correct step-marker spellings
- CUDA 13.x notes + compatibility guidance (ptxas `sm_103`, PTX rules)
- torchao ↔ torch 2.9 compatibility / C++ extension mismatch (why `torchao` extensions are being skipped)

> Keep excerpts short. Do NOT vendor full docs.

---

## Repo context (why we care)

**Empirical symptoms (from repo artifacts):**
- VAE decode perf is stack-dependent:
  - torch `2.8.0+cu129` (CUDA `12.9`): `stream_decode(t=3)` ~`760ms/call` (`outputs/b300_cu129_vae_stream_decode_bench.log`)
  - torch `2.9.0+cu130` (CUDA `13.0`): `stream_decode(t=3)` ~`195ms/call` (`outputs/b300_cu130_vae_stream_decode_bench.log`)
- Generator dominates denoise:
  - `outputs/b300_cu130_fp8_bias03_drilldown_generator_steps.json` shows `call_model_kv_cache` dominates `generator(...)` (conversion is negligible).

**Operational knobs already in-repo:**
- `TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas` (or newer)
- `DISABLE_FLEX_ATTENTION_COMPILE=1` (avoid SM103 torch.compile hard-aborts in flex_attention path)
- `WANVAE_STREAM_DECODE_MODE=chunk` (decode chunk in one call)

---

## Quick link index (external)

### PyTorch TF32 / CUDA notes
- PyTorch CUDA notes (TF32 section): https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices

### FlexAttention / torch.compile guidance (PyTorch forum)
- “Training with flex attention is extremely slow due to torch.compile settings”: https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581

### Blackwell tcgen05 background (helps explain Triton/LLVM failures)
- Modular blog (tcgen05 / 5th-gen tensor cores): https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul

### Triton SM103 issues (primary sources)
- Triton: tcgen05 intrinsic abort on GB300: https://github.com/triton-lang/triton/issues/8481
- Triton: SM103 support and `ptxas` >= CUDA 12.9: https://github.com/triton-lang/triton/issues/8473

### cuDNN release notes index (find 9.13 entries + Blackwell/conv3d notes)
- cuDNN release notes index: https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html
- cuDNN Backend 9.13.0 release notes: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/release-notes.html

### PyTorch 2.9 Conv3d BF16/FP16 regressions (and the cuDNN 9.15+ workaround)
- PyTorch 2.9.1 release notes (explicitly recommends `nvidia-cudnn-cu12>=9.15` if impacted): https://github.com/pytorch/pytorch/releases/tag/v2.9.1
- Conv3d BF16 memory regression issue: https://github.com/pytorch/pytorch/issues/166643
- Severe Conv3d BF16 perf regression issue: https://github.com/pytorch/pytorch/issues/168167
- Conv3d OOM regression (workspace algo selection; includes `nvidia-cudnn-cu12==9.10.2.21` in env): https://github.com/pytorch/pytorch/issues/166790
- Conv3d AMP regression (includes `nvidia-cudnn-cu12==9.10.2.21` in env): https://github.com/pytorch/pytorch/issues/166122

### CUDAGraph Trees “output overwritten” + step marker spelling
- CUDAGraph Trees docs: https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
- Step marker API doc page: https://docs.pytorch.org/docs/stable/generated/torch.compiler.cudagraph_mark_step_begin.html
- Inductor cudagraph env var (v2.9.1): `TORCHINDUCTOR_CUDAGRAPHS=1` in https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_inductor/config.py

### CUDA Toolkit release notes + compatibility guide (Blackwell / sm_103 / ptxas)
- CUDA Toolkit release notes index (13.0 / 13.1 / 13.2+): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA Features Archive (12.8/12.9/13.x feature deltas): https://docs.nvidia.com/cuda/cuda-features-archive/index.html
- CUDA Compatibility Guide: https://docs.nvidia.com/deploy/cuda-compatibility/index.html
- CUDA Blackwell compatibility guide: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
- CUDA compute capability appendix: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- NVIDIA blog: family-specific architectures (CUDA 12.9): https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/

### torchao compatibility / version mismatch
- torchao repo: https://github.com/pytorch/ao
- Issue referenced by our logs: https://github.com/pytorch/ao/issues/2919

---

## 1) PyTorch 2.9 TF32 API change (fp32_precision)

### What we observed in our logs (actionable excerpt)

From `outputs/b300_cu130_fp8_bias03_drilldown_perf.log`:

> “Please use the new API settings to control TF32 behavior, such as
> `torch.backends.cudnn.conv.fp32_precision = 'tf32'` or
> `torch.backends.cuda.matmul.fp32_precision = 'ieee'`.
> Old settings … will be deprecated after Pytorch 2.9.”

### External excerpts (PyTorch docs)

From the PyTorch CUDA notes (TF32 section):
- “Starting in PyTorch 2.7, there is a new way of controlling the precision of float32 matrix multiplications and convolutions…
  `torch.backends.cuda.matmul.fp32_precision = <value>`
  `torch.backends.cudnn.conv.fp32_precision = <value>`”
- “`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` will be deprecated in a future PyTorch release…”

### External doc link
- https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices

### Follow-up to extract (web)
Search that page for:
- `fp32_precision`
- `allow_tf32`
- “Blackwell”, “SM100”, “SM103”

---

## 2) TorchInductor / Triton / flex_attention on Blackwell (SM103): tcgen05 + compile hazards

### Repo-grounded reason `DISABLE_FLEX_ATTENTION_COMPILE=1` exists

The exact error string / context we’ve seen (for web search):
- `tcgen05.wait.st`
- “Triton 3.5 … fails on SM103 when lowering tensor-core dot() to tcgen05 intrinsics”

Relevant code context:
- `src/scope/core/kernels/triton_attention.py`
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (compile toggles + ptxas selection)

### External excerpts (Triton)

Triton compilation on Blackwell-class (GB300) can hard-abort with tcgen05 intrinsics:
- “LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st”
- “Fatal Python error: Aborted”
Source: https://github.com/triton-lang/triton/issues/8481

Triton SM103 support depends on using a new-enough `ptxas`:
- “release 3.4 did support sm103 as long as ptxas shipped with CUDA 12.9 or newer is used.”
Source: https://github.com/triton-lang/triton/issues/8473

### External excerpt (FlexAttention defaults to compile in common usage)

From a PyTorch forum thread discussing FlexAttention performance:
- “`create_block_mask` defaults to `_compile=True`, which uses `torch.compile`…”
Source: https://discuss.pytorch.org/t/training-with-flex-attention-is-extremely-slow-due-to-torch-compile-settings/222581

### External context: what tcgen05 is

From the Modular Blackwell matmul article (use as background, not authority for fixes):
- https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul

### What to extract next (web follow-ups)
Prefer primary sources (PyTorch GitHub / Triton GitHub / NVIDIA forums):
- exact error strings
- recommended workarounds (env vars, version pins, flags)
- which torch/triton versions fix it (if fixed)

Search terms:
- `tcgen05.wait.st triton sm_103`
- `torch.compile flex_attention sm_103`
- `inductor triton blackwell tcgen05`

---

## 3) cuDNN 9.13 release notes: Blackwell + conv3d-heavy improvements

### Why this matters (repo evidence)
WanVAE decode is conv3d-heavy and is the major “stack sensitivity” we’ve measured on B300.

Artifacts:
- `outputs/b300_cu129_vae_stream_decode_bench.log`
- `outputs/b300_cu130_vae_stream_decode_bench.log`

### External doc link (start here)
- https://docs.nvidia.com/deeplearning/cudnn/release-notes/index.html

### External excerpt (cuDNN 9.13.0)

From cuDNN Backend v9.13.0 release notes:
- “Performance of grouped convolution has been improved on Blackwell-architecture data center GPUs.”
Source: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/release-notes.html

### What to extract (targeted)
From cuDNN 9.13 notes, pull the *specific bullets* about:
- Blackwell / SM10x / SM103 support
- 3D convolution / conv3d performance or engine/heuristics changes
- bf16 performance notes

---

## 3b) PyTorch 2.9 Conv3d BF16/FP16 regressions: the official workaround

### Why this matters (repo evidence)
Our pipeline is conv3d-heavy in decode (WanVAE). Even if our local B300 improvements primarily came from CUDA 13 / cuDNN 9.13, upstream reports show that Conv3d BF16/FP16 can catastrophically regress depending on the exact PyTorch+cuDNN combo, so we should keep a “known good workaround” link.

### External excerpt (keep short)
From the PyTorch 2.9.1 release notes (Tracked Regressions section):
- “If you are impacted please install nvidia-cudnn package version 9.15+ from pypi.”

Source: https://github.com/pytorch/pytorch/releases/tag/v2.9.1

---

## 4) CUDA 13.x release notes + compatibility guide: sm_103 and ptxas rules

### Why this matters (repo behavior)
We explicitly pick a `ptxas` that “knows sm_103” in:
- `scripts/run_daydream_b300.sh`
- `scripts/profile_b300_denoise_drilldown.sh`
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

### External docs to use
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/deploy/cuda-compatibility/index.html

### External excerpts (CUDA Toolkit 13.1 release notes)

From CUDA Toolkit 13.1 release notes (general + cuBLAS sections):

- “The minimum required driver version for CUDA minor version compatibility is shown below…”
  - “13.x | >= 580 | N/A”
- “CUDA 13.1 introduces CUDA Tile…”
  - “The initial release targets Blackwell GPUs, with broader architecture support planned across the CUDA 13.x series.”
- Known issue that can block CUDA initialization on some Linux kernels:
  - “Certain Linux kernels with KASLR enabled have a known issue in HMM initialization, causing CUDA initialization to fail…”
  - Workaround includes disabling KASLR (`nokaslr`) or disabling HMM for UVM (`options nvidia_uvm uvm_disable_hmm=1`).
- cuBLAS improvements that explicitly mention SM103 / compute capability 10.3:
  - “Improved performance on Blackwell (`sm_100` and `sm_103`) via heuristics tuning for FP32 GEMMs whose shapes satisfy `M, N >> K`.”
  - “Removed unnecessary overhead related to loading kernels on GPUs with compute capability 10.3.”

Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

### External excerpt (CUDA Toolkit 12.9: SM103 compiler targets + family-specific architectures)

From CUDA Toolkit 12.9 release notes (General CUDA features):
- “CUDA Toolkit 12.9 adds compiler target support for SM architecture 10.3 (`sm_103`, `sm_103f`, `sm_103a`) …”
- “CUDA Toolkit 12.9 introduces compiler support for a new target architecture class: family-specific architectures.”

This is the key “minimum version” anchor for why older toolchains (e.g. CUDA 12.8 `ptxas`) fail to assemble `sm_103`.

Source: https://docs.nvidia.com/cuda/cuda-features-archive/index.html

### External excerpt (CUDA Features Archive: 12.8 vs 12.9)

From CUDA Toolkit 12.8 features:
- “This release adds compiler support for the following Nvidia Blackwell GPU architectures: `SM_100`, `SM_101`, `SM_120`”

From CUDA Toolkit 12.9 features:
- “CUDA Toolkit 12.9 adds compiler target support for SM architecture 10.3 (`sm_103`, `sm_103f`, `sm_103a`) …”
- “CUDA Toolkit 12.9 introduces compiler support for a new target architecture class: family-specific architectures.”

Source: https://docs.nvidia.com/cuda/cuda-features-archive/index.html

### External excerpts (NVIDIA blog: family-specific architectures)

Key compatibility rules (PTX vs cubin):
- “PTX… is designed to be general enough to be forward compatible with future GPU architectures.”
- “PTX compatibility… Any code with PTX of a certain CC will run on GPUs of that CC and any GPU with a later CC.”
- “Cubin compatibility… Any code with a cubin of a certain CC will run on GPUs of that CC and any later GPU with that same major capability…”

`a` suffix (architecture-specific):
- “When building the architecture-specific target using the `a` suffix, the PTX or cubin code is not forward-compatible with any future GPU architecture.”
- “There is no forward-compatibility for either PTX or a cubin when using the architecture-specific `a` suffix.”

`f` suffix (family-specific; Blackwell + CUDA 12.9):
- “Family-specific features… are supported by devices of more than one minor compute capability.”
- “Family-specific features are guaranteed to be available in the same family of GPUs… later GPUs of the same major compute capability and higher minor compute capability.”

Macros for guarded fallbacks:
- `__CUDA_ARCH_FAMILY_SPECIFIC__`
- `__CUDA_ARCH_SPECIFIC__`

Source: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/

### External excerpts (Blackwell Compatibility Guide: cubin/PTX + `sm_100a`)

From the Blackwell compatibility guide (rules + pitfalls):
- “A cubin generated for a certain compute capability is supported to run on any GPU with the same major revision and same or higher minor revision…”
- “PTX is forward-compatible… it is recommended that all applications should include PTX of the kernels to ensure forward-compatibility.”
- “If neither compatible cubin nor PTX is available, kernel launch results in a failure.”
- “Application binaries that include PTX… with architecture conditional features using `sm_100a` or `compute_100a`… are not forward or backward compatible.”

Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/

### External excerpt (CUDA C++ Programming Guide: `compute_100` vs `compute_100f` vs `compute_100a`)

PTX target variants (Blackwell family vs arch-specific):
- “PTX compiled for `compute_100` is supported on devices with compute capability 10.0 and later.”
- “PTX compiled for `compute_100f` is supported on devices with compute capability 10.0 and 10.3.”
- “PTX compiled for `compute_100a` is supported only on devices with compute capability 10.0.”

Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

### What to extract (targeted)
From CUDA 13.0/13.1/13.2+ notes + compatibility docs:
- minimum toolkit version whose `ptxas` supports `sm_103`
- PTX forward-compat rules and common failure modes when toolkit is older than GPU
- anything about `compute_*a` / family-compatible builds for Blackwell (if documented)

---

## 5) torchao compatibility (torch 2.9)

### What our logs say
We see:
> “Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu130 for torchao version …”

This likely means we’re not getting torchao’s compiled fastpaths, which may affect FP8 behavior.

### External links
- https://github.com/pytorch/ao
- https://github.com/pytorch/ao/issues/2919
- torchao v0.15.0 release: https://github.com/pytorch/ao/releases/tag/v0.15.0

### External excerpt (torchao release compatibility table, issue #2919)

Compatibility matrix (build-time ABI compatibility):
- torchao `0.14.1` ↔ torch `2.9.0`
- torchao `0.13.0` ↔ torch `2.8.0`

Python-API-only compatibility (extensions may be disabled, but APIs work):
- torchao `0.14.1` Python API: torch `2.9.0`, `2.8.0`, `2.7.1`
- torchao `0.13.0` Python API: torch `2.8.0`, `2.7.1`, `2.6.0`

Operational note (re: the warning we see in our logs):
- “Note that unless you actually need c++ or CUDA kernels that ship with torchao, you can ignore the warning and use the Python-only APIs without issues.”

Source: https://github.com/pytorch/ao/issues/2919 (compat table)

### External excerpt (torchao + fbgemm_gpu)

- “torchao has an optional runtime dependency on `fbgemm_gpu`… each `fbgemm_gpu` version only supports a single `torch` version.”
- “if you are using torchao together with `fbgemm_gpu`, you should use the torch version corresponding to your `fbgemm_gpu` version.”

Source: https://github.com/pytorch/ao/issues/2919

### What to extract next (web)
Find an official torchao statement for:
- supported torch versions per torchao release
- correct install procedure for matching wheels/builds for torch `2.9.0+cu130`

---

## 6) Blackwell Optimization Guides (blog collection)

> **Source folder:** `notes/research/2025-12-24/incoming/perf/blogs/`
> These are saved blog posts with optimization strategies relevant to B300.

### Quick reference: Blackwell-specific blogs

| File | Topic | Key Insight |
|------|-------|-------------|
| `verda-b200-b300.md` | B200/B300 architecture specs | SM103 specs, TMEM, CTA pairs, profiling config |
| `verda-flux-on-b200.md` | WaveSpeedAI FLUX on B200 | cuDNN 2x faster than FA3; end-to-end GPU execution |
| `thunderkittens-blackwell.md` | ThunderKittens on B200 | **Tensor cores are 128×128 systolics** - need M,N=128 |
| `gau-nerst-tcgen05.md` | tcgen05 PTX tutorial | Low-level Blackwell matmul, 98% cuBLAS speed |

### Quick reference: General optimization blogs

| File | Topic | Key Insight |
|------|-------|-------------|
| `flexattn-for-inference.md` | FlexDecoding backend | `max-autotune` for score_mods; KV offset pattern |
| `torch-compile-and-diffusers.md` | torch.compile for diffusers | Regional compilation 7x faster cold start |
| `accel-genai-pytorch-1.md` | SAM optimization (part 1) | GPU syncs from index ops; use `torch.where` |
| `accel-genai-pytorch-2.md` | SAM optimization (part 2) | Quantization, sparsity |
| `accel-genai-pytorch-3.md` | SAM optimization (part 3) | NestedTensor, batching |
| `flexattn-1.md` | FlexAttention intro | score_mod/mask_mod API |
| `flexattention_guide.md` | FlexAttention patterns | Common attention variants |

### Key excerpts: Blackwell tensor core sizing

From `thunderkittens-blackwell.md`:
> "The B200 tensor cores aren't just faster than the H100 tensor cores – they're also much larger. From our microbenchmarking, they seem to behave like **128 × 128 systolics**."
>
> "To get full FLOP utilization, you want M and N to be **128** (or larger multiples of 128)."
>
> "Smaller values of M and N run at the corresponding fraction of 128; e.g., a 64×64×64 GEMM will run at one-quarter the FLOP rate of a 128×128×64 GEMM."

**Implication for Scope:** Check that attention head dimensions and batch tile sizes align to 128 multiples for full tensor core utilization on B300.

### Key excerpts: torch.compile best practices

From `torch-compile-and-diffusers.md`:
> "Regional compilation: compile smaller, repeated blocks... On an H100, this cuts compile latency from **67.4 seconds to 9.6 seconds**, reducing the cold start by **7x** while still delivering the 1.5x runtime speedup."
>
> "The DiT dominates the compute budget. You _could_ compile every component in the pipeline, but that only adds compile latency... For these reasons, we restrict torch.compile to the DiT component."

From `flexattn-for-inference.md`:
> "For optimal performance, compile FlexAttention using `max-autotune`, especially when dealing with complex `score_mods` and `mask_mods`:"
> ```python
> flex_attention = torch.compile(flex_attention, dynamic=True, mode='max-autotune')
> ```

### Key excerpts: GPU sync elimination

From `accel-genai-pytorch-1.md`:
> "aten::index is launching two kernels, and a blocking cudaStreamSynchronize is happening in between. This means the CPU is waiting for the GPU to finish processing until it launches the second kernel."
>
> "We can use `torch.where` to rewrite this..."

**Implication for Scope:** Profile for `cudaStreamSynchronize` calls; rewrite CPU-side index operations that cause GPU syncs.

### Key excerpts: TMEM and CTA pairs

From `verda-b200-b300.md`:
> "**TMEM (Tensor Memory):** A new memory space introduced in Blackwell, residing conceptually between Registers and Shared Memory. Very similar to register memory but requires explicit user management/allocation. Reserved for advanced kernel programming (e.g., ThunderKittens)."
>
> "**CTA Pairs (2CTA):** The PTX model allows two Cooperative Thread Arrays (CTAs) to execute tensor core instructions together. One CTA can access the TMEM of the other CTA in the pair."

### Actionable checklist (derived from blogs)

- [x] Verify attention shapes are 128×128 multiples for tensor cores
- [x] Use `max-autotune` for flex_attention compilation
- [ ] Consider regional compilation for faster cold start
- [ ] Profile for GPU syncs (`cudaStreamSynchronize`)
- [ ] Evaluate ThunderKittens attention kernel as alternative

---

## 7) Codebase Correlation: Blog Advice vs Scope Implementation

> **Purpose:** Cross-reference optimization advice from blogs against actual Scope code.
> **Generated:** 2025-12-25

### 7.1 Attention Head Dimensions (✓ ALIGNED)

**Blog advice** (thunderkittens-blackwell.md):
> "Blackwell tensor cores behave like 128×128 systolics. To get full FLOP utilization, you want M and N to be 128."

**Scope implementation:**
- **Wan2.1-T2V-14B**: `dim=5120`, `num_heads=40` → `head_dim=128` ✓
- **Wan2.1-T2V-1.3B**: `dim=1536`, `num_heads=12` → `head_dim=128` ✓

Both models have `head_dim=128`, which aligns perfectly with Blackwell tensor core requirements.

**Source:** `notes/research/2025-12-24/wan2.1-14b-integration-observations.md`

### 7.2 flex_attention Compilation (✓ ALIGNED)

**Blog advice** (flexattn-for-inference.md):
> "For optimal performance, compile FlexAttention using `max-autotune`"

**Scope implementation** (`src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py:491-493`):
```python
_FLEX_ATTENTION_COMPILED = torch.compile(
    _FLEX_ATTENTION_EAGER, dynamic=False, mode="max-autotune-no-cudagraphs"
)
```

Uses `max-autotune-no-cudagraphs` which includes the autotuning benefit. Also has `FLEX_ATTENTION_ALIGNMENT = 128` for padding.

### 7.3 GPU Sync Patterns (⚠️ OPPORTUNITIES)

**Blog advice** (accel-genai-pytorch-1.md):
> "aten::index causes cudaStreamSynchronize between kernels. Use torch.where to rewrite."

**Scope findings:**

| Pattern | Location | Impact |
|---------|----------|--------|
| `torch.cuda.synchronize()` | causal_model.py:409,1277,1300,1321,1345 | Profiling only (acceptable) |
| `kv_cache["..."].item()` | Multiple causal_model.py locations | **Hot path sync** |
| `.numpy()` calls | test files + causal_model.py:1678 | Test code acceptable |

**Potential optimization:**
The `.item()` calls in KV cache management (e.g., `kv_cache["global_end_index"].item()`) may cause GPU syncs in the hot path. Consider:
- Pre-computing these values where possible
- Using tensor comparisons instead of scalar comparisons

**Files with `.item()` in hot path:**
- `streamdiffusionv2/modules/causal_model.py:165,179,180,186,189,211,213,223,225`
- `longlive/modules/causal_model.py:135,249,267,271,273,279,282,289,291,351,353`
- `reward_forcing/modules/causal_model.py:194,207-228`
- `krea_realtime_video/modules/causal_model.py:835,922,985`

### 7.4 non_blocking Device Transfers (✓ GOOD PRACTICE)

**Scope implementation** (`src/scope/core/pipelines/wan2_1/modules/attention.py`):
```python
.to(q.device, non_blocking=True)
```

Uses `non_blocking=True` for device transfers, which is the recommended pattern to avoid unnecessary syncs.

### 7.5 Regional Compilation (❓ NOT YET IMPLEMENTED)

**Blog advice** (torch-compile-and-diffusers.md):
> "Regional compilation cuts compile latency from 67.4s to 9.6s (7x faster cold start)"

**Scope status:**
- Not currently using regional compilation
- DiT blocks could benefit from this pattern
- Would reduce cold start time for first frame

**Potential implementation:**
```python
# Instead of compiling entire transformer:
# pipe.transformer.compile(fullgraph=True)

# Compile repeated blocks once:
# pipe.transformer.compile_repeated_blocks(fullgraph=True)
```

### Summary: Optimization Alignment Score

| Category | Status | Notes |
|----------|--------|-------|
| Tensor core sizing (128×128) | ✓ Aligned | head_dim=128 for both models |
| flex_attention max-autotune | ✓ Aligned | Uses max-autotune-no-cudagraphs |
| non_blocking transfers | ✓ Aligned | attention.py uses non_blocking=True |
| GPU sync elimination | ⚠️ Opportunity | `.item()` calls in hot path |
| Regional compilation | ❓ Not implemented | Could reduce cold start |
| ThunderKittens kernels | ❓ Not evaluated | Alternative to FA4/cuDNN |

---

## TODO queue (web search gaps to fill)

1) A primary-source issue/release note that explicitly ties:
   - Triton + Blackwell tcgen05 + LLVM failure
   - recommended fix / version where it’s fixed
2) cuDNN 9.13 “Blackwell + 3D conv” bullets from release notes
3) CUDA toolkit version that first introduced:
   - `ptxas` support for `sm_103`
   - any `*a` compatibility mode for Blackwell (if applicable)
4) torchao official compatibility statement for torch 2.9 (not just issues)
