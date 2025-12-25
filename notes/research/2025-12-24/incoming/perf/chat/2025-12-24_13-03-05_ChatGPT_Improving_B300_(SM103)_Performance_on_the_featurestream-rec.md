# Improving B300 (SM103) Performance on the feature/stream-recording Branch

> **Disclaimer:** This is an LLM-generated document (ChatGPT). Recommendations should be verified
> against the actual codebase. Some information about SM100/SM103 compatibility is drawn from
> NVIDIA docs and may be accurate but generic—always test on real hardware.
>
> **Quality Assessment:** ⚠️ Mixed. User provided excellent structured context (`<task>`,
> `<architecture>`, `<selected_context>`). SM103 compatibility info is accurate but general.
> The FA4 score_mod implementation plan references real file paths.
>
> **Key Finding (Verified):** B300 is SM103 (CC 10.3), B200 is SM100 (CC 10.0). Both are Blackwell
> family but some architecture-specific builds (`compute_100a`) may not work across both.

Source: [ChatGPT](https://chatgpt.com/g/g-p-6943a1900f6481918b389cc8e6c966ce/c/694a597e-ba8c-8332-a7e4-23391a3c44e2)
Extracted: 2025-12-24T21:03:05.837Z
🚀 Exported with [Save my Chatbot](https://save.hugocolin.com)!

---

## User

<taskname="FA4 score_mod"/>
<task>Prepare context to implement Kernel-B piecewise bias as a FlashAttention-4 score_mod (FA4/CUTE) on the WAN2.1 attention path, gated by a feature flag, and keep existing triton/flex_attention fallbacks for kv_cache_attention_bias. Prefer constant-closure for frame_seqlen/current_block_start/log_bias to avoid aux_tensors and preserve vec_size=2; inference-only so no bwd score_mod needed. FA4 score_mod is incompatible with softcap.</task>
<architecture>
- WAN2.1 attention wrapper selects FA4/FA3/FA2/Sage backends and exposes attention() used across multiple pipelines.
- KreaRealtimeVideo CausalWanSelfAttention handles KV-cache bias: triton Kernel B when enabled, otherwise flex_attention with score_mod; bias disabled (1.0) routes to attention().
- Kernel B piecewise bias rule: no bias for first frame and current block; apply log_bias to past-frame tokens.
- kv_cache_attention_bias is set in pipeline state and passed through generator -> model -> blocks -> self_attn.
</architecture>
<selected_context>
src/scope/core/pipelines/wan2_1/modules/attention.py: flash_attention()/attention() backend selection (FA4/FA3/FA2/Sage), varlen handling, environment gating.
src/scope/core/kernels/triton_attention.py: Triton Kernel B implementation of piecewise bias (log_bias on past frames).
src/scope/core/kernels/__init__.py: exports triton_kernel_b used by CausalWanSelfAttention.
src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py: CausalWanSelfAttention bias path (triton Kernel B + flex_attention score_mod), attention() fallback, and CausalWanModel._forward_inference wiring of kv_cache_attention_bias.
src/scope/core/pipelines/krea_realtime_video/pipeline.py: default kv_cache_attention_bias and warmup behavior.
src/scope/core/pipelines/wan2_1/components/generator.py: passes kv_cache_attention_bias into CausalWanModel.
src/scope/core/pipelines/wan2_1/blocks/denoise.py: forwards kv_cache_attention_bias from pipeline state into generator.
</selected_context>
<relationships>
- DenoiseBlock -> WanDiffusionWrapper.forward -> CausalWanModel._forward_inference -> CausalWanAttentionBlock.self_attn -> CausalWanSelfAttention KV-cache bias path.
- CausalWanSelfAttention uses triton_kernel_b if USE_TRITON_KERNEL_B=1 else flex_attention score_mod; bias disabled routes to attention() (FA/Sage).
- attention() in wan2_1/modules/attention.py is imported by krea_realtime_video, longlive, reward_forcing, streamdiffusionv2; changes affect all users of this wrapper.
</relationships>
<ambiguities>
- Decision point: implement FA4 score_mod using constant-closure (preferred) vs aux_tensors if values must be dynamic.
- Need to pick a feature flag location (env var in attention.py vs in CausalWanSelfAttention) to gate FA4 score_mod usage.
- FA4 score_mod + softcap are mutually exclusive; verify whether any softcap path exists before wiring.
- q_idx/kv_idx are local in score_mod; current sequences appear fixed/padded, so offsets likely unnecessary unless varlen is introduced.
</ambiguities>

## Message

###### ChatGPT said:

## User

# Phase 4: FA4/CuTe DSL `score_mod` for KV-cache Bias (Kernel B)

## Why this exists

We already have a strong B200 path for Krea Realtime self-attention **KV-cache bias** (“Kernel B”) via a custom Triton kernel.

The next obvious “maybe bigger” win is to see whether **FlashAttention-4 (CuTe DSL)** can run the same bias path faster (or more simply) by using a custom `score_mod` that is inlined into FA4’s SM100 kernel.

This doc is a plan + integration sketch so other models/people can critique it quickly.

## Current repo status (2025-12-23)

### What’s working
- **RoPE Phase 3 Step 2 (v2 fused)** is complete and the default; perf is back to ~20 FPS on B200. See `notes/FA4/phase3-triton-rope.md`.
- **Kernel B Triton** is integrated in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` and beats flex_attention on the real shape.

### What’s missing for “CuTe score_mod”
- The installed wheel `flash-attn==2.8.3` exposes `flash_attn.cute.flash_attn_varlen_func(...)`, but **it does not accept `score_mod`** (signature check confirms this).
- We *do* have a newer FA4 CuTe implementation with `score_mod` support in `flash-attention.bak/flash_attn/cute/interface.py`.
- `src/scope/core/pipelines/wan2_1/modules/attention.py` has an `_extend_flash_attn_path()` hook that will prefer a local `flash-attention/flash_attn` tree if present — but the repo currently has `flash-attention.bak/`, not `flash-attention/`.

**Implication:** before we can even prototype FA4 `score_mod`, we must make sure the runtime imports a `flash_attn.cute.flash_attn_varlen_func` that supports it (either by switching to a newer wheel, or by using the local `flash-attention.bak` code via the path hook).

## Goal (Phase 4)

Implement and benchmark a FA4/CuTe DSL `score_mod` that matches Krea’s KV-cache bias rule, and optionally integrate it behind a feature flag:

Backend order goal:
1) FA4 `score_mod` (if available + faster)
2) Triton Kernel B (current default)
3) flex_attention fallback

We only flip defaults if FA4 clearly wins and is stable.

## KV-cache bias semantics (what we must exactly reproduce)

In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` the flex_attention fallback currently applies:

- `log_bias = log(kv_cache_attention_bias)` (natural log)
- Bias applies for keys where:
 - exclude first frame: `kv_idx >= frame_seqlen`
 - exclude current semantic block: `kv_idx < cache_current_block_start`

Where:
- `frame_seqlen = (height//16) * (width//16)`
- `block_size = frame_seqlen * num_frame_per_block`
- `cache_current_block_start = Lk - block_size` (in token units)

So biased region is `kv_idx ∈ [frame_seqlen, Lk - block_size)`.

## Key insight from CuTe DSL score_mod guide

From `notes/FA4/DeepResearch/2025-12-23/CuTeDSL_score_mod.md` + FA4 code:

- `score_mod` is called with a `seqlen_info` object that exposes **runtime** `seqlen_k`.
- Passing `aux_tensors` usually forces `vec_size=1` (less vectorization), so we want **zero aux tensors**.

**Therefore:** we can avoid recompiling per `Lk` and avoid aux tensors by computing `cache_current_block_start` inside the score_mod from `seqlen_info.seqlen_k`.

## Proposed score_mod (in CuTe DSL)

We want an inference-only score modifier:

```
@cute.jit
def score_mod_kv_bias(scores, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 Lk = seqlen_info.seqlen_k
 current_block_start = Lk - block_size_tokens
 mask = (kv_idx >= frame_seqlen_tokens) & (kv_idx < current_block_start)
 return scores + where(mask, log_bias, 0)
```

Notes:
- `scores` passed to `score_mod` are already multiplied by `softmax_scale` inside FA4 (see `apply_score_mod_inner` in `flash-attention.bak/flash_attn/cute/softmax.py`), so adding a natural-log bias is correct.
- `block_size_tokens = frame_seqlen * num_frame_per_block` is constant for a given pipeline resolution/config → closure constant.
- `log_bias` is constant for a run/config → closure constant.
- We must handle early frames where `Lk < block_size_tokens` by clamping:
 - `current_block_start = max(0, Lk - block_size_tokens)`

## Implementation plan

### Step 0 — Make `score_mod` importable (plumbing)

We need `flash_attn.cute.flash_attn_varlen_func` to accept `score_mod=...` (and maybe `aux_tensors=...`).

Two options:

**Option 0A (preferred for iteration): use local `flash-attention.bak` code**
- Create `flash-attention/` (symlink to `flash-attention.bak/`) so `_extend_flash_attn_path()` in `src/scope/core/pipelines/wan2_1/modules/attention.py` finds it.
- In a fresh process, verify:
 - `import inspect; from flash_attn.cute import flash_attn_varlen_func; inspect.signature(...)` includes `score_mod`.
- Keep this behind an env guard if you want safety (e.g., `SCOPE_USE_LOCAL_FLASH_ATTN_CUTE=1`), but the directory is already `.gitignore`’d.

**Option 0B: upgrade/replace the wheel**
- Install a flash-attn build that exposes the score_mod CuTe interface.
- Higher risk (binary build/ABI) and slower iteration.

### Step 1 — Add a FA4 score_mod entrypoint in Scope

Touchpoint: `src/scope/core/pipelines/wan2_1/modules/attention.py`

- Add a function (or extend `flash_attention`) to accept:
 - `score_mod: Optional[Callable]`
 - `aux_tensors: Optional[list[torch.Tensor]]`
- Only attempt FA4 `score_mod` if:
 - `FLASH_ATTN_4_AVAILABLE` is True
 - the imported `flash_attn_4_varlen_func` signature supports the kwargs (guard with `inspect.signature` or `try/except TypeError`).
- If unsupported, fall back to:
 - Triton Kernel B (preferred), else flex_attention (existing).

### Step 2 — Implement the CuTe `score_mod` for KV bias

Touchpoint: new helper module, recommended:
- `src/scope/core/pipelines/krea_realtime_video/modules/cute_score_mod.py`

Responsibilities:
- Build and cache the `@cute.jit` callable.
- Avoid aux_tensors.
- Capture only constants that should specialize the kernel:
 - `frame_seqlen_tokens` (int)
 - `block_size_tokens` (int)
 - `log_bias` (float32)

Cache key suggestion:
`(frame_seqlen_tokens, block_size_tokens, round(log_bias, 6))`

### Step 3 — Microbench + correctness harness

We need an apples-to-apples benchmark on the true shape:
- `B=1, H=16, D=128`
- Typical: `Lq=4680`, `Lk=9360` (3-frame query, 6-frame KV)

Recommended new script:
- `scripts/bench_fa4_score_mod.py` (or extend `scripts/triton_sdpa.py --kernel-b` with a `--fa4-score-mod` mode)

Comparisons:
1) FA4 `score_mod` (new)
2) Triton Kernel B (current)
3) flex_attention `score_mod` (baseline)

Correctness:
- Compare outputs vs flex_attention reference for small and medium shapes (bf16 tolerance).
- Include “early cache” cases: `Lk` in `{frame_seqlen, 2*frame_seqlen, …, 6*frame_seqlen}`.

Perf:
- Warm up first-call JIT (CuTe compiles on first use).
- Report steady-state average over many iters.

### Step 4 — Integrate into Krea pipeline behind a flag

Touchpoint: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

Add a new backend selector, suggested env var:
- `SCOPE_KV_BIAS_BACKEND=triton|fa4|flex` (default `triton`)

Behavior:
- If `fa4`: attempt FA4 score_mod; if it errors, log once and fall back to `triton` for the rest of the process (“trip breaker”).
- If `triton`: current behavior.

### Step 5 — Nsight validation (optional but recommended)

Once FA4 score_mod runs:
- Use NCU to confirm kernel is on SM100 FA4 path and check occupancy/reg pressure.
- Use NSYS to ensure there aren’t unexpected CPU-side overheads (e.g., repeated compilation).

## How hard is this? (risk / effort)

**Core score_mod logic:** low complexity (hours).

**Plumbing + deps:** medium to high complexity (days), because:
- Our current installed FA4 API doesn’t expose score_mod.
- CuTe DSL versions and signatures vary; we need robust guards.
- `nvidia-cutlass-dsl` can conflict with `torch._inductor` in some environments (see `notes/TODO-next-session.md`), so we must keep fallbacks and avoid destabilizing the main pipeline.

**Performance outcome uncertainty:** medium.
- We might get a win if FA4’s SM100 kernel handles this shape better than our Triton implementation.
- But we should expect “maybe marginal” because Kernel B is already good and appears occupancy-limited in profiling.

## Acceptance criteria

- **Correctness:** Matches flex_attention reference within bf16 tolerance across early + steady-state `Lk` values.
- **No silent fallback:** logs clearly indicate which backend is used.
- **Performance:** FA4 score_mod is faster than Triton Kernel B on the real shape, or we keep Triton as default.
- **Stability:** no regressions in non-score_mod paths; failures trip-break back to Triton.

## RepoPrompt bundle (for external review)

**Files to include**
- `src/scope/core/pipelines/wan2_1/modules/attention.py`
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- `scripts/triton_sdpa.py`
- `scripts/tune_kernel_b.py`
- `notes/FA4/DeepResearch/2025-12-23/CuTeDSL_score_mod.md`
- `flash-attention.bak/flash_attn/cute/interface.py`
- `flash-attention.bak/flash_attn/cute/softmax.py`
- `flash-attention.bak/tests/cute/score_mod_definitions.py`

**Query (paste into RepoPrompt)**

Goal: implement FA4/CuTe DSL `score_mod` for Krea KV-cache bias attention (Kernel B), and benchmark it against Triton Kernel B.

1) Confirm why our current runtime FA4 (`flash-attn==2.8.3`) doesn’t expose `score_mod`, and the cleanest way to enable the newer `flash-attention.bak/flash_attn/cute/interface.py` implementation via Scope’s `_extend_flash_attn_path()` hook.
2) Design the CuTe `score_mod` to reproduce Scope’s bias rule:
 - bias keys where `kv_idx >= frame_seqlen` and `kv_idx < seqlen_info.seqlen_k - block_size_tokens`
 - add `log_bias = log(kv_cache_attention_bias)` (natural log)
 - avoid `aux_tensors` to keep `vec_size=2`
3) Propose exact code changes in:
 - `src/scope/core/pipelines/wan2_1/modules/attention.py` (FA4 score_mod call + guards)
 - `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (backend selection + fallback)
4) Propose a microbench harness (new script or extend `scripts/triton_sdpa.py`) that compares:
 - FA4 score_mod vs Triton Kernel B vs flex_attention score_mod
 using the real shape: `B=1,H=16,D=128,Lq=4680,Lk=9360` and several early-cache `Lk` values.

## ChatGPT

Got it—I’m on it and have updated the task accordingly.

## User

For everything we're doing, what are the chances that we could adapt it to the B300, right? Which is SM what exactly? things.

## ChatGPT

B300 is **compute capability 10.3**, which in CUDA toolchain naming is **SM103** (you’ll see it as `sm_103`). B200 is **compute capability 10.0**, **SM100** (`sm_100`). [NVIDIA Developer](https://developer.nvidia.com/cuda/gpus)

### So can we adapt “everything we’re doing” from B200 to B300?

High odds, with one important caveat.

#### What should “just work” (very likely)

**Your semantics and plumbing** (feature flags, bias rule, fallbacks) are GPU-agnostic.

**Triton Kernel B** should be fine. Triton typically emits PTX and the driver JITs for the resident GPU, so you naturally recompile to SM103 on first run.

The **score_mod constant-closure idea** (no aux_tensors, compute `current_block_start` from `seqlen_info.seqlen_k`) is also architecture-agnostic.

If you stick to baseline or “family” targets, CUDA explicitly treats 10.0 and 10.3 as the same Blackwell family for compatibility purposes. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

#### The main risk

**Prebuilt FA4 / CuTe binaries and “architecture-specific” builds.**

Blackwell introduces the notion that some “architecture-specific” targets (the ones with an **`a` suffix**, like `compute_100a` / `sm_100a`) are *not* compatible across minor versions. Code built for `compute_100a` is intended to run only on CC 10.0 devices, not 10.3. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

So your success on B300 depends on how FA4 in your environment is packaged:

If the FA4 wheel includes **PTX** or a **family-compatible** build (for example `compute_100f` / `sm_100f` style compatibility), it should run across **10.0 and 10.3**. [NVIDIA Docs+1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

If it is built with **only SM100a cubins** and no compatible PTX, it may fail on B300 and you’d need a rebuild that includes SM103 or family/ptx coverage.

### Practical “do we have a problem?” checks

On a B300 box:

Confirm device CC:

`torch.cuda.get_device_capability()` should return `(10, 3)`.

Force PTX JIT to detect missing PTX in a binary package:

Set `CUDA_FORCE_PTX_JIT=1` and run your minimal FA4 attention call. If it breaks, the binary likely lacks PTX coverage for forward compatibility. [NVIDIA Docs](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)

For your CuTe score_mod specifically:

Do the same runtime smoke test you planned (signature includes `score_mod`, and an actual `score_mod` changes output).

### Bottom line

**SM for B300:** **SM103**, CC **10.3**. [NVIDIA Developer](https://developer.nvidia.com/cuda/gpus)

**Portability odds:** **very high** for the Triton path and for the high-level design; **medium-to-high** for FA4 score_mod depending on whether your FA4 build is packaged in a way that supports **10.3** (PTX or family-compatible cubins), versus being locked to **10.0a**.

If you tell me which exact FA4 build you’re using on B200 (wheel name or commit hash of your `flash-attention.bak` tree, plus CUDA toolkit version), I can point to the most likely compatibility mode and what to change in your build flags to cover both SM100 and SM103.

## User

So, how hard will it be to do a rebuild, do you think?

## Message

###### ChatGPT said:

Something went wrong.

## User

We’re on CUDA 12.8 (PyTorch 2.8.0+cu128, Python 3.12). FA4/CUTE is coming from a local flash-attention repo at commit
 eacbc560be4811b40dee21c4449ab226d40a2edc (/root/scope/flash-attention.bak). For pip flash-attn (FA2/FA3), the wheel in use is flash_attn-
 2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl as pinned in pyproject.toml.

## ChatGPT

Got it – I’ve updated the context with your CUDA/PyTorch/FA4 commit details.

## User

Please try again

## Message

###### ChatGPT said:

## User

What actually blocks B300 today

 - PyTorch flex_attention compile on SM103 fails (NoValidChoicesError), because torch.compile(flex_attention, mode="max-autotune-no-
 cudagraphs") happens at import time and SM103 kernels aren’t there yet. This is independent of FA4.
 - FA4/CUTE on SM103 works only after patching nvidia-cutlass-dsl to accept sm_103 (scripts/patch_cutlass_sm103.py). But those deps conflict
 with PyTorch Inductor’s cutlass module, so FA4 must be isolated/uninstalled for normal runs.
 - Local flash-attention repo injection can override pip FA4 and change behavior. You already symlinked it to .bak, so this is under control.

 What we need for “B200 and B300” flexibility

 - Avoid compiling flex_attention at import time on SM103. Gate it on capability or disable for SM103; this removes the immediate B300
 failure.
 - Keep FA4 separate from normal runs (current state): either dedicated env or install/uninstall flow. Coexistence is broken due to cutlass
 module shadowing.
 - Decide on FA4 for B300: either rebuild/patch for SM103 (works for benchmarks) or skip FA4 on SM103 until upstream support stabilizes.

 Concrete changes to consider

 - Runtime dispatch: if torch.cuda.get_device_capability() == (10, 3):
 - Skip torch.compile for flex_attention.
 - Optionally set kv_cache_attention_bias=1.0 to avoid score_mod path if needed.
 - FA4 gating: only enable FA4 when nvidia-cutlass-dsl is patched and we’re in a “FA4-only” env.

 Questions to unblock decisions

 1. Do you want a single environment that runs on both B200/B300 (no FA4), or are you okay with a FA4-only env for benchmarks?
 2. For B300, is it acceptable to disable torch.compile for flex_attention (and take perf hit) until PyTorch adds SM103 kernels?
 3. Are you willing to rebuild FA4 with SM103 targets (and accept the conflict with Inductor), or should FA4 be B200-only for now?

## ChatGPT

Got it – I’m on it and have updated the context accordingly.

## User

Also <this>Summary: FA4 Journey & B300 Adaptation Analysis

 Current State (B200/SM100) - What's Working

 | Component | Status | Performance |
 |-----------------------------------------|------------------|-----------------------------------|
 | Kernel B (KV-bias) - FA4/CUTE score_mod | ✅ Opt-in | 0.54ms (1.89x faster than Triton) |
 | Kernel B (KV-bias) - Triton | ✅ Default | 1.02ms |
 | RoPE v2 fused | ✅ Default | 0.069ms (2.2x faster than Step 0) |
 | Kernel A (block_mask) | ⏸️ Deprioritized | Only 3.4% of self_attn time |
 | End-to-end FPS | | 20.2-21.4 FPS |

 Key Dependencies at Risk for SM103

 1. FA4/CUTE DSL — HIGH RISK

 From b300-investigation.md:62-80, the CUTE kernels target sm_100a/sm_100f specifically. The fix requires patching:
 - cute/arch/mbar.py, cute/arch/elect.py
 - cute/nvgpu/tcgen05/mma.py, cute/nvgpu/tcgen05/copy.py, cute/nvgpu/cpasync/copy.py

 Question: Does the existing scripts/patch_cutlass_sm103.py cover the score_mod path used in Phase 4?

 2. PyTorch flex_attention — KNOWN BROKEN on SM103

 From b300-investigation.md:110-125:
 # This works on B200, FAILS on B300:
 flex_attention = torch.compile(_flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs')
 Workaround: Skip torch.compile on SM103 (see proposed fix at lines 236-260).

 3. Triton Kernels — LIKELY OK, MAY NEED RETUNING

 Triton emits PTX that JITs for the target GPU. Your Kernel B and RoPE v2 should work, but:
 - Block sizes tuned for B200 (BLOCK_M=64, BLOCK_N=64, warps=8, stages=2) may not be optimal for B300
 - Different L2 cache size, memory bandwidth, SM count

 4. nvidia-cutlass-dsl ↔ torch._inductor conflict — MEDIUM RISK

 From b300-investigation.md:88-103:
 - FA4 deps install nvidia-cutlass-dsl which shadows PyTorch's internal cutlass module
 - On SM100 this coexists; on SM103 it caused NoValidChoicesError

 ---
 Questions to Resolve

 Critical Path Questions

 1. FA4 Wheel vs Local flash-attention.bak:
 - Phase 4 uses flash-attention.bak/flash_attn/cute/interface.py (local, supports score_mod)
 - Is this local code SM103-compatible after patching, or only the pip wheel?
 2. CUTLASS/CUTE SM103 Coverage:
 - Does patch_cutlass_sm103.py patch ALL the files touched by _flash_attn_fwd() with score_mod?
 - Specifically: flash_fwd_sm100.py, softmax.py (apply_score_mod_inner)
 3. PTX vs Cubin in FA4 Build:
 - Does your flash-attention.bak generate PTX (portable) or SM100a cubins (architecture-specific)?
 - If cubins only → needs rebuild for SM103

 Performance Questions

 4. Triton Autotune on SM103:
 - Should we re-run scripts/tune_kernel_b.py on B300 once available?
 - Memory hierarchy differences may shift optimal block sizes
 5. RoPE v2 Kernel Stability:
 - v2 uses BLOCK_M=8, BLOCK_H=2 tuned on B200
 - Worth a quick smoke test on SM103

 Fallback Questions

 6. Backend Selection Robustness:
 - Current: SCOPE_KV_BIAS_BACKEND=fa4|triton|flex
 - If FA4 fails on SM103, does it trip-breaker to Triton correctly?
 - If flex_attention compile fails, is there a working fallback?

 ---
 Recommended Verification Steps (when B300 available)

 # 1. Check device capability
 python -c "import torch; print(torch.cuda.get_device_capability())" # Should be (10, 3)

 # 2. Test Triton Kernel B (should work)
 uv run python scripts/triton_sdpa.py --kernel-b

 # 3. Test FA4 score_mod (after patching)
 uv run python scripts/test_fa4_kv_bias.py

 # 4. Test flex_attention (expected to fail without fix)
 SCOPE_KV_BIAS_BACKEND=flex uv run python scripts/triton_sdpa.py --kernel-b

 # 5. Force PTX JIT to detect missing PTX
 CUDA_FORCE_PTX_JIT=1 uv run python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('OK')"

 ---
 My Assessment

 | Component | SM103 Risk | Likely Fix |
 |---------------------------|-------------|-------------------------------|
 | Triton Kernel B | Low | May need retune |
 | Triton RoPE v2 | Low | May need retune |
 | FA4/CUTE score_mod | Medium-High | Extend patch_cutlass_sm103.py |
 | flex_attention (compiled) | High | Skip torch.compile on SM103 |
 | FA4 → Triton fallback | Low | Already implemented |

 The safest path is: on SM103, default to SCOPE_KV_BIAS_BACKEND=triton and skip flex_attention compilation. This gives up the FA4 1.89x win but preserves functionality.</this> the context.

## ChatGPT

Got it – I’m on it and have updated the context accordingly.

## User

<info>print(torch.cuda.get_device_capability())"
(10, 3)
(daydream-scope) root@wide-earth-fades-fin-03:~/scope# uv run python scripts/triton_sdpa.py --kernel-b
warning: The `extra-build-dependencies` option is experimental and may change without warning. Pass --preview-features
extra-build-dependencies to disable this warning.

============================================================
Kernel B: Correctness Test (Piecewise Bias)
============================================================

Test: B=1, H=1, Lq=64, Lk=128, D=16
 frame_seqlen=32, current_block_start=64
 beta=0.3, log_bias=-1.2040
Traceback (most recent call last):
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 424, in make_cubin
 subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
 File "/usr/lib/python3.12/subprocess.py", line 571, in run
 raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['/root/scope/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas', '-lineinfo',
'-v', '--gpu-name=sm_103a', '/tmp/tmpcneds6n3.ptx', '-o', '/tmp/tmpcneds6n3.ptx.o']' returned non-zero exit status 255.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
 File "/root/scope/scripts/triton_sdpa.py", line 1232, in <module>
 test_kernel_b_correctness()
 File "/root/scope/scripts/triton_sdpa.py", line 1101, in test_kernel_b_correctness
 out = triton_kernel_b(Q, K, V, frame_seqlen, current_block_start, log_bias)
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/root/scope/scripts/triton_sdpa.py", line 1027, in triton_kernel_b
 kernel_b_bias_attention[grid](
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 390, in <lambda>
 return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/runtime/autotuner.py", line 251, in run
 ret = self.fn.run(
 ^^^^^^^^^^^^
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 594, in run
 kernel = self.compile(src, target=target, options=options.__dict__)
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/compiler/compiler.py", line 359, in compile
 next_module = compile_ir(module, metadata)
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 461, in <lambda>
 stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 File "/root/scope/.venv/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py", line 442, in make_cubin
 raise PTXASError(f"{error}\n"
triton.runtime.errors.PTXASError: PTXAS error: Internal Triton PTX codegen error
`ptxas` stderr:
ptxas fatal : Value 'sm_103a' is not defined for option 'gpu-name'

Repro command: /root/scope/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas -lineinfo -v --gpu-name=sm_103a
/tmp/tmpcneds6n3.ptx -o /tmp/tmpcneds6n3.ptx.o<</here>

● Very informative! This reveals a more fundamental issue than CUTLASS patching:

 Problem: Triton Doesn't Support SM103 Yet

 ptxas fatal: Value 'sm_103a' is not defined for option 'gpu-name'

 The bundled ptxas in Triton doesn't know about SM103. This is the Triton version packaged in your venv.

 Potential Fixes

 Option 1: Use System ptxas (Quick Test)

 CUDA 12.8 should have SM103 support. Try using the system ptxas instead of Triton's bundled one:

 # Find system ptxas
 which ptxas
 # or
 /usr/local/cuda/bin/ptxas --version

 # If it exists, try:
 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas uv run python scripts/triton_sdpa.py --kernel-b

 Option 2: Check Triton Version

 uv run python -c "import triton; print(triton.__version__)"

 SM103 support was likely added in a newer Triton version. The nightly or a recent release may have it.

 Option 3: Force SM100 Target (Compatibility Mode)

 Since SM103 is in the same Blackwell family as SM100, PTX compiled for SM100 might JIT to SM103:

 # Try forcing sm_100 target
 TRITON_CUDA_TARGET=sm_100 uv run python scripts/triton_sdpa.py --kernel-b

 ---
 Can you run these to gather more info?

 # 1. Check Triton version
 uv run python -c "import triton; print(triton.__version__)"

 # 2. Check system CUDA version
 nvcc --version

 # 3. Check if system ptxas knows sm_103
 /usr/local/cuda/bin/ptxas --help 2>&1 | grep -i "103\|gpu-name"

 # 4. Try system ptxas
 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas uv run python scripts/triton_sdpa.py --kernel-b

 This is actually a Triton ecosystem issue, not just our code. Both Triton Kernel B and any Triton-based FA4 would hit this same wall</info>

## ChatGPT

Got it – I’m on it and have updated the task accordingly.

## User

<file_map>
/Users/daviddickinson/Projects/Lora/scope
├── notes
│ ├── FA4
│ │ ├── DeepResearch
│ │ │ ├── 2025-12-23
│ │ │ │ ├── CuTeDSL_score_mod.md *
│ │ │ │ └── B300_01.md
│ │ │ ├── 2025-12-22
│ │ │ │ ├── MSU.md
│ │ │ │ ├── NVTX_Capture.md
│ │ │ │ ├── Phase3_01.md
│ │ │ │ ├── Phase3_02.md
│ │ │ │ ├── Phase3_03.md
│ │ │ │ └── wafer.md
│ │ │ ├── MSU_chat.md
│ │ │ ├── summary.md
│ │ │ └── wafer.md
│ │ ├── B300-FA4-PATCHES.md *
│ │ ├── b300-investigation.md *
│ │ ├── B300-SESSION-STATE.md *
│ │ ├── B300-SETUP-GUIDE.md *
│ │ ├── kernel-dev-log.md *
│ │ ├── phase4-fa4-score-mod.md *
│ │ ├── b200-bringup-log.md
│ │ ├── phase3-triton-rope-step2.md
│ │ ├── phase3-triton-rope-step3.md
│ │ ├── phase3-triton-rope.md
│ │ └── rope-optimization.md
│ ├── TODO-next-session.md *
│ ├── 5pro_chat_01.md
│ ├── 5pro_chat_02.md
│ ├── input-mode-fix.md
│ ├── krea_rt.md
│ ├── krea-offline-render-tuning.md
│ ├── offline-timeline-renderer-report.md
│ ├── offline-timeline-renderer.md
│ └── render-timeline-guide.md
├── scripts
│ ├── patch_cutlass_sm103.py * +
│ ├── test_fa4_kv_bias.py * +
│ ├── bench_blockwise_attn.py +
│ ├── bench_triton_rope.py +
│ ├── patch_cutlass_sm103.sh
│ ├── profile_attention.py +
│ ├── test_rope_correctness.py +
│ ├── triton_sdpa.py +
│ └── tune_kernel_b.py +
├── src
│ └── scope
│ ├── core
│ │ ├── compat
│ │ │ ├── sm103.py * +
│ │ │ └── __init__.py +
│ │ ├── pipelines
│ │ │ ├── krea_realtime_video
│ │ │ │ ├── modules
│ │ │ │ │ ├── causal_model.py * +
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── model.py +
│ │ │ │ │ ├── vae_block3.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── prepare_context_frames.py +
│ │ │ │ │ └── recompute_kv_cache.py +
│ │ │ │ ├── components
│ │ │ │ │ └── __init__.py
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── __init__.py +
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ └── test.py +
│ │ │ ├── wan2_1
│ │ │ │ ├── modules
│ │ │ │ │ ├── attention.py * +
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── clip.py +
│ │ │ │ │ ├── sage.py +
│ │ │ │ │ ├── t5.py +
│ │ │ │ │ ├── tokenizers.py +
│ │ │ │ │ └── xlm_roberta.py +
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── auto_prepare_latents.py +
│ │ │ │ │ ├── clean_kv_cache.py +
│ │ │ │ │ ├── decode.py +
│ │ │ │ │ ├── denoise.py +
│ │ │ │ │ ├── embedding_blending.py +
│ │ │ │ │ ├── noise_scale_controller.py +
│ │ │ │ │ ├── prepare_latents.py +
│ │ │ │ │ ├── prepare_next.py +
│ │ │ │ │ ├── prepare_video_latents.py +
│ │ │ │ │ ├── preprocess_video.py +
│ │ │ │ │ ├── set_timesteps.py +
│ │ │ │ │ ├── set_transformer_blocks_local_attn_size.py +
│ │ │ │ │ ├── setup_caches.py +
│ │ │ │ │ └── text_conditioning.py +
│ │ │ │ ├── components
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── generator.py +
│ │ │ │ │ ├── scheduler.py +
│ │ │ │ │ └── text_encoder.py +
│ │ │ │ ├── lora
│ │ │ │ │ ├── strategies
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ ├── module_targeted_lora.py +
│ │ │ │ │ │ ├── peft_lora.py +
│ │ │ │ │ │ └── permanent_merge_lora.py +
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── manager.py +
│ │ │ │ │ ├── mixin.py +
│ │ │ │ │ └── utils.py +
│ │ │ │ ├── vace
│ │ │ │ │ ├── blocks
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ └── vace_encoding.py +
│ │ │ │ │ ├── models
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ ├── attention_blocks.py +
│ │ │ │ │ │ └── causal_vace_model.py +
│ │ │ │ │ ├── utils
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ ├── encoding.py +
│ │ │ │ │ │ └── weight_loader.py +
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── design.md
│ │ │ │ │ └── mixin.py +
│ │ │ │ ├── vae
│ │ │ │ │ ├── modules
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ └── vae.py +
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── constants.py +
│ │ │ │ │ └── wan.py +
│ │ │ │ ├── __init__.py
│ │ │ │ └── utils.py +
│ │ │ ├── base
│ │ │ │ └── __init__.py
│ │ │ ├── longlive
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── prepare_recache_frames.py +
│ │ │ │ │ └── recache_frames.py +
│ │ │ │ ├── components
│ │ │ │ │ └── __init__.py
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── causal_model.py +
│ │ │ │ │ ├── model.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── __init__.py +
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ ├── test_vace.py +
│ │ │ │ └── test.py +
│ │ │ ├── passthrough
│ │ │ │ ├── __init__.py +
│ │ │ │ └── pipeline.py +
│ │ │ ├── reward_forcing
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ └── prepare_next.py +
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── causal_model.py +
│ │ │ │ │ └── model.py +
│ │ │ │ ├── __init__.py
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ └── test.py +
│ │ │ ├── streamdiffusionv2
│ │ │ │ ├── components
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── causal_model.py +
│ │ │ │ │ ├── model.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── __init__.py +
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ ├── test_r2v.py +
│ │ │ │ └── test.py +
│ │ │ ├── __init__.py +
│ │ │ ├── blending.py +
│ │ │ ├── components.py +
│ │ │ ├── defaults.py +
│ │ │ ├── helpers.py +
│ │ │ ├── interface.py +
│ │ │ ├── memory.py +
│ │ │ ├── process.py +
│ │ │ ├── registry.py +
│ │ │ ├── schema.py +
│ │ │ ├── utils.py +
│ │ │ └── video.py +
│ │ ├── kernels
│ │ │ ├── __init__.py +
│ │ │ ├── triton_attention.py +
│ │ │ ├── triton_rope_fused.py +
│ │ │ └── triton_rotary.py +
│ │ ├── __init__.py
│ │ └── config.py +
│ ├── cli
│ │ ├── __init__.py
│ │ └── render_timeline.py +
│ ├── server
│ │ ├── spout
│ │ │ ├── __init__.py +
│ │ │ ├── receiver.py +
│ │ │ └── sender.py +
│ │ ├── __init__.py
│ │ ├── app.py +
│ │ ├── artifacts.py +
│ │ ├── build.py +
│ │ ├── credentials.py +
│ │ ├── download_models.py +
│ │ ├── download_progress_manager.py +
│ │ ├── frame_processor.py +
│ │ ├── logs_config.py +
│ │ ├── models_config.py +
│ │ ├── pipeline_artifacts.py +
│ │ ├── pipeline_manager.py +
│ │ ├── schema.py +
│ │ ├── tracks.py +
│ │ └── webrtc.py +
│ └── __init__.py
├── vendored
│ ├── cutlass-cute
│ │ ├── python
│ │ │ ├── CuTeDSL
│ │ │ │ ├── ampere
│ │ │ │ │ ├── elementwise_add.py * +
│ │ │ │ │ ├── call_bypass_dlpack.py +
│ │ │ │ │ ├── call_from_jit.py +
│ │ │ │ │ ├── dynamic_smem_size.py +
│ │ │ │ │ ├── elementwise_apply.py +
│ │ │ │ │ ├── flash_attention_v2.py +
│ │ │ │ │ ├── hstu_attention.py +
│ │ │ │ │ ├── inline_ptx.py +
│ │ │ │ │ ├── sgemm.py +
│ │ │ │ │ ├── smem_allocator.py +
│ │ │ │ │ └── tensorop_gemm.py +
│ │ │ │ ├── cute
│ │ │ │ │ ├── tvm_ffi
│ │ │ │ │ │ ├── jit_and_use_in_torch.py * +
│ │ │ │ │ │ ├── ampere_gemm_with_fake_tensor.py +
│ │ │ │ │ │ ├── aot_export.py +
│ │ │ │ │ │ ├── aot_use_in_cpp_bundle.cpp +
│ │ │ │ │ │ ├── aot_use_in_cpp_bundle.sh
│ │ │ │ │ │ ├── aot_use_in_jax.py +
│ │ │ │ │ │ ├── aot_use_in_torch.py +
│ │ │ │ │ │ ├── compile_with_fake_tensor.py +
│ │ │ │ │ │ ├── error_reporting.py +
│ │ │ │ │ │ ├── jit_and_use_in_jax.py +
│ │ │ │ │ │ └── requirements.txt
│ │ │ │ │ ├── ffi
│ │ │ │ │ │ ├── CMakeLists.txt
│ │ │ │ │ │ ├── jit_argument.py +
│ │ │ │ │ │ └── tensor.cpp +
│ │ │ │ │ ├── print_latex.py +
│ │ │ │ │ └── torch_fake_tensor.py +
│ │ │ │ ├── blackwell
│ │ │ │ │ ├── blockwise_gemm
│ │ │ │ │ │ ├── blockwise_gemm.py +
│ │ │ │ │ │ ├── contiguous_grouped_gemm.py +
│ │ │ │ │ │ └── masked_grouped_gemm.py +
│ │ │ │ │ ├── mamba2_ssd
│ │ │ │ │ │ ├── mamba2_ssd_reference.py +
│ │ │ │ │ │ ├── mamba2_ssd_tile_scheduler.py +
│ │ │ │ │ │ └── mamba2_ssd.py +
│ │ │ │ │ ├── mixed_input_fmha
│ │ │ │ │ │ ├── mixed_input_fmha_decode.py +
│ │ │ │ │ │ └── mixed_input_fmha_prefill.py +
│ │ │ │ │ ├── tutorial_gemm
│ │ │ │ │ │ ├── fp16_gemm_0.py +
│ │ │ │ │ │ ├── fp16_gemm_1.py +
│ │ │ │ │ │ └── README.md
│ │ │ │ │ ├── dense_blockscaled_gemm_persistent_prefetch.py +
│ │ │ │ │ ├── dense_blockscaled_gemm_persistent.py +
│ │ │ │ │ ├── dense_gemm_alpha_beta_persistent.py +
│ │ │ │ │ ├── dense_gemm_persistent_prefetch.py +
│ │ │ │ │ ├── dense_gemm_persistent.py +
│ │ │ │ │ ├── dense_gemm_software_pipeline.py +
│ │ │ │ │ ├── dense_gemm.py +
│ │ │ │ │ ├── fmha_bwd.py +
│ │ │ │ │ ├── fmha.py +
│ │ │ │ │ ├── grouped_blockscaled_gemm.py +
│ │ │ │ │ ├── grouped_gemm.py +
│ │ │ │ │ ├── grouped_mixed_input_gemm.py +
│ │ │ │ │ ├── mixed_input_gemm.py +
│ │ │ │ │ ├── mla.py +
│ │ │ │ │ └── programmatic_dependent_launch.py +
│ │ │ │ ├── blackwell_geforce
│ │ │ │ │ └── dense_gemm.py +
│ │ │ │ ├── distributed
│ │ │ │ │ ├── all_reduce_one_shot_lamport.py +
│ │ │ │ │ ├── all_reduce_simple.py +
│ │ │ │ │ ├── all_reduce_two_shot_multimem.py +
│ │ │ │ │ ├── distributed_all_gather_gemm_blackwell.py +
│ │ │ │ │ ├── distributed_gemm_all_reduce_blackwell.py +
│ │ │ │ │ ├── distributed_gemm_reduce_scatter_blackwell.py +
│ │ │ │ │ └── README.md
│ │ │ │ ├── helpers
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── fmha_helpers.py +
│ │ │ │ │ ├── sparse_utils.py +
│ │ │ │ │ └── test_sparse_utils.py +
│ │ │ │ ├── hopper
│ │ │ │ │ ├── dense_gemm_persistent.py +
│ │ │ │ │ ├── dense_gemm.py +
│ │ │ │ │ └── fmha.py +
│ │ │ │ ├── notebooks
│ │ │ │ │ ├── images
│ │ │ │ │ │ ├── blocked_gemm.svg
│ │ │ │ │ │ ├── cuda_graphs_image.png
│ │ │ │ │ │ ├── software_pipelining_ab_stages_minus_1.svg
│ │ │ │ │ │ └── software_pipelining_ab_stages_minus_2.svg
│ │ │ │ │ ├── async_pipeline.ipynb
│ │ │ │ │ ├── benchmark_autotune.ipynb
│ │ │ │ │ ├── composed_layout.ipynb
│ │ │ │ │ ├── cuda_graphs.ipynb
│ │ │ │ │ ├── cute_layout_algebra.ipynb
│ │ │ │ │ ├── data_types.ipynb
│ │ │ │ │ ├── elementwise_add.ipynb
│ │ │ │ │ ├── hello_world.ipynb
│ │ │ │ │ ├── print.ipynb
│ │ │ │ │ ├── README.md
│ │ │ │ │ ├── tensor.ipynb
│ │ │ │ │ ├── tensorssa.ipynb
│ │ │ │ │ └── tour_to_sol_gemm.ipynb
│ │ │ │ └── utils
│ │ │ │ ├── __init__.py
│ │ │ │ ├── fmha_helpers.py +
│ │ │ │ ├── sparse_utils.py +
│ │ │ │ └── test_sparse_utils.py +
│ │ │ └── deprecated
│ │ │ ├── 00_basic_gemm.ipynb
│ │ │ ├── 01_epilogue.ipynb
│ │ │ ├── 02_pytorch_extension_grouped_gemm.ipynb
│ │ │ ├── 03_basic_conv2d.ipynb
│ │ │ ├── 04_epilogue_visitor.ipynb
│ │ │ └── README.md
│ │ └── cute
│ │ ├── tutorial
│ │ │ ├── blackwell
│ │ │ │ ├── 01_mma_sm100.cu
│ │ │ │ ├── 02_mma_tma_sm100.cu
│ │ │ │ ├── 03_mma_tma_multicast_sm100.cu
│ │ │ │ ├── 04_mma_tma_2sm_sm100.cu
│ │ │ │ ├── 05_mma_tma_epi_sm100.cu
│ │ │ │ ├── CMakeLists.txt
│ │ │ │ └── example_utils.hpp
│ │ │ ├── hopper
│ │ │ │ ├── CMakeLists.txt
│ │ │ │ ├── wgmma_sm90.cu
│ │ │ │ └── wgmma_tma_sm90.cu
│ │ │ ├── CMakeLists.txt
│ │ │ ├── sgemm_1.cu
│ │ │ ├── sgemm_2.cu
│ │ │ ├── sgemm_sm70.cu
│ │ │ ├── sgemm_sm80.cu
│ │ │ ├── tiled_copy_if.cu
│ │ │ └── tiled_copy.cu
│ │ └── CMakeLists.txt
│ └── rope
│ ├── flash_attn_triton_rotary.py +
│ ├── README.md
│ ├── transformers_modeling_llama.py +
│ ├── vllm_rotary_embedding_base.py +
│ └── vllm_rotary_embedding_common.py +
├── .github
│ ├── ISSUE_TEMPLATE
│ │ └── bug-report.yml
│ └── workflows
│ ├── build-electron.yml
│ ├── docker-build.yml
│ ├── lint.yml
│ └── test.yml
├── app
│ ├── assets
│ │ ├── icon.ico
│ │ ├── icon.png
│ │ └── tray-icon.png
│ ├── build
│ │ ├── linux
│ │ │ ├── after-install.sh
│ │ │ ├── after-remove.sh
│ │ │ └── daydream-scope.desktop
│ │ ├── mac
│ │ │ └── entitlements.mac.plist
│ │ ├── win
│ │ │ └── installer.nsh
│ │ ├── afterPack.js +
│ │ └── update-latest-yml.js +
│ ├── src
│ │ ├── components
│ │ │ ├── ErrorBoundary.tsx +
│ │ │ ├── LogViewer.html
│ │ │ ├── LogViewer.js
│ │ │ ├── ServerLoading.tsx +
│ │ │ └── Setup.tsx +
│ │ ├── services
│ │ │ ├── electronApp.ts +
│ │ │ ├── pythonProcess.ts +
│ │ │ └── setup.ts +
│ │ ├── types
│ │ │ ├── ipc.ts +
│ │ │ └── services.ts +
│ │ ├── utils
│ │ │ ├── config.test.ts +
│ │ │ ├── config.ts +
│ │ │ ├── logger.ts +
│ │ │ └── port.ts +
│ │ ├── index.css
│ │ ├── main.ts +
│ │ ├── preload.ts +
│ │ └── renderer.tsx +
│ ├── .gitignore
│ ├── dev-app-update.yml
│ ├── electron-builder.yml
│ ├── index.html
│ ├── package-lock.json
│ ├── package.json
│ ├── postcss.config.js +
│ ├── tsconfig.json
│ ├── tsconfig.node.json
│ ├── vite.config.ts +
│ ├── vite.main.config.ts +
│ ├── vite.preload.config.ts +
│ ├── vite.renderer.config.ts +
│ └── vitest.config.ts +
├── docs
│ ├── contributing.md
│ ├── lora.md
│ ├── server.md
│ ├── spout.md
│ └── vace.md
├── frontend
│ ├── assets
│ │ └── wireframe.png
│ ├── public
│ │ ├── assets
│ │ │ ├── discord-symbol-white.svg
│ │ │ └── github-mark-white.svg
│ │ ├── icon.svg
│ │ └── vite.svg
│ ├── src
│ │ ├── assets
│ │ │ └── react.svg
│ │ ├── components
│ │ │ ├── shared
│ │ │ │ ├── PromptField.tsx +
│ │ │ │ ├── TemporalTransitionControls.tsx +
│ │ │ │ └── WeightSlider.tsx +
│ │ │ ├── ui
│ │ │ │ ├── badge.tsx +
│ │ │ │ ├── button.tsx +
│ │ │ │ ├── card.tsx +
│ │ │ │ ├── debounced-slider.tsx +
│ │ │ │ ├── dialog.tsx +
│ │ │ │ ├── file-picker.tsx +
│ │ │ │ ├── input.tsx +
│ │ │ │ ├── label-with-tooltip.tsx +
│ │ │ │ ├── play-overlay.tsx +
│ │ │ │ ├── progress.tsx +
│ │ │ │ ├── select.tsx +
│ │ │ │ ├── slider-with-input.tsx +
│ │ │ │ ├── slider.tsx +
│ │ │ │ ├── sonner.tsx +
│ │ │ │ ├── spinner.tsx +
│ │ │ │ ├── textarea.tsx +
│ │ │ │ ├── toggle-group.tsx +
│ │ │ │ ├── toggle.tsx +
│ │ │ │ └── tooltip.tsx +
│ │ │ ├── DenoisingStepsSlider.tsx +
│ │ │ ├── DownloadDialog.tsx +
│ │ │ ├── Header.tsx +
│ │ │ ├── ImageManager.tsx +
│ │ │ ├── InputAndControlsPanel.tsx +
│ │ │ ├── LoRAManager.tsx +
│ │ │ ├── MediaPicker.tsx +
│ │ │ ├── PromptInput.tsx +
│ │ │ ├── PromptInputWithTimeline.tsx +
│ │ │ ├── PromptTimeline.tsx +
│ │ │ ├── ReportBugDialog.tsx +
│ │ │ ├── SettingsPanel.tsx +
│ │ │ ├── StatusBar.tsx +
│ │ │ ├── TimelinePromptEditor.tsx +
│ │ │ └── VideoOutput.tsx +
│ │ ├── data
│ │ │ ├── parameterMetadata.ts +
│ │ │ └── pipelines.ts +
│ │ ├── hooks
│ │ │ ├── useLocalSliderValue.ts +
│ │ │ ├── useLocalVideo.ts +
│ │ │ ├── usePipeline.ts +
│ │ │ ├── usePromptManager.ts +
│ │ │ ├── useStreamRecorder.ts +
│ │ │ ├── useStreamState.ts +
│ │ │ ├── useTimelinePlayback.ts +
│ │ │ ├── useVideoSource.ts +
│ │ │ ├── useWebRTC.ts +
│ │ │ └── useWebRTCStats.ts +
│ │ ├── lib
│ │ │ ├── api.ts +
│ │ │ └── utils.ts +
│ │ ├── pages
│ │ │ └── StreamPage.tsx +
│ │ ├── types
│ │ │ └── index.ts +
│ │ ├── utils
│ │ │ ├── loraHelpers.ts +
│ │ │ ├── promptColors.ts +
│ │ │ ├── promptWeights.ts +
│ │ │ └── timelinePromptSubmission.ts +
│ │ ├── App.css
│ │ ├── App.tsx +
│ │ ├── index.css
│ │ ├── main.tsx +
│ │ └── vite-env.d.ts
│ ├── .gitignore
│ ├── .prettierrc
│ ├── components.json
│ ├── eslint.config.js +
│ ├── index.html
│ ├── package-lock.json
│ ├── package.json
│ ├── postcss.config.js
│ ├── README.md
│ ├── tailwind.config.js
│ ├── tsconfig.app.json
│ ├── tsconfig.json
│ ├── tsconfig.node.json
│ └── vite.config.ts +
├── outputs
│ └── b300_320x576_fa4.log
├── tests
│ ├── __init__.py
│ ├── test_logs_config.py +
│ └── test_render_timeline_cli.py +
├── pyproject.toml *
├── uv.lock *
├── ._.DS_Store
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── build.sh
├── CLAUDE.md
├── Dockerfile
├── LICENSE.md
├── ncu_kernel_b_large.ncu-rep
├── ncu_kernel_b_tune.ncu-rep
├── ncu_kernel_b.ncu-rep
├── notes.zip
├── nsys_kernel_b.nsys-rep
├── nsys_kernel_b.sqlite
├── nsys_smoke.nsys-rep
├── nsys_smoke.sqlite
├── publish.sh
├── README.md
└── testfile

(* denotes selected files)
(+ denotes code-map available)
</file_map>
<file_contents>
File: /Users/daviddickinson/Projects/Lora/scope/uv.lock
(lines 700-780: fa4 dependency group and flash-attn wheel URLs/pins in lockfile.)

```
lock
 { name = "torchvision", version = "0.23.0+cu128", source = { registry = "https://download.pytorch.org/whl/cu128" }, marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
 { name = "transformers" },
 { name = "triton", marker = "sys_platform == 'linux'" },
 { name = "triton-windows", marker = "sys_platform == 'win32'" },
 { name = "twilio" },
 { name = "uvicorn" },
]

[package.dev-dependencies]
dev = [
 { name = "freezegun" },
 { name = "imageio" },
 { name = "imageio-ffmpeg" },
 { name = "pre-commit" },
 { name = "pytest" },
 { name = "ruff" },
 { name = "twine" },
]
fa4 = [
 { name = "cuda-python", marker = "python_full_version >= '3.12'" },
 { name = "nvidia-cutlass-dsl", marker = "python_full_version >= '3.12'" },
]

[package.metadata]
requires-dist = [
 { name = "accelerate", specifier = ">=1.1.1" },
 { name = "aiortc", specifier = ">=1.13.0" },
 { name = "diffusers", specifier = ">=0.31.0" },
 { name = "easydict", specifier = ">=1.13" },
 { name = "einops", specifier = ">=0.8.1" },
 { name = "fastapi", specifier = ">=0.116.1" },
 { name = "flash-attn", marker = "python_full_version = '3.12' and sys_platform == 'linux'", url = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" },
 { name = "flash-attn", marker = "sys_platform == 'win32'", url = "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl" },
 { name = "ftfy", specifier = ">=6.3.1" },
 { name = "httpx", specifier = ">=0.28.1" },
 { name = "huggingface-hub", specifier = ">=0.25.0" },
 { name = "kernels", specifier = ">=0.10.4" },
 { name = "lmdb", specifier = ">=1.7.3" },
 { name = "omegaconf", specifier = ">=2.3.0" },
 { name = "peft", specifier = ">=0.17.1" },
 { name = "pyopengl", marker = "sys_platform == 'win32'", specifier = ">=3.1.10" },
 { name = "safetensors", specifier = ">=0.6.2" },
 { name = "sageattention", marker = "python_full_version == '3.10.*' and sys_platform == 'linux'", url = "https://github.com/daydreamlive/SageAttention/releases/download/v2.2.0-linux/sageattention-2.2.0-cp310-cp310-linux_x86_64.whl" },
 { name = "sageattention", marker = "sys_platform == 'win32'", url = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl" },
 { name = "spoutgl", marker = "sys_platform == 'win32'", specifier = ">=0.1.1" },
 { name = "torch", marker = "sys_platform != 'linux' and sys_platform != 'win32'", specifier = "==2.8.0" },
 { name = "torch", marker = "sys_platform == 'linux' or sys_platform == 'win32'", specifier = "==2.8.0", index = "https://download.pytorch.org/whl/cu128" },
 { name = "torchao", specifier = "==0.13.0" },
 { name = "torchvision", marker = "sys_platform != 'linux' and sys_platform != 'win32'", specifier = "==0.23.0" },
 { name = "torchvision", marker = "sys_platform == 'linux' or sys_platform == 'win32'", specifier = "==0.23.0", index = "https://download.pytorch.org/whl/cu128" },
 { name = "transformers", specifier = ">=4.49.0" },
 { name = "triton", marker = "sys_platform == 'linux'", specifier = "==3.4.0" },
 { name = "triton-windows", marker = "sys_platform == 'win32'", specifier = "==3.4.0.post21" },
 { name = "twilio", specifier = ">=9.8.0" },
 { name = "uvicorn", specifier = ">=0.35.0" },
]

[package.metadata.requires-dev]
dev = [
 { name = "freezegun", specifier = ">=1.5.5" },
 { name = "imageio", specifier = ">=2.37.0" },
 { name = "imageio-ffmpeg", specifier = ">=0.6.0" },
 { name = "pre-commit", specifier = ">=4.0.0" },
 { name = "pytest", specifier = ">=8.4.2" },
 { name = "ruff", specifier = ">=0.8.0" },
 { name = "twine", specifier = ">=5.0.0" },
]
fa4 = [
 { name = "cuda-python", marker = "python_full_version >= '3.12'", specifier = ">=12.9.0" },
 { name = "nvidia-cutlass-dsl", marker = "python_full_version >= '3.12'", specifier = "==4.1.0" },
]

[[package]]
name = "diffusers"
version = "0.35.2"
source = { registry = "https://pypi.org/simple" }
dependencies = [
 { name = "filelock" },
```

File: /Users/daviddickinson/Projects/Lora/scope/scripts/test_fa4_kv_bias.py

```
py
#!/usr/bin/env python3
"""
Test FA4/CUTE score_mod for KV-cache bias (Kernel B replacement).

This is a standalone test that validates:
1. CUTE score_mod compiles and runs
2. Output matches FlexAttention reference within tolerance
3. Performance comparison (optional)

Usage:
 uv run python scripts/test_fa4_kv_bias.py
"""

import sys
import os
import math
import inspect
from pathlib import Path
import torch

def _extend_flash_attn_path_for_score_mod() -> None:
 import flash_attn as _flash_attn

 try:
 base_path = Path(__file__).resolve()
 except Exception:
 return
 for parent in base_path.parents:
 for repo_dir in ("flash-attention", "flash-attention.bak"):
 candidate = parent / repo_dir / "flash_attn"
 if candidate.is_dir():
 candidate_str = str(candidate)
 if candidate_str not in _flash_attn.__path__:
 _flash_attn.__path__.insert(0, candidate_str)
 return

import cutlass
import cutlass.cute as cute
import operator

_extend_flash_attn_path_for_score_mod()
from flash_attn.cute.interface import _flash_attn_fwd
if "score_mod" not in inspect.signature(_flash_attn_fwd).parameters:
 raise SystemExit(
 "flash_attn.cute.interface._flash_attn_fwd() does not support score_mod. "
 "Ensure the local flash-attention repo is available (flash-attention.bak)."
 )

# Try to import FlexAttention for reference
try:
 from torch.nn.attention.flex_attention import flex_attention, create_block_mask
 HAS_FLEX = True
except ImportError:
 HAS_FLEX = False
 print("Warning: FlexAttention not available, skipping reference comparison")

# =============================================================================
# CUTE score_mod for KV-cache bias
# =============================================================================

def make_kv_bias_score_mod(frame_seqlen: int, block_size: int, log_bias: float):
 """
 Factory that creates a score_mod with constants captured in closure.

 This avoids aux_tensors, keeping vec_size=2 for better performance.

 Bias rule (3 regions):
 - Region 1: First frame (kv_idx = 0
 zero_tensor = cute.full_like(kv_idx, 0)
 block_start_tensor = cute.where(
 operator.ge(block_start_tensor, zero_tensor),
 block_start_tensor,
 zero_tensor
 )

 # Build Region 2 mask: frame_seqlen ≤ kv_idx = frame_seqlen_t) & (kv_idx preset > timeline
- preview, standard, quality, highres, max

### Docs Created
- `notes/render-timeline-guide.md` - CLI usage guide
- `notes/krea-offline-render-tuning.md` - Tuning knobs reference
```

File: /Users/daviddickinson/Projects/Lora/scope/vendored/cutlass-cute/python/CuTeDSL/cute/tvm_ffi/jit_and_use_in_torch.py
(lines 50-83: CuTe DSL example showing enable_tvm_ffi usage and --enable-tvm-ffi option.)

```
py

@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
 """b = a + 1"""
 device_add_one(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))

def main():
 # compile the kernel with "--enable-tvm-ffi" option
 a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
 b_torch = torch.zeros(10, dtype=torch.float32, device="cuda")
 a_cute = from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
 b_cute = from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()
 # compile the kernel with "--enable-tvm-ffi" option
 compiled_add_one = cute.compile(add_one, a_cute, b_cute, options="--enable-tvm-ffi")
 # run the compiled function by passing in cute.Tensor as input
 # you need to set enable_tvm_ffi=True for now
 compiled_add_one(a_cute, b_cute)
 # print the result
 print("result of b after compiled_add_one(a, b)")
 print(b_torch)
 a_torch = a_torch + 1
 # We can directly pass in torch.Tensor as input
 # the call overhead is optimized so it is very fast to pass in torch.Tensor as input
 # takes about less than 0.5us per call likely in terms of API overhead
 compiled_add_one(a_torch, b_torch)
 # print the result
 print("result of b after compiled_add_one(a, b)")
 print(b_torch)

if __name__ == "__main__":
 main()
```

File: /Users/daviddickinson/Projects/Lora/scope/pyproject.toml
(lines 100-140: fa4 dependency group and cutlass-dsl pin (4.1.0) plus notes about Python 3.12.)

```
toml
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[dependency-groups]
dev = [
 "imageio>=2.37.0",
 "imageio-ffmpeg>=0.6.0",
 "ruff>=0.8.0",
 "pre-commit>=4.0.0",
 "twine>=5.0.0",
 "pytest>=8.4.2",
 "freezegun>=1.5.5",
]
# Flash Attention 4 (CUTE) dependencies for Blackwell GPUs (B200/B300, SM100+)
# Requires Python 3.12+. After installing, run: python scripts/patch_cutlass_sm103.py
fa4 = [
 "cuda-python>=12.9.0; python_version >= '3.12'",
 "nvidia-cutlass-dsl==4.1.0; python_version >= '3.12'",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 88
target-version = "py310"
exclude = [
 "*/vendor/*",
 "**/vendor/**",
 "*/modules/*",
 "**/modules/**",
 ".venv",
 "__pycache__",
]

[tool.ruff.lint]
select = [
 "E", # pycodestyle errors
 "W", # pycodestyle warnings
 "F", # pyflakes
 "I", # isort
```

File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/B300-FA4-PATCHES.md

```
md
# FA4/CUTE on B300 (SM103) - Patch Guide

**Date**: 2025-12-23
**Status**: ✅ FA4 basic works, ❌ score_mod needs version matching

## Summary

FA4/CUTE can run on B300 (SM103) after patching `nvidia-cutlass-dsl` to accept the `sm_103a` architecture. Without patches, CUTLASS DSL rejects SM103 because it only allows `sm_100a` and `sm_100f`.

### Results After Patching

| Kernel | B300 (SM103) | Notes |
|--------|--------------|-------|
| **FA4 basic (causal)** | 0.074 ms | ✅ 13x faster than Triton |
| **Triton Kernel B** | 0.977 ms | ✅ Baseline |
| **flex_attention** | 1.094 ms | ✅ Works |
| **FA4 with score_mod** | ❌ | Version mismatch |

## Prerequisites

1. **CUDA 12.9** installed (for `ptxas` SM103 support)
2. **nvidia-cutlass-dsl==4.1.0** (required by flash_attn==2.8.3)
3. **TRITON_PTXAS_PATH** set to CUDA 12.9's ptxas
```

bash
# Install correct cutlass version
uv pip install nvidia-cutlass-dsl==4.1.0

# Set ptxas path
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

```
## Patches Required

### 1. Patch `impl_utils.py` (arch validation function)

This is the core fix - patches the `check_value_in` function to allow `sm_103a` wherever `sm_100a/sm_100f` is allowed.

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/impl_utils.py`
```

python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

def check_value_in(
 value, possible_values: list, value_description: str, prefix=""
) -> None:
 # SM103 patch: allow sm_103a wherever sm_100a/sm_100f is allowed
 if value == 'sm_103a' and value_description == 'arch':
 if 'sm_100a' in possible_values or 'sm_100f' in possible_values:
 possible_values = list(possible_values) + ['sm_103a']
 if value not in possible_values:
 err_msg = prefix
 if err_msg != "":
 err_msg += ": "
 err_msg += f"invalid {value_description}, got {value}, must be one of {possible_values}"
 raise ValueError(err_msg)

def check_type_in(ty, possible_types: list, type_description: str, prefix="") -> None:
 if not isinstance(ty, type):
 ty = type(ty)
 if ty not in possible_types:
 err_msg = prefix
 if err_msg != "":
 err_msg += ": "
 err_msg += f"invalid type for {type_description}, got {ty}, must be one of {possible_types}"
 raise TypeError(err_msg)

```
### 2. Patch `tcgen05/mma.py` (TensorCore MMA ops)

Add `sm_103a` to all `admissible_archs` lists.

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/mma.py`
```

bash
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' mma.py
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' mma.py

```
### 3. Patch `tcgen05/copy.py` (TensorCore copy ops)

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/tcgen05/copy.py`
```

bash
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' copy.py
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' copy.py

```
### 4. Patch `cpasync/copy.py` (async copy ops)

**File**: `.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/nvgpu/cpasync/copy.py`
```

bash
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' copy.py
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' copy.py

```
### 5. Clear Python cache
```

bash
find .venv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name "*.pyc" -delete
find .venv/lib/python3.12/site-packages/nvidia_cutlass_dsl -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

```
## Automated Patch Script

Save as `scripts/patch_cutlass_sm103.sh`:
```

bash
#!/bin/bash
# Patch nvidia-cutlass-dsl for SM103 (B300) support
# Requires: nvidia-cutlass-dsl==4.1.0

set -e

VENV_PATH="${1:-.venv}"
CUTLASS_PATH="$VENV_PATH/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass"

if [ ! -d "$CUTLASS_PATH" ]; then
 echo "Error: CUTLASS path not found: $CUTLASS_PATH"
 exit 1
fi

echo "Patching nvidia-cutlass-dsl for SM103 support..."

# 1. Patch impl_utils.py
IMPL_FILE="$CUTLASS_PATH/impl_utils.py"
cp "$IMPL_FILE" "${IMPL_FILE}.bak"

cat > "$IMPL_FILE" << 'EOF'
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
# PATCHED: Added SM103 support for B300 GPUs

def check_value_in(
 value, possible_values: list, value_description: str, prefix=""
) -> None:
 # SM103 patch: allow sm_103a wherever sm_100a/sm_100f is allowed
 if value == 'sm_103a' and value_description == 'arch':
 if 'sm_100a' in possible_values or 'sm_100f' in possible_values:
 possible_values = list(possible_values) + ['sm_103a']
 if value not in possible_values:
 err_msg = prefix
 if err_msg != "":
 err_msg += ": "
 err_msg += f"invalid {value_description}, got {value}, must be one of {possible_values}"
 raise ValueError(err_msg)

def check_type_in(ty, possible_types: list, type_description: str, prefix="") -> None:
 if not isinstance(ty, type):
 ty = type(ty)
 if ty not in possible_types:
 err_msg = prefix
 if err_msg != "":
 err_msg += ": "
 err_msg += f"invalid type for {type_description}, got {ty}, must be one of {possible_types}"
 raise TypeError(err_msg)
EOF
echo " Patched: impl_utils.py"

# 2. Patch tcgen05/mma.py
MMA_FILE="$CUTLASS_PATH/cute/nvgpu/tcgen05/mma.py"
cp "$MMA_FILE" "${MMA_FILE}.bak"
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$MMA_FILE"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$MMA_FILE"
echo " Patched: tcgen05/mma.py"

# 3. Patch tcgen05/copy.py
TCGEN_COPY="$CUTLASS_PATH/cute/nvgpu/tcgen05/copy.py"
cp "$TCGEN_COPY" "${TCGEN_COPY}.bak"
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$TCGEN_COPY"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$TCGEN_COPY"
echo " Patched: tcgen05/copy.py"

# 4. Patch cpasync/copy.py
CPASYNC_COPY="$CUTLASS_PATH/cute/nvgpu/cpasync/copy.py"
cp "$CPASYNC_COPY" "${CPASYNC_COPY}.bak"
sed -i 's/"sm_100a",/"sm_100a", "sm_103a",/g' "$CPASYNC_COPY"
sed -i 's/"sm_100f",/"sm_100f", "sm_103a",/g' "$CPASYNC_COPY"
echo " Patched: cpasync/copy.py"

# 5. Clear pycache
find "$CUTLASS_PATH" -name "*.pyc" -delete 2>/dev/null || true
find "$CUTLASS_PATH" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo " Cleared: __pycache__"

echo ""
echo "SM103 patches applied successfully!"
echo ""
echo "To verify:"
echo " TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas python -c \\"
echo " \"from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 OK')\""

```
## Verification
```

bash
# Set environment
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

# Test FA4 import
python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 imported')"

# Benchmark
python -c "
import torch
from flash_attn.cute.interface import _flash_attn_fwd
import time

B, H, S, D = 1, 16, 4680, 128
q = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(5):
 _flash_attn_fwd(q, k, v, causal=True, softmax_scale=1.0/D**0.5)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(50):
 _flash_attn_fwd(q, k, v, causal=True, softmax_scale=1.0/D**0.5)
torch.cuda.synchronize()
print(f'FA4: {(time.perf_counter()-start)/50*1000:.3f} ms')
"

```
## What Works vs What Doesn't

### ✅ Works on B300 after patches

1. **FA4 basic attention** (causal, non-causal)
 - Forward pass: ✅
 - Performance: 0.074ms (13x faster than Triton)

2. **All Blackwell tcgen05 ops**
 - MMA (matrix multiply-accumulate)
 - TMA copy operations
 - Async copy operations

### ❌ Doesn't work yet

1. **FA4 with score_mod**
 - The `flash-attention.bak` repo (with score_mod) expects different cutlass-dsl API
 - `FastDivmodDivisor` not in cutlass-dsl 4.1.0
 - Need matched versions of flash-attn wheel + cutlass-dsl

2. **FA4 backward pass**
 - Not tested on B300
 - May need additional patches

## Version Compatibility Matrix

| flash-attn | cutlass-dsl | score_mod | B300 Status |
|------------|-------------|-----------|-------------|
| 2.8.3 wheel | 4.1.0 | ❌ | ✅ Basic works after patches |
| flash-attention.bak | 4.1.0 | ✅ | ❌ API mismatch |
| flash-attention.bak | ??? | ✅ | 🔍 Need to find matching version |

## Next Steps

1. **Find matching cutlass-dsl for flash-attention.bak**
 - Check what version of cutlass-dsl exports `FastDivmodDivisor`
 - May need cutlass-dsl 4.2.x or 4.3.x with different patches

2. **Integrate FA4 into daydream-scope**
 - Add `SCOPE_KV_BIAS_BACKEND=fa4` option
 - Auto-detect SM103 and apply patches at runtime
 - Fallback to Triton if FA4 fails

3. **Benchmark full pipeline**
 - Test FA4 in render_timeline on B300
 - Compare FPS: FA4 vs Triton vs flex_attention

## Files Modified

On B300 machine, these files were patched:
```

.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/
├── impl_utils.py # check_value_in SM103 support
└── cute/nvgpu/
 ├── tcgen05/
 │ ├── mma.py # admissible_archs += sm_103a
 │ └── copy.py # admissible_archs += sm_103a
 └── cpasync/
 └── copy.py # admissible_archs += sm_103a

```
Backups created with `.bak` extension.

## References

- [NVIDIA CUDA 12.9 SM103 Support](https://docs.nvidia.com/cuda/cuda-features-archive/index.html)
- [Triton SM103 Issue](https://github.com/triton-lang/triton/issues/8473)
- [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
```

File: /Users/daviddickinson/Projects/Lora/scope/vendored/cutlass-cute/python/CuTeDSL/ampere/elementwise_add.py
(lines 170-210: CuTe DSL example using cute.make_rmem_tensor (API expected by flash-attention.bak).)

```
py
 thr_copy_A = tiled_copy_A.get_slice(tidx)
 thr_copy_B = tiled_copy_B.get_slice(tidx)
 thr_copy_C = tiled_copy_C.get_slice(tidx)

 thrA = thr_copy_A.partition_S(blkA)
 thrB = thr_copy_B.partition_S(blkB)
 thrC = thr_copy_C.partition_S(blkC)

 # allocate fragments for gmem->rmem
 frgA = cute.make_fragment_like(thrA)
 frgB = cute.make_fragment_like(thrB)
 frgC = cute.make_fragment_like(thrC)

 thrCrd = thr_copy_C.partition_S(blkCrd)
 frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)

 print("[DSL INFO] Sliced Tensors per thread:")
 print(f"[DSL INFO] thrA = {thrA.type}")
 print(f"[DSL INFO] thrB = {thrB.type}")
 print(f"[DSL INFO] thrC = {thrC.type}")
 print(f"[DSL INFO] thrCrd = {thrCrd.type}")

 for i in range(0, cute.size(frgPred), 1):
 val = cute.elem_less(thrCrd[i], shape)
 frgPred[i] = val

 # Print per thread predicate mask
 # if tidx == 0 and bidx == 0:
 # cute.printf("block_dim = {}", cute.arch.grid_dim())
 # cute.printf("shape = {}", shape)
 # cute.print_tensor(thrA)
 # cute.print_tensor(thrB)
 # cute.print_tensor(frgPred)

 ##########################################################
 # Move data to reg address space
 ##########################################################

 cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
 cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)
```

File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/B300-SETUP-GUIDE.md

```
md
# B300 (SM103) Setup Guide

**Date**: 2025-12-23
**Status**: ✅ WORKING

## Quick Setup for B300 Machines
```

bash
# 1. Install CUDA 12.9 (has SM103 support)
sudo apt update && sudo apt install -y cuda-toolkit-12-9

# 2. Add ptxas path to shell profile (RECOMMENDED - works with uv run)
echo 'export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas' >> ~/.bashrc
source ~/.bashrc

# 3. Verify it works
uv run python scripts/triton_sdpa.py --kernel-b
uv run daydream-scope # Full server

```
### Alternative: Per-command (no persistence)
```

bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope

```
### Note on `.venv/bin/activate`
Adding to `.venv/bin/activate` only works when manually activating the venv with `source .venv/bin/activate`. It does NOT work with `uv run` which bypasses activate scripts. Use `~/.bashrc` instead for `uv run` compatibility.

## Why This Is Needed

- **CUDA 12.8's ptxas** doesn't support `sm_103a` (B300's architecture)
- **Triton 3.4.0** bundles an older ptxas that also doesn't support SM103
- **CUDA 12.9** added SM103 support
- Setting `TRITON_PTXAS_PATH` tells Triton to use the system ptxas instead of bundled

## Verified Working on B300

| Component | Status | Performance |
|-----------|--------|-------------|
| **Triton Kernel B** | ✅ PASS | 0.977 ms |
| **flex_attention** | ✅ PASS | 1.095 ms |
| **RoPE fused** | ✅ PASS | 0.029 ms |
| **Correctness tests** | ✅ ALL PASS | |

**Triton Kernel B is 10.8% faster than flex_attention on B300!**

## Environment Details
```

GPU: NVIDIA B300 SXM6 AC
Compute Capability: (10, 3) = SM103
CUDA Toolkit: 12.9 (for ptxas)
PyTorch: 2.8.0+cu128
Triton: 3.4.0
Python: 3.12

```
## What Works on B300

| Feature | B200 (SM100) | B300 (SM103) | Notes |
|---------|--------------|--------------|-------|
| Triton Kernel B | ✅ | ✅ | 0.977 ms (with CUDA 12.9 ptxas) |
| flex_attention | ✅ | ✅ | 1.095 ms |
| RoPE fused | ✅ | ✅ | 0.029 ms |
| FA4/CUTE basic | ✅ | ✅ | **0.074 ms** (13x faster!) - needs patches |
| FA4/CUTE score_mod | ✅ | ❌ | Version mismatch, see B300-FA4-PATCHES.md |
| daydream-scope | ✅ 20 FPS | ✅ 8.8 FPS | Baseline, room for optimization |

## FA4/CUTE on B300 (Optional)

FA4 basic attention works on B300 after patching `nvidia-cutlass-dsl`. It's **13x faster** than Triton for causal attention!
```

bash
# 1. Install correct cutlass version
uv pip install nvidia-cutlass-dsl==4.1.0

# 2. Apply SM103 patches
./scripts/patch_cutlass_sm103.sh

# 3. Verify FA4 works
python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4 OK')"

```
**Note**: FA4 with `score_mod` (for Kernel B equivalent) doesn't work yet due to version mismatches. See `notes/FA4/B300-FA4-PATCHES.md` for details.

## Troubleshooting

### Error: `ptxas fatal: Value 'sm_103a' is not defined`

**Cause**: Using old ptxas that doesn't know SM103

**Fix**:
```

bash
# Verify CUDA 12.9 is installed
/usr/local/cuda-12.9/bin/ptxas --help | grep sm_103

# Set the env var
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

```
### Error: `NoValidChoicesError` for flex_attention

**Cause**: torch.compile failing on SM103

**Fix**: The TRITON_PTXAS_PATH fix should resolve this. If not, the SM103 compat layer in `src/scope/core/compat/sm103.py` can help gate compilation.

## Files Modified/Created

1. `.venv/bin/activate` - Added `TRITON_PTXAS_PATH`
2. `src/scope/core/compat/sm103.py` - SM103 compatibility utilities
3. `src/scope/core/compat/__init__.py` - Compat module init
4. `notes/FA4/b300-investigation.md` - Full investigation log
5. `notes/FA4/B300-FA4-PATCHES.md` - FA4/CUTE SM103 patch documentation
6. `scripts/patch_cutlass_sm103.sh` - Automated CUTLASS patching script

### Files Patched for FA4 (by patch script)
```

.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/
├── impl_utils.py # check_value_in SM103 support
└── cute/nvgpu/
 ├── tcgen05/
 │ ├── mma.py # admissible_archs += sm_103a
 │ └── copy.py # admissible_archs += sm_103a
 └── cpasync/
 └── copy.py # admissible_archs += sm_103a

```
## Next Steps

1. **Improve B300 FPS** (currently 8.8 FPS vs B200's 20 FPS)
 - Profile to find bottlenecks
 - Test with FA4 basic attention in pipeline

2. **FA4 score_mod on B300**
 - Find matching cutlass-dsl version for flash-attention.bak
 - Or wait for upstream FA4 wheel with score_mod support

3. **Performance tuning**
 - Block sizes may benefit from B300-specific tuning
 - SM103 has 148 SMs vs B200's 160 SMs
```

File: /Users/daviddickinson/Projects/Lora/scope/src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py
(lines 1-430: KV-bias backend selection, FA4 score_mod path, flash segment-combine, flex_attention compile guards, and helper functions used by bias path.)

```
py
# Modified from https://github.com/krea-ai/realtime-video
import atexit
import functools
import logging
import math
import os
import subprocess
from collections import defaultdict

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import (
 BlockMask,
 create_block_mask,
 flex_attention,
)

from scope.core.pipelines.wan2_1.modules.attention import attention

logger = logging.getLogger(__name__)

def _is_sm103() -> bool:
 """Return True on B300 (SM103)."""
 try:
 return (
 torch.cuda.is_available()
 and torch.cuda.get_device_capability(0) == (10, 3)
 )
 except Exception:
 return False

# KV-cache bias backend selection
# SCOPE_KV_BIAS_BACKEND:
# - fa4: FA4/CUTE score_mod (fastest when available; requires CUTLASS DSL)
# - flash: Bias via FlashAttention segment-combine (no score_mod; SM103 default)
# - triton: Triton Kernel B (SM100 default)
# - flex: torch.nn.attention.flex_attention fallback
_env_kv_bias_backend = os.getenv("SCOPE_KV_BIAS_BACKEND")
_KV_BIAS_BACKEND = (
 _env_kv_bias_backend.lower()
 if _env_kv_bias_backend
 else ("flash" if _is_sm103() else "triton")
)

# FA4 + CUTE score_mod: 1.89x faster than Triton Kernel B on B200
_fa4_available = False
_fa4_fwd = None
_fa4_score_mod_cache = {}

if _KV_BIAS_BACKEND == "fa4":
 try:
 import inspect
 import operator
 from pathlib import Path

 import flash_attn as _flash_attn

 def _extend_flash_attn_path_for_score_mod() -> None:
 try:
 base_path = Path(__file__).resolve()
 except Exception:
 return
 for parent in base_path.parents:
 for repo_dir in ("flash-attention", "flash-attention.bak"):
 candidate = parent / repo_dir / "flash_attn"
 if candidate.is_dir():
 candidate_str = str(candidate)
 if candidate_str not in _flash_attn.__path__:
 _flash_attn.__path__.insert(0, candidate_str)
 return

 _extend_flash_attn_path_for_score_mod()
 from flash_attn.cute.interface import _flash_attn_fwd as _fa4_fwd
 if "score_mod" not in inspect.signature(_fa4_fwd).parameters:
 raise ImportError(
 "FA4/CUTE score_mod requires the local flash-attention repo; "
 "_flash_attn_fwd() lacks score_mod in this environment."
 )
 import cutlass
 import cutlass.cute as cute
 _fa4_available = True
 logger.info("FA4/CUTE score_mod enabled for KV-cache attention bias (1.89x faster than Triton)")
 except Exception as e:
 logger.warning(f"FA4/CUTE not available, falling back to Triton: {e}")
 _KV_BIAS_BACKEND = "triton"

# Triton Kernel B: fast fallback for KV-cache bias path
USE_TRITON_KERNEL_B = _KV_BIAS_BACKEND in ("triton", "fa4", "flash") # fa4/flash fall back to triton
_triton_kernel_b = None

if USE_TRITON_KERNEL_B:
 try:
 from scope.core.kernels import triton_kernel_b as _triton_kernel_b
 logger.info("Triton Kernel B available for KV-cache attention bias")
 except ImportError as e:
 logger.warning(f"Triton Kernel B not available: {e}")
 USE_TRITON_KERNEL_B = False

def _get_fa4_score_mod(frame_seqlen: int, block_size: int, log_bias: float):
 """
 Get or create a CUTE score_mod for FA4 KV-cache bias.

 Uses closure constants to avoid aux_tensors (keeps vec_size=2 for better perf).
 Caches compiled score_mods by (frame_seqlen, block_size, log_bias).
 """
 if not _fa4_available:
 return None

 # Round log_bias to avoid cache explosion
 cache_key = (frame_seqlen, block_size, round(log_bias, 6))
 if cache_key in _fa4_score_mod_cache:
 return _fa4_score_mod_cache[cache_key]

 # Closure constants (compile-time)
 _frame_seqlen = frame_seqlen
 _block_size = block_size
 _log_bias = log_bias

 @cute.jit
 def score_mod_kv_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 # Get runtime Lk from seqlen_info
 Lk = seqlen_info.seqlen_k

 # Compute block_start = max(0, Lk - block_size)
 block_start_val = Lk - _block_size
 block_start_tensor = cute.full_like(kv_idx, 0) + block_start_val
 zero_tensor = cute.full_like(kv_idx, 0)
 block_start_tensor = cute.where(
 operator.ge(block_start_tensor, zero_tensor),
 block_start_tensor,
 zero_tensor
 )

 # Build Region 2 mask: frame_seqlen ≤ kv_idx torch.Tensor:
 key = (str(device), int(batch_size), int(seqlen))
 cached = _cu_seqlens_cache.get(key)
 if cached is not None:
 return cached
 cu = torch.arange(
 0,
 (batch_size + 1) * seqlen,
 step=seqlen,
 device=device,
 dtype=torch.int32,
 )
 _cu_seqlens_cache[key] = cu
 return cu

def _flash_attn_with_lse(
 q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
 """
 Compute attention output + LSE for segment-combine KV-cache bias.

 Returns:
 out: [B, Lq, H, D]
 lse: [B, Lq, H] (float32)
 """
 fa4_fwd = _get_fa4_fwd()
 if fa4_fwd is not None:
 out, lse = fa4_fwd(
 q,
 k,
 v,
 softmax_scale=softmax_scale,
 causal=False,
 return_lse=True,
 )
 if lse is None:
 raise RuntimeError("FA4 _flash_attn_fwd returned lse=None with return_lse=True")
 if lse.dim() == 3:
 # [B, H, Lq] -> [B, Lq, H]
 lse = lse.transpose(1, 2)
 elif lse.dim() == 2:
 # [H, total_q] -> [B, Lq, H]
 b, lq = q.shape[:2]
 lse = lse.transpose(0, 1).reshape(b, lq, -1)
 else:
 raise RuntimeError(f"Unexpected lse shape from FA4: {tuple(lse.shape)}")
 return out, lse

 from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

 b, lq, _h, _d = q.shape
 lk = k.shape[1]
 q_flat = q.reshape(b * lq, q.shape[2], q.shape[3])
 k_flat = k.reshape(b * lk, k.shape[2], k.shape[3])
 v_flat = v.reshape(b * lk, v.shape[2], v.shape[3])
 cu_seqlens_q = _get_cu_seqlens(q.device, b, lq)
 cu_seqlens_k = _get_cu_seqlens(q.device, b, lk)
 out_flat, softmax_lse, _s_dmask, _rng_state = _flash_attn_varlen_forward(
 q_flat,
 k_flat,
 v_flat,
 cu_seqlens_q,
 cu_seqlens_k,
 lq,
 lk,
 0.0, # dropout_p
 softmax_scale,
 False, # causal
 )
 out = out_flat.reshape(b, lq, q.shape[2], v.shape[3])
 lse = softmax_lse.transpose(0, 1).reshape(b, lq, -1)
 return out, lse

def _merge_out_lse(
 out_a: torch.Tensor,
 lse_a: torch.Tensor,
 out_b: torch.Tensor,
 lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
 """
 Merge two attention results from disjoint KV segments.

 Shapes:
 out_*: [B, Lq, H, D]
 lse_*: [B, Lq, H] (float32)
 """
 lse_a = lse_a.to(torch.float32)
 lse_b = lse_b.to(torch.float32)
 lse_new = torch.logaddexp(lse_a, lse_b)
 w_a = torch.exp(lse_a - lse_new)
 w_b = torch.exp(lse_b - lse_new)
 out_new = out_a.to(torch.float32) * w_a.unsqueeze(-1) + out_b.to(torch.float32) * w_b.unsqueeze(-1)
 return out_new, lse_new

def _kv_bias_flash_combine(
 q: torch.Tensor,
 k: torch.Tensor,
 v: torch.Tensor,
 frame_seqlen: int,
 current_block_start: int,
 log_bias: float,
) -> torch.Tensor:
 """
 Exact KV-bias attention using FlashAttention segment combine.

 This reproduces the score_mod rule:
 add log_bias for kv_idx in [frame_seqlen, current_block_start)
 without requiring a score_mod-capable kernel.
 """
 lk = int(k.shape[1])
 if lk == 0 or log_bias == 0.0:
 return attention(q, k, v)

 frame_end = max(0, min(int(frame_seqlen), lk))
 block_start = max(0, min(int(current_block_start), lk))
 if block_start 0 else 0
 logger.info(f" {name}: {time_ms:.1f}ms ({pct:.1f}%) [{count} calls, {time_ms/count:.2f}ms/call]")
 logger.info(f" TOTAL: {total:.1f}ms")

 # Compute p_bias vs p_recompute
 bias_time = _profile_times.get("self_attn_kv_bias", 0)
 recompute_time = _profile_times.get("self_attn_block_mask", 0)
 plain_time = _profile_times.get("self_attn_kv_plain", 0)
 attn_total = bias_time + recompute_time + plain_time
 if attn_total > 0:
 logger.info("=== p_bias vs p_recompute ===")
 logger.info(f" p_bias (Kernel B): {100 * bias_time / attn_total:.1f}%")
 logger.info(f" p_recompute (Kernel A): {100 * recompute_time / attn_total:.1f}%")
 logger.info(f" p_plain (FA path): {100 * plain_time / attn_total:.1f}%")

# Register atexit handler to print profiling report on exit
atexit.register(profile_report)

class _ProfileBlock:
 """Context manager for profiling a code block."""
 __slots__ = ('name', 'start_event', 'end_event')

 def __init__(self, name: str):
 self.name = name
 self.start_event = None
 self.end_event = None

 def __enter__(self):
 if _should_profile():
 self.start_event = torch.cuda.Event(enable_timing=True)
 self.end_event = torch.cuda.Event(enable_timing=True)
 self.start_event.record()
 return self

 def __exit__(self, *args):
 if self.start_event is not None:
 self.end_event.record()
 torch.cuda.synchronize()
 elapsed_ms = self.start_event.elapsed_time(self.end_event)
 _profile_times[self.name] += elapsed_ms
 _profile_counts[self.name] += 1

from .model import (
 USE_TRITON_ROTARY,
 USE_TRITON_ROPE_FUSED,
 WAN_CROSSATTENTION_CLASSES,
 MLPProj,
 WanLayerNorm,
 WanRMSNorm,
 get_rope_cos_sin,
 rope_apply,
 rope_params,
 sinusoidal_embedding_1d,
 triton_apply_rotary,
 triton_rope_fused_3way,
)

_FLEX_ATTENTION_EAGER = flex_attention
_FLEX_ATTENTION_COMPILED = None
_FLEX_ATTENTION_USE_EAGER = False
_FLEX_ATTENTION_DISABLE_COMPILE = os.getenv("DISABLE_FLEX_ATTENTION_COMPILE", "0") == "1"

def _maybe_set_triton_ptxas_path() -> None:
 if os.getenv("TRITON_PTXAS_PATH"):
 return

 candidates = [
 "/usr/local/cuda-13.1/bin/ptxas",
 "/usr/local/cuda-13.0/bin/ptxas",
 "/usr/local/cuda-12.9/bin/ptxas",
 "/usr/local/cuda/bin/ptxas",
 "/usr/local/cuda-12.8/bin/ptxas",
 ]

 def supports_sm103(ptxas_path: str) -> bool:
 try:
 proc = subprocess.run(
 [ptxas_path, "--help"],
 capture_output=True,
 text=True,
 check=False,
 timeout=2,
```

(lines 960-1135: KV-cache bias application in CausalWanSelfAttention: FA4 score_mod call, flash-combine fallback, Triton Kernel B, flex_attention score_mod fallback.)

```
py
 ):
 with _ProfileBlock("cache_eviction"):
 # Calculate the number of new tokens added in this step
 # Shift existing cache content left to discard oldest tokens
 # Clone the source slice to avoid overlapping memory error
 num_evicted_tokens = (
 num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
 )
 num_rolled_tokens = (
 kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
 )
 kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
 kv_cache["k"][
 :,
 sink_tokens + num_evicted_tokens : sink_tokens
 + num_evicted_tokens
 + num_rolled_tokens,
 ].clone()
 )
 kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
 kv_cache["v"][
 :,
 sink_tokens + num_evicted_tokens : sink_tokens
 + num_evicted_tokens
 + num_rolled_tokens,
 ].clone()
 )
 # Insert the new keys/values at the end
 local_end_index = (
 kv_cache["local_end_index"]
 + current_end
 - kv_cache["global_end_index"]
 - num_evicted_tokens
 )
 local_start_index = local_end_index - num_new_tokens
 kv_cache["k"][:, local_start_index:local_end_index] = roped_key
 kv_cache["v"][:, local_start_index:local_end_index] = v
 else:
 with _ProfileBlock("cache_update"):
 # Assign new keys/values directly up to current_end
 local_end_index = (
 kv_cache["local_end_index"]
 + current_end
 - kv_cache["global_end_index"]
 )
 local_start_index = local_end_index - num_new_tokens
 kv_cache["k"][:, local_start_index:local_end_index] = roped_key
 kv_cache["v"][:, local_start_index:local_end_index] = v

 kv_start_idx = max(0, local_end_index - self.max_attention_size)
 cached_k = kv_cache["k"][:, kv_start_idx:local_end_index]
 cached_v = kv_cache["v"][:, kv_start_idx:local_end_index]

 if kv_cache_attention_bias != KV_CACHE_ATTENTION_BIAS_DISABLED:
 # Use flex_attention with bias to mitigate error accumulation in past frames
 # log_scale in (0, 1]: smaller values = less attention to past frame tokens
 log_scale = math.log(kv_cache_attention_bias)

 # Exclude first frame and current block from bias
 cache_len = local_end_index - kv_start_idx
 cache_current_block_start = (
 cache_len - frame_seqlen * self.num_frame_per_block
 )

 q_len = roped_query.shape[1]
 block_size = frame_seqlen * self.num_frame_per_block

 x = None

 if _fa4_available and _KV_BIAS_BACKEND == "fa4":
 # FA4 + CUTE score_mod: 1.89x faster than Triton Kernel B
 # FA4 expects layout: [B, L, H, D] (already correct)
 score_mod_cute = _get_fa4_score_mod(frame_seqlen, block_size, log_scale)
 with _ProfileBlock("self_attn_kv_bias_fa4"):
 x, _ = _fa4_fwd(
 roped_query,
 cached_k,
 cached_v,
 score_mod=score_mod_cute,
 causal=False,
 return_lse=False,
 )

 elif _KV_BIAS_BACKEND == "flash":
 global _flash_bias_tripped
 if not _flash_bias_tripped:
 try:
 with _ProfileBlock("self_attn_kv_bias_flash"):
 x = _kv_bias_flash_combine(
 roped_query,
 cached_k,
 cached_v,
 frame_seqlen=frame_seqlen,
 current_block_start=cache_current_block_start,
 log_bias=log_scale,
 )
 except Exception as e:
 _flash_bias_tripped = True
 logger.warning(
 "KV-bias flash backend failed; falling back to Triton: %s",
 e,
 )

 if x is None and USE_TRITON_KERNEL_B and _triton_kernel_b is not None:
 # Triton Kernel B: no padding needed
 # Input: (B, L, H, D) -> (B, H, L, D)
 with _ProfileBlock("transpose_contiguous"):
 Q_t = roped_query.transpose(2, 1).contiguous()
 K_t = cached_k.transpose(2, 1).contiguous()
 V_t = cached_v.transpose(2, 1).contiguous()
 with _ProfileBlock("self_attn_kv_bias"):
 x = _triton_kernel_b(
 Q=Q_t,
 K=K_t,
 V=V_t,
 frame_seqlen=frame_seqlen,
 current_block_start=cache_current_block_start,
 log_bias=log_scale,
 ).transpose(2, 1) # (B, H, L, D) -> (B, L, H, D)

 if x is None:
 # Fallback: flex_attention with padding
 kv_len = cached_k.shape[1]
 target_padded_length = (
 math.ceil(max(q_len, kv_len) / FLEX_ATTENTION_ALIGNMENT)
 * FLEX_ATTENTION_ALIGNMENT
 )

 padded_roped_query = _pad_tensor_for_flex_attention(
 roped_query, target_padded_length, pad_dim=1
 )
 padded_k = _pad_tensor_for_flex_attention(
 cached_k, target_padded_length, pad_dim=1
 )
 padded_v = _pad_tensor_for_flex_attention(
 cached_v, target_padded_length, pad_dim=1
 )

 # Convert scalars to tensors to avoid ShapeAsConstantBuffer dtype issues during compilation
 # This is critical when using torch.compile with flex_attention
 frame_seqlen_tensor = torch.as_tensor(
 frame_seqlen, dtype=torch.int32, device=roped_query.device
 )
 cache_current_block_start_tensor = torch.as_tensor(
 cache_current_block_start, dtype=torch.int32, device=roped_query.device
 ).squeeze()
 log_scale_tensor = torch.as_tensor(
 log_scale, dtype=roped_query.dtype, device=roped_query.device
 )

 def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
 # Apply bias only to past frames (exclude first frame and current block)
 return torch.where(
 (kv_idx >= frame_seqlen_tensor)
 & (kv_idx None:
 try:
 base_path = Path(__file__).resolve()
 except Exception:
 return
 for parent in base_path.parents:
 candidate = parent / "flash-attention" / "flash_attn"
 if candidate.is_dir():
 candidate_str = str(candidate)
 if candidate_str not in _flash_attn.__path__:
 _flash_attn.__path__.insert(0, candidate_str)
 break

_extend_flash_attn_path()
flash_attn_func = _flash_attn.flash_attn_func

def is_hopper_gpu():
 if not torch.cuda.is_available():
 return False
 device_name = torch.cuda.get_device_name(0).lower()
 return "h100" in device_name or "hopper" in device_name

def is_b200_gpu():
 if not torch.cuda.is_available():
 return False
 device_name = torch.cuda.get_device_name(0).lower()
 return "b200" in device_name

def is_blackwell_gpu():
 if not torch.cuda.is_available():
 return False
 major, _minor = torch.cuda.get_device_capability(0)
 return major >= 10

flash_attn_4_varlen_func = None
FLASH_ATTN_4_AVAILABLE = False
if os.getenv("DISABLE_FLASH_ATTENTION_4", "0") == "0":
 try:
 from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func

 FLASH_ATTN_4_AVAILABLE = is_blackwell_gpu()
 except Exception:
 pass

FLASH_ATTN_3_AVAILABLE = False

try:
 import flash_attn_interface

 FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
 pass

if not FLASH_ATTN_3_AVAILABLE and platform.system() != "Windows":
 try:
 from kernels import get_kernel

 flash_attn_3_hub = get_kernel(
 "kernels-community/flash-attn3", revision="fake-ops-return-probs"
 )
 flash_attn_interface = flash_attn_3_hub
 FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
 except Exception:
 pass

try:
 flash_attn = _flash_attn

 FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
 FLASH_ATTN_2_AVAILABLE = False

sageattn_func = None
SAGEATTN_AVAILABLE = False
# Do not try to load SageAttention on Hopper GPUs because at the moment
# loading SageAttention 2.2.0 in the sage module causes static on a H100
# Do not try to load SageAttention on B200 GPUs because at the moment
# SageAttention 2.2.0 is not supported on B200 GPUs
if not is_hopper_gpu() and not is_b200_gpu():
 from .sage import SAGEATTN_AVAILABLE, sageattn_func

import warnings

__all__ = [
 "flash_attention",
 "attention",
 "sageattn_func",
 "SAGEATTN_AVAILABLE",
]

logger.info(f"Attention backends: FA2={FLASH_ATTN_2_AVAILABLE}, FA3={FLASH_ATTN_3_AVAILABLE}, FA4={FLASH_ATTN_4_AVAILABLE}, Sage={SAGEATTN_AVAILABLE}")

# Track which attention backend was used (for one-time runtime logging)
_attention_backend_logged = False

def flash_attention(
 q,
 k,
 v,
 q_lens=None,
 k_lens=None,
 dropout_p=0.0,
 softmax_scale=None,
 q_scale=None,
 causal=False,
 window_size=(-1, -1),
 deterministic=False,
 dtype=torch.bfloat16,
 version=None,
):
 """
 q: [B, Lq, Nq, C1].
 k: [B, Lk, Nk, C1].
 v: [B, Lk, Nk, C2]. Nq must be divisible by Nk.
 q_lens: [B].
 k_lens: [B].
 dropout_p: float. Dropout probability.
 softmax_scale: float. The scaling of QK^T before applying softmax.
 causal: bool. Whether to apply causal attention mask.
 window_size: (left right). If not (-1, -1), apply sliding window local attention.
 deterministic: bool. If True, slightly slower and uses more memory.
 dtype: torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
 """
 if not FLASH_ATTN_3_AVAILABLE and not FLASH_ATTN_4_AVAILABLE:
 return flash_attn_func(
 q,
 k,
 v,
 )
 half_dtypes = (torch.float16, torch.bfloat16)
 assert dtype in half_dtypes
 assert q.device.type == "cuda" and q.size(-1) flash-attention.bak` symlink (the directory is gitignored).
 - KV-bias FA4 path imports and calls `flash_attn.cute.interface._flash_attn_fwd(...)` directly (not the wheel’s `flash_attn.cute.flash_attn_varlen_func`).

## Goal (Phase 4)

Implement and benchmark a FA4/CuTe DSL `score_mod` that matches Krea’s KV-cache bias rule, and optionally integrate it behind a feature flag:

Backend order goal:
1) FA4 `score_mod` (if available + faster)
2) Triton Kernel B (current default)
3) flex_attention fallback

We only flip defaults if FA4 clearly wins and is stable.

## KV-cache bias semantics (what we must exactly reproduce)

In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` the flex_attention fallback currently applies:

- `log_bias = log(kv_cache_attention_bias)` (natural log)
- Bias applies for keys where:
 - exclude first frame: `kv_idx >= frame_seqlen`
 - exclude current semantic block: `kv_idx = frame_seqlen` and `kv_idx Tuple[int, int]:
 """
 Get the CUDA compute capability of the current device.

 Returns:
 Tuple of (major, minor) version, e.g., (10, 0) for B200, (10, 3) for B300
 """
 global _device_capability_cache
 if _device_capability_cache is not None:
 return _device_capability_cache

 try:
 import torch
 if torch.cuda.is_available():
 _device_capability_cache = torch.cuda.get_device_capability()
 return _device_capability_cache
 except Exception as e:
 logger.warning(f"Failed to get device capability: {e}")

 return (0, 0)

def is_sm100() -> bool:
 """Check if running on SM100 (B200, compute capability 10.0)."""
 major, minor = get_compute_capability()
 return major == 10 and minor == 0

def is_sm103() -> bool:
 """Check if running on SM103 (B300, compute capability 10.3)."""
 major, minor = get_compute_capability()
 return major == 10 and minor == 3

def is_blackwell() -> bool:
 """Check if running on Blackwell architecture (SM100 or SM103)."""
 major, _ = get_compute_capability()
 return major == 10

def is_hopper() -> bool:
 """Check if running on Hopper architecture (SM90)."""
 major, _ = get_compute_capability()
 return major == 9

@functools.lru_cache(maxsize=1)
def patch_cutlass_for_sm103() -> bool:
 """
 Patch nvidia-cutlass-dsl to support SM103 (B300).

 The CUTLASS/CUTE DSL library has architecture validation lists that
 only include SM100 by default. This patches them to also allow SM103.

 Returns:
 True if patching was successful or not needed, False if failed
 """
 if not is_sm103():
 logger.debug("Not SM103, skipping CUTLASS patch")
 return True

 logger.info("Detected SM103 (B300), applying CUTLASS compatibility patch...")

 try:
 import cutlass
 import cutlass.cute as cute
 except ImportError:
 logger.warning("cutlass not installed, cannot patch for SM103")
 return False

 # Files to patch and their arch validation attributes
 # Each entry: (module_path, attr_path, valid_arch_values)
 patches = [
 # cute/arch/mbar.py - mbarrier operations
 ("cute.arch.mbar", "_VALID_ARCHS", [100, 103]),
 # cute/arch/elect.py - elect operations
 ("cute.arch.elect", "_VALID_ARCHS", [100, 103]),
 # cute/nvgpu/tcgen05/mma.py - TensorCore Gen 05 MMA
 ("cute.nvgpu.tcgen05.mma", "_VALID_ARCHS", [100, 103]),
 # cute/nvgpu/tcgen05/copy.py - TensorCore Gen 05 copy
 ("cute.nvgpu.tcgen05.copy", "_VALID_ARCHS", [100, 103]),
 # cute/nvgpu/cpasync/copy.py - async copy
 ("cute.nvgpu.cpasync.copy", "_VALID_ARCHS", [100, 103]),
 ]

 patched_count = 0
 for module_path, attr_name, valid_archs in patches:
 try:
 # Import the module
 parts = module_path.split(".")
 module = cutlass
 for part in parts:
 module = getattr(module, part, None)
 if module is None:
 break

 if module is None:
 logger.debug(f"Module {module_path} not found, skipping")
 continue

 # Check if attribute exists
 if not hasattr(module, attr_name):
 # Try to find the validation in different places
 # Some modules may have it as a local variable in functions
 logger.debug(f"Attribute {attr_name} not found in {module_path}")
 continue

 current_value = getattr(module, attr_name)
 if isinstance(current_value, (list, set, tuple)):
 # Add 103 if not present
 if 103 not in current_value:
 if isinstance(current_value, list):
 current_value.append(103)
 elif isinstance(current_value, set):
 current_value.add(103)
 else:
 setattr(module, attr_name, tuple(current_value) + (103,))
 patched_count += 1
 logger.debug(f"Patched {module_path}.{attr_name}")
 else:
 logger.debug(f"Unexpected type for {module_path}.{attr_name}: {type(current_value)}")
 except Exception as e:
 logger.debug(f"Failed to patch {module_path}.{attr_name}: {e}")

 if patched_count > 0:
 logger.info(f"Applied {patched_count} SM103 compatibility patches to CUTLASS")
 else:
 logger.info("No CUTLASS patches needed (already compatible or different structure)")

 return True

def get_recommended_backend() -> str:
 """
 Get the recommended KV-cache attention backend for the current GPU.

 Returns:
 One of: "fa4", "triton", "flex"

 Recommendations:
 - SM100 (B200): "fa4" if available, else "triton"
 - SM103 (B300): "triton" (fa4 may work after patching, but safer default)
 - SM90 (Hopper): "triton"
 - Other: "flex"
 """
 env_backend = os.getenv("SCOPE_KV_BIAS_BACKEND", "").lower()
 if env_backend:
 # User explicitly set backend, respect it
 return env_backend

 if is_sm100():
 # B200: FA4 is 1.89x faster than Triton, but default to triton for safety
 # User can set SCOPE_KV_BIAS_BACKEND=fa4 to enable
 return "triton"
 elif is_sm103():
 # B300: Triton should work, FA4 needs patching and testing
 return "triton"
 elif is_hopper():
 # Hopper: Triton works well
 return "triton"
 else:
 # Other: flex_attention as fallback
 return "flex"

def check_fa4_availability() -> Tuple[bool, str]:
 """
 Check if FA4/CUTE is available and working on the current GPU.

 Returns:
 Tuple of (is_available, reason_if_not)
 """
 if not is_blackwell():
 return False, "FA4/CUTE requires Blackwell GPU (SM100 or SM103)"

 # Apply SM103 patch if needed
 if is_sm103():
 if not patch_cutlass_for_sm103():
 return False, "Failed to patch CUTLASS for SM103"

 try:
 # Try importing FA4
 import cutlass
 import cutlass.cute as cute
 from flash_attn.cute.interface import _flash_attn_fwd
 return True, "FA4/CUTE available"
 except ImportError as e:
 return False, f"Import error: {e}"
 except Exception as e:
 return False, f"Unexpected error: {e}"

def check_flex_attention_availability() -> Tuple[bool, str]:
 """
 Check if flex_attention is available and working on the current GPU.

 Note: On SM103, torch.compile with flex_attention may fail due to
 inductor not having SM103 kernel support.

 Returns:
 Tuple of (is_available, reason_if_not)
 """
 try:
 import torch
 from torch.nn.attention.flex_attention import flex_attention

 if is_sm103():
 # flex_attention with torch.compile is known to fail on SM103
 return True, "flex_attention available but torch.compile may fail on SM103"

 return True, "flex_attention available"
 except ImportError as e:
 return False, f"Import error: {e}"

def get_capability_info() -> dict:
 """
 Get comprehensive GPU capability information for debugging.

 Returns:
 Dict with capability details
 """
 major, minor = get_compute_capability()

 fa4_available, fa4_reason = check_fa4_availability()
 flex_available, flex_reason = check_flex_attention_availability()

 return {
 "compute_capability": (major, minor),
 "sm_version": f"SM{major * 10 + minor}" if major > 0 else "Unknown",
 "architecture": (
 "Blackwell" if major == 10 else
 "Hopper" if major == 9 else
 "Ampere" if major == 8 else
 "Unknown"
 ),
 "gpu_name": _get_gpu_name(),
 "is_sm100": is_sm100(),
 "is_sm103": is_sm103(),
 "is_blackwell": is_blackwell(),
 "recommended_backend": get_recommended_backend(),
 "fa4_available": fa4_available,
 "fa4_reason": fa4_reason,
 "flex_available": flex_available,
 "flex_reason": flex_reason,
 }

def _get_gpu_name() -> str:
 """Get the GPU device name."""
 try:
 import torch
 if torch.cuda.is_available():
 return torch.cuda.get_device_name()
 except Exception:
 pass
 return "Unknown"

# Auto-apply SM103 patch on import if detected
def _auto_patch():
 """Auto-apply patches on module import."""
 if is_sm103():
 # Only log at debug level during auto-patch
 try:
 patch_cutlass_for_sm103()
 except Exception as e:
 logger.debug(f"Auto-patch for SM103 failed: {e}")

# Run auto-patch on import
_auto_patch()
```

File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/DeepResearch/2025-12-23/CuTeDSL_score_mod.md

```
md
# Guide to CuTe DSL for Custom FlashAttention Score Modifiers

## CuTe DSL Syntax Basics

**CuTe DSL Overview:** CuTe (CUDA Templates for Linear Algebra) is a Python DSL in the CUTLASS 4.x library that enables writing GPU kernels with high-level Python code. Functions decorated with @cute.jit are compiled into optimized device code (MLIR IR) that can be invoked from Python or from other CuTe DSL functions[\[1\]](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#:~:text=%60%40jit%60). CuTe DSL uses native types (e.g. cutlass.Float16, cutlass.Int32) and provides Python operators and utilities that mimic NumPy/PyTorch semantics but execute at compile-time or on the GPU.

**TensorSSA:** In CuTe DSL, arithmetic on tensor values is done with TensorSSA objects – immutable, thread-local tensor values in SSA form (Static Single Assignment) that carry shape and data type metadata[\[2\]](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5787-L5795). You can think of a TensorSSA as a vector register containing one or more elements of a given shape and type. Standard Python arithmetic and comparison operators are overloaded to work on TensorSSA (with broadcasting rules similar to NumPy). For example, if `q_idx` and `kv_idx` are TensorSSA of type Int32, then `q_idx - kv_idx` produces a new TensorSSA with the difference for each element, and `q_idx >= kv_idx` yields a Boolean TensorSSA mask[\[3\]](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5892-L5900)[\[4\]](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5900-L5908). These boolean tensors can then be used for masking via `cute.where`.

**Key DSL Functions:** CuTe provides utilities like `cute.where(cond, x, y)` to select elements from x or y based on a boolean tensor cond[\[5\]](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L6648-L6656), and `cute.full_like(t, value)` to produce a tensor filled with a given value, matching shape of tensor t[\[6\]](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L6569-L6578). These are analogous to `torch.where` and `torch.full_like`. For example, `cute.full_like(tSrS_ssa, float("-inf"))` returns a tensor shaped like `tSrS_ssa` with all entries set to negative infinity (useful for masking out scores)[\[7\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L18-L22). CuTe also exposes math operations (e.g. `cute.math.exp2` for exponent base 2) and type conversion via `.to(new_dtype)` on tensors.

**@cute.jit vs @cute.kernel:** The CuTe DSL defines two decorators: @jit for JIT-compiled functions and @kernel for full GPU kernels[\[8\]](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#:~:text=1.%20%60%40jit%60%20%E2%80%94%20Host,functions)[\[9\]](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#:~:text=%60%40kernel%60). In practice, when writing FlashAttention score modifiers, we use @cute.jit since these functions will be inlined into the larger attention kernel rather than launched on their own. You do not need to manually specify grid or block sizes for @cute.jit functions – they become device functions called from the main FlashAttention kernel. Simply decorate the Python function with @cute.jit and use CuTe operations inside; the DSL will compile it to device code when FlashAttention is invoked.

## Understanding the score_mod Function Signature

In FlashAttention v4 with CuTe DSL, you can supply a custom scoring modification function (and its backward counterpart) to alter the attention scores before softmax. The function signature is:
```

python
@cute.jit
def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 ...

```
**Parameters:**

* **tSrS_ssa** – A TensorSSA containing the raw attention score(s) for a small tile of positions. This is the value you will modify. Think of it as the "Score" tensor computed from the Q·K^T product for one or a few elements. If the kernel processes multiple score elements per thread, `tSrS_ssa` may be a vector (e.g. length 1 or 2) of those scores in registers. For example, on Blackwell GPUs with certain tile sizes, each thread might handle 2 scores at a time (`vec_size=2` when no aux tensors)[\[10\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134). If aux tensors are used (see below), the implementation often falls back to `vec_size=1`[\[10\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134), meaning one score at a time.

* **b_idx, h_idx** – The **batch index** and **head index** (within the batch) for the current score(s). These are typically TensorSSA of type Int32 (often scalar shape). Use these if your modification needs to vary per batch or per attention head. For instance, you might use `b_idx` to index a batch-specific bias vector[\[11\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L80-L88), or `h_idx` to apply a head-dependent scaling (as in ALiBi where bias depends on head index[\[12\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L44-L52)).

* **q_idx, kv_idx** – The **query token index** and **key/value token index** for the score(s) being computed. These are crucial for position-based masking or biases. They are Int32 indices relative to the start of the current processing block (for fixed-length inputs, this is the absolute position in the sequence; for variable-length, see `seqlen_info` below). For example, to implement causal masking you compare `q_idx` and `kv_idx` and mask out where `kv_idx > q_idx`[\[7\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L18-L22). These indices can be used in arithmetic: e.g. `diff = q_idx - kv_idx` yields the relative position offset, which you can take absolute value of, etc.[\[13\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L24-L29).

* **seqlen_info** – A struct carrying sequence length metadata, used mainly for **variable-length sequences** (when different sequences in a batch have different lengths or when using packed sequences). It typically contains offsets like `offset_q` and `offset_k` that convert local indices to global indices[\[14\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L124-L132). In a multi-batch varlen scenario, `q_idx` and `kv_idx` reset to 0 at the start of each sequence or each block; adding `seqlen_info.offset_q` (or `offset_k`) yields the **global index** in the entire batch's padded sequence. If you are not dealing with packed sequences or paged attention, you might not need to use `seqlen_info` at all (it could be ignored or None). However, if implementing global attention biases (e.g. absolute position biases across the whole batch sequence), use these offsets. For example, FlashAttention provides `score_mod_global_kv_bias` which uses `kv_idx_global = kv_idx + seqlen_info.offset_k` to index into a global bias table[\[14\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L124-L132).

* **aux_tensors** – A list of auxiliary tensors for any additional data your score modification needs (passed at runtime without requiring kernel recompilation). This is how you supply **runtime parameters** such as learned bias vectors or masks. The CuTe DSL treats `aux_tensors` as an optional list of global memory buffers that you can index inside the `score_mod` function. For instance, you might pass in a tensor of length B (batch biases) or length L (position biases) in the `aux_tensors` list. Inside `score_mod`, you can read from these by first creating a fragment and storing the index, then loading from the global tensor. For example, to add a per-batch bias from an aux tensor, you can do:
```

python
batch_bias = aux_tensors[0] # global tensor of shape [batch_size]
dtype = batch_bias.element_type # e.g. cutlass.Float16
b_frag = cute.make_fragment(1, cutlass.Int32)
b_frag.store(b_idx) # store batch index into fragment
bias_frag = cute.make_fragment(1, dtype)
bias_frag[0] = batch_bias[b_frag[0]] # load bias for this batch index
bias_val = (bias_frag.load()).to(cutlass.Float32)
return tSrS_ssa + bias_val # add the bias to the scores

```
[\[11\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L80-L88). In this snippet, `cute.make_fragment(1, T)` creates a register fragment of length 1. We store the index in an Int32 fragment, then use it to index the `batch_bias` tensor, storing the result in a fragment of the bias's dtype. Finally, we load that fragment (getting a TensorSSA scalar) and cast to Float32 for accumulation before adding to `tSrS_ssa`. This pattern is used because direct indexing of a global memory tensor with a TensorSSA index isn't allowed; we must go through a fragment (which acts like a register). If you have multiple aux tensors, they will appear as `aux_tensors[0]`, `aux_tensors[1]`, etc. You can pass any number of aux tensors (FlashAttention threads them through as a list and marks the kernel to use `vec_size=1` if any aux is present[\[10\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134)).

**Return value:** The `score_mod` function should return a modified score tensor (same shape as `tSrS_ssa`). Typically you will either add biases to `tSrS_ssa` or replace certain elements with $-\infty$ (for masking) or other computed values. For correctness, ensure you **do not change the shape** or data type of the score tensor (aside from upcasting to float for accumulation if needed). Also, avoid side effects – treat the inputs as read-only except for constructing the returned tensor.

## Examples of Common score_mod Patterns

Below are several typical attention score modifications and how to implement them in the CuTe DSL. Each example is a self-contained `@cute.jit` function that could be passed as `score_mod`:

* **Causal Masking (no attending to future tokens):** Set scores for any key position greater than the query position to $-\infty$. You can do this by comparing indices and using `cute.where` to select either the original score or a large negative value. For example:
```

python
@cute.jit
def score_mod_causal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 mask = operator.ge(q_idx, kv_idx) # True where q_idx >= kv_idx
 return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

```
[\[7\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L18-L22). Here `operator.ge` produces a Boolean mask (TensorSSA) of the same shape, and we select `tSrS_ssa` when the mask is True, or $-\infty$ otherwise. This effectively zeroes out attention to future positions after softmax (since $\exp(-\infty) \approx 0$). Note that `cute.full_like(tSrS_ssa, float("-inf"))` will create a tensor of negative infinities with the appropriate shape and dtype.

* **Sliding Window (local attention):** Only allow attending to tokens within a certain window of the query (e.g. +/- 256 positions), mask out distant positions. This can be done by checking the absolute difference `|q_idx - kv_idx|`. For instance, to enforce a window of 256 on each side:
```

python
@cute.jit
def score_mod_sliding_window(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 diff = q_idx - kv_idx
 abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
 mask = operator.le(abs_diff, cute.full_like(abs_diff, 256))
 return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

```
[\[15\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L57-L62). We compute `abs_diff` by invoking the MLIR math absolute value (`mlir_math.absi`) on the difference. The mask is True for pairs within 256 positions, False otherwise, and we mask out scores outside that range. (CuTe's broadcasting and vectorized ops handle that `abs_diff` could be a vector.) This approach is flexible – by changing the constant or making it an aux parameter, you can adjust window size.

* **ALiBi (Position-Dependent Bias per Head):** *ALiBi* (Attention Linear Bias) is a strategy where each attention head has a fixed slope by which it linearly penalizes farther keys. In practice, this means subtracting a term $s_h \cdot |q - k|$ from the score, where $s_h$ is a head-specific slope. In CuTe DSL, you can compute the slope based on `h_idx` and apply the bias. For example:
```

python
@cute.jit
def score_mod_alibi(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 score = tSrS_ssa.to(cutlass.Float32) # upcast scores to float for accumulation
 # Compute slope = 2^(- (h_idx+1) / 8). Here 8 is an example factor controlling slope differences.
 slope_exp = (h_idx + cute.full_like(h_idx, 1)) * cute.full_like(h_idx, -8)
 slope = cute.math.exp2(slope_exp.to(cutlass.Float32) * cute.full_like(score, 0.125))
 # Apply bias = slope * |q_idx - kv_idx|
 diff = q_idx - kv_idx
 abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype).to(cutlass.Float32)
 return score - slope * abs_diff

```
This example follows the FlashAttention implementation[\[16\]](https://github.com/Jackmin801/flash-attention/blob/caa34bdef07079c1bc721721131d996bfa4e2317/tests/cute/test_score_mod.py#L66-L74)[\[17\]](https://github.com/Jackmin801/flash-attention/blob/caa34bdef07079c1bc721721131d996bfa4e2317/tests/cute/test_score_mod.py#L76-L84): it calculates slope = $2^{-(h+1)/8}$ (the constants 0.125 etc. implement that) and then subtracts `slope * |q - k|` from the score. We use `.to(Float32)` to avoid overflow/precision issues since the biases might be fractional. The result is that higher heads (larger `h_idx`) get a smaller slope (hence smaller penalty for distance) while lower heads heavily penalize distant positions.

* **Block-Diagonal Mask (e.g. for sparse patterns or group-by-group attention):** Only allow attention within certain blocks of the sequence. For instance, if queries and keys are divided into blocks of size 64 that should attend only within the same block, you can mask across blocks. Implementation: compute block indices by integer division and compare:
```

python
@cute.jit
def score_mod_block_diagonal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 q_block = q_idx // 64
 kv_block = kv_idx // 64
 mask = operator.eq(q_block, kv_block)
 return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

```
[\[18\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L65-L70). This yields $-\infty$ for any query-key pair not in the same 64-length block. You can generalize this to arbitrary block sizes or patterns. This pattern is useful for block-sparse attention or grouping heads into segments.

* **Custom Piecewise-Constant Bias (use case – KV cache sampling):** If you want to *add a bias* that changes at certain positions (rather than a mask), you can use a condition to select between different bias values. One approach is to predefine the bias values and thresholds and pass them via `aux_tensors`. For example, suppose you want to add bias B1 to tokens in the first region of the cache and B2 to the rest. You could pass a tensor like `region_bias = [B1, B2]` as `aux_tensors[0]`, and a cutoff index as an integer constant or aux as well. Then:
```

python
@cute.jit
def score_mod_two_region_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 kv_idx_global = kv_idx + seqlen_info.offset_k # global position if needed
 cutoff = cute.full_like(kv_idx_global, CUT_OFF) # CUT_OFF is an int or from aux_tensors
 mask_segment = operator.lt(kv_idx_global, cutoff)
 bias_vals = aux_tensors[0] # tensor of [B1, B2]
 # Select B1 or B2 based on mask
 frag = cute.make_fragment(1, cutlass.Int8) # or Int32 if index larger
 frag.store(mask_segment.to(cutlass.Int8)) # store bool as 0/1
 bias_frag = cute.make_fragment(1, bias_vals.element_type)
 bias_frag[0] = bias_vals[frag[0]]
 bias_val = bias_frag.load().to(cutlass.Float32)
 return tSrS_ssa + bias_val

```
In this pseudocode, we map the boolean condition to 0 or 1 and index into `bias_vals` to pick the corresponding bias. The result is that for keys before the cutoff, B1 is added; for keys after, B2 is added. You can extend this idea to multiple regions (though at some point a lookup table approach might be clearer).

Another approach is to directly encode piecewise logic with nested `where` or logical operations. For example, consider combining a causal mask and a range-based mask: FlashAttention's "stress test" example shows how to compose conditions[\[19\]](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L275-L284). They compute `is_causal = (q_idx >= kv_idx)` and `is_nearby = (|q_idx_global - kv_idx_global| =4.3.3`, `apache-tvm-ffi`
 - Details: `notes/FA4/phase4-fa4-score-mod.md`
 - Test script: `scripts/test_fa4_kv_bias.py`
- **RoPE Phase 3 Step 2 COMPLETE**: v2 fused kernel achieves **20.2 FPS** (fixed 17.8 FPS regression from v1 padding overhead)
 - v2 uses unified 64-pair arange (no per-chunk padding)
 - 2.2x faster than Step 0 baseline in microbench
 - Tag: `v0.3.0-phase3-step2`
 - Details: `notes/FA4/phase3-triton-rope.md`
- Kernel B (KV-cache bias): Triton integrated + pinned for B200; real runs show ~20% FPS uplift at 320x576; profiling shows `p_bias≈89.5%` of **attention-kernel** time (Kernel A is small).
- Fine-grained `self_attn` breakdown (targeting "what moves the needle"): `self_attn_kv_bias` 27.4%, `qkv_projection` 20.8%, `rope_apply` 15.8%, `self_attn_block_mask` 3.4% → kernel work needs to shift toward RoPE/QKV (and/or a much faster Kernel B backend).
- Next bets (highest upside per effort): (1) ~~RoPE fusion~~ **DONE**, (2) ~~FA4/CUTE score_mod~~ **DONE**, (3) Make FA4 the default backend, (4) RoPE Step 3 tuning (BLOCK_M/H, Q+K fusion), (5) Update pyproject.toml with FA4 deps.

Amdahl cheat sheet (based on latest fine-grained profile; assumes `self_attn≈51%` of total step time)
- 2× `self_attn_kv_bias` (~14% of total) → ~7–8% faster end-to-end
- 2× `qkv_projection` (~11% of total) → ~6% faster end-to-end
- 2× `rope_apply` (~8% of total) → ~4% faster end-to-end

Context (why this exists)
- Krea Realtime’s self-attn uses `torch.nn.attention.flex_attention` for:
 - KV recomputation with a block-causal `block_mask`
 - KV-cache sampling with a piecewise-constant `score_mod` bias on past-frame tokens
- Those FlexAttention paths are currently the main blocker to “FA4-class” performance for Krea Realtime self-attn; cross-attn is already on FA4 in other parts of the stack.
- Primary codepaths:
 - Krea self-attn callsites: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
 - Microbench harness for the same semantics: `scripts/bench_blockwise_attn.py`
 - Triton prototype kernel track: `scripts/triton_sdpa.py`
 - FA4 backend integration (for non-Flex paths): `src/scope/core/pipelines/wan2_1/modules/attention.py`
```

File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/B300-SESSION-STATE.md

```
md
# B300 Session State - 2025-12-23 (Updated)

## Current Status

**B300 is at 8.8 FPS - suspiciously constant regardless of changes made**

## Key Discovery This Session

The `flash-attention` symlink was shadowing the working FA4 package. Removed it, FA4 now imports correctly:
- `FLASH_ATTN_4_AVAILABLE: True`
- FA4 benchmarks show 0.38ms for KV-cache attention (vs 1.6ms Triton)

**BUT FPS stays at exactly 8.8 regardless of:**
- Enabling FA4 for main attention
- Setting `SCOPE_KV_CACHE_ATTENTION_BIAS=1.0` (bypasses Triton Kernel B)
- Changing codec MAX_FRAME_RATE from 8 to 30

This suggests a **hard limiter we haven't found yet**.

## Two Codex Agents Working

### Codex1: FA4 score_mod fix (preferred approach)
- Patching `flash-attention.bak` to replace `FastDivmodDivisor` with custom `FastDivmod`
- Files modified: `fast_math.py`, `tile_scheduler.py`, `paged_kv.py`, `flash_fwd.py`, `flash_fwd_sm100.py`, `flash_fwd_combine.py`, `flash_bwd_sm100.py`
- **Status: Still working**

### Codex2: Segment-combine fallback
- Added `_kv_bias_flash_combine()` - splits KV into 3 segments, runs FA4 on each, merges with LSE
- Auto-detects SM103 → defaults to "flash" backend
- **Status: Complete, merged into causal_model.py**

## Current Code State

| Component | Status |
|-----------|--------|
| `flash-attention` symlink | Missing (removed) |
| `flash-attention.bak/` | Has Codex1's FastDivmod patches (incomplete) |
| `causal_model.py` | Has Codex2's flash-combine + Codex1's path extension |
| `webrtc.py` | MAX_FRAME_RATE changed 8→30 |
| `pipeline.py` | Added SCOPE_KV_CACHE_ATTENTION_BIAS env var |

## Backend Selection on B300
```

python
_KV_BIAS_BACKEND = "flash" if _is_sm103() else "triton"

```
- `fa4`: FA4 with score_mod (needs Codex1 to finish + symlink)
- `flash`: Segment-combine (Codex2, currently active on B300)
- `triton`: Triton Kernel B (slow on B300: 1.6ms)
- `flex`: flex_attention fallback

## To Test When Codex1 Finishes
```

bash
# Recreate symlink
ln -s flash-attention.bak flash-attention

# Run with FA4 score_mod
SCOPE_KV_BIAS_BACKEND=fa4 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope

```
## To Test Current State (Codex2's approach)
```

bash
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas uv run daydream-scope

```
## Mystery: Why Always 8.8 FPS?

Investigated but not found:
- Codec cap (was 8, changed to 30 - no effect)
- KV-cache bias path (bypassed with bias=1.0 - no effect)
- Attention backend (FA4 vs Triton - no effect on FPS)

Possible unexplored causes:
- FPS calculation/reporting bug
- Some synchronization in frame_processor.py
- Input frame rate limiting
- Something in the actual pipeline execution path

## Files to Investigate

- `/root/scope/src/scope/server/frame_processor.py` - FPS calculation at line 296
- `/root/scope/src/scope/server/tracks.py` - Frame rate control
- The actual pipeline call path that generates frames

## Environment
```

bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
# Optional overrides:
export SCOPE_KV_BIAS_BACKEND=flash # or fa4, triton
export SCOPE_KV_CACHE_ATTENTION_BIAS=1.0 # disable bias for plain FA4
export PROFILE_ATTENTION=1 # enable profiling

```

```

File: /Users/daviddickinson/Projects/Lora/scope/scripts/patch_cutlass_sm103.py

```
py
#!/usr/bin/env python3
"""
Patch nvidia-cutlass-dsl to support SM103 (B300/B300A) architecture.

FA4 (Flash Attention 4) with CUTE DSL only officially supports sm_100a/sm_100f.
The B300 uses sm_103, which requires patching the arch validation checks.

⚠️ WARNING: nvidia-cutlass-dsl conflicts with PyTorch's inductor!
 The `cutlass` module shadows torch._inductor's cutlass utilities.
 FA4 deps must be uninstalled to run normal pipelines on B300.

 To uninstall:
 uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder

Usage:
 python scripts/patch_cutlass_sm103.py

Or:
 uv run python scripts/patch_cutlass_sm103.py
"""

import os
import re
import sys
from pathlib import Path

def find_venv_site_packages():
 """Find the site-packages directory for the current venv."""
 # Try common locations
 candidates = [
 Path(".venv/lib/python3.12/site-packages"),
 Path(".venv/lib/python3.11/site-packages"),
 Path(".venv/lib/python3.10/site-packages"),
 ]

 for candidate in candidates:
 if candidate.exists():
 return candidate

 # Fallback to sys.path
 for p in sys.path:
 if "site-packages" in p:
 return Path(p)

 return None

def patch_file(filepath: Path, search: str, replace: str) -> bool:
 """Patch a file by replacing search pattern with replace string."""
 if not filepath.exists():
 return False

 content = filepath.read_text()
 if search in content and replace not in content:
 new_content = content.replace(search, replace)
 filepath.write_text(new_content)
 return True
 return False

def main():
 site_packages = find_venv_site_packages()
 if not site_packages:
 print("ERROR: Could not find site-packages directory")
 sys.exit(1)

 cutlass_base = site_packages / "nvidia_cutlass_dsl/python_packages/cutlass"

 if not cutlass_base.exists():
 print("ERROR: nvidia-cutlass-dsl not installed")
 sys.exit(1)

 print(f"Patching cutlass at: {cutlass_base}")

 # Files that need arch list patches
 patch_patterns = [
 # Pattern: add sm_103 variants after sm_100f
 ('"sm_100f",', '"sm_100f", "sm_103", "sm_103a",'),
 # Pattern: add sm_103 variants in multi-line lists ending with sm_100f
 ('"sm_100f",\n ]', '"sm_100f",\n "sm_103",\n "sm_103a",\n ]'),
 ('"sm_100f",\n ]', '"sm_100f",\n "sm_103",\n "sm_103a",\n ]'),
 # Pattern: add sm_103 to lists ending with sm_100a (MX format ops)
 ('"sm_100a",\n ]', '"sm_100a",\n "sm_103",\n "sm_103a",\n ]'),
 ]

 files_to_patch = [
 cutlass_base / "cute/arch/mbar.py",
 cutlass_base / "cute/arch/elect.py",
 cutlass_base / "cute/nvgpu/tcgen05/mma.py",
 cutlass_base / "cute/nvgpu/tcgen05/copy.py",
 cutlass_base / "cute/nvgpu/cpasync/copy.py",
 ]

 patched_count = 0
 for filepath in files_to_patch:
 if not filepath.exists():
 print(f" SKIP: {filepath.name} (not found)")
 continue

 file_patched = False
 for search, replace in patch_patterns:
 if patch_file(filepath, search, replace):
 file_patched = True

 if file_patched:
 print(f" PATCHED: {filepath.relative_to(cutlass_base)}")
 patched_count += 1
 else:
 # Check if already patched
 content = filepath.read_text()
 if "sm_103" in content:
 print(f" OK: {filepath.relative_to(cutlass_base)} (already patched)")
 else:
 print(f" SKIP: {filepath.relative_to(cutlass_base)} (no changes needed)")

 print(f"\nPatched {patched_count} files for SM103 support")
 print("\nTo test FA4:")
 print(' uv run python -c "from flash_attn.cute import flash_attn_func; print(\'FA4 OK\')"')

if __name__ == "__main__":
 main()
```

File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/b300-investigation.md

```
md
# B300 (SM103) Investigation Notes

## Environment
- **GPU**: NVIDIA B300 SXM6 AC → now testing on **B200**
- **Compute Capability**: SM103 (10.3) for B300, **SM100 (10.0) for B200**
- **PyTorch**: 2.8.0+cu128
- **CUDA**: 12.8 (historical on B300)
- **Python**: 3.12.3

## Update: SM103 requires newer `ptxas` (CUDA 12.9+)

The earlier B300 failures included:
- `ptxas fatal: Value 'sm_103a' is not defined for option 'gpu-name'`
- `NoValidChoicesError: target: flex_attention` during `torch.compile` autotune

These can happen when Triton/Inductor is assembling with an older `ptxas` (e.g., CUDA 12.8).

**Fix:** ensure Triton uses a CUDA toolkit whose `ptxas` supports SM103, e.g. CUDA **12.9+**.

On the machine backing this repo, CUDA 12.9 is installed and supports SM103:
```

bash
/usr/local/cuda-12.9/bin/ptxas --help | rg "sm_103"

```
Recommended env var (set before importing anything that triggers Triton compilation):
```

bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

```
Note: Triton often uses its bundled `ptxas` by default (inside the Python wheel), which may be
too old for SM103. The above override forces the system toolkit `ptxas` instead.

If `torch.compile(flex_attention, ...)` still fails after this, *then* it’s likely a genuine SM103 support gap in the PyTorch/Triton stack.

### B200 vs B300
- B200 = SM100 (officially supported by PyTorch flex_attention and FA4/CUTE)
- B300 = SM103 (NOT supported, requires patches)

---

## B200 (SM100) Test Results ✅

### Compiled flex_attention WORKS on SM100
```

bash
# This works on B200, fails on B300:
flex_attention = torch.compile(_flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs')

```
Autotune output shows kernel selection working:
```

AUTOTUNE flex_decoding(1x8x1x64x64, ...)
 triton_flex_decoding_1 0.0061 ms 100.0%

```
### Attention Backend Status on B200
- FA2: ✅ Available
- FA3: ❌ (H100 only)
- FA4: ✅ Available (local flash-attention repo)
- SageAttn: ❌ (no cp312 wheel)

### nvidia-cutlass-dsl Present but No Conflict on SM100
Even with nvidia-cutlass-dsl 4.3.3 installed, flex_attention compilation succeeds.
The conflict only manifests on SM103 (B300).

### Conclusion
**SM100 (B200) works out of the box with PyTorch 2.8.** No patches needed for flex_attention.
SM103 (B300) requires either:
1. Skip torch.compile for flex_attention on SM103
2. Wait for PyTorch to add SM103 kernel support

### FA4 Version Conflict on B200
The local `flash-attention/` repo breaks FA4! Here's why:

1. `attention.py` has `_extend_flash_attn_path()` that inserts local repo into path
2. Local repo's cute requires `FastDivmodDivisor` (cutlass-dsl >= 4.3.3)
3. Pip flash_attn's cute requires `cutlass.utils.ampere_helpers` (cutlass-dsl 4.1.0)

**Fix**: Rename or remove local flash-attention repo:
```

bash
mv /root/scope/flash-attention /root/scope/flash-attention.bak

```
Then FA4 will use the pip-installed version which works with cutlass-dsl 4.1.0.

---

## Issue 1: FA4 (CUTE) - RESOLVED FOR BENCHMARKS

### Problem
FA4 (Flash Attention 4 via CUTE DSL) only officially supports sm_100a/sm_100f, not sm_103.

### Solution
Created `scripts/patch_cutlass_sm103.py` that patches nvidia-cutlass-dsl to add sm_103 support:
- Patches arch validation lists in `cute/arch/mbar.py`, `cute/arch/elect.py`
- Patches `cute/nvgpu/tcgen05/mma.py`, `cute/nvgpu/tcgen05/copy.py`, `cute/nvgpu/cpasync/copy.py`

### Result
FA4 benchmarks work after patching:
```

Config Batch Seq Heads Dim Time (ms) TFLOPS
Small 2 512 8 64 0.046 11.59
Medium 2 1024 8 64 0.042 51.05
Large 2 2048 8 128 0.043 404.03
XL 1 4096 8 128 0.046 739.98

```
---

## Issue 2: nvidia-cutlass-dsl Conflicts with PyTorch Inductor - BLOCKER

### Problem
When FA4 deps are installed, `nvidia-cutlass-dsl` provides a `cutlass` module that shadows PyTorch's internal cutlass utilities in `torch._inductor`.

### Symptom
```

NoValidChoicesError: target: flex_attention
...
AttributeError: module 'cutlass' has no attribute 'CACHE_FILE'

```
### Workaround
Uninstall FA4 deps before running normal pipelines:
```

bash
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder

```
### Status
FA4 can only be used in isolation (benchmarks). Cannot coexist with render_timeline.

---

## Issue 3: flex_attention on B300 - UNDER INVESTIGATION

### Problem
Even without FA4 deps, `render_timeline` fails on B300 with `NoValidChoicesError` for `flex_attention`.

### Root Cause Hypothesis
Most likely: Triton/Inductor is using an older `ptxas` that doesn’t recognize `sm_103a`, so all candidate kernels fail to assemble and autotune surfaces `NoValidChoicesError`.

Secondary possibility (if using CUDA 12.9+ `ptxas` doesn’t fix it): PyTorch 2.8’s inductor lacks complete SM103 kernel support for flex_attention’s `"max-autotune-no-cudagraphs"` mode.

### What We Tried

#### Attempt 1: TORCH_COMPILE_DISABLE=1
**Result**: Different error - `'tuple' object has no attribute 'unflatten'`

**Analysis**: When torch.compile is disabled, something in the attention path returns a tuple instead of a tensor. The code at `causal_model.py:675` does:
```

python
x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

```
Where `y = self.self_attn(...)`. So `self_attn` returns a tuple when uncompiled.

**Key insight**: flex_attention itself works fine uncompiled (tested independently), so the tuple must come from elsewhere in the model.

#### Attempt 2: Direct flex_attention test
```

python
TORCH_COMPILE_DISABLE=1 python -c "
from torch.nn.attention.flex_attention import flex_attention
q = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device='cuda')
out = flex_attention(q, k, v)
print(type(out)) # <class 'torch.Tensor'> - works!
"

```
**Result**: Works fine, returns tensor.

### Hypotheses to Test Next

1. **Something else in `self_attn` returns tuple**: The `CausalWanSelfAttention.forward()` method may have conditional paths that return different types

2. **PyTorch version mismatch**: flex_attention API may have changed between versions, and the code assumes compiled behavior

3. **SM103-specific codepath**: There may be architecture detection that triggers different behavior on B300

4. **The `attention()` fallback path**: Line 571 has `x = attention(roped_query, cached_k, cached_v)` as fallback - this might return tuple

### Code Locations to Check
- `causal_model.py:562-579` - flex_attention call and return
- `causal_model.py:571` - `attention()` fallback function
- `attention.py` - the `attention()` function implementation
- Check if `kv_cache_attention_bias` affects which path is taken

---

## Key Files

### Modified for FA4 Support
- `pyproject.toml` - Added cp312 flash-attn wheels, fa4 dependency group
- `.python-version` - Changed to 3.12
- `scripts/patch_cutlass_sm103.py` - SM103 arch patches

### Relevant Source Files
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` - flex_attention usage
- `src/scope/core/pipelines/wan2_1/modules/attention.py` - attention backends (FA2/FA3/FA4)

---

## LATEST FINDINGS (Claude session - save state)

### Local flash-attention Directory Found!
There's a cloned `flash-attention` repo at `/root/scope/flash-attention/` that gets injected into the path via `_extend_flash_attn_path()` in attention.py:
```

python
for parent in base_path.parents:
 candidate = parent / "flash-attention" / "flash_attn"
 if candidate.is_dir():
 _flash_attn.__path__.insert(0, candidate_str)

```
This local repo has its own `flash_attn/cute/` directory which may work differently!

### Current Attention Backend Status on B300 (Python 3.12)
```

SAGEATTN_AVAILABLE: False (no cp312 wheel)
FLASH_ATTN_2_AVAILABLE: True
FLASH_ATTN_3_AVAILABLE: False (H100 only)
FLASH_ATTN_4_AVAILABLE: True (from local flash-attention repo!)

```
**KEY INSIGHT**: FA4 shows as available because the LOCAL `flash-attention/flash_attn/cute/` is being loaded, NOT the pip-installed package. The local version might have different dependencies or behavior!

### The Actual Error Flow
1. Without FA4 deps: `render_timeline` fails with `NoValidChoicesError` for `flex_attention` in torch.compile
2. With `TORCH_COMPILE_DISABLE=1`: Different error - `'tuple' object has no attribute 'unflatten'`
3. The tuple error comes from `y = self.self_attn(...)` at causal_model.py:658
4. With `DISABLE_FLASH_ATTENTION_4=1`: **SAME flex_attention error** - this is NOT about FA4!

### ROOT CAUSE IDENTIFIED
The problem is in `causal_model.py` lines 26-28:
```

python
flex_attention = torch.compile(
 flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

```
This compiles flex_attention at MODULE IMPORT TIME. PyTorch 2.8's inductor doesn't have SM103 kernel support for flex_attention's max-autotune mode.

**This is a PyTorch + B300/SM103 compatibility issue, NOT an FA4 issue!**

### Next Investigation Steps
1. **Check local flash-attention cute interface** - does it return tuple?
2. **Remove/rename local flash-attention dir** to use only pip package
3. **Check if FA4 path is being taken** despite cutlass-dsl not installed
4. **The `attention()` function** falls back to flash_attn when sage not available

### Working Commands
```

bash
# Check attention backends
uv run python -c "from scope.core.pipelines.wan2_1.modules.attention import *; print(f'FA4: {FLASH_ATTN_4_AVAILABLE}')"

# Test without torch.compile
TORCH_COMPILE_DISABLE=1 uv run render_timeline timeline.json out.mp4 --preset standard

# Uninstall FA4 deps (if needed)
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder

```
---

## POTENTIAL FIX

Modify `causal_model.py` to conditionally compile flex_attention:
```

python
import torch

def is_sm103():
 if torch.cuda.is_available():
 major, minor = torch.cuda.get_device_capability(0)
 return major == 10 and minor == 3
 return False

from torch.nn.attention.flex_attention import flex_attention as _flex_attention

# Don't compile on SM103 (B300) - inductor doesn't support it yet
if not is_sm103():
 flex_attention = torch.compile(
 _flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
 )
else:
 flex_attention = _flex_attention # Use uncompiled version

```
OR use environment variable:
```

python
import os
if os.getenv("DISABLE_FLEX_ATTENTION_COMPILE", "0") == "0":
 flex_attention = torch.compile(...)
else:
 flex_attention = _flex_attention

```
**CAVEAT**: When we tested with `TORCH_COMPILE_DISABLE=1` (global), we got:
```

'tuple' object has no attribute 'unflatten'

```
So uncompiled flex_attention may also have issues - need to debug why `self_attn` returns tuple.

---

## Next Steps

1. **Try the fix above** - modify causal_model.py to skip compile on SM103

2. **Test uncompiled flex_attention** - might have the tuple issue still

3. **Check if H100 works**: Compare behavior on different GPU

4. **Debug the tuple return**: If uncompiled fails, trace where tuple comes from

---

## 2025-12-23 Analysis: FA4 + score_mod SM103 Compatibility

### Overview
Analysis of what's needed to run the FA4/CUTE `score_mod` path (Kernel B) on B300 (SM103).

### Current State on B200 (SM100)
- **Kernel B with FA4/CUTE score_mod**: 0.54ms (1.89x faster than Triton)
- **Kernel B with Triton**: 1.02ms
- Backend selection: `SCOPE_KV_BIAS_BACKEND=fa4|triton|flex`

### CUTE Files Touched by `_flash_attn_fwd` with `score_mod`

The FA4 score_mod path invokes these files when `compute_capability == 10`:
```

flash_attn/cute/
├── interface.py # Entry point - _flash_attn_fwd()
├── flash_fwd_sm100.py # FlashAttentionForwardSm100 class
│ ├── imports tcgen05 (mma, copy)
│ ├── imports cpasync (copy)
│ └── uses apply_score_mod_inner from softmax.py
├── softmax.py # SoftmaxSm100, apply_score_mod_inner()
│ └── score_mod is called here with (batch_idx, head_idx, q_idx, kv_idx)
├── blackwell_helpers.py # tcgen05 gemm operations
├── mma_sm100_desc.py # MMA instruction descriptors
└── pipeline.py # Pipeline utilities

```
### nvidia-cutlass-dsl Files Requiring SM103 Patches

The existing `scripts/patch_cutlass_sm103.py` patches:

| File | Purpose | Patch Type |
|------|---------|------------|
| `cute/arch/mbar.py` | mbarrier operations | Add 103 to `_VALID_ARCHS` |
| `cute/arch/elect.py` | elect operations | Add 103 to `_VALID_ARCHS` |
| `cute/nvgpu/tcgen05/mma.py` | TensorCore Gen 05 MMA | Add 103 to `_VALID_ARCHS` |
| `cute/nvgpu/tcgen05/copy.py` | TensorCore Gen 05 copy | Add 103 to `_VALID_ARCHS` |
| `cute/nvgpu/cpasync/copy.py` | Async copy | Add 103 to `_VALID_ARCHS` |

**Verdict**: The patch covers ALL tcgen05/cpasync modules used by the score_mod path.

### New SM103 Compatibility Layer

Created: `src/scope/core/compat/sm103.py`

Features:
1. **Auto-detection**: `is_sm100()`, `is_sm103()`, `is_blackwell()`
2. **Auto-patching**: `patch_cutlass_for_sm103()` - patches cutlass-dsl on import
3. **Backend recommendation**: `get_recommended_backend()` - returns safe defaults
4. **Capability info**: `get_capability_info()` - debug helper

Usage:
```

python
from scope.core.compat.sm103 import (
 is_sm103, patch_cutlass_for_sm103, get_recommended_backend
)

# Auto-detect and get safe backend
backend = get_recommended_backend() # "triton" on SM103, "fa4" or "triton" on SM100

```
### SM103 Adaptation Risks

| Component | Risk Level | Notes |
|-----------|------------|-------|
| Triton Kernel B | **Low** | JIT compiles to target GPU, may need retuning |
| Triton RoPE v2 | **Low** | Same as above |
| FA4/CUTE score_mod | **Medium** | Needs patching, untested on real SM103 |
| flex_attention (compiled) | **High** | torch._inductor lacks SM103 kernels |
| FA4 → Triton fallback | **Low** | Already implemented in causal_model.py |

### Recommended Testing on B300
```

bash
# 1. Check device capability
python -c "import torch; print(torch.cuda.get_device_capability())"
# Expected: (10, 3)

# 2. Check capability info from new compat module
python -c "from scope.core.compat.sm103 import get_capability_info; import pprint; pprint.pprint(get_capability_info())"

# 3. Test Triton Kernel B (should work)
SCOPE_KV_BIAS_BACKEND=triton uv run python scripts/triton_sdpa.py --kernel-b

# 4. Test FA4 score_mod (after auto-patching)
SCOPE_KV_BIAS_BACKEND=fa4 uv run python -c "
from scope.core.compat.sm103 import patch_cutlass_for_sm103
patch_cutlass_for_sm103()
from flash_attn.cute.interface import _flash_attn_fwd
import torch
q = torch.randn(1, 64, 8, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 256, 8, 128, dtype=torch.bfloat16, device='cuda')
out, _ = _flash_attn_fwd(q, k, v, causal=False)
print('FA4 basic test passed:', out.shape)
"

# 5. Force PTX JIT to detect missing PTX coverage
CUDA_FORCE_PTX_JIT=1 SCOPE_KV_BIAS_BACKEND=fa4 uv run python -c "
from flash_attn.cute.interface import _flash_attn_fwd
print('PTX JIT test passed')
"

```
### Questions to Resolve

1. **Does SM103 share the same tcgen05 instruction set as SM100?**
 - Likely yes (same Blackwell family), but untested
 - PTX should JIT to SM103 if cubin doesn't match

2. **Will Triton kernels need retuning for B300?**
 - B300 may have different memory bandwidth, L2 size, SM count
 - Block sizes tuned for B200 may not be optimal

3. **Can we use FA4 as primary on SM103?**
 - Only after validation on real B300 hardware
 - Default should remain Triton until confirmed

---

## 2025-12-23: BREAKTHROUGH - Triton Works on B300!

### The Fix: CUDA 12.9 ptxas

**Root cause**: CUDA 12.8's ptxas doesn't support `sm_103a`. Triton 3.4.0 bundles an older ptxas.

**Solution**: Install CUDA 12.9 and point Triton to its ptxas:
```

bash
# Install CUDA 12.9 (has SM103 support)
sudo apt install cuda-toolkit-12-9

# Set the env var (added to .venv/bin/activate)
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

```
### Verified Working on B300 (SM103)
```

GPU: NVIDIA B300 SXM6 AC
Compute Capability: (10, 3) = SM103

Kernel B Correctness Tests: ALL PASS
- Test 1 (B=1, H=1, Lq=64, Lk=128, D=16): PASS (max error: 0.0039)
- Test 2 (B=1, H=4, Lq=128, Lk=256, D=64): PASS (max error: 0.0039)
- Test 3 (B=1, H=16, Lq=4680, Lk=9360, D=128): PASS (max error: 0.0020)

Kernel B Benchmark (4680x9360, H=16, D=128):
- flex_attention: 1.095 ms
- Triton Kernel B: 0.977 ms (10.8% faster)

```
### B300 Environment Setup
```

bash
# In .venv/bin/activate, add:
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

```
### Updated Status

| Component | B200 (SM100) | B300 (SM103) |
|-----------|--------------|--------------|
| Triton Kernel B | ✅ Works | ✅ Works (with CUDA 12.9 ptxas) |
| flex_attention | ✅ Works | ✅ Works |
| FA4/CUTE score_mod | ✅ Works | ❓ Needs cutlass-dsl |
| RoPE fused | ✅ Works | ✅ Works (0.029 ms) |

### RoPE Benchmark on B300
```

shape: B=1 L=4680 H=16 D=128 dtype=bf16
PyTorch (prefix only): 0.091 ms
Triton rotary (prefix only): 0.067 ms
rope_apply (end-to-end): 0.030 ms
triton_rope_fused_3way (direct): 0.029 ms
Correctness: max_err=0.031250 mean_err=0.000595

```
### What's Left for Full B300 Support

1. ~~**Test RoPE v2 kernel** on B300~~ ✅ DONE
2. **Full pipeline test** (`render_timeline`) on B300
3. **FA4/CUTE** requires separate venv with nvidia-cutlass-dsl (conflicts with pip flash-attn)
4. **Performance tuning** - block sizes may benefit from B300-specific tuning

---

## Git Commits
- `701afd6` - Add Python 3.12 support and FA4 (CUTE) dependencies for Blackwell
- `28aa3d9` - Update .gitignore with video files and cloned repos
- `b76bc4b` - Fix fa4 dependency group to use Python version markers
- `858ddf2` - Add warning about nvidia-cutlass-dsl conflict with PyTorch inductor
```

</file_contents>
<user_instructions>
<taskname="FA4 score_mod B300"/>
<task>Identify the nvidia-cutlass-dsl version(s) compatible with flash-attention.bak CuTe score_mod (missing APIs like TensorSSA.broadcast_to, cute.make_rmem_tensor, enable_tvm_ffi), then recommend an integration strategy for B200 (SM100) + B300 (SM103) and concrete commands/patches so _flash_attn_fwd(..., score_mod=...) runs on B300 without breaking normal FA2/FA4 attention.</task>

<architecture>
- Attention backend selection lives in `src/scope/core/pipelines/wan2_1/modules/attention.py` (FA2/FA3/FA4 detection, local flash-attention path injection).
- KV-cache bias (Kernel B) path and FA4 score_mod usage live in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (SCOPE_KV_BIAS_BACKEND with fa4/flash/triton/flex).
- SM103 compatibility helpers in `src/scope/core/compat/sm103.py` plus SM103 patch script `scripts/patch_cutlass_sm103.py`.
- Dependency pins/extra group in `pyproject.toml` and lock in `uv.lock` (flash-attn 2.8.3, cutlass-dsl 4.1.0).
- Notes document current B300 status, patching, version conflicts, and FA4 score_mod plans.</architecture>

<selected_context>
- `src/scope/core/pipelines/wan2_1/modules/attention.py`: extends `flash_attn.__path__` to a local `flash-attention/flash_attn`, selects FA2/FA3/FA4 varlen APIs.
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`: FA4 score_mod path (`_get_fa4_score_mod`), flash segment-combine fallback, Triton Kernel B fallback, and flex_attention score_mod fallback in the KV-cache bias path.
- `src/scope/core/compat/sm103.py`: detects SM103, patches CUTLASS arch lists, recommends safe backend defaults.
- `scripts/patch_cutlass_sm103.py`: edits cutlass-dsl files to allow SM103; warns about cutlass module conflict with torch._inductor.
- `scripts/test_fa4_kv_bias.py`: standalone FA4/CUTE score_mod test harness with path injection and benchmark logic.
- `notes/FA4/B300-FA4-PATCHES.md`: SM103 patch steps + version matrix; calls out API mismatch and missing FastDivmodDivisor in cutlass-dsl 4.1.0.
- `notes/FA4/b300-investigation.md`: runtime path injection details, cutlass-dsl vs torch._inductor conflict, and FA4 score_mod file touchpoints.
- `notes/FA4/phase4-fa4-score-mod.md`: score_mod integration plan and semantics; notes local flash-attention.bak usage.
- `notes/FA4/B300-SETUP-GUIDE.md`: CUDA 12.9 ptxas requirement and working baseline for SM103; mentions FA4 score_mod mismatch.
- `notes/FA4/B300-SESSION-STATE.md`: current backend defaults (flash on SM103) and local flash-attention.bak state.
- `notes/FA4/DeepResearch/2025-12-23/CuTeDSL_score_mod.md`: CuTe DSL score_mod patterns, aux_tensors behavior, and interface expectations.
- `notes/TODO-next-session.md`: FA4 dependency group and warning about cutlass-dsl shadowing torch._inductor.
- `notes/FA4/kernel-dev-log.md` (slice): top status with deps for FA4 score_mod (nvidia-cutlass-dsl>=4.3.3, apache-tvm-ffi).
- `pyproject.toml` (slice) + `uv.lock` (slice): pinned versions and FA4 dependency group.
- `vendored/cutlass-cute/python/CuTeDSL/ampere/elementwise_add.py` (slice): example using `cute.make_rmem_tensor`.
- `vendored/cutlass-cute/python/CuTeDSL/cute/tvm_ffi/jit_and_use_in_torch.py` (slice): example using `enable_tvm_ffi` and `--enable-tvm-ffi` option.</selected_context>

<relationships>
- `attention.py` injects local flash-attention repo into `flash_attn.__path__` → FA4 availability can come from local tree rather than pip wheel.
- `causal_model.py` uses `_flash_attn_fwd` when `SCOPE_KV_BIAS_BACKEND=fa4` and checks for `score_mod` in signature; otherwise falls back to segment-combine (`flash`), Triton kernel B (`triton`), or flex_attention.
- `sm103.py` / `patch_cutlass_sm103.py` patch cutlass-dsl arch lists for SM103; notes warn that cutlass-dsl can shadow torch._inductor.
- Notes show current pins: flash-attn 2.8.3 + cutlass-dsl 4.1.0 for basic FA4; score_mod path expects newer APIs (FastDivmodDivisor, make_rmem_tensor, enable_tvm_ffi) implying newer cutlass-dsl (>=4.3.3 per kernel-dev-log) but conflicts with torch._inductor.</relationships>

<ambiguities>
- The requested `flash-attention.bak` source files are not present in this workspace, so exact CuTe API expectations must be inferred from notes or external references.
- No captured stdout/stderr log from running `scripts/test_fa4_kv_bias.py` on B300 is present in-repo.
- Exact cutlass-dsl release that introduces TensorSSA.broadcast_to / make_rmem_tensor / enable_tvm_ffi is not recorded in-repo; may require upstream release notes/commit history.</ambiguities>
</user_instructions>

## ChatGPT

You are basically dealing with two different “CUTLASS Python DSL eras”:

**Your current pinned era**: `nvidia-cutlass-dsl==4.1.0` (works with the pip `flash-attn==2.8.3` FA4 bits for basic attention after SM103 arch patching, but is missing newer DSL APIs).

**The `flash-attention.bak` score_mod era**: expects newer CuTe DSL APIs (the ones you explicitly called out: `TensorSSA.broadcast_to`, `cute.make_rmem_tensor`, `enable_tvm_ffi`, plus the `FastDivmodDivisor` family).

Below is what’s actually required, which versions satisfy it, and a strategy that lets you run `_flash_attn_fwd(..., score_mod=...)` on **B300 (SM103)** while keeping “normal attention” stable.

## 1) Which `nvidia-cutlass-dsl` versions are compatible with `flash-attention.bak` score_mod

You listed three missing APIs. CUTLASS’ own CuTe DSL changelog pins when those arrived:

| Missing API / behavior | First CUTLASS DSL version that includes it | Why it matters for FA4 score_mod |
| --- | --- | --- |
| `TensorSSA.broadcast_to` | **4.2.0** | Newer flash-attention CuTe code uses broadcast utilities to keep everything TensorSSA-shaped instead of scalarizing. |
| `cute.make_rmem_tensor` | **4.3.0** | This is used in several DSL examples and is commonly relied on by newer CuTe kernels for predicate/register-memory fragments. |
| `enable_tvm_ffi` (and `--enable-tvm-ffi`) | **4.3.2** | Newer CuTe integration can optionally go through a TVM FFI path; your own vendored example matches this model, and CUTLASS documents `--enable-tvm-ffi` and the `enable_tvm_ffi=True` flow. |

So the hard minimum for the three APIs you called out is:

**`nvidia-cutlass-dsl >= 4.3.2`**

Now the extra “missing” you cited from notes is **`FastDivmodDivisor`** (flash-attention.bak expects it, 4.1.0 doesn’t have it). Your own kernel-dev-log already says the working score_mod setup uses:

**`nvidia-cutlass-dsl >= 4.3.3`** plus **`apache-tvm-ffi`**

Given that, the safest recommendation is:

**Pin `nvidia-cutlass-dsl==4.3.3`** (or the lowest version you have confirmed contains `FastDivmodDivisor`), because it sits above the 4.3.2 threshold and matches what your repo notes say was used successfully.

One more compatibility landmine that matters for your mixed setup:

CUTLASS DSL **4.3.0 removed `cutlass.utils.ampere_helpers`**. If any code path still imports that (common in older flash-attn CuTe glue), it will explode once you move to 4.3.x.

That removal is one of the reasons “pip flash-attn FA4 + cutlass-dsl 4.3.x” can be fragile unless you make sure you’re loading a flash-attention tree that has already moved off the old imports.

## 2) Integration strategy that works on B200 + B300 without destabilizing normal attention

You want two things simultaneously:

**Score_mod FA4** for KV-bias (Kernel B) via `_flash_attn_fwd(..., score_mod=...)` on SM103.

Keep “normal attention” (FA2 and, ideally, FA4 varlen) working and not randomly shadowed by a local checkout.

The central problem is that today your code sometimes turns `flash_attn` into a “Frankenpackage” by inserting a local `flash-attention/flash_attn` directory into `flash_attn.__path__`. That can be good (score_mod support) but also dangerous (silently shadowing the pip wheel’s modules).

### Recommendation: treat “score_mod CuTe” as an opt-in overlay, not a global default

Concretely:

Keep your default runtime using the **pip wheel** for `flash_attn` unless explicitly enabling score_mod CuTe.

When score_mod is enabled, allow a **controlled overlay** of the local `flash-attention.bak` tree, but only when needed, and only in one place.

This matches your backend design already:

B200 default can stay `triton`, opt-in `fa4`

B300 default can stay `flash` (segment-combine) or `triton`, opt-in `fa4`

### Two dependency “tracks” (this is the cleanest way to avoid constant breakage)

**Track A (production / stable):**

`flash-attn==2.8.3` (your pinned wheels)

`nvidia-cutlass-dsl==4.1.0` only if you need basic FA4 testing

No score_mod CuTe expectation

**Track B (score_mod CuTe):**

`flash-attn==2.8.3` still installed (for FA2 and for shared compiled bits)

`nvidia-cutlass-dsl==4.3.3` (or >= 4.3.3)

`apache-tvm-ffi`

Local `flash-attention.bak` used as the CuTe implementation provider

Track B is what you use when you explicitly want `_flash_attn_fwd(..., score_mod=...)`.

This is the least risky way to keep normal runs stable while still enabling the aggressive path.

## 3) Concrete `pyproject.toml` changes

Add a *new* dependency group for score_mod rather than mutating `fa4` in-place (because `fa4` today is effectively “basic FA4 bring-up” pinned to 4.1.0).

toml

Copy code

[dependency-groups]
# Existing:
fa4 = [
 "cuda-python>=12.9.0; python_version >= '3.12'",
 "nvidia-cutlass-dsl==4.1.0; python_version >= '3.12'",
]

# New:
fa4_scoremod = [
 "cuda-python>=12.9.0; python_version >= '3.12'",
 "nvidia-cutlass-dsl==4.3.3; python_version >= '3.12'",
 "apache-tvm-ffi; python_version >= '3.12'",
]

If you want to be slightly more flexible (but less reproducible), use `>=4.3.3,<4.4` once you’ve tested 4.3.4 etc.

## 4) Concrete install + patch commands for B300 (SM103)

### 4.1 Environment prerequisites (you already have this right)

bash

Copy code

export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

### 4.2 Install the score_mod track dependencies

From repo root:

bash

Copy code

uv sync --group fa4_scoremod

Or explicit:

bash

Copy code

uv pip install "nvidia-cutlass-dsl==4.3.3" "apache-tvm-ffi" "cuda-python>=12.9.0"

### 4.3 Patch CUTLASS DSL for SM103

You have two patch sources in notes:

`impl_utils.py` arch allowlist patch (critical in some versions)

tcgen05 and cpasync admissible arch lists

I would update your **Python** patcher (`scripts/patch_cutlass_sm103.py`) to also patch `impl_utils.py`, because that’s the one that tends to fail earliest and noisily.

#### Minimal patch extension idea (drop-in logic)

In `scripts/patch_cutlass_sm103.py`, add:

Locate `cutlass_base / "impl_utils.py"`

Insert the “allow sm_103/sm_103a whenever sm_100a/sm_100f allowed” logic (you already documented this in `B300-FA4-PATCHES.md`)

Then run:

bash

Copy code

uv run python scripts/patch_cutlass_sm103.py

(If you stick to the `.sh` patcher, run that instead, but I’d prefer one canonical patch script.)

### 4.4 Make sure the local repo exists

You already have `flash-attention.bak/`. Keep it. Do not rely on the `flash-attention` symlink existing in production runs.

## 5) Code patches to keep normal FA2/FA4 stable while enabling score_mod CuTe

### Patch 1: Gate `flash_attn.__path__` injection in `attention.py`

Right now `wan2_1/modules/attention.py` always does `_extend_flash_attn_path()` at import time, which is how you get shadowing surprises.

Make it opt-in:

Only extend the path when you are explicitly trying to use score_mod CuTe.

Also search `flash-attention.bak` (not only `flash-attention`).

Example patch sketch:

py

Copy code

# src/scope/core/pipelines/wan2_1/modules/attention.py

def _extend_flash_attn_path() -> None:
 use_local = (
 os.getenv("SCOPE_FLASH_ATTN_LOCAL", "0") == "1"
 or os.getenv("SCOPE_KV_BIAS_BACKEND", "").lower() == "fa4"
 )
 if not use_local:
 return

 try:
 base_path = Path(__file__).resolve()
 except Exception:
 return

 for parent in base_path.parents:
 for repo_dir in ("flash-attention.bak", "flash-attention"):
 candidate = parent / repo_dir / "flash_attn"
 if candidate.is_dir():
 candidate_str = str(candidate)
 if candidate_str not in _flash_attn.__path__:
 _flash_attn.__path__.insert(0, candidate_str)
 return

Result:

Normal runs do not get shadowed.

When you set `SCOPE_KV_BIAS_BACKEND=fa4` (or `SCOPE_FLASH_ATTN_LOCAL=1`) you get the local CuTe stack.

### Patch 2: Make KV-bias FA4 path explicitly request local CuTe

In `krea_realtime_video/modules/causal_model.py` you already do `_extend_flash_attn_path_for_score_mod()` and check `score_mod` in signature. Keep that, but align it with the same gating behavior so you do not accidentally inject local code when you are not using FA4 score_mod.

The most robust pattern is:

If backend is `fa4`, do the injection and import `_flash_attn_fwd`

Otherwise do not touch `flash_attn.__path__`

### Patch 3: On B300, never rely on `torch.compile(flex_attention)` in the score_mod environment

You already introduced `DISABLE_FLEX_ATTENTION_COMPILE` and SM103 guards. For the “score_mod track” on B300, I’d hard-default to eager flex_attention (or avoid flex_attention entirely by using `fa4`, `flash`, `triton` backends).

Concrete run flag:

bash

Copy code

export DISABLE_FLEX_ATTENTION_COMPILE=1

This is not about performance, it’s about preventing the `cutlass` naming conflict from killing inductor paths on SM103.

## 6) How to run and verify `_flash_attn_fwd(..., score_mod=...)` on B300

### One-shot correctness + perf harness (you already have it)

bash

Copy code

export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export DISABLE_FLEX_ATTENTION_COMPILE=1
export SCOPE_KV_BIAS_BACKEND=fa4
export SCOPE_FLASH_ATTN_LOCAL=1 # optional if you key off KV_BIAS_BACKEND

uv run python scripts/test_fa4_kv_bias.py

Success criteria:

The script does not exit with “_flash_attn_fwd lacks score_mod”

Basic test passes

Flex comparison passes (if available)

Real-shape benchmark runs

## 7) If you still hit `FastDivmodDivisor` errors

If after moving to `nvidia-cutlass-dsl==4.3.3` you still see `FastDivmodDivisor` missing, there are only three plausible realities:

The symbol is in **a later 4.3.x** (so bump to 4.3.4/4.3.5 if they exist in your package index).

The symbol name changed (example: split into `FastDivmod` + `FastDivmodU32` style helpers).

Your local flash-attention.bak is at a commit that expects an even newer CUTLASS DSL snapshot than the PyPI `nvidia-cutlass-dsl` provides.

In that case, the high-leverage fix is to **stop guessing and bind the pair**:

Pick a specific flash-attention commit that you know works on B200 for score_mod.

Pin a specific `nvidia-cutlass-dsl` version that that commit works with.

Treat them as a matched pair in the `fa4_scoremod` group.

You already have a strong hint in `kernel-dev-log.md` that the known-good score_mod setup used `>=4.3.3` and `apache-tvm-ffi`. If `FastDivmodDivisor` is still missing at 4.3.3, bumping within the same minor series is the first move.

## 8) The “frontier” option: making cutlass-dsl coexist with Inductor on SM103

If you want one environment that does everything (score_mod CuTe plus compiled flex_attention plus any Inductor paths), the real blocker is the **top-level module name `cutlass`**.

The only credible ways around that, without upstream changes, are:

**Disable all Inductor paths that import `cutlass`** (most pragmatic, what you are already converging on for B300).

**Import hook shim**: a custom `MetaPathFinder` that routes `import cutlass` differently depending on importer (torch._inductor vs flash-attention). This is doable but brittle and hard to test.

**Process isolation**: run score_mod attention in a helper process with the CuTe stack, keep main process clean for Inductor. GPU tensor handoff is possible but nontrivial and may erase the latency win.

Given your current architecture, (1) is the best “get it working now” path, and (2) is the most interesting experiment if you want a single-env solution later.

## Bottom line

**Minimum cutlass-dsl for your missing APIs**: **4.3.2** (because `enable_tvm_ffi` lands there).

**Practical pin for `flash-attention.bak` score_mod**: **4.3.3** (matches your own kernel-dev-log) plus `apache-tvm-ffi`.

**Make local flash-attention CuTe an opt-in overlay** (gated path injection) so you don’t randomly break normal FA2/FA4.

**On B300**, pair that with your SM103 patching and disable flex_attention compilation in the score_mod environment.

If you want, I can write the exact diffs for:

`wan2_1/modules/attention.py` gating logic

`causal_model.py` “only inject local CuTe when backend=fa4”

`scripts/patch_cutlass_sm103.py` extended to patch `impl_utils.py` too

…in the same style as your repo (minimal, trip-breaker friendly, and safe defaults).

## User

2025-12-23_19-44-35_Claude_Chat_TCGen05_project_discussion.md

File

Any thoughts on the bigger picture here of being stuck and what we might not be seeing? Right, on the B300, what could be happening? Hypothesis there at the end worth following up on?

## Message

###### ChatGPT said:

## User

Note that the B200 was running at 20.4 FPS. PS, right? So, like, somehow the consideration that's influencing this needs to be present for the B300, but not for the B200. That's the confusion, right? Because framerate and other stuff... Yeah, I mean, we can go and double check it on an H100. But yeah, so like somehow profiling plus checking on different hardware, plus forming definite hypotheses that then we sort of investigate.

## ChatGPT

Got it – I’m on it and have updated the task accordingly.

## User

Great, how is this as a runbook?

```
# B300 Investigation Runbook

**Purpose:** Systematic approach to diagnose why B300 is stuck at 8.8 FPS while B200 hits ~20 FPS.

**Key insight:** Attention backend changes don't move FPS. The bottleneck is elsewhere.

---

## Quick Reference
```

bash
# Output directory (create per-GPU subdirs)
export INVESTIGATION_DIR=/tmp/gpu-investigation
mkdir -p $INVESTIGATION_DIR/{b200,b300,h100}

# Set GPU tag for this session
export GPU_TAG=b300 # or b200, h100
export OUT_DIR=$INVESTIGATION_DIR/$GPU_TAG

```
---

## Hypotheses

| ID | Hypothesis | Symptom if true |
|----|------------|-----------------|
| H1 | Output pacing limit (not compute) | gpu_ms << wall_ms |
| H2 | CPU bound | Low GPU util, high CPU |
| H3 | Power/thermal throttling | High GPU util, low clocks |
| H4 | GPU not exclusive (MIG/vGPU) | Other processes visible |
| H5 | Non-attention op falling back | GEMMs dominate profile |

---

## Test 0: Environment Baseline

Run on each GPU before any other tests.
```

bash
# Capture system info
nvidia-smi -q > $OUT_DIR/nvidia-smi-full.txt
nvidia-smi -L > $OUT_DIR/nvidia-smi-devices.txt
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,power.limit --format=csv > $OUT_DIR/gpu-info.csv

# Check for other processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > $OUT_DIR/gpu-processes.csv

# CPU info
lscpu > $OUT_DIR/cpu-info.txt
free -h > $OUT_DIR/memory-info.txt

```
**Checkpoint:** If `gpu-processes.csv` shows other processes, H4 may be in play.

---

## Test 1: GPU vs Wall Time (H1 - Pacing)

This is the most important discriminating test.
```

bash
# Run the block profiler with timing
uv run python scripts/profile_krea_pipeline_blocks.py \
 --iters 20 --skip 3 \
 --profile-blocks \
 --profile-blocks-json $OUT_DIR/block-profile.json \
 2>&1 | tee $OUT_DIR/profile-output.log

```
**What to look for:**
- Compare GPU time vs wall time per block
- If `gpu_ms << wall_ms` consistently → pacing/backpressure (H1)
- If `gpu_ms ≈ wall_ms` → compute bound, continue to H2-H5

---

## Test 2: Live GPU Monitoring (H2, H3)

Run in a separate terminal while inference is running.
```

bash
# Start monitoring (run for 60+ seconds during inference)
nvidia-smi dmon -s pucvmet -d 1 > $OUT_DIR/dmon.log &
DMON_PID=$!

# Run inference
uv run python scripts/profile_krea_pipeline_blocks.py --iters 50 --skip 3

# Stop monitoring
kill $DMON_PID

```
**Columns in dmon output:**
- `pwr` - power draw (watts)
- `gtemp` - GPU temperature
- `sm` - SM utilization %
- `mem` - memory utilization %
- `enc/dec` - encoder/decoder utilization
- `mclk/pclk` - memory/graphics clock

**What to look for:**
- Low `sm` with spiky pattern → CPU bound (H2)
- High `sm` but low `pclk` → throttling (H3)
- `pwr` way below TDP → power capped (H3)

---

## Test 3: Power and Clocks Snapshot (H3)
```

bash
# Detailed power/clock query during inference
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $OUT_DIR/power-clocks.txt

# Also capture in a loop during inference
for i in {1..30}; do
 nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate \
 --format=csv >> $OUT_DIR/clocks-timeseries.csv
 sleep 1
done &
CLOCK_PID=$!

# Run inference
uv run python scripts/profile_krea_pipeline_blocks.py --iters 30 --skip 3

kill $CLOCK_PID

```
**What to look for:**
- P-state should be P0 under load
- SM clocks should be near max boost
- Power should be near TDP

---

## Test 4: Bypass WebRTC (H1 - Pacing Isolation)

If H1 is suspected, run inference without WebRTC to isolate.
```

bash
# Direct pipeline test (no streaming)
uv run python scripts/profile_krea_pipeline_blocks.py \
 --iters 50 --skip 5 \
 --profile-blocks \
 --profile-blocks-json $OUT_DIR/no-webrtc-profile.json \
 2>&1 | tee $OUT_DIR/no-webrtc-output.log

```
**What to look for:**
- If FPS jumps up without WebRTC → pacing is in streaming layer
- If FPS stays at 8.8 → bottleneck is in model/pipeline

---

## Test 5: GEMM Microbenchmark (H5)

If compute-bound but attention doesn't matter, test raw GEMM perf.
```

bash
# Quick matmul benchmark
python3 -c "
import torch
import time

# Typical shapes from QKV projection
M, N, K = 4680, 2048, 2048
A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(10):
 C = torch.mm(A, B)
torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
 C = torch.mm(A, B)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f'GEMM {M}x{N}x{K}: {elapsed/100*1000:.3f} ms/iter')
print(f'TFLOPS: {2*M*N*K*100/elapsed/1e12:.2f}')
" 2>&1 | tee $OUT_DIR/gemm-benchmark.txt

```
---

## Test 6: Full Nsight Systems Profile (H5 - Deep Dive)

Only if other tests point to GPU compute issues.
```

bash
# Requires nsys installed
nsys profile -o $OUT_DIR/nsys-profile \
 --stats=true \
 uv run python scripts/profile_krea_pipeline_blocks.py --iters 10 --skip 2

# Generate report
nsys stats $OUT_DIR/nsys-profile.nsys-rep > $OUT_DIR/nsys-stats.txt

```
---

## Comparison Checklist

After running on multiple GPUs, compare:

| Metric | B200 | B300 | H100 |
|--------|------|------|------|
| FPS (block profiler) | | | |
| gpu_ms per frame | | | |
| wall_ms per frame | | | |
| SM utilization % | | | |
| SM clock (MHz) | | | |
| Power draw (W) | | | |
| P-state | | | |
| GEMM TFLOPS | | | |

---

## Decision Tree
```

gpu_ms << wall_ms?
├── YES → H1: Pacing/backpressure
│ └── Next: Find the pacer (WebRTC, queue, sleep)
│
└── NO (gpu_ms ≈ wall_ms) → Compute bound
 │
 ├── SM util low?
 │ └── YES → H2: CPU bound
 │ └── Next: Profile CPU, find single-threaded bottleneck
 │
 ├── SM util high, clocks low?
 │ └── YES → H3: Throttling
 │ └── Next: Check power limits, thermals
 │
 ├── Other GPU processes?
 │ └── YES → H4: Shared GPU
 │ └── Next: Get exclusive access
 │
 └── None of above?
 └── H5: Kernel fallback
 └── Next: Nsight profile, check GEMM perf

```
---

## H100 as Discriminator

H100 result interpretation:

| H100 FPS | Meaning |
|----------|---------|
| ~8.8 FPS (same as B300) | Pacing/scheduling issue in code, not GPU-specific |
| ~15-20 FPS (like B200) | B300-specific issue (SM103 stack, driver, hardware) |

---

## File Checklist

After investigation, each GPU dir should have:
```

$INVESTIGATION_DIR/<gpu>/
├── nvidia-smi-full.txt
├── nvidia-smi-devices.txt
├── gpu-info.csv
├── gpu-processes.csv
├── cpu-info.txt
├── memory-info.txt
├── block-profile.json
├── profile-output.log
├── dmon.log
├── power-clocks.txt
├── clocks-timeseries.csv
├── gemm-benchmark.txt
└── (optional) nsys-profile.nsys-rep

```

```

## ChatGPT

This is a solid runbook. It is already structured in the right way: hypotheses that are mutually distinguishable, minimal tests first, then deeper tooling only if needed.

If I were to tighten it so it converges faster and avoids a couple traps you have already hit (uv run env, WebRTC pacing, FA4 deps conflicts), I would add and tweak the following.

## 1) Add a single sentence definition of “FPS” per test

Right now “FPS” is implicit. For the B300 mystery, the killer ambiguity is:

“model throughput FPS” (how fast the pipeline can produce frames if you let it run), versus

“delivered FPS” (what the server reports after pacing, encoding, client demand, or track timestamps).

I would add this in the header:

**Model FPS:** measured around the model forward path.

**Delivered FPS:** measured at the WebRTC send boundary (or wherever frame_processor reports it).

Then make every test state which one it is producing.

Why this matters: if the 8.8 number is delivered FPS but your profiling script measures model FPS, you can chase the wrong bottleneck for hours.

## 2) Baseline must capture the “B300 gotcha” env: TRITON_PTXAS_PATH

You discovered this already. Put it in Test 0 so you never forget:

bash

Copy code

# Critical for SM103: ensure ptxas supports sm_103a
echo "TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH" | tee $OUT_DIR/env-triton-ptxas.txt
if [ -n "$TRITON_PTXAS_PATH" ]; then
 $TRITON_PTXAS_PATH --version | tee $OUT_DIR/ptxas-version.txt
 $TRITON_PTXAS_PATH --help | grep -i "sm_103" | tee $OUT_DIR/ptxas-sm103.txt || true
fi

Also record Python and torch stack versions, because small deltas can change kernel selection:

bash

Copy code

uv run python - <<'PY' | tee $OUT_DIR/python-stack.txt
import torch, triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda", torch.version.cuda)
print("cap", torch.cuda.get_device_capability())
print("gpu", torch.cuda.get_device_name())
PY

## 3) Baseline should explicitly detect the “cutlass shadowing” situation

Because FA4 deps can silently change imports, add a guard that prints where `cutlass` came from:

bash

Copy code

uv run python - <<'PY' | tee $OUT_DIR/cutlass-import.txt
import importlib.util
spec = importlib.util.find_spec("cutlass")
print("cutlass spec:", spec)
if spec:
 print("cutlass origin:", spec.origin)
PY

This prevents you from comparing B200 and B300 runs where one is in the FA4 bench env and the other is not.

## 4) Test 1 needs a fallback path if that profiling script does not exist

Your runbook references `scripts/profile_krea_pipeline_blocks.py`. I do not see it in the repo map you posted, so make the runbook resilient:

If it exists, great.

If it does not, the runbook should say “instrument the real server path directly” or provide a tiny drop-in snippet.

The key is the discriminating measurement:

GPU time for “model forward for one frame”

Wall time for “everything required to produce one frame”

If you do nothing else, do this.

### Implementation detail to avoid lying to yourself

If you time GPU with events, make sure you synchronize on the end event before reading elapsed time. Otherwise you will undercount.

## 5) Add an explicit “tight loop, no I/O” control test

Your WebRTC bypass test is good, but it still depends on that script.

I would add a very explicit control that isolates “pure compute throughput”:

run N forward steps

no encoding

no networking

no disk writes

discard outputs

Even if this is an ad hoc script for now, it becomes the gold reference for “what the GPU can do”.

This is the fastest way to confirm or kill H1.

## 6) Expand Test 2 interpretation with encoder utilization

You already capture `enc/dec` in `dmon`. Make it explicit in the “what to look for” section:

If `enc` is high (near 100) and `sm` is low, you might be **encode bound**, not model bound.

If `enc` is low and `sm` is low with idle gaps, you are likely **pacing or waiting for input**.

If `sm` is high but `pclk` low, it is **throttling**.

Given you are in a real time app, encoder or pacing is genuinely likely.

## 7) Add a “pacer hunt” mini checklist if H1 triggers

If `gpu_ms << wall_ms`, your next step should not be “profile harder”, it should be “find the sleep or backpressure”. Add a quick grep driven checklist:

bash

Copy code

# Look for explicit pacing controls
rg -n "sleep\\(|MAX_FRAME_RATE|frame_rate|fps|pacer|pts|time_base|await asyncio\\.sleep" src/scope/server -S > $OUT_DIR/pacing-grep.txt

And a note:

If FPS is stuck at ~8.8, check if any code is pacing to 9 FPS, 10 FPS with overhead, or a timebase rounding artifact.

## 8) Add MIG and compute mode queries to Test 0

You already check for processes, but MIG can hide in plain sight. Add:

bash

Copy code

nvidia-smi --query-gpu=mig.mode.current,compute_mode --format=csv > $OUT_DIR/gpu-mig-compute-mode.csv
nvidia-smi -q -d MIG > $OUT_DIR/nvidia-smi-mig.txt || true

This makes H4 much easier to confirm.

## 9) Nsight Systems command: include traces that answer your hypotheses

Your current `nsys profile` line is fine, but for pacing and CPU stalls you want OS runtime and NVTX if you have ranges.

I would use:

bash

Copy code

nsys profile -o $OUT_DIR/nsys-profile \
 --stats=true \
 --trace=cuda,nvtx,osrt \
 uv run python scripts/profile_krea_pipeline_blocks.py --iters 10 --skip 2

If you do not have NVTX ranges, `osrt` still helps show sleeps, waits, and synchronization.

## 10) Decision tree: add the “encode bound” branch

Right now it goes CPU bound vs throttling vs shared vs fallback. Add:

`enc high and sm low` -> encode bound (either NVENC saturated or CPU encode slow)

That can absolutely pin delivered FPS and ignore attention changes.

## Suggested small edits to your runbook text

If you want the minimal patch, it is basically:

Add “FPS definition” section

Add TRITON_PTXAS_PATH capture

Add cutlass import origin capture

Add MIG queries

Add encode bound branch

Add pacer hunt grep step

Everything else is optional.

## One meta point for the B200 vs B300 comparison

Make sure you run the same “mode” on both devices:

same client attached or no client attached

same resolution

same codec path

same pipeline preset

If B200’s 20.4 was measured offline or in a different endpoint than B300’s 8.8, your first job is to remeasure with a single consistent harness.

If you paste the exact line in the code where the server prints “FPS” (file and function), I can tell you which category that measurement is in, and what other timing would sit next to it to disambiguate H1 immediately.

## User

Check the GitHub repo davidrd123/scope, the feature/stream-recording branch to see the work that I've done so far and integrate with what we've been talking about. notes/FA4 folder has notes that point to files and profiling I did - I think codex had a breakthrough on the b300, we ended up at 14 frames per second last night

## ChatGPT

Thanks—could you clarify what you’d like me to look for or help with in the davidrd123/scope repo, specifically in the feature/stream-recording branch? For example:

Are you looking for a summary of the FA4 integration work?

Do you want a review of the profiling results or analysis of the B300 improvements?

Should I generate a prioritized TODO list based on the current repo state?

Let me know your goal and I’ll dig in accordingly.

## User

Yeah, specifically the B300 improvements in light of kind of what we've talked about.

## ChatGPT

Great — I’ll examine the davidrd123/scope GitHub repository on the `feature/stream-recording` branch to understand recent improvements on B300 (SM103), especially in light of our previous attention backend and FPS bottleneck discussion.

I’ll focus on profiling data, patches to attention paths, environment fixes (e.g., TRITON_PTXAS_PATH, patch_cutlass_sm103), and any changes in pacing or backend selection logic that might explain the observed jump from 8.8 to 14 FPS.

I’ll get back to you shortly with a breakdown of what’s been done, what worked, and what remaining areas might yield further gains.

## User

![](https://chatgpt.com/backend-api/estuary/content?id=file_00000000e9a471f5be36fc04321df8a3&ts=490725&p=fs&cid=1&sig=ce3867021405e7decf0f41e9133b83adc8be95e345504d2e2511c0085e4e4ff6&v=0)

I'd like you to go through all the links here as potential performance optimization resources or approaches, and give me a comprehensive integrated report of all the most promising techniques. Essentially, an all-in-one resource that I can drop into my project and have my coding agents look at the code in more detail to estimate which we could make use of.

## ChatGPT

Below is an integrated, “drop-into-the-project” performance optimization report based on every unique link in your screenshot. I’m going to treat this as a menu of techniques, then translate that into a practical triage plan for your specific situation (B300 lagging, attention backend swaps not moving FPS, then a jump from ~8.8 to ~14 FPS).

# Integrated performance optimization report (PyTorch stack)

## Unifying theme across all links

Across these posts, PyTorch’s current performance playbook is:

**Make the workload more compiler-friendly** so `torch.compile` (TorchDynamo + Inductor) can fuse and schedule aggressively.

**Eliminate host-side pacing and synchronization** that quietly caps throughput even if kernels are fast.

**Use higher-level kernel authoring and kernel search** only after you have a measured, stable hotspot.

**Treat the full pipeline as the unit of performance**, including decode/encode, transfers, and queuing, not just model math.

That framing matters for your B300 story: “attention backend changes don’t move FPS” is often the signature of a bottleneck that sits above kernels (pacing, graph breaks, CPU orchestration, transfers, non-attention ops, or fixed-rate streaming).

## Technique inventory by link

### 1) KernelFalcon (autonomous kernel generation via deep agents)

**What it is**
KernelFalcon is presented as an approach for **autonomously generating GPU kernels via “deep agents”**, targeting performance through iterative synthesis and evaluation.

**When it helps**

You have a **small number of expensive hotspots** that are not already dominated by cuBLAS/cuDNN.

You can write down a kernel spec (inputs/outputs, constraints) and objectively benchmark it.

You suspect you are hitting **fallback kernels**, poor fusion, or a weird op that Inductor will not fuse well.

**How to apply in your project**

Use your block profiler + Nsight Systems to identify 1 to 3 kernels or subgraphs that dominate GPU time.

Extract a minimal reproduction (tensor shapes, dtype, layout).

Use KernelFalcon style tooling to explore alternatives (often Triton variants), then re-integrate as a custom op.

**Pitfalls**

This is not a first-line fix for “FPS is capped by wall time” problems.

Best used after you’ve proven the bottleneck is compute-side, not pacing.

### 2) KernelAgent (Meta PyTorch GitHub)

**What it is**
KernelAgent is described as a framework to accelerate GPU kernel development by using **LLMs plus a search algorithm** to generate and optimize **Triton kernels**.

**When it helps**

Same as KernelFalcon, but more direct and pragmatic if you already accept Triton kernels as an integration path.

You want to iterate quickly on kernel variants for a specific shape regime (common in diffusion and attention).

**How to apply**

Treat it as an “optimization workbench”:

Export benchmark harness for a hotspot op.

Let KernelAgent produce candidates.

Keep only kernels that win in your real shapes and are stable.

Wrap as a custom op or Inductor extern kernel.

**What to assign an agent**

“Identify top 3 non-attention hotspots, create microbench harness, run KernelAgent search, report best kernel and integration plan.”

### 3) Helion (high-level GPU kernel DSL)

**What it is**
Helion is positioned as a **high-level DSL for GPU kernels**: write once, get high performance via compilation and scheduling decisions.

**Why it is relevant to B200/B300**

If your bottleneck ends up being a custom kernel problem, Helion is aimed at making kernel authoring less painful than raw CUDA, while still getting performance.

It pairs naturally with the Blackwell-era reality that the fastest paths increasingly require careful layout and async movement strategies.

**When it helps**

You need something more structured than “raw Triton scripts everywhere”, especially if you are translating kernels toward CuTe-like thinking.

You want a path to performance without writing tcgen05 PTX directly.

**How to apply**

Use Helion for kernels that:

Are too custom for cuBLAS/cuDNN.

Are too awkward for Inductor fusion.

Have clear, stable shapes.

**Pitfalls**

Like KernelAgent, this is “downstream” of having solid profiling proof.

You still need a benchmark harness and shape discipline.

### 4) torch.compile + Diffusers hands-on guide

**What it is**
A practical guide to applying `torch.compile` in Diffusers pipelines to reach peak performance.
**Why this is likely one of your top levers** 

If your B300 issue is not attention-specific, the big wins often come from:

**reducing graph breaks**

**compiling the heavy modules**

**reducing Python overhead**

**making execution more static so Inductor can fuse well**

This guide is basically a recipe book for doing exactly that in diffusion-style pipelines.

**High-probability actions**

Compile the biggest blocks (typically UNet, VAE decode, maybe text encoder depending on how often it runs).

Make shapes static where possible (consistent resolution, fixed batch, fixed guidance strategy).

Remove or isolate dynamic control flow from compiled regions.

Avoid hidden sync points (debug prints that call `.item()`, explicit synchronizes, CPU conversions).

**What to assign an agent**

“Measure graph breaks and compiled coverage. Produce a plan to compile UNet and VAE with minimal breaks, and report wall-time vs GPU-time before and after.”

### 5) Flux Fast (H100-focused diffusion acceleration case study)

**What it is**
A performance case study aimed at making Flux run extremely fast on H100, with a “full-stack” mindset.
**Why you should care even if you are on B300** 

Even if some tactics are H100-specific, the meta-lessons tend to transfer:

Identify the dominant blocks.

Push compilation and fusion.

Make execution stable (so scheduling and caching pay off).

Hunt non-obvious overhead outside the math.

**How to use it**

Treat it as a checklist for:

Which parts to compile first

What kinds of overhead hide outside attention

How to validate wins (not just “it feels faster”)

**What to assign an agent**

“Extract the transferable Flux Fast tactics into a checklist, then apply that checklist to our pipeline in order of expected ROI.”

### 6) FlexAttention (core concept)

**What it is**
FlexAttention is a PyTorch mechanism for expressing attention flexibly while still targeting performance, rather than being locked into one fixed pattern.

**When it helps**

You need non-standard attention masks (sliding window, block patterns, custom biasing) but want a path to fused/optimized execution.

You want a “compiler-friendly” attention interface that can specialize to your pattern.

**Relevance given your observation** 

You already saw “attention backend changes don’t move FPS.” That suggests FlexAttention is not your first lever for the B300 delta. But it can still matter if:

Your real bottleneck is not attention overall, but a specific stage that becomes attention-heavy at certain settings.

You later discover attention is indirectly forcing graph breaks or preventing fusion.

### 7) FlexAttention for inference (inference-specific improvements)

**What it is**
An inference-focused writeup on using FlexAttention in deployment contexts.

**Where it can help**

Serving paths where you care about:

predictable latency

stable shapes

reduced overhead in attention execution

**How to use it in your workflow**

Treat as a targeted option if profiling later shows a real attention kernel or attention-adjacent overhead dominating.

### 8) Accelerating Generative AI (general)

**What it is**
A broader PyTorch post about accelerating generative AI workloads.
**Why it matters** 

This is where a lot of the “boring but decisive” optimizations usually live:

removing synchronizations

keeping tensors on GPU

avoiding unnecessary transfers

structuring code so compilers can do their job

It’s especially relevant if you suspect a pacing or orchestration cap rather than raw kernel speed.

### 9) Segment Anything 2 acceleration (specific applied optimization story)

**What it is**
A concrete example of accelerating a large generative/vision model workload (SAM2) in PyTorch.
**Why it matters to your case** 

Applied posts like this tend to reveal the “real” wins:

what they had to change in code structure

where compilation helped

what they had to do to make execution stable

Use it as an existence proof: performance jumps often come from pipeline-level changes, not just swapping one kernel backend.

### 10) Warp specialization (kernel design pattern)

**What it is**
Warp specialization is described as a kernel design pattern that splits warps into specialized roles (commonly: some warps focused on data movement, others on compute) to improve GPU utilization.
**Why it might matter for B300** 

On Blackwell-era GPUs, the performance frontier increasingly comes from:

async memory movement (TMA-like concepts)

deep pipelining

explicit overlap

Warp specialization is one of the recurring patterns used to get that overlap.

**How to apply**

This is mainly for custom kernel work (Helion, Triton, CUDA, CuTe), not a simple knob.

Use it when you have a proven kernel hotspot and you are ready to redesign the kernel.

### 11) TorchCodec (video decode into PyTorch)

**What it is**
TorchCodec is introduced as a PyTorch-oriented library for working with video, with an emphasis on efficient ingestion into torch workflows.
**Why it might be directly relevant to your situation** 

If your pipeline has any of:

CPU video decode

CPU color conversion

extra copies to get frames into GPU tensors

WebRTC-related staging buffers

Then TorchCodec-style ingestion can remove CPU bottlenecks and reduce transfer overhead.

**How to apply**

Audit your pipeline for any CPU decode or conversions.

Aim for “decode straight to a tensor format that your model expects” with minimal intermediate copies.

## A practical prioritization for your project (B300 stuck, then improved)

Given what you’ve said (B200 ~20.4 FPS, B300 started ~8.8 FPS, attention backend did not move FPS, then a breakthrough to ~14 FPS), the best way to use these resources is to sort them by “likelihood of explaining the delta”:

### Tier 1: Most likely to explain a big FPS cap

**torch.compile playbook (Diffusers guide)**
Because graph breaks, Python overhead, and “not actually GPU-bound” behavior often presents exactly like “backend swaps don’t matter.”

**Pipeline-level acceleration mindset (Accelerating Generative AI, SAM2 post, Flux Fast case study)**
These are where you get tactics that change wall time even if kernels are fine.

**TorchCodec** (only if you have video/decode/stream ingest on the critical path)

### Tier 2: Likely, but depends on what profiling says

**FlexAttention for inference**
Bring in if profiling shows attention is relevant again, or attention is causing graph breaks or preventing fusion.

### Tier 3: Heavy artillery, only after you have a measured hotspot

**KernelAgent / KernelFalcon**
Use when a specific kernel or op family is truly dominating and you need custom kernels.

**Helion + warp specialization mindset**
Use when you are in the “we are going to redesign this kernel for Blackwell” phase.

# “Drop-in” checklist for coding agents

This is the part you can paste into your repo for agents to execute.

## Agent Task A: Determine if you are wall-time capped or GPU-time capped

**Goal:** Prove whether the remaining gap (B300 14 vs B200 20) is pacing, CPU, transfers, or compute.

**Deliverables**

A table per block: `wall_ms`, `gpu_ms`, and ratio.

A list of “sync points” (anything that forces CPU to wait for GPU).

**Likely code smells**

`.item()`, `.cpu()`, `.numpy()`, PIL conversions in the hot path

`torch.cuda.synchronize()`

queue backpressure or sleeps in streaming loop

implicit sync via timing calls or logging

## Agent Task B: Apply torch.compile systematically

**Goal:** Increase compiled coverage, reduce graph breaks, and stabilize shapes.

**Deliverables**

List of graph breaks and exact causes.

Compile plan by module: UNet, VAE decode, other heavy blocks.

Before/after FPS and a note whether the speedup came from lower wall time or lower GPU time.

**Reference:** torch.compile + Diffusers guide

## Agent Task C: Validate pipeline-level wins from applied case studies

**Goal:** Use Flux Fast, SAM2, and Accelerating GenAI as checklists.

**Deliverables**

A checklist of tactics pulled from these posts.

For each tactic: “already doing it / not doing it / blocked by X.”

Measured impact for any changes.

**References:**

## Agent Task D: If media ingest or streaming is in the loop, audit it

**Goal:** Determine whether decode/encode/copy is gating FPS.

**Deliverables**

CPU profile of ingest + preproc

GPU copy timeline (HtoD/DtoH) count per frame

Proposal to use TorchCodec or a GPU-native ingest path

**Reference:** TorchCodec

## Agent Task E: Only if compute-bound persists, go after kernels

**Goal:** For top non-attention hotspots, generate faster kernels.

**Deliverables**

Minimal benchmark harness per hotspot

Candidate kernels from KernelAgent or KernelFalcon style workflow

Integration plan and maintenance risk assessment

**References:** KernelAgent, KernelFalcon

# How this connects to “what might we not be seeing” on B300

Based on your description, the most suspicious bigger-picture failure mode is:

**You are not truly compute-bound on B300**, so changing attention backends does nothing.

The jump from ~8.8 to ~14 FPS strongly hints you removed one major limiter (often pacing, sync, or a major graph break) but there is still at least one more limiter.

So the highest-value follow-up hypothesis is still the one your runbook already encodes:

**H1: pacing/backpressure or wall-time cap**, validated by `gpu_ms << wall_ms`.

If that’s true, none of the kernel-level tools matter until you fix pacing.

If it’s not true, then Tier 1 becomes torch.compile coverage and graph-break elimination, because that can shift you from “lots of small kernels plus Python overhead” into “fused kernels at high occupancy,” which often looks like a sudden step-function improvement.

If you want, paste your current “what changed to get 14 FPS” diff (or even just the bullet list of changes), and I’ll map each change onto the framework above and tell you what it most likely fixed, and what the next most likely limiter is.

