# B300 / SM103 Notes (Quick Index)

This folder is the “working set” for the B300 (SM103) optimization thread: how to reproduce, what’s currently true, and where to record new measurements.

## “At Hand” Navigation (How to Use This Folder)

Think of the docs as **four roles**:
- **Truth** (what works today): `session-state.md`
- **Protocol** (how we measure): `investigation-runbook.md`
- **Logbook** (what we tried): `experiments.md`
- **Strategy / R&D** (what we do next): `optimization-vision.md` + `development-plan.md` + Level 6 docs below

If you’re ever unsure what to read next, follow this loop:
1) reproduce baseline → 2) measure (block + op profile) → 3) decide next lever → 4) make one change → 5) log an experiment card → repeat.

## Where to Start

| If you want to… | Start here |
|---|---|
| Run the current “known-good” baseline | `session-state.md` |
| Measure correctly (and not lie to yourself) | `investigation-runbook.md` |
| Log a single change / result | `experiments.md` |
| See the longer narrative / hypotheses | `investigation.md` |
| Read the “what we learned” distillation | `lessons.md` |
| See the roadmap / next levers | `optimization-vision.md` |
| See the learning ladder framing | `optimization-ladder.md` |
| Align on how we operate (learning-first) | `philosophy.md` |
| External references / stack maturity links | `blackwell-docs.md` |
| Document layout contracts (blocks Level 6) | `layout-contracts.md` |
| Break down `other_in_self` with stacks | `other-in-self-breakdown.md` |
| Quick reference for Blackwell primitives | `blackwell-primitives-cheatsheet.md` |
| Map the vendored reference kernels (CuTe/CUTLASS/FA4) | `vendored-kernel-resources.md` |
| VAE decode graph + cuDNN planning notes | `vae-decode-architecture.md` |
| Kernel experiment gates (perf + quality) | `kernel-experiment-template.md` |

## Doc Map (By Workflow)

- **Reproduce / baseline**
  - `session-state.md` (blessed commands + pitfalls)
  - `setup-guide.md` (how to get onto the cu130 “fast stack”)
  - `fa4-patches.md` (FA4/CuTe setup + SM103 patch notes)
- **Measure**
  - `investigation-runbook.md` (block profiler + op profiler + decision points)
  - `scripts/profile_krea_pipeline_blocks.py` (block timing)
  - `scripts/profile_krea_pipeline_ops.py` (operator + stack attribution)
- **Decide next lever**
  - `optimization-vision.md` (ordered options; “what’s left”)
  - `development-plan.md` (workstreams + acceptance criteria)
- **Record**
  - `experiments.md` (one-change cards only)
- **Distill**
  - `lessons.md` (what we learned + pointers to evidence)
- **Level 6 groundwork (only after measurement)**
  - `layout-contracts.md` (hard constraints at QKV/RoPE/cache/attn boundaries)
  - `other-in-self-breakdown.md` (fill once with real stack data; don’t guess)
  - `vendored-kernel-resources.md` (where to learn patterns from real kernels in-repo)
  - `kernel-experiment-template.md` (quality/perf gates for any kernel work)
  - `blackwell-primitives-cheatsheet.md` (TMA/mbarrier/tcgen05/TMEM reference)
  - `vae-decode-architecture.md` (decode-specific map + planning ideas)

## Ground Truth Conventions

- **Current best:** ~34.5 FPS (BF16 + compile + FA4 varlen + resample-contiguous). **~3.9× baseline.**
- **Canonical perf comparisons:** `320x576`, 4 denoise steps, `kv_cache_attention_bias=0.3`, quality-preserving settings.
- **Quality gate:** FP8 quantization is currently **off-limits** on B300 (garbage output). Use `--quantization none` (BF16). Details: `session-state.md`.
- **One-change rule:** change exactly one thing per experiment card and write down:
  - command, env, baseline, result, artifacts, and the lesson.

## Tiny Desk Reference (Copy/Paste)

Pick the Python for your cu130 env (example):

```bash
PY=.venv-b300-cu130-decode/bin/python
```

Offline timeline render (uses B300 “known-good” env defaults + BF16):

```bash
scripts/render_timeline_b300.sh ~/.daydream-scope/recordings/session_*.timeline.json out.mp4
```

Canonical BF16 env vars (SM103-safe defaults):

```bash
export SCOPE_KV_BIAS_BACKEND=fa4
export DISABLE_FLEX_ATTENTION_COMPILE=1
export WANVAE_STREAM_DECODE_MODE=chunk
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

Optional (B300-specific) knobs:

```bash
# Optional: whole-model compilation (usually neutral; can increase warmup).
export SCOPE_TORCH_COMPILE_STRATEGY=model  # default: blocks

# B300 hazard: fused to_qkv(...).chunk(...) can create strided Q/K views and
# trigger extra copies in eager mode. For the compiled path we currently prefer
# fused projections ON; use this knob for eager/debug.
export SCOPE_DISABLE_FUSED_PROJECTIONS=1

# Experimental: write RoPE(K) directly into the KV cache window (neutral so far).
export SCOPE_ROPE_K_TO_CACHE=1

# Small denoise win (compiled B300): opt into FA4/CuTe varlen for the *plain* (non-bias)
# attention path even when `SCOPE_KV_BIAS_BACKEND=fa4`. Increases warmup/JIT time.
export SCOPE_ENABLE_FA4_VARLEN=1

# Small decode win: use channels-last 3D activations for Conv3d-heavy VAE decode.
export WANVAE_DECODE_CHANNELS_LAST_3D=1

# Big decode win: keep streaming Resample outputs contiguous so Conv3d stays on
# cuDNN/CUTLASS (avoids `aten::slow_conv_dilated3d` / vol2col fallback).
export WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1
```

Benchmark harness (BF16, no compile):

```bash
$PY scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 10 --skip 3 \
  --kv-cache-attention-bias 0.3 \
  --quantization none \
  --cudnn-benchmark
```

Benchmark harness (BF16 + `--compile`):

```bash
$PY scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 10 --skip 3 \
  --kv-cache-attention-bias 0.3 \
  --quantization none \
  --cudnn-benchmark \
  --compile
```

Op profile (stack attribution for `copy_`/`to` hotspots):

```bash
$PY scripts/profile_krea_pipeline_ops.py \
  --height 320 --width 576 \
  --iters 1 --pre-iters 1 \
  --kv-cache-attention-bias 0.3 \
  --kv-bias-backend fa4 \
  --quantization none \
  --cudnn-benchmark \
  --with-stack --stack-n 12 \
  --summary outputs/b300_ops_summary.md \
  --json outputs/b300_ops_profile.json
```

When numbers disagree: trust `session-state.md` → `investigation-runbook.md` → recent `outputs/` artifacts, and sanity-check GPU sharing + warmup time.

## Upstream References (Shortlist)

If you’re trying to answer “is this a us-bug or an upstream landmine?”, this is the minimal set:

- `nvidia-cutlass-dsl` module shadowing `torch._inductor` cutlass utilities (can break `torch.compile` / `flex_attention`): see [`setup-guide.md`](setup-guide.md) and [`investigation.md`](investigation.md) (Issue 2).
- TorchAO FP8 + `torch.compile` + `aten.as_strided` hole: see the “Upstream refs” section in `session-state.md`.
- Conv3d BF16/FP16 regressions + “install cuDNN 9.15+”: PyTorch v2.9.1 release notes (linked in `session-state.md` and `blackwell-docs.md`).
- CUDAGraph “output overwritten” + step-marker spelling: `torch.compiler.cudagraph_mark_step_begin()` docs (linked in `session-state.md`); on SM103 we auto-ignore `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` unless `SCOPE_ALLOW_REDUCE_OVERHEAD_SM103=1`.

## Deep Research Packets (Optional, but source-backed)

If you want the “receipts” (issue IDs / exact spellings) in one place:
- [`claude_dr.md`](../DeepResearch/2025-12-26/B300_optim_ladder/round02/claude_dr.md)
- [`README.md`](../DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/README.md)
- [`claude01.md`](../DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md)
- [`README.md`](../DeepResearch/2025-12-27/climbing_the_mountain/README.md) (Blackwell/SM103 doc pack; mbarrier vs CUtensorMap alignment; `cuTensorMapEncodeTiled` rank 3–5)
