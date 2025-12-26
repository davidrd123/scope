# Backend Selection & Tuning Knobs (What Controls What)

> **Explainer #17** — A “knobs map” for this repo: which environment variables and runtime parameters control which attention backend, which caches reset, and which codepaths are risky on SM103.
> **Updated:** 2025-12-26

---

## TL;DR

There are *two* backend-selection problems in the realtime pipeline:

1) **Plain attention** (bias disabled): handled in `src/scope/core/pipelines/wan2_1/modules/attention.py`.
2) **KV-bias attention** (bias enabled): handled in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`.

If you only remember one rule: **always record which backend actually ran** (to avoid “silent fallback” benchmarking).

---

## Quick Tables (Copy/Paste Friendly)

### Runtime knobs (server / playlist params)

| Knob | Default | What it does | Notes |
|------|---------|--------------|-------|
| `kv_cache_attention_bias` | `0.3` | `<1.0` enables KV-bias attention path; `1.0` disables bias path (plain attention) | Must be in `(0, 1]` (log-bias math); `1.0` is “no bias” |
| `hard_cut` / `reset_cache` | `false` | Full cache reset at next chunk boundary | Jarring but clean |
| `soft_cut` | `false` | Temporary bias override for `N` chunks | Implemented in `FrameProcessor` (worker-thread state machine) |
| `soft_cut_bias` | (UI default) | Bias value used during soft cut | Lower = forget past faster |
| `soft_cut_chunks` | (UI default) | Number of chunks to hold the soft bias before auto-restore | Avoids manual “undo” |

### Env var knobs (backend selection + toolchain)

| Env var | Values | Default | What it controls | Notes |
|--------|--------|---------|------------------|-------|
| `SCOPE_KV_CACHE_ATTENTION_BIAS` | float | `0.3` | Default `kv_cache_attention_bias` when the runtime doesn’t override it | `1.0` disables bias path |
| `SCOPE_KV_BIAS_BACKEND` | `fa4|flash|triton|flex` | SM100→`triton`, SM103→`flash` | KV-bias backend when bias is enabled | **Read at import time**; restart process when changing |
| `DISABLE_FLASH_ATTENTION_4` | `0|1` | `0` | Disables FA4 varlen selection for plain attention | Useful to avoid CuTe mixing on SM103 |
| `SCOPE_ENABLE_FA4_VARLEN` | `0|1` | `0` | Allows FA4 varlen for *plain* attention even when `SCOPE_KV_BIAS_BACKEND=fa4` | Opt-in because mixing CuTe module sources can break |
| `SCOPE_COMPILE_KREA_PIPELINE` | `0|1` | (auto: Hopper only) | Enables `torch.compile` of diffusion blocks in the server pipeline | On B200 (SM100), compile is typically a big steady-state win |
| `SCOPE_COMPILE_KREA_PIPELINE_ALLOW_QUANTIZATION` | `0|1` | `0` | Allows `torch.compile` even when FP8 quantization is enabled | On SM100 we allow compile+FP8 without this; elsewhere this is an explicit experiment override |
| `SCOPE_TORCH_COMPILE_MODE` | string | unset | Controls `torch.compile(..., mode=...)` | Some modes are footguns on SM103 |
| `TRITON_PTXAS_PATH` | path | unset | Points Triton/Inductor to a newer `ptxas` | **SM103:** often required for correct codegen/autotune |
| `DISABLE_FLEX_ATTENTION_COMPILE` | `0|1` | unset | Prevents flex_attention compilation in known-bad modes | **SM103:** set to `1` to avoid tcgen05 LLVM aborts |
| `WANVAE_STREAM_DECODE_MODE` | `chunk|full|...` | unset | VAE streaming decode mode | Mostly affects B300 decode throughput |

---

## 1) Plain Attention Backend Selection (Bias Disabled)

When `kv_cache_attention_bias == 1.0`, `CausalWanSelfAttention` uses:

- `src/scope/core/pipelines/wan2_1/modules/attention.py::attention(q, k, v)`

The selection order is:

1) **SageAttention** (if available and supported on this GPU)
2) **FlashAttention** (FA4 varlen on Blackwell if available; else FA3/FA2)
3) **PyTorch SDPA** fallback

### Knobs that affect this path

- `DISABLE_FLASH_ATTENTION_4=1`  
  Disables FA4 varlen selection even on Blackwell.

- `SCOPE_ENABLE_FA4_VARLEN=1` (special case)  
  When `SCOPE_KV_BIAS_BACKEND=fa4`, we often hot-swap `flash_attn.cute` to a vendored score_mod-capable CuTe implementation. Mixing that with the wheel’s FA4 varlen implementation can cause runtime DSL mismatches.  
  So FA4 varlen is **opt-in** in that configuration.

---

## 2) KV-bias Backend Selection (Bias Enabled)

When `kv_cache_attention_bias != 1.0`, the self-attention path switches to the KV-bias implementation:

- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py::CausalWanSelfAttention.forward()`

That path selects a backend via:

- `SCOPE_KV_BIAS_BACKEND` (string)

Backends:

- `fa4` — FA4/CuTe `score_mod` via `_flash_attn_fwd(...)`
  - Fastest when available.
  - Requires CUTLASS DSL / CuTe toolchain to support your GPU (SM103 may require patching).

- `flash` — FlashAttention segment-combine bias
  - Doesn’t use `score_mod`; bias is applied by shifting per-segment LSE.
  - Often the safest “works on SM103” option.

- `triton` — Triton Kernel B
  - Historically strong on SM100.
  - Can be catastrophically slow or unsafe on SM103 depending on Triton/ptxas/toolchain (see #12, #18).

- `flex` — `torch.nn.attention.flex_attention` fallback
  - Used as an escape hatch when others fail.
  - Often introduces padding/alignment overhead and may trigger compilation/autotune hazards.

### Knobs that affect this path

- `SCOPE_KV_CACHE_ATTENTION_BIAS` (default bias used by the pipeline)  
  - `1.0` disables KV-bias path (plain attention).
  - `< 1.0` enables KV-bias path (and triggers the separate backend selection).

- `SCOPE_KV_BIAS_BACKEND`  
  Picks the KV-bias backend (`fa4`, `flash`, `triton`, `flex`).

---

## 3) Toolchain / Compile Knobs (Mostly SM103 Pain Avoidance)

These don’t “choose a kernel” directly, but they decide whether compilation succeeds or which fallbacks are forced.

- `TRITON_PTXAS_PATH=/path/to/newer/ptxas`  
  On SM103, using a `ptxas` that recognizes `sm_103a` can be the difference between:
  - autotune success, and
  - `NoValidChoicesError` / silent fallback behavior.

- `DISABLE_FLEX_ATTENTION_COMPILE=1`  
  Prevents flex_attention compilation-at-import or compilation in known-bad modes on SM103 (tcgen05 LLVM aborts).

- `SCOPE_TORCH_COMPILE_MODE=...`  
  Changes `torch.compile` behavior; some modes are high-upside, some are “footgun on SM103”.

---

## 4) Runtime “Knobs” (Hard Cut vs Soft Cut vs Bias)

These are not env vars; they are runtime controls sent through the server:

- **Hard cut**: full cache reset (clean break, jarring).
- **Soft cut**: temporary bias override for N chunks (intermediate).
- **Bias level**: steady-state “how much do we forget the past?”

Where they land:

- server endpoints: `src/scope/server/app.py`
- message bridge: `src/scope/server/webrtc.py`
- applied at chunk boundary: `src/scope/server/frame_processor.py`

---

## 5) Recommended Defaults (Learning-Oriented)

If your goal is “keep it working while learning”:

- Prefer **stable backends** first (on SM103: often `SCOPE_KV_BIAS_BACKEND=flash`).
- Use **soft cut** for experimenting with transition dynamics without nuking caches.
- Treat compile/autotune mode changes as **explicit experiments** with a write-up.

---

## References

- Call path map: `notes/FA4/explainers/15-scope-to-fa4-call-path.md`
- SM103 reality + guardrails: `notes/FA4/explainers/12-sm103-notes.md`
- Plain attention selection: `src/scope/core/pipelines/wan2_1/modules/attention.py`
- KV-bias selection: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- Debugging cookbook: `notes/FA4/explainers/18-debugging-cookbook.md`
