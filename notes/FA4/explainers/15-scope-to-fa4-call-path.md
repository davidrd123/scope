# Scope → FA4: The End-to-End Call Path (Where the Kernel Actually Runs)

> **Explainer #15** — A practical map from “I pressed a key in the TUI / hit a REST endpoint” to “a specific attention kernel ran on GPU”, including where KV-bias and backend selection plug in.
> **Updated:** 2025-12-26

---

## TL;DR

- **Control plane:** `video_cli.py` → FastAPI (`app.py`) → `apply_control_message()` → `FrameProcessor.update_parameters()`.
- **Data plane:** `FrameProcessor.process_chunk()` → `pipeline.prepare()` → `pipeline(**call_params)` → pipeline blocks → transformer → attention kernel.
- **Two attention worlds:**
  - **Bias disabled** (`kv_cache_attention_bias == 1.0`): `wan2_1/modules/attention.py::attention()` picks Sage / FlashAttention (FA4 varlen on Blackwell) / PyTorch SDPA.
  - **Bias enabled** (`kv_cache_attention_bias != 1.0`): `krea_realtime_video/modules/causal_model.py::CausalWanSelfAttention.forward()` uses the KV-bias path and picks `SCOPE_KV_BIAS_BACKEND` (FA4 score_mod / Flash segment-combine / Triton Kernel B / flex_attention fallback).

If you only need one “where do I put my debugger / profiler?” answer: start at `src/scope/server/frame_processor.py` and follow the call into `src/scope/core/pipelines/krea_realtime_video/pipeline.py`.

---

## 1) Big Picture Diagram

```
TUI / REST client
  |
  v
FastAPI endpoints (src/scope/server/app.py)
  |
  v
apply_control_message() (src/scope/server/webrtc.py)
  |
  v
FrameProcessor.update_parameters() (src/scope/server/frame_processor.py)
  |
  v   (worker thread; chunk boundaries)
FrameProcessor.process_chunk()
  |
  +--> pipeline.prepare(...)  (inputs required? chunk size?)
  |
  +--> pipeline(**call_params)
          |
          v
    KreaRealtimeVideoPipeline._generate()
          |
          v
    KreaRealtimeVideoBlocks (modular block graph)
      - setup_caches
      - recompute_kv_cache
      - denoise   <-- transformer hot path
      - decode    <-- VAE hot path on some stacks
          |
          v
    CausalWanModel / CausalWanSelfAttention
          |
          v
    attention backend (FA4/FA3/FA2/Sage/SDPA, or KV-bias special path)
```

---

## 2) Control Plane: How “Hard Cut / Soft Cut / Prompt Change” Reaches the Worker

### Where requests enter

- REST endpoints live in `src/scope/server/app.py` (e.g. prompt, playlist, hard cut; soft cut is additive).
- These endpoints don’t touch the pipeline directly; they emit a **control message** dict.

### The control message bridge

`src/scope/server/webrtc.py::apply_control_message(session, msg)`:

- Ensures there is a single active session (connected WebRTC).
- Ensures the `VideoProcessingTrack` has a `FrameProcessor`.
- Calls `FrameProcessor.update_parameters(msg)` which enqueues the update.

### Why this indirection exists

The pipeline runs on a dedicated worker thread. Control messages are queued so:

- updates are thread-safe,
- and semantics are consistent (“apply at chunk boundary”, not mid-kernel).

---

## 3) FrameProcessor: The Chunk Boundary Is the Commit Point

The worker loop is `src/scope/server/frame_processor.py::process_chunk()`.

Key behavior (high level):

1) **Drain updates:** it drains *all* pending control messages and merges them **last-write-wins** into `merged_updates`.
2) **Handle reserved keys:** some keys are consumed and never forwarded to the pipeline (e.g. snapshot/restore, step, and soft transition state).
3) **Order + apply events:** prompt changes, LoRA changes, etc. are applied in a deterministic order.
4) **Prepare and run:** `pipeline.prepare(...)` then `pipeline(**call_params)`.

### Hard cut (cache reset)

- REST uses `reset_cache=True` (via the hard cut endpoint or playlist `hard_cut=true`).
- `FrameProcessor` pops `reset_cache` and passes it as `init_cache` into the pipeline call.
- It also flushes output queues to avoid old frames leaking through during a reset.

### Soft cut (temporary KV-bias override)

- Soft cut is a *temporary override* to `kv_cache_attention_bias` for a fixed number of chunks.
- It’s implemented as a reserved-key state machine on the `FrameProcessor` worker thread so it can auto-restore cleanly at chunk boundaries.

This is intentionally “between”:
- hard cut (full reset) and
- a steady-state bias like `0.3`.

---

## 4) Pipeline: From `pipeline(**kwargs)` to “the transformer ran”

The realtime pipeline is `src/scope/core/pipelines/krea_realtime_video/pipeline.py::KreaRealtimeVideoPipeline`.

Important waypoints:

- `__call__()` delegates to `_generate(**kwargs)`.
- `_generate()` writes inputs into a `PipelineState` (including `prompts`, `init_cache`, `kv_cache_attention_bias`, etc.).
- Then it runs a fixed block graph: `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`.

The blocks include (order matters):

- `setup_caches` (KV cache management)
- `recompute_kv_cache` (when needed)
- `denoise` (**transformer hot path**)
- `decode` (VAE decode; can dominate on the wrong runtime stack)

If you’re hunting performance wins, `PROFILE_PIPELINE_BLOCKS=1` gives a reliable “where time goes” breakdown (but it synchronizes GPU work, so don’t interpret it as peak throughput).

---

## 5) The Attention Fork: Bias Disabled vs Bias Enabled

### A) Bias disabled (`kv_cache_attention_bias == 1.0`)

In `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`:

- When bias is disabled, `CausalWanSelfAttention` calls:
  - `src/scope/core/pipelines/wan2_1/modules/attention.py::attention(q, k, v)`

That function chooses (roughly):

1) SageAttention (when available and supported)
2) FlashAttention (FA4 varlen on Blackwell if available; else FA3/FA2)
3) PyTorch SDPA fallback

### B) Bias enabled (`kv_cache_attention_bias != 1.0`)

The KV-bias path lives inside:

- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py::CausalWanSelfAttention.forward()`

Key idea:

- We want the model to “forget” older cached frames a bit so errors don’t accumulate indefinitely.
- We implement that by adding a **log-bias** to attention scores for a slice of the KV cache (“past frames but not first frame, and not the current block”).

Backend selection for the KV-bias attention is controlled separately by:

- `SCOPE_KV_BIAS_BACKEND` (see #17 for full knob list)

Options:

- `fa4`: FA4/CuTe `_flash_attn_fwd(..., score_mod=...)`
- `flash`: FlashAttention segment-combine (bias applied in the combine stage)
- `triton`: Triton Kernel B
- fallback: `torch.nn.attention.flex_attention` with a `score_mod`

---

## 6) “If I Change X, What Actually Happens?”

| You change… | Where it lands | What it affects |
|---|---|---|
| Prompt / playlist step | `FrameProcessor` prompt event → pipeline state | text embeddings + denoise conditioning |
| `hard_cut` | `reset_cache=True` → pipeline `init_cache=True` | KV cache reset, clean transition |
| `soft_cut` | reserved key → temporary `kv_cache_attention_bias` | faster adaptation for N chunks |
| `kv_cache_attention_bias` | pipeline state + `CausalWanSelfAttention` | chooses bias vs non-bias attention path |
| `SCOPE_KV_BIAS_BACKEND` | `CausalWanSelfAttention` KV-bias branch | which bias implementation runs |
| `DISABLE_FLASH_ATTENTION_4` | `wan2_1/modules/attention.py` | prevents FA4 varlen selection |
| `TRITON_PTXAS_PATH` | Triton/Inductor toolchain | can be the difference between “works” and “NoValidChoicesError” on SM103 |

---

## References (Most Useful Jump Points)

- Control message bridge: `src/scope/server/webrtc.py`
- Chunk boundary application + pipeline call: `src/scope/server/frame_processor.py`
- Pipeline entry and block graph: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`, `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py`
- KV-bias implementation + backend selection: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- Plain attention backend selection: `src/scope/core/pipelines/wan2_1/modules/attention.py`
- FA4 internals: `notes/FA4/explainers/01-how-fa4-works.md`, `notes/FA4/explainers/02-blackwell-path.md`
