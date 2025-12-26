# Claude Session State — B300 Investigation (2025-12-25, Session 2)

> **Purpose:** Handoff document for context compaction. Resume from here.
> **Status:** Historical snapshot from 2025-12-25. For the current “what to run today” state (updated numbers, BF16 baseline, compile status), see `notes/FA4/b300/session-state.md` and `notes/FA4/b300/optimization-vision.md`.

---

## Current Status

**As of 2025-12-25, B300 @ 320×576:** ~15.0 FPS with FA4 (up from 8.8 baseline = +70%)

Current reality has moved since this session (patch-embed Conv2d fastpath + BF16 `--compile` wins). Treat the rest of this file as a time capsule.

**Branch:** `feature/stream-recording`

---

## What Was Accomplished This Session

### 1. Attention Profiling Deep Dive

Ran `PROFILE_ATTENTION=1` to get transformer breakdown:

```
Transformer Block Split (top-level):
├── self_attn:  ~56%
├── cross_attn: ~22%
└── ffn:        ~22%

Within self_attn (with FA4):
├── kv_bias_fa4:   22% (0.42ms/call) ← the kernel itself
└── other_in_self: 74% ← QKV projections, RoPE, cache, memory movement
```

**Key insight:** The attention kernel is now fast with FA4. The bottleneck shifted to "other_in_self" (74% of self_attn).

### 2. Created Optimization Vision Doc

`notes/FA4/b300/optimization-vision.md` — Strategic roadmap with:
- Progress log (8.8 → 13.5 → 15.0 FPS)
- 6 strategic options (A-F)
- Option A (FA4 score_mod) marked COMPLETE
- Phased roadmap toward 24+ FPS target

### 3. Codex Proposed Recompute Cadence Change — REJECTED

Codex found `SCOPE_KV_CACHE_RECOMPUTE_EVERY=2` gives +12% FPS (15→16.8).

**We rejected this** after reading KREA's blog (`notes/krea/blog-krea-realtime-14b.md`):
- KV Cache Recomputation is CORE to their error accumulation solution
- They explicitly tried cheaper alternatives and concluded recompute-every-frame is essential
- Quote: "despite many attempts to devise cheaper solutions, we found these two techniques to be crucial for long, stable generations"

**Decision:** Recompute cadence is OFF LIMITS as an optimization lever.

### 4. Discovered User's Context Editing Vision

`notes/research/2025-12-24/incoming/context_editing_and_console_spec.md`

The OPPOSITE approach to Codex's proposal:
- KREA re-encodes anchor frame from RGB during recompute
- This creates an **edit surface** — modify `decoded_frame_buffer[:, :1]` before recompute
- Edits propagate through KV cache to future frames
- Enables: error correction, retroactive insertion, character modification

**Status:** Speculative — needs validation spike (blue tint test in the spec)

### 5. Found Orphaned VACE-14B Integration Docs

`notes/vace-14b-integration/plan.md` — Ready to implement:
- Add reference image conditioning to Krea 14B
- Upstream weights verified (6.1 GB from Kijai)
- 5 implementation steps documented
- Status: "engineering + wiring work, not blocked"

### 6. torch.compile Is Blocked

Codex identified two blockers:
1. SM103 + fp8: torchao `Float8Tensor` hits `aten.as_strided` under AOTAutograd
2. FA4/CuTe: breaks Dynamo tracing via DLpack fake-tensor paths

**Decision:** Defer torch.compile until these are resolved.

---

## Key Files

| Purpose | File |
|---------|------|
| Vision/roadmap | `notes/FA4/b300/optimization-vision.md` |
| Session state | `notes/FA4/b300/session-state.md` |
| Investigation runbook | `notes/FA4/b300/investigation-runbook.md` |
| KREA blog (recompute rationale) | `notes/krea/blog-krea-realtime-14b.md` |
| Context editing spec | `notes/research/2025-12-24/incoming/context_editing_and_console_spec.md` |
| VACE-14B plan | `notes/vace-14b-integration/plan.md` |

---

## Updated Optimization Roadmap

**OFF LIMITS:**
- `SCOPE_KV_CACHE_RECOMPUTE_EVERY` — breaks KREA's core stability mechanism

**BLOCKED:**
- torch.compile — SM103 + fp8 + FA4 issues

**NEXT (Performance):**
- Fix torchao for cu130 (install 0.14.1 for FP8 fastpaths)
- Benchmark cuDNN attention backend
- Try SageAttention

**NEXT (Capability — Orphaned, Needs Grooming):**
- VACE-14B integration (ready to implement)
- Context editing validation (speculative, needs spike)

---

## Current Bottleneck Understanding

```
Pipeline Total
├── Transformer (denoise + recompute_kv): ~79%
│   └── Within Transformer:
│       ├── self_attn: 56% ─┐
│       │   ├── kv_bias (FA4): 22% ← OPTIMIZED
│       │   └── other_in_self: 74% ← NEW BOTTLENECK
│       ├── cross_attn: 22%
│       └── ffn: 22%
└── VAE decode: ~20% (FIXED by cu130)
```

**"other_in_self" contains:**
- QKV projections (GEMMs)
- RoPE application
- Output projection
- Cache updates
- Memory movement / reshape

**Relevant blogs for other_in_self:**
- `torch-compile-and-diffusers.md` — regional compile, epilogue fusion
- `accel-genai-pytorch-1.md` — GPU sync elimination, CUDA graphs

But torch.compile is blocked, so torchao FP8 fastpaths may be more immediate.

---

## Quick Resume Commands

```bash
# Best config for B300
SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 --iters 6 --skip 2 \
  --quantization fp8_e4m3fn --kv-cache-attention-bias 0.3 --cudnn-benchmark

# With attention profiling
PROFILE_ATTENTION=1 [above command]
```

---

## Open Questions for Next Session

1. Should we create `capability-roadmap.md` separate from `optimization-vision.md`?
2. Update vision doc to mark Option D (recompute) as OFF LIMITS?
3. Priority: torchao fix vs cuDNN benchmark vs capability features?
4. Context editing spike — is it worth validating now or defer?
