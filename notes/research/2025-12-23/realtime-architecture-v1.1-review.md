Version 1.1 is a real step forward. It’s now “implementable as written” in the ways that matter for your sprint, and it explicitly incorporates the key fixes that were previously missing (world recompilation bug, resume loop guard, kwargs coverage, correct KV bias default, chunk-based FrameBus sizing, and a first pass at continuity-aware snapshots).

What changed from “good concept doc” to “useful canonical doc” is that it now acknowledges the ugly parts (snapshot tiers, determinism limits, token budgeting, streaming ambiguity) and turns them into explicit upgrade paths instead of hidden assumptions.

That said, there are still a few places where the document is directionally correct but the concrete implementation details are not yet aligned with how KREA actually stores and advances state.

## What is solid and I would keep as-is

### The two-axis framing is a real win

Separating semantic layers (Presentation, World, Style, Render) from operational planes (Control, Generation, Media, Timeline, Analysis) makes the system legible when it gets complicated. It prevents the classic failure mode where “world logic” ends up entangled with “how we stream frames.”

### “Branch only at chunk boundaries” remains the right constraint

Given the pipeline’s blockwise nature and cache recomputation behavior, chunk boundaries are the natural “transaction” boundary. This keeps your control semantics sane.

### The StyleManifest + PromptCompiler split is structurally correct

Codifying LoRA-specific prompt knowledge into manifests is exactly how you turn your prompt experiments into reusable assets, and it makes the “content vs style” separation enforceable.

### The Known Limitations section is excellent

This is the part that will save you time. It clearly labels what’s “good enough now” versus what’s “true but expensive later.”

## The biggest remaining mismatch: where continuity state actually lives

You added “generator_continuity” to snapshots and explicitly named the right buffers: `first_context_frame`, `context_frame_buffer`, `decoded_frame_buffer`. That is the correct conceptual set.

But the current capture/restore method is likely incorrect in practice because it tries to read them as direct attributes on the pipeline object:

```python
if hasattr(self.pipeline, 'context_frame_buffer'):
    continuity["context_frame_buffer"] = self.pipeline.context_frame_buffer
```

In the KREA pipeline, those buffers are produced by pipeline blocks and stored in the pipeline’s internal `state` as keys (they are described explicitly as outputs of `PrepareContextFramesBlock`).

### What I would change in the canonical doc

Add a small “Pipeline Adapter” abstraction whose whole job is:

1. Convert `ControlState` into the exact kwargs that `KreaRealtimeVideoPipeline.__call__` expects.
2. Extract and restore continuity buffers from `pipeline.state` using known keys.

Why this matters: it avoids fragile `hasattr()` checks and turns “seamless continuation snapshot” into something you can actually test deterministically.

If you do only one revision to v1.1, do this.

## Second biggest gap: ControlBus exists, but it is not actually driving the GeneratorDriver

Right now you have:

* A fully specified ControlBus with typed events and apply modes.
* A GeneratorDriver that does not consume ControlBus events at all. It uses `_pending_world_updates` and `_pending_control_updates` lists instead.

That’s fine for an early prototype, but it creates an architectural split-brain: the doc says “events are the control plane,” but the code path shown is “mutable state with ad hoc queues.”

### The practical risk

Branching and replay get much harder if your “source of truth” is not the event log.

### The clean fix

Make GeneratorDriver own a `ControlBus`, and at each chunk boundary do:

* `events = control_bus.drain_pending(is_paused=(state==PAUSED))`
* apply them in a deterministic order (for example: lifecycle, then style switch, then world updates, then direct control overrides)
* record “applied at chunk_index K” in event history (this becomes your replay skeleton)

Also, there is a specific bug in the helper `pause_event`: it sets `apply_mode=IMMEDIATE_IF_PAUSED`, which is the opposite of what you want for pausing a running system. That mode only applies immediately when already paused.

## Prompt ramps are described but not implemented

The doc includes transition fields (`transition_chunks_remaining`, `transition_from_prompts`) and the API allows `transition: {type: ramp, chunks: 4}`.

But nothing in the shown driver loop applies that ramp over successive chunks, and `to_pipeline_kwargs()` does not express “blend A to B” in any way.

This is important because your real “instrument feel” depends on smooth changes, not hard prompt cuts.

### Two viable approaches

* **Approach A (engine-native):** If Scope’s pipeline already supports embedding blending for prompt transitions (there is an `EmbeddingBlendingBlock` in the KREA block sequence), then the ramp should be expressed in the kwargs/state the block expects.
* **Approach B (control-layer):** Implement ramping in the Control Plane by progressively mixing weights or generating intermediate prompt states across chunk boundaries.

The canonical doc should pick one and state the exact mechanics, otherwise “ramp” will stay aspirational.

## Rollouts “in parallel” are not realistic with one pipeline instance

The sequence diagram shows `par Branch 1..4` and repeatedly calls `restore(snapshot)` and generates chunks.

On a single GPU-backed pipeline object with mutable internal state, you cannot truly run those in parallel without:

* multiple pipeline instances (each with its own state and caches), or
* strict sequential rollouts (restore, run horizon, store preview, restore again), or
* a job system that serializes access to the generator while still producing “as-if parallel” previews.

For hackathon mode, sequential rollouts are totally acceptable, but the doc should say “rollouts are executed sequentially unless you provision multiple generator workers.”

## One more subtle correctness point: pipeline kwarg names and cache reset semantics

You improved `to_pipeline_kwargs()` to include `lora_stack` and `kv_cache_attention_bias` and set bias default to 0.3, which matches the KREA default described in the pipeline guide.

But there is still a risk: the true KREA pipeline flow talks about `lora_scales` triggering `init_cache=True` when LoRA scales change.
Your canonical doc uses `lora_stack` as the input contract. That might be correct for Scope’s wrapper, but it might not.

This is another reason the “Pipeline Adapter” layer is worth making explicit: it is the right place to map `lora_stack` to whatever the pipeline actually expects, and to enforce “LoRA changes force cache reset” deterministically.

## My updated verdict

* As a canonical architecture doc: v1.1 is strong, coherent, and sprint-aligned. It’s now the kind of doc you can hand to someone else and they can build the skeleton without guessing what you meant.
* As a literal implementation spec: it still needs two “make it real” clarifications:

  1. Continuity snapshot capture/restore should operate on pipeline state keys, not guessed attributes.
  2. ControlBus should be the actual mechanism by which GeneratorDriver changes state, or the doc should drop it until it is.

If you want a fast next revision (v1.2), I would add a short section titled “Pipeline Adapter Contract” and another titled “Chunk Boundary Event Semantics,” and tighten the rollout diagram to say sequential by default.

If you want, paste your current Scope pipeline call signature (or the wrapper that accepts kwargs) and I’ll draft the exact adapter interface and the snapshot key list that will actually work with KREA’s `state` layout.
