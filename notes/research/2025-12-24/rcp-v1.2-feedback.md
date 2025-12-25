## High-level read: your proposal mostly *sits on top of* what the repo already does

The doc in `notes/realtime_video_architecture.md` is essentially proposing a **session-level control/runtime layer** (WorldState → PromptCompiler/StyleManifest → ControlState → GeneratorDriver + ControlBus + FrameBus) around a generator that emits **3-frame chunks** and maintains continuity via **context buffers + KV cache recompute**.

That underlying generator model is already present in the repo today:

* The “GeneratorDriver” role is effectively being played by **`FrameProcessor`** (running in its own worker thread), which:

  * merges parameter updates,
  * gates execution on “chunk readiness” (video input availability),
  * calls `pipeline.prepare()` and `pipeline(**call_params)` once per chunk,
  * handles pause/reset/LoRA updates/transition lifecycle,
  * pushes frames into an output queue.
* The “Generator” is already a modular pipeline: **`KreaRealtimeVideoPipeline`** with `PipelineState` and blocks including **`RecomputeKVCacheBlock`** and **`PrepareContextFramesBlock`**.
* The “Control plane” is already wired end-to-end over WebRTC data channel: **`WebRTCManager` → `VideoProcessingTrack` → `FrameProcessor.update_parameters()`**.

So the doc is directionally aligned — but a lot of the “new runtime” you describe is already embodied in `FrameProcessor + pipeline.state`. The main architectural question is whether you **wrap/extend** what exists, or you **replace** it with a new session-driver abstraction.

---

## Where the proposal aligns strongly with the current implementation

### 1) Chunk-boundary semantics already exist (and match your intent)

Your doc insists on **chunk-transactional control**: apply control changes at boundaries between pipeline calls.

That’s already how `FrameProcessor.process_chunk()` behaves:

* It drains at most one update from `parameters_queue` at the top of each processing attempt, merges it, and then calls the pipeline once per chunk.
* It blocks on enough input frames when video mode requires it (`pipeline.prepare()` → `requirements.input_size`).

Net effect: parameter updates are naturally “committed” at chunk boundaries, not mid-call.

### 2) Your “continuity buffer keys” are real and live in `pipeline.state`

The doc lists continuity keys like:

* `current_start_frame`
* `first_context_frame`
* `context_frame_buffer`
* `decoded_frame_buffer`
* `context_frame_buffer_max_size`
* `decoded_frame_buffer_max_size`

These aren’t theoretical — they’re exactly what the Krea realtime pipeline blocks use:

* `PrepareContextFramesBlock` initializes `first_context_frame` and appends to both buffers.
* `RecomputeKVCacheBlock` allocates those buffers on the first frame and uses them thereafter to rebuild cache context frames.

This is a big alignment win: your “snapshot tier that stores buffers, not full KV tensors” matches the pipeline’s actual continuity mechanism.

### 3) The transition lifecycle you describe is already (mostly) implemented

Your proposed contract:

* send a `transition` dict with `target_prompts`, `num_steps`, interpolation method,
* pipeline blocks manage it internally,
* runtime clears transition when complete.

`FrameProcessor` already manages the lifecycle:

* It **clears stale transitions** when new prompts arrive without a transition.
* It **clears transitions on completion**, by checking `pipeline.state.get("_transition_active", False)` and then replacing `prompts` with `target_prompts`.

And the pipeline itself also protects against stale transitions by clearing `transition` from state if it wasn’t provided on a call (`KreaRealtimeVideoPipeline._generate`).

### 4) Your “edge-trigger LoRA updates” advice matches real side-effects

Your doc warns that `lora_scales` should be edge-triggered because it can force cache resets.

That is exactly true in the pipeline:

* `KreaRealtimeVideoPipeline._generate()` explicitly forces `init_cache = True` on LoRA scale updates when `manage_cache` is enabled.

And `FrameProcessor` already behaves like your `PipelineAdapter` idea:

* It pops `lora_scales` out of `parameters` so it **won’t be resent every chunk**.

### 5) Mode defaults + schema introspection are already first-class

Your doc’s intent to drive UI defaults from pipeline metadata is aligned with the existing `/api/v1/pipelines/schemas` endpoint, which returns each pipeline config’s JSON schema plus supported modes and mode-specific defaults.

That’s a rare case where the “architecture doc” and “actual codebase ergonomics” are already converged.

---

## Key gaps / mismatches you should call out (and adjust expectations around)

### 1) “Session lifecycle + step mode” is not present as an API concept today

Your doc proposes REST endpoints like:

* `POST /v1/sessions`
* `POST /v1/sessions/{id}/step`
* `POST /pause`, `/resume`, etc.

Today the repo’s “sessions” are **WebRTC sessions**, and the only public control surface is:

* WebRTC offer/ICE endpoints
* data channel messages that mutate `FrameProcessor.parameters`

There is no explicit server-side “session object” with a step API. The closest thing is:

* `FrameProcessor.paused` (driven by `"paused"` parameter updates)
* `reset_cache` behavior
* pipeline load/unload lifecycle at the server level

**Practical implication:** if you keep the proposed REST session API, it’s an *additional* control plane that currently doesn’t exist — and you should decide whether you want **two** control planes (REST + WebRTC) or you want to model everything as “messages over data channel”.

### 2) Your GeneratorDriver pseudocode assumes “no input required”

In your doc’s `GeneratorDriver._generate_chunk()`, you call the pipeline each tick. That matches **text-to-video** mode.

But the current runtime has a hard distinction:

* In **video mode**, the pipeline `prepare()` returns requirements (`input_size`), and `FrameProcessor` **waits** until it has enough frames.
* Video input can come from:

  * WebRTC incoming video track (`VideoProcessingTrack.input_loop()` → `FrameProcessor.put()`), and/or
  * Spout receiver mode (exists in `FrameProcessor`, even though you didn’t focus on it in the doc).

So your “driver” abstraction, if you formalize it, needs to explicitly incorporate “input readiness” as a first-class state (your doc mentions it in planes, but the pseudocode driver elides it).

### 3) Multi-session correctness is a major risk in the current architecture (and your doc assumes per-session ownership)

This is the biggest “alignment risk” I see:

* `WebRTCManager` supports multiple sessions (`self.sessions: dict[str, Session]`).
* Each session creates its own `VideoProcessingTrack` and `FrameProcessor`.
* But **all of them share the same `PipelineManager`**, and (very likely) the same underlying pipeline instance, because `PipelineManager.get_pipeline()` takes no session id and the manager looks singleton-ish.

If the pipeline object is shared across sessions, you’ll have:

* shared `pipeline.state` across users,
* shared KV cache and context buffers,
* races between multiple `FrameProcessor` threads calling `pipeline(**kwargs)` concurrently,
* cross-session contamination of temporal continuity.

Your proposed architecture strongly implies **a pipeline instance (and continuity state) owned by a session/driver**. That’s the right mental model for branching/snapshots — but it may not match the repo’s current runtime assumption unless the product is “single active session at a time.”

**Recommendation for the doc (no implementation, just clarity):**
Add an explicit statement like:

* *“Current server runtime is effectively single-session; multi-session will require per-session pipeline ownership or strict serialization.”*

### 4) “FrameBus / ChunkStore / Timeline” are not present today (you only have ephemeral queues)

The doc assumes you can:

* scrub,
* build previews,
* branch,
* request historical ranges (`get_last_n_seconds`, etc.).

The current code has:

* `FrameProcessor.output_queue` (transient, drop-on-overflow behavior)
* optional Spout sender queue (also transient)
* no persistent ring buffer for rendered chunks

So features like timeline playback and branching previews require **a retention layer** that doesn’t exist in the server runtime today.

This doesn’t invalidate the doc — it just means that your “FrameBus” plane is currently “a queue feeding WebRTC,” not “a buffer enabling time-travel.”

### 5) Negative prompts are still aspirational in this pipeline path

Your `ControlState` includes `negative_prompt` and the doc keeps it “for forward compatibility.”

That’s correct: the current Krea realtime pipeline path you’re using does not obviously consume a negative prompt (and your own doc even notes it).

So StyleManifest authoring should not assume “negatives fix artifacts” unless you confirm other pipelines/backends use them.

### 6) Naming mismatches between config schema and runtime parameters could bite the “auto UI” story

This repo has *two vocabularies*:

* Pipeline config models (`scope/core/pipelines/schema.py`) use fields like `denoising_steps`, `base_seed`.
* WebRTC runtime `Parameters` (`scope/server/schema.py`) uses runtime-ish fields like `denoising_step_list`, `reset_cache`, `lora_scales`, etc.

Your doc leans into the runtime vocabulary (which is correct for control), but it also wants to drive defaults from the config schema endpoint.

**Risk:** the UI (or your compiler layer) needs an explicit mapping between:

* config defaults (`denoising_steps`) and
* runtime call args (`denoising_step_list`)

Otherwise you’ll get “the schema says one thing, the runtime expects another” drift.

### 7) LoRA “permanent merge” vs runtime updates is a product-level constraint your doc should surface

The server load schema defaults `lora_merge_mode` to `PERMANENT_MERGE`, and its own docstring says:

* permanent merge: max FPS, **no runtime updates**
* runtime_peft: allows runtime updates at reduced FPS

Your architecture assumes runtime `lora_scales` changes are part of the live control surface.

That’s only *reliably* true if the loaded pipeline is in a runtime-update-capable mode.

**Doc tweak suggestion:** add a “capabilities matrix” per pipeline + merge mode: which knobs are “live-editable” vs “requires reload.”

---

## Subtle behavioral mismatches to watch (practical fit)

### 1) Pause semantics exist in two places

* Data channel handler calls `session.video_track.pause(data["paused"])` (freezes output frames).
* `FrameProcessor.process_chunk()` also uses `"paused"` to stop generation work.

This is fine, but it means your “Pause/Resume” in the proposal is really two concepts:

* pause output playback (repeat last frame),
* pause generation (stop advancing state / buffers).

Your doc treats pause as a control-state machine concept; the repo implements it as both a **track behavior** and a **generator behavior**. Worth explicitly acknowledging so future “step” mode doesn’t accidentally unpause playback or vice versa.

### 2) Transition completion detection depends on a private-ish state key

`FrameProcessor` checks `pipeline.state.get("_transition_active", False)`.

Your doc assumes blocks signal completion. That’s probably true today, but it’s fragile unless you treat `_transition_active` as part of the pipeline contract (or you add a more explicit signal later).

The practical doc-level point: treat transition lifecycle as **pipeline-specific capability**, not a universal invariant across pipelines.

### 3) Snapshots that restore “buffers only” assume shape/device compatibility

Your doc is correct that you don’t need to serialize the full KV cache tensors because recompute exists.

But in practice, buffer snapshots are only valid if:

* same pipeline id (and same internal model config like `kv_cache_num_frames`, downsample factors),
* same resolution (`height/width`),
* same device and dtype expectations,
* LoRA configuration hasn’t fundamentally changed the model structure (usually fine, but worth noting).

Your doc mentions some of this; I’d make the “compatibility checks” more explicit because the real code allocates buffers based on config and expects consistent shapes.

---

## What I’d change in the architecture doc to better match the repo (still no implementation)

### 1) Recast existing components into your proposed abstractions

Right now the doc reads as if you’ll build a new driver from scratch. But the repo already has a workable “driver”:

* **`FrameProcessor` = GeneratorDriver + ControlBus (simple) + Output queue**
* **WebRTC data channel = Control API**
* **`pipeline.state` = continuity store**

If you update the doc to explicitly map those, you’ll reduce accidental duplication and make the plan feel “incremental over existing runtime,” which is more realistic.

### 2) Add an explicit “single-session vs multi-session” section

Given the likelihood of shared pipeline state, you should decide and document:

* “Scope is single-active-session by design” **or**
* “Scope supports N sessions and needs per-session pipeline ownership / serialization.”

This is essential before investing in branching/snapshots, because branching implies *session-local continuity state*.

### 3) Make “input readiness” a first-class concept in GeneratorDriver

Even in the doc-level pseudocode, include the notion that:

* in video mode, generation is gated by input frames,
* input source can be WebRTC track or Spout receiver,
* “step” behavior differs by mode (step without new input frames may mean “reuse last frames,” or “stall,” depending on design).

### 4) Add a short “capabilities by pipeline” appendix

Your doc is mostly written from a KreaRealtimeVideo perspective, but the repo supports multiple pipelines with different behaviors (VACE-enabled pipelines, passthrough, etc.).

Even a simple statement like:

* “World/Style/Branching layer is pipeline-agnostic, but snapshot continuity is only guaranteed for pipelines that expose these state keys…”

…would prevent future confusion.

### 5) Add a “config-schema vs runtime-params mapping” note

If the UI uses `/pipelines/schemas` to populate defaults, your doc should acknowledge the mapping needed between:

* config defaults (e.g., `denoising_steps`) and
* runtime args (e.g., `denoising_step_list`)

Otherwise your “auto UI generation” story will fail in subtle ways.

---

## Bottom line

* The architectural direction is compatible with the codebase: **KV cache recompute + context buffers + chunk processing + prompt transitions + LoRA scale updates** are already real and match your proposed abstractions.
* The main gaps are **session semantics (especially multi-session isolation), step/timeline/snapshot retention layers, and integrating mode/input gating into the driver abstraction.**
* The largest risk is **pipeline state ownership**: branching and continuity only make sense if a session “owns” its pipeline state, or you serialize access strictly.

If you want, I can also provide a “doc-diff style” checklist of exact sections in `realtime_video_architecture.md` that should be amended to reflect these realities (still without proposing code).
