Below is a technical vet of `notes/proposals/transition-prompts.md` against the **current** code you included (playlist parser, playlist API, frame processor, modular pipeline blocks, EmbeddingBlender).

I’m focusing on (a) the **Semantics / Backward Compatibility** claims near the top, and (b) the **Implementation Context (Code Exploration)** section near the bottom, as requested.

---

## Mismatches and incorrect assumptions

### 1) Backward-compat claim “`>` lines are ignored by current parser” is false

**Doc says (Backward Compatibility section):**

* “Lines starting with `>` are simply ignored by current parser”

**Reality in `src/scope/realtime/prompt_playlist.py`:**

* `PromptPlaylist.from_file()` currently appends **every non-empty line** as a prompt (after optional trigger swap).
* There is **no filtering** for `>` lines.
* There is also **no filtering** for comment lines like `#`.

**Impact:**

* A playlist file that includes `>` lines today will treat those lines as literal prompts, and playlist navigation will happily send them to the model.
* So the proposed `>` syntax is **not forward-compatible** with current builds, and “existing playlists work unchanged” is only true for playlists that **don’t** use `>` lines.

✅ **Action for doc:** Replace the “ignored by current parser” line with an explicit note that `>` lines are currently treated as normal prompts and require an updated Scope version.

---

### 2) Implementation Context step “Each frame: `get_next_embedding()` …” is inaccurate

**Doc says (Current Transition Flow):**

* “Each frame: `get_next_embedding()` returns interpolated embedding”

**Reality:**

* The transition queue is consumed once per **pipeline call / chunk**, not per frame.
* The pipeline generates a batch of frames for the chunk, but the conditioning embedding chosen for that call is what the denoiser sees for that chunk.

✅ **Action for doc:** Change wording to “each chunk / pipeline call”.

---

### 3) `num_steps` semantics are subtly different than the doc’s implied “N chunks of smooth motion”

In `EmbeddingBlender.start_transition()`:

```py
t_values = torch.linspace(0, 1, steps=num_steps)
```

This **includes both endpoints** (0 and 1). That implies:

* The **first** queued embedding is exactly the **source** embedding (t=0).
* The **last** queued embedding is exactly the **target** embedding (t=1).
* With `num_steps=2`, you effectively get: `[source, target]` (no intermediate).

Your proposal’s “Two-stage (simple) chunks 1-2: A → T” reads like those two chunks contain interpolation. With the current implementation, **2 steps isn’t really a blend**, it’s a delayed snap.

✅ **Action for doc:** Add a note that with the current blender implementation, `num_steps` includes endpoints; recommend allocating ≥3 steps per stage if you want visible intermediate motion.

---

### 4) “Two-stage transition via two messages” needs a scheduling warning

The Implementation Sketch suggests:

> send msg1 (A→T), then send msg2 (T→B) “after delay”

**What’s missing:**

* There is no existing mechanism in `_apply_playlist_prompt()` / playlist endpoints to schedule a delayed follow-up message (without adding new state/tasks).
* Also, sending two control messages back-to-back risks them being **merged/overwritten before the next chunk boundary**, meaning the first transition may never actually run.

FrameProcessor semantics reinforce this risk:

* updates are merged, translated to `SET_PROMPT`, and applied at the next chunk boundary in deterministic order.
* “last write wins” within that boundary.

✅ **Action for doc:** Add an explicit note: “requires orchestration” (server-side state machine or client-driven second message triggered by completion/elapsed chunks).

---

### 5) Missing detail: transition messages don’t set `prompts` immediately (old prompts remain until completion)

In `_apply_playlist_prompt()` when `transition=true`, the message is:

```py
msg = { "transition": { "target_prompts": ... } }
```

It does **not** send `prompts`.

That means during the transition:

* `self.parameters["prompts"]` in the frame processor still reflects the “current” prompt,
* but `_get_current_effective_prompt()` and the pipeline blocks treat `transition.target_prompts` as the effective target.

Then on completion, frame_processor promotes `target_prompts` into `prompts` and clears `transition`.

✅ **Action for doc:** Mention this, because it affects how UIs/recorders interpret “current prompt” during transitions, and it matters for a two-stage scheme (you probably want to know when stage 1 has truly “landed” on T).

---

### 6) Minor doc hygiene: line-number anchors are brittle

Your Implementation Context cites specific line ranges (e.g., `app.py:1644-1712`). That might be correct today, but it will drift easily.

✅ **Action for doc:** Prefer anchoring by function name (`_apply_playlist_prompt`, `playlist_next`, `FrameProcessor` transition completion block) instead of exact line numbers, or mark ranges as “approx”.

---

## Proposed edits to `transition-prompts.md` (no code changes)

Below are concrete patch-style edits you can apply to the doc.

### Edit 1: Fix Backward Compatibility section (top of doc)

**Replace this:**

```md
### Backward Compatibility

- Existing playlists work unchanged
- Lines starting with `>` are simply ignored by current parser
- Feature is opt-in per-scene
```

**With this (accurate to current code):**

```md
### Backward Compatibility / Versioning

- Existing playlists **without** `>` transition lines work unchanged.
- **Important:** In the current codebase, `PromptPlaylist.from_file()` appends every non-empty line as a prompt.  
  That means lines starting with `>` (and `#`) are **not ignored** today — they will be treated as literal prompts.
- Therefore, playlists using `>` transition prompts will require a Scope build that implements this proposal (or an external preprocessor).
- Feature remains opt-in per scene once implemented, but it is **not forward-compatible** with older builds.
```

---

### Edit 2: Tighten Semantics section around multiple `>` lines

Right now you say “Multiple `>` lines could chain (TBD)”, but your Implementation Sketch parser shows a single `pending_transition` (meaning **last one wins**).

**Change this bullet:**

```md
- Multiple `>` lines before a scene could chain (TBD)
```

**To something that matches the sketch:**

```md
- Multiple `>` lines before a scene: **MVP behavior = last `>` wins** (stores a single pending transition prompt).  
  Chaining multiple transition prompts is a possible extension (TBD).
```

---

### Edit 3: Fix “Current Transition Flow” wording (Implementation Context)

**Replace step 6:**

```md
6. **Each frame:** `get_next_embedding()` returns interpolated embedding
```

**With:**

```md
6. **Each pipeline call / chunk:** `get_next_embedding()` returns the next interpolated embedding (one dequeue per generation call)
```

---

### Edit 4: Add a note about `num_steps` including endpoints (Implementation Context)

Insert under “Current Transition Flow” (after step 6 is fine):

```md
> Note on `num_steps`: the current `EmbeddingBlender.start_transition()` uses `torch.linspace(0, 1, steps=num_steps)`,
> which includes both endpoints. This means the first step is exactly the **source** embedding (t=0) and the last step
> is exactly the **target** embedding (t=1). With `num_steps=2`, the transition is effectively `[source, target]` with
> no intermediate. For visible interpolation, prefer `num_steps >= 3`.
```

---

### Edit 5: Add a scheduling warning to the “Two-stage” implementation sketch

In the “Navigation Changes” / two-stage pseudocode area, add:

```md
**Scheduling note:** A two-stage transition cannot be implemented by sending two transition messages back-to-back,
because control messages are merged and applied at chunk boundaries (last write wins). The second message must be
sent only after stage 1 has progressed (e.g., after N generated chunks) or after the pipeline signals the transition
has completed. This requires orchestration (server-side pending state machine or client-driven follow-up message).
```

---

### Edit 6: Clarify that transition messages don’t overwrite `prompts` immediately

Add a short clarification under “Current Transition Flow”:

```md
> While a transition is active, the server may still have `parameters["prompts"]` set to the prior prompt.
> The effective prompt for recording/UI precedence comes from `transition.target_prompts` until the frame processor
> promotes `target_prompts` into `prompts` on completion.
```

---

## Optional doc improvements (still “doc-only”, but helpful)

Not required for correctness, but these would prevent future confusion:

* In “Interpolation Behavior” diagrams, label steps as **chunks** rather than frames.
* In the “Format” section, add a one-liner:

  * “This syntax is **not supported by current playlist parsing**; do not use in production playlists until implemented.”

---

If you want, I can also produce a single “clean” replacement block for the entire **Implementation Context** section (keeping your structure but incorporating the corrections above), so you can paste it wholesale.
