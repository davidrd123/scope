# Proposal: Transition Prompts in Playlists

> Status: Draft
> Date: 2025-12-26

## Problem

Currently, embedding interpolation goes directly from prompt A to prompt B. These prompts are designed as discrete "shots" - they weren't written to blend into each other. The semantic path A→B may pass through awkward intermediate states.

## Proposal

Allow explicit **transition prompts** in playlists that describe the in-between state or transformation itself.

## Format

Use `>` prefix for transition prompts:

```
A serene forest at dawn
> Morning mist dissolving, forms shifting into geometry
A busy city street at noon
> Neon bleeding into shadows, urban decay
A quiet bedroom at night
```

### Semantics

- `>` prefix = transition prompt belonging to the *following* scene
- Transition prompts are optional - not every scene needs one
- Multiple `>` lines before a scene: **MVP behavior = last `>` wins** (stores a single pending transition prompt). Chaining multiple transition prompts is a possible extension (TBD).

### Backward Compatibility / Versioning

- Existing playlists **without** `>` transition lines work unchanged.
- **Important:** In the current codebase, `PromptPlaylist.from_file()` appends every non-empty line as a prompt. That means lines starting with `>` (and `#`) are **not ignored** today — they will be treated as literal prompts.
- Therefore, playlists using `>` transition prompts will require a Scope build that implements this proposal.
- Feature remains opt-in per scene once implemented, but it is **not forward-compatible** with older builds.

> **Note:** This syntax is not supported by current playlist parsing; do not use in production playlists until implemented.

## Interpolation Behavior

When navigating A → B where B has transition prompt T:

### Option 1: Two-stage (simple)
```
chunks 1-2: A ────────► T
chunks 3-4: T ────────► B
```

### Option 2: Three-point blend (smoother)
```
chunk 1: A
chunk 2: blend(A, T, 0.5)
chunk 3: T
chunk 4: blend(T, B, 0.5)
chunk 5: B
```

### Option 3: Weighted midpoint
```
t=0.0: A
t=0.3: lerp(A, T)
t=0.5: T (transition prompt at midpoint)
t=0.7: lerp(T, B)
t=1.0: B
```

## Edge Cases

| Case | Behavior |
|------|----------|
| First prompt has `>` | Ignore (no source to transition from) |
| Multiple `>` before scene | Last `>` wins (MVP); chaining is future extension |
| `>` without following scene | Ignore (dangling transition) |
| Hard cut to scene with `>` | Skip transition (no continuity) |

## Implementation Sketch

### 1. Playlist Parsing

```python
@dataclass
class PlaylistEntry:
    prompt: str
    transition_prompt: str | None = None  # The `>` line before this entry
```

### 2. PromptPlaylist Changes

```python
def _parse_lines(self, lines: list[str]) -> list[PlaylistEntry]:
    entries = []
    pending_transition = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            pending_transition = line[1:].strip()
        else:
            entries.append(PlaylistEntry(
                prompt=line,
                transition_prompt=pending_transition
            ))
            pending_transition = None

    return entries
```

### 3. Navigation Changes

When `transition=True` and target has `transition_prompt`:

```python
if target.transition_prompt and transition:
    # Two-stage transition
    half = transition_chunks // 2

    # Stage 1: current → transition_prompt
    msg1 = build_transition_msg(target.transition_prompt, half)

    # Stage 2: transition_prompt → target (after delay)
    msg2 = build_transition_msg(target.prompt, transition_chunks - half)
```

Or handle entirely in `EmbeddingBlender` with waypoints.

## Creative Applications

### Scene Bridges
```
Underwater coral reef
> Bubbles rising, water surface approaching
Sunny beach from below the waves
```

### Mood Transitions
```
Joyful celebration, bright colors
> Colors fading, energy dissipating
Melancholic empty room
```

### Abstract Morphs
```
Geometric crystalline structures
> Crystals shattering into organic forms
Flowing organic shapes, cellular
```

### Style Transitions
```
Photorealistic cityscape
> Reality dissolving into brushstrokes
Impressionist painting of a city
```

## Open Questions

1. **Timing control**: Should transition prompts have their own chunk count?
   ```
   >3 Morning mist dissolving  # 3 chunks for this transition
   ```

2. **Bidirectional transitions**: Same transition for A→B and B→A? Or separate?

3. **Transition libraries**: Pre-built transition prompts for common patterns?
   ```
   >@dissolve  # Expands to a standard dissolve description
   ```

4. **LLM-generated**: Auto-generate transition prompts from A and B?

## Related

- Soft cut (KV bias) - opens plasticity window
- Embedding transition (LERP/SLERP) - interpolates embeddings
- This proposal - adds semantic waypoints to the interpolation path

---

## Implementation Context (Code Exploration)

> Added 2025-12-27 from codebase exploration

### File Locations

| Component | File | Key Functions/Classes |
|-----------|------|----------------------|
| **PromptPlaylist** | `src/scope/realtime/prompt_playlist.py` | `PromptPlaylist.from_file()`, `next()`, `current` |
| **EmbeddingBlender** | `src/scope/core/pipelines/blending.py` | `start_transition()`, `get_next_embedding()`, `slerp()` |
| **EmbeddingBlendingBlock** | `src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py` | `__call__()`, `TransitionConfig` |
| **TextConditioningBlock** | `src/scope/core/pipelines/wan2_1/blocks/text_conditioning.py` | Prompt encoding |
| **API Endpoints** | `src/scope/server/app.py` | `playlist_next()`, `_apply_playlist_prompt()` |
| **Frame Processor** | `src/scope/server/frame_processor.py` | `process_chunk()`, transition completion block |

### Current Data Structure

```python
# src/scope/realtime/prompt_playlist.py — PromptPlaylist class
@dataclass
class PromptPlaylist:
    source_file: str = ""
    prompts: list[str] = field(default_factory=list)  # Just strings, no metadata
    current_index: int = 0
    trigger_swap: tuple[str, str] | None = None
    original_count: int = 0
```

**Key observation:** Prompts are stored as bare strings. No per-prompt metadata exists yet.

### Current Parsing Flow

```python
# src/scope/realtime/prompt_playlist.py — from_file() method
prompts = []
for line in lines:
    line = line.strip()
    if skip_empty and not line:
        continue
    # Apply trigger swap...
    prompts.append(line)  # Just appends the string
```

**Hook point:** The loop in `from_file()` — this is where `>` parsing would go.

### Current Transition Flow

1. **User calls** `POST /api/v1/realtime/playlist/next?transition=true&transition_chunks=4`
2. **`playlist_next()`** → calls `_apply_playlist_prompt(transition=True, transition_chunks=4)`
3. **`_apply_playlist_prompt()`** builds message:
   ```python
   msg = {
       "transition": {
           "target_prompts": [{"text": prompt, "weight": 1.0}],
           "num_steps": transition_chunks,
           "temporal_interpolation_method": "linear"
       }
   }
   ```
4. **Frame processor** receives message, forwards to pipeline
5. **EmbeddingBlendingBlock** calls `blender.start_transition(source, target, num_steps)`
6. **Each pipeline call / chunk:** `get_next_embedding()` returns the next interpolated embedding (one dequeue per generation call)
7. **Transition complete:** Frame processor promotes `target_prompts` into `prompts` and clears `transition`

> **Note on `num_steps`:** The current `EmbeddingBlender.start_transition()` uses `torch.linspace(0, 1, steps=num_steps)`, which **includes both endpoints**. This means the first step is exactly the **source** embedding (t=0) and the last step is exactly the **target** embedding (t=1). With `num_steps=2`, the transition is effectively `[source, target]` with no intermediate. For visible interpolation, prefer `num_steps >= 3`.

> **Note on `prompts` during transition:** While a transition is active, `parameters["prompts"]` still reflects the prior prompt. The effective prompt for recording/UI comes from `transition.target_prompts` until the frame processor promotes it on completion.

### What Needs to Change

#### 1. New Data Structure

```python
# src/scope/realtime/prompt_playlist.py
@dataclass
class PlaylistEntry:
    prompt: str
    transition_prompt: str | None = None  # The `>` line before this entry

@dataclass
class PromptPlaylist:
    source_file: str = ""
    entries: list[PlaylistEntry] = field(default_factory=list)  # Changed from list[str]
    current_index: int = 0
    # ...
```

#### 2. Parser Changes

```python
# src/scope/realtime/prompt_playlist.py — updated from_file() loop
entries = []
pending_transition = None

for line in lines:
    line = line.strip()
    if skip_empty and not line:
        continue
    if line.startswith("#"):  # Comments
        continue
    if line.startswith(">"):
        pending_transition = line[1:].strip()
    else:
        # Apply trigger swap to main prompt...
        entries.append(PlaylistEntry(
            prompt=line,
            transition_prompt=pending_transition
        ))
        pending_transition = None
```

#### 3. Navigation Changes

When navigating to an entry with `transition_prompt`, the transition becomes two-stage:

```python
# In _apply_playlist_prompt() or equivalent
entry = playlist.current_entry

if entry.transition_prompt and transition:
    # Stage 1: current → transition_prompt
    half = transition_chunks // 2
    # ... send transition to entry.transition_prompt

    # Stage 2: transition_prompt → entry.prompt (scheduled after stage 1)
    # ... send transition to entry.prompt
```

**Option A:** Handle in `_apply_playlist_prompt()` with delayed second message
**Option B:** Extend `EmbeddingBlender` to support waypoints (list of embeddings, not just source→target)

> **Scheduling warning:** A two-stage transition **cannot** be implemented by sending two transition messages back-to-back, because control messages are merged and applied at chunk boundaries (last write wins). The second message must be sent only after stage 1 has progressed (e.g., after N chunks) or after the pipeline signals the transition has completed. This requires orchestration: either a server-side pending state machine, or client-driven follow-up message triggered by completion/elapsed chunks.

#### 4. API Surface

No new endpoints needed. The existing `transition=true` parameter triggers the behavior automatically when the target entry has a `transition_prompt`.

### Backward Compatibility

- Entries without `>` prefix work exactly as before
- `current` property returns `entry.prompt` (the main prompt text)
- `to_dict()` could optionally expose `transition_prompt` for UI display

### Testing Strategy

1. **Unit test:** Parse playlist with `>` lines, verify entries have correct `transition_prompt`
2. **Integration test:** Navigate with `transition=true`, verify two-stage interpolation
3. **Edge cases:** First entry with `>`, multiple `>` lines, `>` without following prompt

### Effort Estimate

| Task | Effort |
|------|--------|
| Data structure change | S |
| Parser modification | S |
| Two-stage transition logic | M |
| Tests | S |
| **Total** | **S-M** |

### Risk Assessment

- **Low:** Parser change is isolated
- **Low:** Backward compatible (existing playlists work)
- **Medium:** Two-stage transition timing needs tuning (how long for each stage?)

### Decision Needed

**Two-stage timing:** Should each stage get half the chunks, or should transition prompts have their own duration?

Option A: Split evenly (`transition_chunks // 2` each)
Option B: Transition prompt gets fixed 2 chunks, remainder goes to target
Option C: Allow syntax like `>3 dissolving mist` for explicit chunk count
