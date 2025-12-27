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
- Multiple `>` lines before a scene could chain (TBD)

### Backward Compatibility

- Existing playlists work unchanged
- Lines starting with `>` are simply ignored by current parser
- Feature is opt-in per-scene

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
| Multiple `>` before scene | Chain them? Use last? TBD |
| `>` without following scene | Ignore |
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

| Component | File | Purpose |
|-----------|------|---------|
| **PromptPlaylist** | `src/scope/realtime/prompt_playlist.py` | Playlist parsing and navigation |
| **EmbeddingBlender** | `src/scope/core/pipelines/blending.py` | Core interpolation logic (LERP/SLERP) |
| **EmbeddingBlendingBlock** | `src/scope/core/pipelines/wan2_1/blocks/embedding_blending.py` | Pipeline integration, transition lifecycle |
| **TextConditioningBlock** | `src/scope/core/pipelines/wan2_1/blocks/text_conditioning.py` | Prompt encoding |
| **API Endpoints** | `src/scope/server/app.py:1644-1712` | `/playlist/next` with transition params |
| **Frame Processor** | `src/scope/server/frame_processor.py:858-1519` | Control message handling, transition completion |

### Current Data Structure

```python
# src/scope/realtime/prompt_playlist.py:20-32
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
# src/scope/realtime/prompt_playlist.py:59-80
prompts = []
for line in lines:
    line = line.strip()
    if skip_empty and not line:
        continue
    # Apply trigger swap...
    prompts.append(line)  # Just appends the string
```

**Hook point:** Lines 59-80 in `from_file()` — this is where `>` parsing would go.

### Current Transition Flow

1. **User calls** `POST /api/v1/realtime/playlist/next?transition=true&transition_chunks=4`
2. **`app.py:1644-1712`** → `_apply_playlist_prompt(transition=True, transition_chunks=4)`
3. **`app.py:1512-1567`** builds message:
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
6. **Each frame:** `get_next_embedding()` returns interpolated embedding
7. **Transition complete:** Frame processor promotes target to active prompts

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
# src/scope/realtime/prompt_playlist.py:59-80
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

**Option A:** Handle in `app.py:_apply_playlist_prompt()` with delayed second message
**Option B:** Extend `EmbeddingBlender` to support waypoints (list of embeddings, not just source→target)

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
