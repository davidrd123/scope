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
