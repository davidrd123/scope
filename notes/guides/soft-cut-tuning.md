# Soft Cut Tuning Guide

> Quick reference for tuning soft cut transitions in playlist navigation.

## What Soft Cut Does

Soft cut temporarily lowers `kv_cache_attention_bias` when transitioning between prompts, making the scene more responsive to the new prompt without a full cache reset (hard cut).

- **Normal operation**: bias = 0.3 (default)
- **Soft cut**: bias drops to temp value for N chunks, then auto-restores

```
Prompt A ──────────────────────────────────────────────────▶
                    │
                    │ Navigate to Prompt B (with soft cut)
                    ▼
           ┌─────────────────┐
           │  SOFT WINDOW    │  ← bias lowered (e.g., 0.15)
           │  (N chunks)     │    scene is "loose", adapts faster
           └─────────────────┘
                    │
                    ▼
Prompt B ◀──────────────────────────────────────────────────
           bias restored to 0.3, scene "locks in"
```

More chunks = longer soft window = more time for scene to shift before locking in.

## Chunk Duration vs FPS

A chunk = 3 frames. Actual duration depends on your current FPS:

| Pipeline FPS | 1 Chunk | 2 Chunks | 3 Chunks | 4 Chunks | 5 Chunks |
|--------------|---------|----------|----------|----------|----------|
| 6 FPS        | 0.50s   | 1.00s    | 1.50s    | 2.00s    | 2.50s    |
| 10 FPS       | 0.30s   | 0.60s    | 0.90s    | 1.20s    | 1.50s    |
| 15 FPS       | 0.20s   | 0.40s    | 0.60s    | 0.80s    | 1.00s    |
| 20 FPS       | 0.15s   | 0.30s    | 0.45s    | 0.60s    | 0.75s    |
| 24 FPS       | 0.125s  | 0.25s    | 0.375s   | 0.50s    | 0.625s   |

**Formula**: `chunk_duration = (3 frames / FPS) * num_chunks`

## TUI Controls

In `video-cli playlist nav`:

| Key | Action |
|-----|--------|
| `s` | Toggle soft cut mode ON/OFF |
| `h` | Toggle hard cut mode (mutually exclusive) |
| `x` | One-shot hard cut (doesn't change mode) |
| `1` | Set bias = 0.05 (very aggressive) |
| `2` | Set bias = 0.10 (aggressive, default) |
| `3` | Set bias = 0.15 (moderate) |
| `4` | Set bias = 0.20 (conservative) |
| `5` | Set bias = 0.25 (light touch) |
| `!` | Set duration = 1 chunk |
| `@` | Set duration = 2 chunks (default) |
| `#` | Set duration = 3 chunks |
| `$` | Set duration = 4 chunks |
| `%` | Set duration = 5 chunks |

Use `x` when you need a single hard reset without leaving soft cut mode (e.g., scene got stuck, need to break out).

## Bias Reference

| Value | Key | Cache Reliance | Behavior |
|-------|-----|----------------|----------|
| 0.05  | `1` | Minimal | Scene very malleable, fastest adaptation, higher artifact risk |
| 0.10  | `2` | Low | Fast adaptation, good for distinct scene changes |
| 0.15  | `3` | Reduced | Balanced - noticeable speedup with stability |
| 0.20  | `4` | Moderate | Gentle speedup, maintains more continuity |
| 0.25  | `5` | Slightly reduced | Subtle nudge, nearly normal behavior |
| 0.30  | — | Normal | Default operation (no soft cut) |

## Comparison: No Cut vs Soft Cut vs Hard Cut

| Mode | What Happens | Use When |
|------|--------------|----------|
| **No cut** | Bias stays 0.3, scene morphs slowly | Smooth continuous flow, same scene |
| **Soft cut** | Bias drops temporarily, faster morph | Scene changes within same "world" |
| **Hard cut** | Cache reset, clean break | Completely different scene/setting |

## Tuning Strategy

### Step 1: Establish baseline
Run a few transitions with soft cut OFF to see how long morphs take normally.

### Step 2: Start with defaults
Enable soft cut (`s`), use bias=0.10, chunks=2.

### Step 3: Adjust based on results

| Problem | Try |
|---------|-----|
| Transition still too slow | Lower bias (`1`) or more chunks (`#`, `$`) |
| Artifacts during transition | Higher bias (`3`, `4`) or fewer chunks (`!`) |
| Scene doesn't fully settle | More chunks (`#`, `$`, `%`) |
| Transition feels abrupt | Higher bias (`3`, `4`, `5`) |

### Step 4: Consider FPS
At lower FPS (6-10), each chunk is longer, so fewer chunks may suffice.
At higher FPS (20+), you may need more chunks for the same effect.

## Recommended Starting Points

| Scenario | Bias | Chunks | Notes |
|----------|------|--------|-------|
| Similar scenes, fast FPS | 0.10 | 2 | Quick, responsive |
| Similar scenes, slow FPS | 0.15 | 2 | Slightly gentler |
| Different scenes, want continuity | 0.10 | 3-4 | More settling time |
| Subtle prompt tweaks | 0.20 | 2 | Light touch |
| Testing/exploration | 0.15 | 3 | Good middle ground |

## API Reference

```bash
# Standalone soft cut
curl -X POST "http://localhost:8000/api/v1/realtime/soft-cut" \
  -H "Content-Type: application/json" \
  -d '{"temp_bias": 0.15, "num_chunks": 3, "prompt": "optional new prompt"}'

# With playlist navigation
curl -X POST "http://localhost:8000/api/v1/realtime/playlist/next?soft_cut=true&soft_cut_bias=0.15&soft_cut_chunks=3"
```

## Logs

Watch soft cut activity:
```bash
tail -f ~/.daydream-scope/logs/scope-logs-*.log | grep -i "soft transition"
```

You'll see:
- `Soft transition: bias -> X for N chunks (will restore to Y)` - started
- `Soft transition complete: restored bias to Y` - finished
- `Soft transition canceled: explicit kv_cache_attention_bias update received` - overridden

## Embedding Transition (LERP/SLERP)

Separate from KV cache bias, there's also **embedding interpolation** (`t` key in TUI). This smoothly morphs the prompt embedding from old→new over N chunks.

Note: embedding transitions rely on continuity. If you do a **hard cut** (cache reset), the transition is effectively suppressed (there’s no “source embedding” to interpolate from). Use **soft cut + transition** for the “in-between” mode.

### How It Works

```
Prompt A embedding ─────────────────────────────────────────────────▶
                    │
                    │ Navigate to Prompt B (with transition=true)
                    │
                    │         interpolation over N chunks
                    │    ┌──────────────────────────────────┐
                    ▼    │  A      A→B    A→B    A→B    B   │
Prompt B embedding ◀────┴──────────────────────────────────┴────────
                         chunk 1  chunk 2 chunk 3 chunk 4
```

The embedding gradually shifts from A to B, causing the scene to smoothly morph.

### Interpolation Methods

| Method | Key | Description |
|--------|-----|-------------|
| **Linear (LERP)** | default | Straight-line path: `(1-t) * A + t * B` |
| **Spherical (SLERP)** | `T` toggle | Arc path on unit sphere, preserves angular relationships |

**SLERP** often produces smoother semantic transitions but only works with exactly 2 embeddings. Press `T` to toggle between methods.

### TUI Controls

| Key | Action |
|-----|--------|
| `t` | Toggle transition mode ON/OFF |
| `T` | Toggle interpolation method (linear ↔ slerp) |
| `6` | Set transition chunks = 1 |
| `7` | Set transition chunks = 2 |
| `8` | Set transition chunks = 3 |
| `9` | Set transition chunks = 4 (default) |
| `0` | Set transition chunks = 5 |

## Combining Soft Cut + Transition

The **combo effect** layers both mechanisms:

```
           ┌─────────────────────────────────────────────────┐
           │                COMBO EFFECT                     │
           │                                                 │
           │   SOFT CUT: Lower KV bias → scene is "loose"   │
           │   TRANSITION: Embedding morphs A → B            │
           │                                                 │
           │   Together: Scene is receptive while prompt     │
           │             gradually shifts = smooth change    │
           └─────────────────────────────────────────────────┘
```

**Best practice**: Align durations for optimal effect:
- `soft_cut_chunks = 4` + `transition_chunks = 4`
- Soft cut opens the "plasticity window" while transition morphs the embedding

### Recommended Combo Settings

| Scenario | Soft Cut | Transition | Notes |
|----------|----------|------------|-------|
| Fast scene changes | bias=0.10, chunks=2 | chunks=2 | Quick, responsive |
| Smooth morphs | bias=0.15, chunks=4 | chunks=4 | Balanced, cinematic |
| Dramatic shifts | bias=0.05, chunks=3 | chunks=3 | Very malleable |

### API Reference

```bash
# With both soft_cut and transition
curl -X POST "http://localhost:8000/api/v1/realtime/playlist/next?\
soft_cut=true&soft_cut_bias=0.15&soft_cut_chunks=4&\
transition=true&transition_chunks=4&transition_method=linear"
```
