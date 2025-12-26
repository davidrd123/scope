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
