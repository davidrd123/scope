# Session State - 2025-12-25

## What We Did This Session

### Commits Made (12 total on feature/stream-recording)
1. FA4 varlen opt-in + vendored CuTe sources
2. torch.compile mode experiments for B300
3. FA4/B300 documentation updates
4. FA4 kernel optimization docs
5. Project documentation (notes index, overview, daydream cohort)
6. VACE architecture explainer guide (`notes/guides/vace-architecture-explainer.md`)
7. Creative workflows concept doc (`notes/concepts/creative-workflows.md`)
8. Consolidated concept docs into `notes/concepts/`
9. **Hard-cut endpoint** (`POST /api/v1/realtime/hard-cut`)
10. **Hard-cut playlist integration** (`?hard_cut=true` on all playlist endpoints)
11. **Hard-cut TUI toggle** (`H` key in `video-cli playlist nav`)
12. Roadmap updates

### Hard Cut Implementation

**Done:**
- `POST /api/v1/realtime/hard-cut` - standalone endpoint
- `?hard_cut=true` parameter on playlist/next, prev, goto, apply
- `H` key toggle in interactive nav mode (`video-cli playlist nav`)
- Debug logging added (not yet committed):
  - `frame_processor.py:1056` - "HARD CUT: reset_cache=True received"
  - `setup_caches.py:137` - "HARD CUT: SetupCachesBlock received init_cache=True"
  - `setup_caches.py:215` - "HARD CUT: Executing cache reset"

**Flow:**
```
API (reset_cache=True)
  → frame_processor.py pops reset_cache, passes as init_cache
  → pipeline._generate() receives init_cache
  → SetupCachesBlock.__call__() checks init_cache
  → If True: reinitialize KV cache, reset current_start_frame=0, clear VAE cache
```

**Testing:**
- Logs are in `~/.daydream-scope/logs/scope-logs-*.log`
- User needs to restart server to pick up debug logging
- Then: `tail -f ~/.daydream-scope/logs/*.log | grep -i "hard cut"`

### Open Question: Intermediate Transition Modes

User asked about something between hard cut and regular morph. Ideas to explore:

1. **KV cache attention bias** (`kv_cache_attention_bias`)
   - Currently 0.3 means less reliance on past frames
   - Could temporarily set to very low (0.1?) for one chunk then restore
   - Wouldn't fully reset but would "loosen" the memory

2. **Partial cache reset**
   - Reset only recent frames in cache, keep older context
   - Would need to modify `initialize_kv_cache` to support partial reset

3. **Soft transition with prompt interpolation**
   - Already exists: `transition` parameter does embedding interpolation
   - Could combine with lower bias for smoother scene changes

4. **VAE cache only reset**
   - Reset VAE decoder cache without touching KV cache
   - Lighter weight than full reset

5. **"Fade through noise"**
   - Temporarily increase noise_scale for a few chunks
   - Scene becomes more malleable, then settles into new prompt

The key insight from `prompt-sequences.md`:
> "Do not hard cut prompts. For the cleanest transitions, do a short transition window where you gradually rewrite from old to new across a few blocks"

So the existing `transition` system might already be the "intermediate" mode - just not exposed well for playlist navigation.

### Files Changed (uncommitted debug logging)
- `src/scope/server/frame_processor.py` - added HARD CUT logging
- `src/scope/core/pipelines/wan2_1/blocks/setup_caches.py` - added HARD CUT logging

### Key Files for VACE-14B (next task)
See `notes/guides/vace-architecture-explainer.md` for full details.

Quick list:
- `src/scope/server/pipeline_artifacts.py` - add 14B artifact
- `src/scope/server/pipeline_manager.py` - 14B path selection, `_configure_vace()` call
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py` - add mixin, `_init_vace()`, fusing order
- `src/scope/core/pipelines/krea_realtime_video/modular_blocks.py` - add VaceEncodingBlock

### Docs Created
- `notes/guides/vace-architecture-explainer.md` - comprehensive VACE guide
- `notes/concepts/creative-workflows.md` - explore/record/playback model
- `notes/concepts/tui-director.md` - TUI design (copied from plans/)
- `notes/concepts/context-editing-spec.md` - full CLI/API spec (copied from research/)
- `notes/concepts/prompt-sequences.md` - prompt patterns (copied from research/)
