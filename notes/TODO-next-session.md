# TODO: Next Session

## FA4 Integration - TESTED ON B300 ✅

### Completed
- ✅ FA4 (CUTE) tested and working on B300 (SM103)
- ✅ Python 3.12 required (cutlass-dsl only has cp312 wheels)
- ✅ Updated pyproject.toml with cp312 flash-attn wheels
- ✅ Added `fa4` dependency group with cuda-python and nvidia-cutlass-dsl==4.1.0
- ✅ Created `scripts/patch_cutlass_sm103.py` to patch arch validation for SM103
- ✅ Benchmarked: up to 740 TFLOPS on B300 at 4096 seq length

### SM103 Patching Required
The B300 is SM103, but nvidia-cutlass-dsl==4.1.0 only supports sm_100a/sm_100f.
Run `python scripts/patch_cutlass_sm103.py` after installing fa4 dependencies.

### ⚠️ CRITICAL: nvidia-cutlass-dsl Conflicts with PyTorch Inductor
`SCOPE_KV_BIAS_BACKEND=fa4` uses FA4/CuTe `score_mod` and requires `nvidia-cutlass-dsl`
(top-level `cutlass` module).

**Current status (2025-12-26):** FA4 KV-bias + regional `torch.compile` is working on B300/cu130
by keeping CuTe calls opaque to Dynamo and disabling flex_attention compilation
(see `notes/FA4/b300/session-state.md`).

**Known gotchas:**
- SM103: compiling `flex_attention` can hard-abort (tcgen05 LLVM) → keep `DISABLE_FLEX_ATTENTION_COMPILE=1`
- `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` is unstable on SM103
- Some envs show an ignored atexit error from Inductor’s CUTLASS utils:
  `AttributeError: module 'cutlass' has no attribute 'CACHE_FILE'`

**To uninstall FA4 deps:**
```bash
uv pip uninstall nvidia-cutlass-dsl cuda-python cuda-bindings cuda-pathfinder
```

### Remaining FA4 Work
1. **Robust FA4 Wrapper** - Handle signature mismatches across versions
2. **Runtime Trip Breaker** - Disable FA4 after first failure
3. **First-call JIT latency** - CUTE JIT compiles on first call (~2s), may need warmup for servers
4. **Resolve cutlass module conflict** - Find a way to make FA4 and PyTorch inductor coexist

---

## Higher Quality Pathways (from 5pro_chat_02.md)

### V2V Refine Pass - NOT YET IMPLEMENTED
```
Krea T2V (preview) → Upscale → Krea V2V (strength=0.3, more steps)
```
Would need to add to render_timeline:
- `--input-video` flag for V2V mode
- `--strength` parameter (maps to `noise_scale` in PrepareVideoLatentsBlock)
- Chunk input video to match pipeline requirements

### Other Models to Explore
- **LongLive** (NVlabs) - 1.3B, sequential prompt specialist, 20.7 FPS on H100
- **CausalWan2.2** (FastVideo) - Wan 2.2 autoregressive, preview-grade
- **StreamingT2V** - chunked long-video with transition machinery

---

## Pointers (current state)

- FA4/B300: `notes/FA4/b300/session-state.md` and `notes/FA4/b300/investigation.md`
- Recent changes: `git log --oneline -n 20`
