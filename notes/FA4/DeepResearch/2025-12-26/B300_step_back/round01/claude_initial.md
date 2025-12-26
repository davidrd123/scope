Based on my analysis of your B300 optimization documentation, here are strategic recommendations organized by category:

---

## Promising Directions to Prioritize

### 1. The Conv2d Patch-Embedding Win Suggests a Pattern Worth Generalizing

Your experiment card showing the Conv3d→Conv2d fastpath for `patch_size=(1,2,2)` delivered a **~3 FPS gain** and slashed `aten::copy_`/`aten::fill_` from ~35k to ~9.6k calls. This is your biggest single-experiment win documented.

**Recommendation:** Systematically audit other Conv3d ops (especially in VAE decode) for the same pattern—temporal kernel size 1 means you can reshape and use Conv2d. The WanVAE decoder is still your second-largest block (~25% of pipeline time on cu130), and it's heavily conv3d. Even one more successful conversion could yield another 1-2 FPS.

### 2. The "flash" Segment-Combine Path is Now Viable—Exploit It as a Stable Fallback

Your fix for FA4 `return_lse` ICE on SM103 (defaulting to FA2 varlen for segment-combine) restored `flash` to **~15-18 FPS** range. This gives you a resilient fallback ladder: `fa4` → `flash` → `flex`.

**Recommendation:** Consider adding a startup banner that logs the resolved backend chain (e.g., "KV-bias: fa4 | non-bias: FA2 varlen | fallback: flex") so debugging sessions never start with "which path am I actually on?"

### 3. Your Best Measured Config is ~23 FPS (compile+qnone+fa4)—Close to Your 24 FPS Target

The numbers show `--compile --quantization none` with `SCOPE_KV_BIAS_BACKEND=fa4` hits **~22.8 FPS**. You're within striking distance of real-time.

**Recommendation:** Before chasing more complex optimizations, spend one experiment card confirming this config end-to-end in Daydream (not just the benchmark script) and document any visual quality delta vs the non-compiled baseline. If it's production-quality, you may have already crossed the finish line.

---

## Gaps in Thinking/Approach

### 1. Missing: Cross-Resolution Scaling Analysis

All documented experiments are at `320x576`. You have no experiment cards testing whether your wins scale proportionally at higher resolutions (e.g., `480x864` or `640x1152`), or whether bottlenecks shift.

**Gap:** Attention complexity grows with sequence length, but your biggest wins so far were in non-attention ops (patch embedding, Conv3d→Conv2d). At higher resolutions, attention might become the limiting factor again. One card at a higher resolution would reveal whether your optimization strategy needs to fork.

### 2. Missing: Warmup Time Budget

You note compile adds "~10-30s warmup" but this isn't systematically tracked as a metric. For a real-time video tool, cold-start time matters.

**Gap:** Consider adding warmup_ms to your experiment card template alongside FPS. If two configs have similar steady-state FPS but 3× different warmup, the user experience differs significantly.

### 3. Under-Explored: cuDNN Attention Backend (Option C)

Your optimization-vision flags this as "Low Effort" but I don't see an experiment card actually testing it. `torch.backends.cudnn.sdp_kernel()` context manager or the `TORCH_CUDNN_SDPA_ENABLED` env var might unlock cuDNN 9.13's Blackwell-optimized attention with minimal code.

**Gap:** This could be a free ~1-2 FPS with a single-line env var test. Worth one experiment card before pursuing higher-effort options.

### 4. The "other_in_self" Mystery Remains Underspecified

Your profiler tells you `other_in_self` (QKV projections, non-bias attention, glue ops) is the majority of self_attn time after FA4 KV-bias, but you don't have a stack-attributed breakdown of *which* ops dominate within that bucket.

**Gap:** Run `scripts/profile_krea_pipeline_ops.py --with-stack` focused specifically on the `self_attn` call to produce a ranked list of "other_in_self" constituents. Without this, you're optimizing blind.

---

## Workflow Enhancements

### 1. Add a "Regression Gate" Command

You have excellent setup scripts but no single-command way to detect regressions. A dedicated `scripts/b300_regression_check.sh` that runs a fixed config and compares FPS against a committed baseline (e.g., `outputs/b300_baseline_qnone_fa4.json`) would catch accidental slowdowns early.

### 2. Consolidate Quality Assessment

FP8 is marked "broken (garbage output)" but I don't see a documented visual comparison methodology. Consider adding a `scripts/capture_quality_snapshot.py` that:
- Runs N frames through a fixed prompt
- Saves PNG frames to `outputs/quality_snapshots/{config_name}/`
- Allows side-by-side comparison

This would let you objectively track whether "FP8 is still broken" after upstream fixes.

### 3. Separate Perf-Only vs Quality-Viable Configs in Session-State

The session-state table mixes FPS numbers from configs that produce usable video with configs that produce garbage. Consider splitting into two tables:
- "Production-Viable Configs"
- "Perf-Only Benchmarks (Quality Unknown/Broken)"

This prevents confusion when someone skims for "what's our best FPS?"

---

## Optimization Targets (Ranked by Expected ROI)

### Tier 1: High Confidence, Measured Path

1. **Validate compile+qnone+fa4 in Daydream end-to-end** (1 experiment card) — you may already be done
2. **Test cuDNN attention backend** (`TORCH_CUDNN_SDPA_ENABLED=1`) (1 experiment card)
3. **Audit VAE decode for additional Conv3d→Conv2d opportunities** (grep for `Conv3d` with `kernel_size=(1,*,*)`)

### Tier 2: Medium Confidence, Requires Investigation

4. **Stack-attribute "other_in_self"** to identify the top 3 ops worth fusing/eliminating
5. **Test at higher resolution** to validate scaling assumptions
6. **Profile VAE decode inner ops** more granularly—your `vae_decode_inner` JSON exists but isn't deeply interpreted in the notes

### Tier 3: Research Bets

7. **ThunderKittens attention** — high potential but requires dependency integration and SM103 validation
8. **Regional torch.compile** (compile transformer blocks only, skip flex_attention) — your notes mention this but no experiment card exists
9. **SageAttention** — flagged as "not installed" but never tested

---

## One Tactical Suggestion

Your blackwell-docs.md has a TODO list of "web search gaps to fill" that would benefit from primary-source grounding. Would you like me to research those gaps (Triton+tcgen05+LLVM fixes, cuDNN 9.13 Blackwell conv3d bullets, torchao torch 2.9 compatibility) to produce source-backed answers you can add to that document?
