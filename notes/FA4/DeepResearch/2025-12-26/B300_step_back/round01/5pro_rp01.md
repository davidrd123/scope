I’m going to treat this as: “given these selected B300/FA4 notes, synthesize the *current truth*, call out what’s stale/inconsistent, and propose the next highest‑ROI steps (both for ‘hit 24+ FPS’ and for ‘Level 5/6 learning’).”

## Current truth (as of the docs you pasted)

### 1) There are still two completely different “worlds” on B300

* **Repo-default stack** (older CUDA/cuDNN bundle): **~8.8 FPS @ 320×576** and it’s largely **decode/cuDNN dominated**.
* **SM103-native stack (cu130)** (torch 2.9.0+cu130 + cuDNN 9.13): decode becomes “reasonable”, and the bottleneck shifts to **denoise/transformer**.

This is consistent across:

* `investigation-runbook.md` (decode stack sensitivity, profiling constraints)
* `setup-guide.md` (cu130 decode ~4× faster)
* `session-state.md` (current blessed setup + failure modes)

### 2) The “big” recent win is *not* just FA4 score_mod — it’s the patch-embedding Conv3d→Conv2d fastpath

Your `experiments.md` has a standout “this explains the mysterious copy/fill storm” card:

* **Patch embedding was using a Conv3d path** with `kernel_size=(1,2,2)`, `stride=(1,2,2)` that triggered enormous `aten::copy_` / `aten::fill_` counts.
* Rewriting it as **per-frame Conv2d** when `patch_size[0]==1` dropped:

  * `aten::copy_` / `aten::fill_` from ~**35k** calls → ~**9.6k**
* And it moved end-to-end significantly.

This is the piece that makes some of the earlier “we’re stuck around 15 FPS” narrative now outdated.

### 3) “Best known working” performance numbers (quality-preserving)

From `session-state.md` + `experiments.md` (latest state after patch-embed fastpath):

**B300, cu130 env, `320×576`, steps=4, bias=0.3, `--quantization none` (quality OK):**

* `SCOPE_KV_BIAS_BACKEND=fa4`:

  * **~19.7 FPS** (no compile)
  * **~22.8 FPS** (with `--compile`)
* `SCOPE_KV_BIAS_BACKEND=flash`:

  * **~17.2 FPS** (no compile)
  * **~21.4 FPS** (with `--compile`)

### 4) Quality gate: FP8 is still “perf-only”

Multiple places (especially `session-state.md`) are very explicit:

* **FP8 output quality on B300 is broken** (gray/noise)
* So FP8 numbers are breadcrumbs for upstream/tooling investigation, **not shippable wins**
* Treat BF16 (`--quantization none`) as the canonical baseline for real output

That constraint should be reflected everywhere you talk about “target FPS”.

---

## What’s stale / inconsistent across the selected docs

### `optimization-vision.md` is materially out of date

It’s anchored at **15 FPS** as “current” and frames “attention is the main remaining bottleneck”.

That *was true* before:

* patch-embed Conv2d fastpath
* follow-on glue cleanup (Upsample BF16 path, RMSNorm fused)

But now you have:

* **~19.7 FPS (no compile)** and **~22.8 FPS (compile)** on BF16
* Which puts you within ~5% of the “24+ FPS” target already (at least in the benchmark harness)

So the “strategic options” section should be updated to include:

* **“Option A2: eliminate hidden slow-path ops (Conv3d patch embed → Conv2d)”** as a proven lever
* A reframed bottleneck statement: *not just attention*, but “copies/fills/layout churn + projections + remaining convs”

### `optimization-ladder.md` is directionally right but the “YOU ARE HERE” numbers lag

It says:

* “B300: ~15 FPS typical cu130; ~22–23 FPS with `--compile`”
  That “~22–23 compile” is still consistent, but the baseline **~15** is now superseded by **~19–20** BF16 after the patch-embed fix.

Also: the ladder’s Level 5 framing (“RoPE is ~0.11ms/call; fusion is learning, not huge FPS”) is still good, but the “next wins” section should now explicitly mention:

> we got a *Level 4.5–5 style* win by eliminating glue/slowpaths (patch embed), not by deeper attention kernel work.

### `investigation-runbook.md` should probably absorb one key new “known B300 trap”

The runbook already says “profile; identify decode vs denoise; avoid backend fallbacks.”

Given the experiments log, you now have a concrete additional trap that’s worth elevating to “ground truth”:

* **Large `aten::copy_` / `aten::fill_` counts can come from a single pathological op (Conv3d patch embedding)**, and fixing that can dwarf attention tweaks.

This is exactly the sort of “prevent reruns of the same mystery” guidance the runbook is for.

---

## Repro: the current “known good” BF16 benchmark command (copy/paste)

This is essentially the `session-state.md` recipe, aligned with the new best-known perf:

```bash
SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark
```

And to test the compile win:

```bash
# same as above, add:
  --compile
```

---

## Highest-ROI next steps to reach 24+ FPS (quality-preserving)

You’re at ~22.8 FPS with compile on BF16; you need ~**+5–6%**.

Here are the moves that fit your “one-change card” discipline and don’t require deep kernel R&D.

### 1) Take the “small, likely” win: enable FA4 varlen for non-bias attention (A/B it)

You already have this documented as a ~1–2% gain in some cases, at the cost of warmup/JIT.

Experiment:

* Baseline: `SCOPE_ENABLE_FA4_VARLEN=0` (default)
* Variant: `SCOPE_ENABLE_FA4_VARLEN=1`

Command:

```bash
SCOPE_ENABLE_FA4_VARLEN=1 \
SCOPE_KV_BIAS_BACKEND=fa4 \
TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
DISABLE_FLEX_ATTENTION_COMPILE=1 \
WANVAE_STREAM_DECODE_MODE=chunk \
.venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --quantization none \
  --kv-cache-attention-bias 0.3 \
  --cudnn-benchmark \
  --compile
```

If it’s truly ~1–2%, it won’t get you to 24 alone, but it’s “cheap”.

### 2) Keep shaving the remaining “copy/to/fill” hotspots using the stack-attributed op profiler

You already demonstrated that:

* **one** structural change (patch embed) removed ~25k copy/fill calls and gave a real FPS jump.

The best next bet is to repeat that exact process:

1. Run:

   ```bash
   DISABLE_FLEX_ATTENTION_COMPILE=1 \
   WANVAE_STREAM_DECODE_MODE=chunk \
   TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \
   SCOPE_KV_BIAS_BACKEND=fa4 \
   .venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_ops.py \
     --height 320 --width 576 \
     --iters 1 --pre-iters 1 \
     --kv-cache-attention-bias 0.3 \
     --kv-bias-backend fa4 \
     --quantization none \
     --cudnn-benchmark \
     --with-stack --stack-n 12
   ```
2. Identify the **top 1–2 stack groups** behind `aten::copy_`, `aten::_to_copy`, `aten::fill_`, and “tiny elementwise storms”.
3. For each, try the same pattern you already used successfully:

   * replace dtype roundtrips (`x.float().type_as(x)`) with fused ops (`F.rms_norm`, fused layernorm variants)
   * remove redundant layout conversions
   * introduce small env-var toggles so A/B is safe (`SCOPE_*_IMPL=legacy/auto`)

This is the most credible way to find another ~5% without heroic changes.

### 3) Be strict about compile mode on SM103

Per your experiments:

* `reduce-overhead` is unstable (“output overwritten”)
* some “max-autotune…” settings can hard-abort via tcgen05 LLVM intrinsic errors

So: for the “get to 24+ quickly” track, keep:

* default compile mode (the one already producing ~22.8)
* avoid chasing cudagraph modes until you have a targeted workaround

### 4) Optional “bigger swing” with low code churn: try cuDNN SDPA backend for *plain* attention

Your older vision doc lists this as Option C.

Given you’re now in a regime where:

* attention is still large, but not the only lever
* the stack is already cu130 / cuDNN 9.13

…it’s worth a controlled A/B on the *plain attention path* (self/cross that aren’t KV-bias score_mod), **if** your backend selection code supports it cleanly.

If you don’t already have a knob, the minimal safe approach is:

* introduce a single env-var-driven switch in `wan2_1/modules/attention.py` (or wherever non-bias attention is chosen)
* measure with `PROFILE_ATTENTION=1` to confirm it actually hit cuDNN kernels (and didn’t silently fall back)

I’m not recommending any specific env var name here because your repo clearly already has a backend-selection layer; the important part is: **make it explicit and logged** (like you did for KV-bias).

---

## If the intent is “Level 5 (RoPE+attention fusion) learning”: what’s the cleanest first milestone?

Your ladder/resources already nail the key technical point:

* **score_mod can’t implement RoPE** (score_mod changes scores after QKᵀ; RoPE rotates Q/K before dot-product).
* So “fuse RoPE” means **Q/K preprocessing during load/prologue** in FA4/CuTe.

A pragmatic first milestone that keeps risk contained:

1. **Add a Q/K preprocessing hook in the vendored CuTe path**

* Something like `q_mod`/`k_mod` callbacks invoked at (or immediately after) tile load into registers/shared.
* Start with a tiny, obviously-detectable transform (e.g., scale Q by a constant) to validate the hook plumbing and verify correctness tests catch it.

2. **Only then implement actual RoPE**

* Pass cos/sin (or packed tables) into the kernel via the same mechanism you use for score_mod aux tensors.
* Apply rotation on Q/K fragments before MMA.

3. **Measure what moved**

* Expect RoPE itself to be small (your own docs say ~0.11ms/call in at least one drilldown), so the point is:

  * reduce launches
  * reduce intermediate writes/reads
  * (maybe) enable better layout for downstream ops

This is a solid “Level 5 learning” project, but it is not the shortest path to 24 FPS given you’re already near 24 via compile + glue cleanup.

---

## Suggested doc maintenance (so future-you doesn’t get misled)

If you want the docs to stay coherent with the *actual* latest state, the minimum edits I’d do:

1. **Update `optimization-vision.md`**

* Add the patch-embed Conv2d milestone to the progress table
* Update “current” baseline from 15 FPS → **~19–20 FPS BF16** (and **~22–23 FPS with compile**)
* Reframe bottleneck as: “remaining time is a mix of attention + projections + conversion/memory traffic; fix pathological ops first”

2. **Update the “YOU ARE HERE” section in `optimization-ladder.md`**

* Replace the ~15 FPS narrative with the new BF16 numbers and call out the patch-embed fix as an example of Level 5-ish “fusion/glue elimination”

3. **Add one paragraph to `investigation-runbook.md`**

* Under “Ground Truth”: explicitly mention the patch-embed discovery as a canonical example of why stack-attributed op profiling matters (copy/fill storms can be a single op).

---

If you tell me what the *actual output you want* is (e.g., “write the patch text for updating optimization-vision + ladder”, or “turn this into a one-page session handoff”), I can produce that directly.
