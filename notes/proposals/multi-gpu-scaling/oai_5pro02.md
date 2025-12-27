````diff
diff --git a/notes/proposals/multi-gpu-scaling.md b/notes/proposals/multi-gpu-scaling.md
index 1111111..2222222 100644
--- a/notes/proposals/multi-gpu-scaling.md
+++ b/notes/proposals/multi-gpu-scaling.md
@@ -1,28 +1,91 @@
 # Multi-GPU Scaling for KREA Realtime
 
 > Status: Exploratory / Research
 > Date: 2025-12-27
 > Confidence: Low (needs investigation)
 
 ## Summary
 
-Enable multi-GPU inference for the KREA realtime pipeline to scale FPS beyond single-GPU limits.
+Enable multi-GPU inference for the KREA realtime pipeline to scale **single-stream FPS** beyond single-GPU limits (while controlling added latency + jitter).
 
 Key framing: **multi-GPU only helps if we can overlap work** (or otherwise avoid turning “more devices” into “more copies + more latency”).
 
 Before diving in, clarify which “scaling” you mean:
 - **Scale a single stream’s FPS** (hard; needs overlap + careful orchestration).
 - **Scale total throughput across streams/users** (easy: run independent sessions pinned to GPUs; little/no model partitioning).
 
 This proposal is primarily about **single-stream FPS**, but many of the measurements apply to throughput scaling too.
 
 StreamDiffusionV2 reports multi-GPU viability (including **~58 FPS with a 14B model on 4× H100**) using pipeline-parallel style orchestration, but we should treat those numbers as *reported* until we reproduce comparable behavior in our stack.
+
+## How multi-GPU can (and cannot) increase single-stream FPS
+
+The only “real” ways multiple GPUs make a **single** stream faster:
+
+1) **Overlap independent stages** (e.g., decode on GPU1 while denoise runs on GPU0).  
+2) **Pipeline-parallel the transformer** (GPU0 runs early blocks while GPU1 runs later blocks, overlapped across frames/steps).  
+
+Everything else tends to become:
+- **sequential offload** (no overlap → often neutral/negative), or
+- **comm-bound parallelism** (big activations → PCIe/NVLink becomes the bottleneck), or
+- **jitter amplification** (more synchronization points, more tail latency).
+
+### Concrete bottleneck snapshot (B300, “current best” @ 320×576)
+
+From `notes/FA4/b300/session-state.md`, the best-known BF16 + `--compile` config is ~**30.7 FPS** and the block shares look like:
+
+| Block | Per-call | Share | Implication for multi-GPU |
+|------:|---------:|------:|---------------------------|
+| `denoise` | ~267 ms | ~69% | Primary headroom target; multi-GPU must touch this to move FPS meaningfully |
+| `recompute_kv_cache` | ~62 ms | ~16% | Matters more as denoise improves; likely “moves with transformer” |
+| `decode` | ~60 ms | ~15% | Offload only helps if we can **overlap**, and the ceiling is small at this share |
+
+**Important:** decode share is configuration-dependent. Older/other profiles (different stacks, resolutions, or pre-fix states) show decode much larger (e.g., ~25–35%+). So **VAE offload viability must be decided from the block profile at the target settings**, not from a single historical number.
+
+### Profiling caveat (critical for overlap work)
+
+Our block profiler uses CUDA events + `synchronize()` per block (see runbook). This is great for *breakdowns* but it **prevents async overlap**.  
+For multi-GPU overlap experiments, prefer:
+- `nsys` (if accessible), or
+- custom “overlap-aware” timing (events without per-block synchronize, plus end-of-iteration synchronize), or
+- a microbench harness designed to measure **wall time vs per-device time**.
 
 ## What We Know
 
@@ -45,6 +108,7 @@ This proposal is primarily about **single-stream FPS**, but many of the measurements apply to throughput scaling too.
 
 ### StreamDiffusionV2 Approach ([paper](https://arxiv.org/abs/2511.07399))
 
 | Strategy | Feasibility | Notes |
 |----------|-------------|-------|
 | **Pipeline Parallel** | ✅ Proven | Consecutive DiT stages across GPUs |
 | **Tensor Parallel** | ⚠️ Likely poor fit | Their claim: spatial activations make comm prohibitive |
 | **Sequence Parallel** | ⚠️ Risky | Their claim: unpredictable latency jitter |
@@ -66,6 +130,12 @@ This proposal is primarily about **single-stream FPS**, but many of the measurements apply to throughput scaling too.
 
 Note: their distributed setup uses `torchrun --nproc_per_node=N`, but our **first prototype** (VAE offload) can likely be done in a single process without `torch.distributed`.
 
+Single-stream takeaway (from the long-form research draft):
+- **Pipeline parallel is the “real” scaling lever** for single-stream FPS when the transformer dominates.
+- **Tensor parallel** is usually a bad trade for diffusion/DiT at low batch because the activation traffic is huge.
+- “Temporal/sequence parallel” tends to show up as **latency jitter** unless carefully scheduled.
+
 ### Current KREA Pipeline State
 
 The pipeline assumes single device:
 ```python
@@ -76,20 +146,55 @@ The pipeline assumes single device:
 
 No distributed primitives currently wired.
 
 ### Current Single-GPU Baseline (B300, 2025-12-27)
 
 The B300 baseline moved quickly as we fixed decode slow-path issues. Avoid hard-coding numbers here; treat the B300 “truth” as:
 - `notes/FA4/b300/session-state.md` (best-known config + block shares + caveats)
 
 As of 2025-12-27, the best-known **quality-preserving** config on B300 is **~30+ FPS** (BF16 + `--compile`) and the block profile looks like:
 - `denoise` dominant
 - `recompute_kv_cache` non-trivial
 - `decode` no longer dominant at the canonical `320x576` resolution (≈ “solved”)
 
 This matters for multi-GPU: **VAE offload only pays if decode is a large share at the target resolution/settings.**
+
+#### Measurement methodology to reuse (don’t reinvent it)
+
+The “how we measure” ground truth and known profiler constraints live in:
+- `notes/FA4/b300/investigation-runbook.md` (baseline FPS, block profile harness, CUPTI/torch.profiler limitations, op-level profiling guidance, code map)
+
+At minimum, every multi-GPU experiment should carry:
+- (A) **block profile** at the target settings (for share + Amdahl sanity),
+- (B) a **copy microbench** for the tensors you’ll move,
+- (C) an **overlap microbench** that proves real concurrency.
+
+#### “Fix single-GPU traps first” (multi-GPU won’t save you)
+
+From the runbook: if op-level profiles show large `aten::copy_` / `aten::fill_` storms (e.g., Conv3d slow path in patch embedding), treat that as a **stop sign**:
+- fix the pathological op on single GPU,
+- then revisit multi-GPU once the baseline is “clean”.
 
 ### Vendored Resources
 
 We have Blackwell-specific distributed GEMM patterns in `vendored/cutlass-cute/python/CuTeDSL/distributed/`:
 - `distributed_gemm_all_reduce_blackwell.py`
 - `distributed_gemm_reduce_scatter_blackwell.py`
 - `all_reduce_one_shot_lamport.py`
 
 These are kernel-level primitives, not pipeline orchestration.
@@ -128,7 +233,7 @@ These are kernel-level primitives, not pipeline orchestration.
 
 ## Potential Approaches
 
 ### Approach 0: Multi-Session Scaling (Throughput, Not Single-Stream FPS)
 
 Run independent sessions pinned to different GPUs (no model partitioning). This doesn’t make one stream faster, but it’s the lowest-risk way to “scale on multiple GPUs”.
@@ -154,7 +259,8 @@ Run independent sessions pinned to different GPUs (no model partitioning). This doesn’t make one stream faster, but it’s the lowest-risk way to “scale on multiple GPUs”.
 
 ### Approach A: VAE Offload (Simplest)
 
-Move VAE encode/decode to a second GPU while transformer runs on primary.
+Move VAE encode/decode to a second GPU while transformer runs on primary.
+This only improves **single-stream FPS** if we can **overlap** VAE work with transformer work.
 
 **Pros:**
 - Minimal code changes
 - VAE decode can be a large share on some stacks/resolutions (historically ~25–35% on B300 before the decode fast-path fix)
 - No transformer modification
 
 **Cons:**
 - Limited scaling (2 GPUs max benefit)
 - Cross-GPU tensor copies for latents (and possibly decoded frames)
 - **Only a win if we overlap** decode with denoise across chunks (i.e., pipeline stages); sequential offload can be neutral/negative
 - Layout considerations: VAE benefits from `channels_last_3d` (`WANVAE_DECODE_CHANNELS_LAST_3D=1`), need to ensure cross-GPU copies preserve this
+
+**Useful variants (worth testing explicitly):**
+- **Decode-only offload** (keep encode on GPU0): can cut copies in half depending on where latents originate/terminate.
+- **Encode-only offload**: sometimes helps latency/jitter even if decode share is small.
+- **Keep decoded frames on GPU1 longer** if server plumbing allows (avoid “decode → copy back → immediately upload/convert” on GPU0).
+- **Double-buffer queue**: GPU0 produces latents for N while GPU1 decodes N-1 (the minimum pipelining structure that can pay off).
 
 **Upper bound intuition (if perfectly overlapped, ignoring copy overhead):**
 - Speedup is capped by the larger stage share: `speedup <= 1 / max(denoise_share, decode_share)`
 - On the current best-known B300 config, decode is no longer dominant at `320x576`, so the theoretical bound is much smaller (and copy/jitter can erase it).
 
 **Complexity:** Low
 
 ### Approach B: Pipeline Parallel Transformer (StreamDiffusionV2 style)
 
 Split transformer blocks across GPUs. GPU 0 runs blocks 0-N/2, GPU 1 runs N/2-N.
 
 **Pros:**
 - Proven by StreamDiffusionV2
 - Near-linear scaling claimed
 
 **Cons:**
 - Adds per-frame latency (pipeline fill time)
 - KV cache needs distribution strategy
 - More complex orchestration
+
+**Single-stream reality check (why this is hard):**
+- A naive “split layers across devices, run sequentially” does **not** increase FPS.
+- You only win if you can keep the pipeline full: while stage 2 works on frame *t*, stage 1 works on frame *t+1* (or on a different denoise step), which implies **multiple in-flight frames/steps** → **added latency** and **buffering**.
+
+**What to measure early (before building a big prototype):**
+- Activation tensor size at the stage boundary and **send/recv cost** (PCIe vs NVLink matters).
+- Whether the boundary sits at a point with “nice” layout (avoid expensive reformat/copies on each boundary).
+- Whether we can keep **KV cache local to each stage** (ideal) with only hidden states crossing GPUs.
 
 **Complexity:** Medium-High
 
 ### Approach C: Temporal Parallelism
 
 Different GPUs work on different frames/chunks in the stream.
@@ -203,6 +330,10 @@ Different GPUs work on different frames/chunks in the stream.
 
 **Complexity:** High (unexplored)
 
 ### Approach D: DistriFusion-style Patch Parallel
 
 Split spatial patches across GPUs ([DistriFusion paper](https://arxiv.org/abs/2402.19481)).
@@ -220,11 +351,26 @@ Split spatial patches across GPUs ([DistriFusion paper](https://arxiv.org/abs/2402.19481)).
 
 **Complexity:** High
 
 ## Prototype Prereqs / “Measure Before Building” (Especially for VAE Offload)
 
 Before writing orchestration code, measure the two things that decide if Approach A is viable:
 
 1) **Peer-to-peer feasibility and copy cost**
 - Does `cuda:0 → cuda:1` support P2P? (NVLink vs PCIe matters a lot.)
 - What’s the measured time to copy the relevant latent tensors between devices?
 
 2) **Overlap feasibility**
 - Can decode run concurrently with denoise without fighting for CPU scheduling / streams?
 - Does the server architecture make it easy to pipeline across chunks (queue latents, decode previous while denoising next)?
+
+3) **Baseline sanity (block shares + “no pathological ops”)**
+- Re-run the block profile at the *target* resolution/settings and compute the shares.
+- If the op-level profile indicates a copy/fill storm (runbook), fix that first.
+
+4) **Instrumentation sanity (don’t accidentally measure “forced sync”)**
+- Block profiling uses `synchronize()` per block; do not use it to claim “overlap works”.
+- For overlap proofs: wall-time measurement + per-device events + minimal synchronization.
 
 ## Decision Gates (When to Stop vs Build)
 
 Multi-GPU work can balloon quickly. These gates prevent “endless exploration”:
@@ -233,19 +379,56 @@ Multi-GPU work can balloon quickly. These gates prevent “endless exploration”:
 ### Gate 0: Define the objective
 
 Pick one:
 - **Lower latency** (time-to-first-frame / time-to-new-prompt)
 - **Higher steady-state FPS**
 - **Higher total throughput (multi-session)**
 
 Each objective implies different architecture choices and success metrics.
 
+### Gate 0.5: Reconfirm the target bottleneck (at target settings)
+
+If the current block shares don’t show a “movable” stage:
+- VAE offload won’t help if `decode` is ~10–15% and cannot be overlapped cleanly.
+- Pipeline parallel transformer won’t help if you’re actually dominated by decode or a pathological copy/fill storm.
+
+Use the runbook harness for a fast, repeatable breakdown.
+
 ### Gate 1: Amdahl sanity check (is there even headroom?)
 
 If a candidate stage is only `p` share of steady-state time, the absolute best-case speedup from “perfectly overlapping” it is:
 
 `speedup <= 1 / (1 - p)`
 
 Example: if decode is 15% share, perfect overlap caps at ~1.18× *before* copy overhead/jitter.
 
 If the theoretical bound is small, don’t build it unless it also improves latency/UX.
 
 ### Gate 2: Measured P2P + overlap
 
 Proceed only if:
 - P2P is available (or copy time is still comfortably below the saved compute time), and
 - you can demonstrate *real overlap* with a microbenchmark (not just “two devices exist”).
 
+### Gate 2.5: Transformer pipeline comm sanity (for Approach B)
+
+Before committing to pipeline parallel:
+- Measure the stage-boundary activation bytes and the transfer time.
+- If transfer time is comparable to per-stage compute time, you’re headed for a comm-bound design (especially on PCIe).
+
 ### Gate 3: End-to-end win at equal quality
 
 Only “count” wins that:
 - preserve output quality (BF16 baseline), and
 - improve an end-to-end metric (steady FPS or latency), not just a microbench.
+
+### Gate 4: Jitter + tail latency budget
+
+Single-stream “feels realtime” only if tail latency stays bounded.
+Proceed only if sustained runs show:
+- no periodic stalls (GC, allocator churn, CPU contention),
+- stable memory usage,
+- acceptable “prompt change / transition” jitter (playlist events are a stressor).
 
 ## Suggested Investigation Path
 
 ### Phase 1: Measure (Before Building)
 
 1. Profile single-GPU to understand where time goes at higher resolutions
 2. Identify natural partition points (VAE boundary, transformer block boundaries)
 3. Estimate communication costs for candidate splits
+
+Concrete measurement checklist (reuse the runbook discipline):
+- Capture a card in `notes/FA4/b300/experiments.md` (hypothesis → command → result → lesson).
+- For each target setting: record (a) FPS, (b) block shares, (c) top op stacks if something looks wrong.
+- If `torch.profiler` is broken due to CUPTI constraints (runbook), prefer: block events + `nsys` + purpose-built microbenches.
 
 ### Phase 2: Reference Study
 
 1. Read StreamDiffusionV2 multi-GPU implementation
 2. Understand their block scheduler
 3. Check if their approach transfers to our architecture
+
+Specific “reference study” questions to answer (actionable):
+- Where do they cut the DiT? (block index, attention boundary, MLP boundary?)
+- What tensor is sent between stages (shape, dtype, bytes)?
+- How do they keep the pipeline full for a *single stream* (how many in-flight frames/steps)?
+- How do they handle the rolling KV cache across pipeline stages?
+- What knobs exist for static vs dynamic scheduling (e.g., `--schedule_block`) and what metrics drive it?
 
 ### Phase 3: Prototype (Simplest First)
 
 1. **VAE decode offload** - run decode on GPU 1 *with pipelining* (decode chunk N-1 while denoise chunk N), measure impact
 2. If beneficial, try encode offload too (or keep encode on GPU 0 if it’s not dominant)
 3. Only then consider transformer partitioning
+
+Recommended ordering within Phase 3 (to reduce false starts):
+1) VAE **decode-only** offload (smallest set of copies)
+2) Full VAE offload (encode+decode) if decode-only wins
+3) Transformer pipeline parallel **2-stage static split** microbench (no scheduler)
+4) Transformer PP integrated into the real loop (measure latency + steady FPS + jitter)
 
 ## Testing Program (Concrete)
 
 Treat this as an experiment ladder: each stage produces artifacts (logs, profiles) and has a pass/fail.
 
+### Test 0: Reconfirm baseline and shares (target settings)
+
+Artifacts:
+- Block profile JSON + the summary table in `notes/FA4/b300/session-state.md` (or a new dated entry)
+- FPS number (steady-state) for the exact settings you’ll target in multi-GPU
+
+Pass:
+- You can state “what stage we’re trying to overlap/parallelize” with a measured share.
+
+### Test 0b: “No pathological ops” sanity (runbook trap)
+
+Artifacts:
+- `scripts/profile_krea_pipeline_ops.py --with-stack --summary` output (or equivalent summary)
+
+Pass:
+- No obvious copy/fill storm (or you intentionally fixed it first).
+
 ### Test 1: Hardware topology + P2P
 
 Artifacts:
 - `nvidia-smi topo -m` output (NVLink vs PCIe)
 - `torch.cuda.can_access_peer(0, 1)` (and enabling peer access if needed)
@@ -265,6 +448,11 @@ Pass:
 - P2P is supported and enabled for the target device pair.
 
 ### Test 2: Copy microbench (latents and/or frames)
 
 Measure end-to-end copy time for the *actual tensors you’d move*:
 - latents: GPU0 → GPU1 (decode input)
 - frames: GPU1 → GPU0 (if required by the server plumbing)
 
 Pass:
 - Copy time is comfortably below the compute you’re trying to overlap (otherwise you just moved the bottleneck).
 
 ### Test 3: Overlap microbench
 
 Goal: prove we can overlap “denoise-like work” on GPU0 with “decode-like work” on GPU1.
@@ -278,6 +466,38 @@ Pass:
 - Wall time is close to `max(t_denoise, t_decode)` (not `t_denoise + t_decode`).
 
+### Test 3b: Overlap with the *real modules* (not just synthetic kernels)
+
+Goal: ensure overlap survives real-world factors: Python scheduling, allocator behavior, stream usage, layout conversions.
+
+Artifacts:
+- A small harness that runs a representative `denoise` call on GPU0 while running real `vae.decode` on GPU1
+- Wall time + per-device event times + CPU utilization
+
+Pass:
+- Sustained overlap for many iterations (not just one lucky run).
+
+### Test 6: Transformer pipeline-parallel comm microbench (Approach B feasibility)
+
+Goal: quantify the “activation handoff” cost at a candidate cut point.
+
+Artifacts:
+- Activation tensor shape/dtype/bytes at the boundary
+- Measured send/recv time (or `.to(other_device)` time) under realistic conditions
+
+Pass:
+- Transfer time is “small enough” relative to per-stage compute to plausibly scale.
+
 ### Test 4: End-to-end offload prototype (Approach A)
 
 Minimal implementation idea:
 - GPU0 produces latents for chunk N and enqueues them
 - A background worker on GPU1 decodes chunk N-1 concurrently
 - The main loop consumes decoded frames and continues streaming
 
 Pass:
 - Measured end-to-end improvement on the canonical harness (and a “looks correct for 1–2 minutes” quality sanity check).
 
+### Test 7: End-to-end transformer pipeline parallel prototype (Approach B, 2-stage static split)
+
+Minimal implementation idea:
+- Two ranks/processes (or two modules in one process if feasible) where each rank owns a contiguous block range.
+- Rank 0 runs early blocks, sends hidden state to rank 1; rank 1 runs late blocks, returns output.
+- Pipeline across frames/steps so both stages are concurrently busy.
+
+Pass:
+- Steady-state FPS improves at equal quality, and added latency is within an acceptable budget.
+
 ### Test 5: Stress + jitter
 
 Measure jitter under:
 - playlist nav events (hard/soft cuts, transitions)
 - variable prompt lengths
 - sustained run (minutes)
@@ -288,11 +508,25 @@ Pass:
 - No periodic stalls, no obvious runaway latency, and stable memory usage.
 
 ### Phase 4: Production Hardening (If Warranted)
 
 1. Dynamic load balancing
 2. Graceful degradation to single-GPU
 3. Heterogeneous GPU support
+
+Additions for “real product” readiness:
+- Partition presets (2-GPU, 4-GPU) with known-good cut points
+- Observability: per-stage time, queue depths, dropped frames, P95/P99 latency
+- Clear runtime toggles: disable PP, disable VAE offload, disable compile if it breaks multi-GPU
+- Per-stage `torch.compile` evaluation (compile each partition independently) and fallback path
 
 ## Open Questions
 
 - Does our streaming KV cache complicate pipeline parallelism?
 - What's the minimum GPU memory per device for useful partitioning?
 - Does NVLink vs PCIe matter significantly?
 - How does `torch.compile` interact with multi-GPU?
   - *Partial answer:* `max-autotune` modes can hard-abort on SM103 with Triton <3.5.1 (tcgen05 LLVM intrinsic). Guards exist in pipeline.py. Multi-GPU would need similar guards per device.
+
+- What is our **latency budget** for “pipeline fill”? (Approach B will trade throughput for added lag; the acceptable lag should be explicit.)
 
 ## Non-Goals (For Now)
 
 - Distributed training (inference only)
 - Multi-node scaling (single machine, multiple GPUs)
 - Automatic partitioning (manual configuration is fine initially)
````

**What this patch adds (high level):**

* Pulls in the concrete, current B300 block share snapshot from `session-state.md` and explicitly ties multi-GPU viability (esp. VAE offload) to *measured shares at target settings*.
* Pulls in runbook measurement discipline + known constraints (sync-based block profiler, CUPTI issues, copy/fill storm “stop sign”).
* Adds more **single-stream**-specific guidance for transformer pipeline parallelism (what to measure early, why naive sharding doesn’t help, what “keeping the pipeline full” implies).
* Adds additional experiments + gates (baseline reconfirmation, op-level sanity, comm feasibility test for PP, jitter gate, overlap-with-real-modules test).
* Adds a more explicit staged plan ordering (decode-only offload → full offload → PP microbench → PP integration).

If you want, I can also generate a small “Experiment Card Template” snippet (hypothesis/command/result/lesson) to paste into `notes/FA4/b300/experiments.md`, but I kept this change scoped strictly to `notes/proposals/multi-gpu-scaling.md` as requested.
