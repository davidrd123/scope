Got it – I’m on it and have updated the task accordingly.


Yeah, that framing is exactly right: if B200 is genuinely doing ~20.4 FPS end to end, but B300 is pinned at ~8.8 FPS regardless of attention backend, then something is different between the two runs that is either:

* Not in attention at all, or
* Not even in “compute” at all (pacing / scheduling / measurement), or
* A B300 specific fallback or throttling mode.

The mistake mode here is “we keep swapping attention kernels, but the thing enforcing the ceiling lives somewhere else”.

## The bigger picture mental model

There are only three places a hard ceiling like this can come from:

1. **The model step is actually ~113 ms on B300** (true compute bound, attention changes do not move it because something else dominates).
2. **The pipeline is pacing output to ~8.8 FPS** (WebRTC track timestamping, queue consumption rate, intentional sleep, backpressure, etc).
3. **The system is time sliced or throttled** (power cap, thermal, multi tenant GPU sharing, MIG/vGPU, or driver bug causing low clocks).

Your “B200 is 20.4 FPS” observation is the key: it strongly suggests (2) is not universally present in the codebase, unless the B200 number was measured in a different mode (offline loop, different endpoint, different FPS definition). So the first thing is to make the B200 vs B300 numbers comparable with one unambiguous measurement.

## Turn this into definite hypotheses with discriminating tests

Below are hypotheses that are genuinely different, plus the single fastest test that would confirm or kill each.

### H1: You are hitting an output pacing limit (not compute)

This is still my top suspicion because “8.8 no matter what” smells like a scheduler, not math.

**Discriminating test (5 minutes): measure GPU time vs wall time per frame.**

Add two timings around the exact same region on B300:

* **GPU elapsed** using `torch.cuda.Event` around the model forward for one frame.
* **Wall elapsed** using `time.perf_counter()` around the entire “produce a frame” path.

If GPU time is, say, 30 to 60 ms but wall is ~113 ms, you are not compute limited. Something is pacing or stalling.

Concrete instrumentation pattern:

* `t_wall0 = perf_counter()`
* record `evt0`
* run model forward
* record `evt1`, `evt1.synchronize()`, `gpu_ms = evt0.elapsed_time(evt1)`
* `t_wall1 = perf_counter()`
* log: `gpu_ms`, `wall_ms = (t_wall1 - t_wall0)*1000`

If `gpu_ms << wall_ms` consistently, you stop thinking about kernels and go find the pacer/backpressure.

**Next isolating test:** bypass WebRTC entirely and run a tight loop that generates N frames and discards them. If that runs faster than 8.8 FPS, the cap is absolutely in streaming, encoding, queueing, or recv/send pacing.

### H2: B300 node is CPU bound (different host, different preprocessing, encode path)

This would also make attention backend irrelevant.

**Discriminating test:** watch GPU utilization and SM clocks while the pipeline runs.

Run:

* `nvidia-smi dmon -s pucvmet -d 1`

What you want to see:

* If GPU util is low or “spiky” with long idle gaps while CPU is busy, that screams CPU bound or synchronization/backpressure.
* If GPU util is high but clocks are low, that screams throttling (H3).

Also quickly compare CPU saturation:

* If one CPU core is pegged and GPU is underutilized, you probably have a single threaded bottleneck in frame processing, tokenization, image conversion, encoder, or Python scheduling.

### H3: Power / thermal / clocks are clamping B300

B300 being pinned to a low performance state can create a consistent ceiling.

**Discriminating test:** capture P state, clocks, and power while running.

* `nvidia-smi -q -d POWER,CLOCK,PERFORMANCE`
* plus the `dmon` line above during inference

If you see something like P2/P3 with unexpectedly low SM clocks under load, or power draw way below expected, that is not a kernel issue.

### H4: The B300 is not actually exclusive (time slicing, MIG, vGPU, other tenants)

Cloud reality check. This can produce stable but lower throughput.

**Discriminating test:**

* `nvidia-smi`

  * Look for other processes.
  * Look for MIG mode / partitioning indicators.
* `nvidia-smi -L`

  * See whether you are on a MIG slice or full device.

If you are not alone on the GPU, 8.8 FPS “no matter what” becomes very plausible.

### H5: On B300, some big non-attention op is falling back to a terrible kernel

This is the “SM103 support gap” possibility: cuBLASLt, cuDNN, or some fused op taking a slow path on SM103, so attention swaps do not matter.

**Discriminating test:** a profile that tells you the top 3 GPU kernels by time on B300, and whether they are GEMMs or something else.

* With Nsight Systems, you want to see:

  * Is the GPU busy almost all the time?
  * Which kernel names dominate?

If the dominant kernels are GEMM and are slower than expected, you then benchmark just those GEMMs in isolation with cublasLt or a tiny torch matmul microbench on B300.

## Why I think the “pacing cap” hypothesis is worth following up

Because the exact symptom you described is “changing compute does not change FPS”.

When that happens, the most common root causes in real time pipelines are:

* A track `recv()` loop that sleeps to maintain a target PTS cadence
* A queue that is consumed at a fixed rate (or a rate derived from client demand)
* Backpressure from the encoder or network
* Some “max fps” setting being applied in a different module than the one you edited

You already saw a version of this: changing a codec max frame rate did not move the needle. That is a classic sign that the limiting pacer is somewhere else.

If you do just one thing next, do the “GPU ms vs wall ms” logging. It turns this from vibes into physics.

## Using H100 as a decisive fork in the road

Testing on H100 is useful, but only if you use it as a fork with a clear interpretation:

* **If H100 is also ~8.8 FPS in the same end to end path**, then it is almost certainly pacing / scheduling / measurement, not Blackwell at all.
* **If H100 jumps back up near B200 levels**, then the cap is B300 node or SM103 software stack specific (driver, clocks, tenancy, kernel fallback).

So the H100 test is not just “more data”, it is a strong discriminator if you keep the rest of the pipeline identical.

## A concrete “one session” plan that converges fast

1. On B300, add the dual timing log: `gpu_ms` for model forward and `wall_ms` for the full frame path.
2. On B300, run `nvidia-smi dmon -s pucvmet -d 1` while generating for 30 to 60 seconds.
3. Run the same exact code path on H100 and compare those two logs.

From those 3, you will almost always know which branch you are on:

* pacing/backpressure,
* CPU bottleneck,
* GPU throttling/tenancy,
* or GPU kernel fallback.

Once you know the branch, the next step becomes obvious instead of speculative.

## One more subtle trap to check

Make sure the B200 “20.4 FPS” and the B300 “8.8 FPS” are the same metric: generated frames per second at the model boundary, or sent frames per second at the WebRTC boundary. It is very easy for one number to be “model throughput” and the other to be “delivered fps”, and then you end up chasing ghosts.

If you want, paste just the code location where FPS is computed (or the log line) and I can tell you what it is actually measuring.

Reference context I am carrying from earlier:
