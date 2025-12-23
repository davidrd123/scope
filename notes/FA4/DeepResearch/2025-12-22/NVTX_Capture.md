Got it – I’m on it and have updated the context accordingly.


Here’s the most likely explanation and the shortest path to a working capture.

## 1) Why `--capture-range=nvtx` did not trigger (and why no new `.nsys-rep` appeared)

With `--capture-range=nvtx`, Nsight Systems will not start collecting anything until it sees the specific NVTX range begin. If it never sees that trigger, you can end up with little or no collected data and, depending on version and settings, you may not get a new report file at all. Nsight Systems explicitly documents that capture only starts when the start trigger happens.

The main gotcha in your exact setup is this:

### A. Registered-string requirement (the usual culprit)

Nsight Systems’ NVTX capture trigger is optimized to look for NVTX “registered strings” by default. Many Python NVTX helpers (including PyTorch’s `torch.cuda.nvtx.range_push/pop`) commonly emit unregistered string messages.

NVIDIA’s NVTX docs even warn that some tools may require registered strings for start/stop capture. ([NVIDIA GitHub][1])

And the Nsight Systems user guide PDF states the key behavior: by default only messages provided by NVTX registered strings are considered, and you must disable that behavior to match unregistered strings. ([NVIDIA Docs][2])

That is exactly why you can see NVTX “work” in a smoke test but still fail to trigger `--capture-range=nvtx`.

**Fix:** set:

* `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`

HackMD also shows this exact pattern in a working `--capture-range=nvtx` command.

### B. Domain mismatch (second most common)

`--nvtx-capture` can be specified as `range@domain`, `range`, or `range@*`.
If your range is being emitted in some domain you did not expect, matching `@*` avoids surprises.

### C. You are profiling the wrong process (possible with `uv run`)

If `uv run` spawns a child process (instead of `exec`-ing Python in-place), NVTX ranges might be emitted by the child Python process while Nsight is effectively attached to the parent wrapper for capture triggering. Then the trigger never happens.

This is easy to rule out by running `nsys` against `python` directly once.

### D. Range never ends (less likely given your description)

With `--capture-range-end=stop-shutdown`, the session shuts down when the range ends.
If an exception or early return skips the `range_pop()`, Nsight may never see the end condition and behavior gets weird. In your harness, you are already synchronizing, so just ensure the `range_pop()` runs in a `finally:`.

## 2) Minimal robust command lines that should work

### Option A: Most robust, no `uv`, matches unregistered strings, matches any domain

```bash
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile -t cuda,nvtx,osrt \
  --capture-range=nvtx \
  --nvtx-capture='kernel_b_triton_bench@*' \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true \
  -o /root/scope/nsys_kernel_b_nvtx \
  -- python scripts/tune_kernel_b.py --iters 50 --warmup 5
```

Why this is robust:

* Uses the environment variable to allow unregistered-string matching. ([NVIDIA Docs][2])
* Uses `@*` to avoid domain mismatches.
* Forces overwrite so you do not confuse “old report” vs “new report”.
* Profiles `python` directly to remove wrapper process ambiguity.

### Option B: Keep `uv run`, but still robust

```bash
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile -t cuda,nvtx,osrt \
  --capture-range=nvtx \
  --nvtx-capture='kernel_b_triton_bench@*' \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true \
  -o /root/scope/nsys_kernel_b_nvtx \
  -- uv run python scripts/tune_kernel_b.py --iters 50 --warmup 5
```

If Option A works and Option B fails, you have a process-tree / wrapper issue and should either:

* keep profiling `python` directly, or
* make `uv` exec in-place (if possible), or
* use a different runner for profiling sessions.

### Small code-side recommendation (to make capture deterministic)

Put the NVTX around only the section you care about, and guarantee pop:

```python
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("kernel_b_triton_bench")
try:
    # timed benchmark loop
    for _ in range(iters):
        kernel.run(...)
    torch.cuda.synchronize()
finally:
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
```

If you put the NVTX around the entire sweep, Nsight will stop after the first matching range ends (because you asked for stop-shutdown), which is often not what you want.

## 3) Pitfalls with `nsys profile` output and `--capture-range` on Linux

* **No trigger = no capture.** With `--capture-range=nvtx`, collection starts only when the NVTX trigger happens.
  So if the trigger is never matched, you may see “no new report” behavior.
* **`-o` is a base name, not always a full filename.** Nsight will append extensions (commonly `.nsys-rep`, and sometimes intermediate files). So check with:

  ```bash
  ls -lah /root/scope/nsys_kernel_b_nvtx*
  ```
* **Overwrite confusion.** If you re-run the same `-o` without `--force-overwrite=true`, you can end up inspecting an older report and thinking nothing changed. Adding force-overwrite eliminates that ambiguity.
* **Registered strings default.** This is the big one for NVTX-triggered capture. Use `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` when your instrumentation is unregistered strings. ([NVIDIA Docs][2])
* **Domain mismatch.** Use `range@*` until you confirm the domain.
* **Using `stop-shutdown`.** The session will shut down when the range ends.
  If you accidentally pop too early, you will capture almost nothing. For repeated regions, use a repeat mode (see below).

If you want to capture multiple occurrences, use repeat mode:

```bash
--capture-range-end=repeat-shutdown:3
```

The repeat options are documented under `--capture-range-end`.

## 4) Fallback approaches if NVTX capture still refuses to trigger

### Fallback 1: Profile the whole run, but make the run tiny

For microbench kernels, the cleanest fallback is: do 1 config, 10 to 50 iterations, profile everything, no gating.

```bash
nsys profile -t cuda,nvtx,osrt \
  --force-overwrite=true \
  -o /root/scope/nsys_kernel_b_fullrun \
  -- python scripts/tune_kernel_b.py --iters 50 --warmup 5 --top 1
```

Then you do not need capture-range at all, and NVTX is just a label in the timeline.

### Fallback 2: Use `cudaProfilerStart/Stop` instead of NVTX gating

Nsight supports `--capture-range=cudaProfilerApi`.
If you can call CUDA profiler start/stop from Python (via PyTorch or ctypes), that trigger path avoids the NVTX registered-string issue entirely.

### Fallback 3: Forget Nsight Systems for “kernel only”, use Nsight Compute

If your real goal is “Kernel B: why is it slower/faster, are we memory bound, what is occupancy”, Nsight Compute (`ncu`) is often the correct tool. Nsight Systems is best when you need CPU/GPU correlation and launch structure, not deep kernel metrics.

## Quick triage checklist (fastest way to pinpoint which failure mode you hit)

1. **Does NVTX show up at all without capture-range?**
   Run without `--capture-range` and check the timeline for NVTX labels. If you do not see any, you are not actually emitting NVTX in that run or you are not tracing nvtx.

2. **Does capture-range start working when you set `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`?**
   If yes, you just hit the registered-string default. ([NVIDIA Docs][2])

3. **Does capture-range work with `python ...` but not with `uv run python ...`?**
   If yes, it is a wrapper/child-process issue.

If you paste the exact `nsys profile ...` stdout/stderr from a failing run (even just the last 30 lines), I can tell you whether it never triggered capture, failed to write output, or wrote output to a different basename.

[1]: https://nvidia.github.io/NVTX/ "NVTX - NVIDIA Tools Extension Library"
[2]: https://docs.nvidia.com/nsight-systems/pdf/UserGuide.pdf?utm_source=chatgpt.com "User Guide"
