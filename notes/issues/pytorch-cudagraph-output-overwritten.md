# Issue Note: `torch.compile(mode="reduce-overhead")` CUDAGraph Trees “output overwritten”

**Status:** Upstream behavior/bug (we can usually work around it locally, but it’s not a one-line fix).

## Symptom

At runtime (typically with `torch.compile(mode="reduce-overhead")` which enables CUDAGraph Trees), you can hit:

- `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.`

The message usually suggests:
- clone the tensor **outside** of `torch.compile()`, and/or
- call `torch.compiler.cudagraph_mark_step_begin()` before each model invocation.

## Upstream tracking

- `pytorch/pytorch#158551` (open as of 2025-12-26): “Accessing Overwritten Tensor Output Despite Clone and cudagraph_mark_step_begin…”

## Common gotchas (why suggested fixes “don’t work”)

- **Cloning inside the compiled function doesn’t help.** The clone needs to happen at the Python boundary (i.e., outside the region Inductor captures into the graph).
- **`cudagraph_mark_step_begin()` must be called before *each* compiled invocation**, not once per epoch / once per outer loop.
- The error can appear even in “simple-looking” modules if *any* graph output escapes and is read after a later invocation has reused/overwritten that memory.

## Where it shows up in this repo

- We experimented with calling `torch.compiler.cudagraph_mark_step_begin()` behind `SCOPE_CUDAGRAPH_MARK_STEP_BEGIN=1` (see `notes/FA4/b300/session-state.md`).
- Today, we treat `mode="reduce-overhead"` as **unstable** on B300/SM103 due to this error and stick with the default compile mode.

## Practical mitigations (things we can try in our code)

1. **Call `torch.compiler.cudagraph_mark_step_begin()` at the correct boundary**
   - Ensure it runs in Python *before* invoking the compiled function(s), not inside compiled regions.
2. **Clone “escaping” outputs outside compile**
   - If we store outputs in session state / caches and use them after later compiled calls, clone before storing.
3. **Avoid CUDAGraph Trees for now**
   - Prefer default `torch.compile(...)` mode over `reduce-overhead`.

## Research TODOs (high value)

1. Capture our exact full traceback(s) when this happens and identify which tensor is being accessed after overwrite.
2. Verify our `mark_step_begin` placement is outside compiled regions for the reduce-overhead experiment.
3. If we can’t make it stable with correct marking + external clones, consider writing a minimal repro and reporting upstream with our stack (PyTorch 2.9.x + CUDA 13.0 + SM103).

