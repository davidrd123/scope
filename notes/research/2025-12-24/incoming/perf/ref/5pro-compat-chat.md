Here are the two missing NVIDIA “primary-source” excerpts, with just enough surrounding context to justify the repo knobs and the FA4 portability story.

## 1) “ptxas first supports sm_103 starting CUDA 12.9 / 13.0”

The clean NVIDIA phrasing is “compiler target support” (they do not separately headline ptxas in the release note bullet), but this is exactly the toolchain boundary that makes older ptxas choke on `sm_103`.

**NVIDIA source: CUDA Toolkit 12.9 Release Notes (CUDA Compiler section)**

Short excerpt:

> “CUDA Toolkit 12.9 adds compiler target support for SM architecture 10.3 (`sm_103`, `sm_103f`, `sm_103a`).” ([NVIDIA Docs][1])

Why this closes your gap:

* Triton and nvcc ultimately depend on a toolchain that recognizes the GPU “name” (the `sm_103*` targets). If the toolkit “adds compiler target support” in 12.9, then a pre-12.9 toolchain (including its ptxas) will not know what `sm_103` is.
* This is the NVIDIA-native justification for your repo behavior: “use ptxas from CUDA 12.9+ (or CUDA 13.x) for SM103”.

If you want to quote the “first supported in 12.9” claim very literally in your brief, you can use exactly that sentence. It is already “first-introduction language” because it says 12.9 “adds” the target.

## 2) PTX vs cubin forward-compat rules, plus compute_*a and what breaks portability

You asked specifically for “compute_*a vs family/PTX compatibility rules” that matter for FA4 portability. The most direct NVIDIA doc for that is the Blackwell Compatibility Guide (it spells out the cubin rules, the PTX forward-compat promise, and the special-case warning for `sm_100a` / `compute_100a`).

**NVIDIA source: Blackwell Compatibility Guide 13.0, “Application Compatibility on Blackwell Architecture”**

Short excerpts (each stands alone in a brief):

> “A cubin … is supported to run on any GPU with the same major revision and same or higher minor revision.” ([NVIDIA Docs][2])

> “Unlike cubin, PTX is forward-compatible.” ([NVIDIA Docs][2])

> “It is recommended that all applications should include PTX … to ensure forward-compatibility.” ([NVIDIA Docs][2])

And the key “compute_*a” portability warning:

> “PTX … using `sm_100a` or `compute_100a` … are not forward or backward compatible.” ([NVIDIA Docs][2])

How to translate that into the exact FA4 portability rule-of-thumb:

* **If you ship only cubins:** you are locked to “same major, same-or-higher minor” and you will fail outright on a new major (and can fail on a lower minor). ([NVIDIA Docs][2])
* **If you include PTX:** you get NVIDIA’s forward-compat path (driver JIT) for future GPUs. ([NVIDIA Docs][2])
* **If you generate “architecture-conditional” PTX (`compute_*a` / `sm_*a`):** NVIDIA explicitly warns it is **not** forward compatible, so it is the wrong sole artifact for portability. ([NVIDIA Docs][2])

That last bullet is the crux for “portable FA4 binaries”: you can use `*a` targets for peak performance, but only if you also include a non-`*a` PTX fallback that the driver can JIT on whatever it lands on.

## 3) The “family” variant (compute_*f) versus the “architecture-conditional” variant (compute_*a)

For the “family” nuance you referenced (compute_*a vs family), NVIDIA lays it out in the CUDA C++ Programming Guide PTX compatibility text.

**NVIDIA source: CUDA C++ Programming Guide, PTX Compatibility**

Short excerpts:

> “PTX compiled for `compute_100` is supported on devices with compute capability 10.0 and later.” ([NVIDIA Docs][3])

> “PTX compiled for `compute_100f` is supported on devices with compute capability 10.0 and 10.3.” ([NVIDIA Docs][3])

> “PTX compiled for `compute_100a` is supported only on devices with compute capability 10.0.” ([NVIDIA Docs][3])

So the portability mapping is:

* **Baseline PTX (`compute_100`)**: forward-compatible to later compute capabilities (this is your safest “runs anywhere newer” artifact). ([NVIDIA Docs][3])
* **Family PTX (`compute_100f`)**: guaranteed across the 10.x family members NVIDIA names (10.0 and 10.3 in the doc), but it is not the same promise as the baseline PTX. ([NVIDIA Docs][3])
* **Architecture-conditional (`compute_100a`)**: explicitly narrow, not a portability artifact. ([NVIDIA Docs][3])

## What I would write in the repo brief as the “decision justification”

If you want a tight “why our knobs exist” paragraph, grounded only in the above NVIDIA text:

* CUDA 12.9 is the first toolkit that “adds compiler target support” for SM 10.3 (`sm_103*`), so ptxas needs to come from CUDA 12.9+ (or 13.x) when assembling code for SM103. ([NVIDIA Docs][1])
* For portability, NVIDIA recommends embedding PTX because PTX is forward-compatible, while cubins are only compatible within the same major compute capability (and same-or-higher minor). ([NVIDIA Docs][2])
* PTX built with architecture-conditional targets like `compute_100a` / `sm_100a` is “not forward or backward compatible”, so performance-specialized kernels must keep a baseline PTX fallback (`compute_100`) if the binary is expected to run broadly. ([NVIDIA Docs][2])

If you paste these excerpts into your B300 external-docs brief, you will have clean NVIDIA quotes for both: (a) the “sm_103 toolchain boundary” and (b) the exact compatibility model that explains why PTX inclusion matters and why `*a` targets are a portability hazard.

If you want, I can also help you turn this into a 6 to 10 line “Policy” block for FA4 builds (what targets to emit, when to allow `*a`, when to force PTX inclusion) that stays fully consistent with those NVIDIA quotes.

[1]: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html "1. CUDA 12.9 Release Notes — Release Notes 12.9 documentation"
[2]: https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-compatibility-guide/index.html "1. Blackwell Architecture Compatibility — Blackwell Compatibility Guide 13.0 documentation"
[3]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html "CUDA C++ Programming Guide (Legacy) — CUDA C++ Programming Guide"
