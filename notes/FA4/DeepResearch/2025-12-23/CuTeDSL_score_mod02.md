You are basically dealing with two different “CUTLASS Python DSL eras”:

* **Your current pinned era**: `nvidia-cutlass-dsl==4.1.0` (works with the pip `flash-attn==2.8.3` FA4 bits for basic attention after SM103 arch patching, but is missing newer DSL APIs).
* **The `flash-attention.bak` score_mod era**: expects newer CuTe DSL APIs (the ones you explicitly called out: `TensorSSA.broadcast_to`, `cute.make_rmem_tensor`, `enable_tvm_ffi`, plus the `FastDivmodDivisor` family).

Below is what’s actually required, which versions satisfy it, and a strategy that lets you run `_flash_attn_fwd(..., score_mod=...)` on **B300 (SM103)** while keeping “normal attention” stable.

---

## 1) Which `nvidia-cutlass-dsl` versions are compatible with `flash-attention.bak` score_mod

You listed three missing APIs. CUTLASS’ own CuTe DSL changelog pins when those arrived:

| Missing API / behavior                    | First CUTLASS DSL version that includes it | Why it matters for FA4 score_mod                                                                                                                                                            |
| ----------------------------------------- | -----------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TensorSSA.broadcast_to`                  |                                  **4.2.0** | Newer flash-attention CuTe code uses broadcast utilities to keep everything TensorSSA-shaped instead of scalarizing.                                                                        |
| `cute.make_rmem_tensor`                   |                                  **4.3.0** | This is used in several DSL examples and is commonly relied on by newer CuTe kernels for predicate/register-memory fragments.                                                               |
| `enable_tvm_ffi` (and `--enable-tvm-ffi`) |                                  **4.3.2** | Newer CuTe integration can optionally go through a TVM FFI path; your own vendored example matches this model, and CUTLASS documents `--enable-tvm-ffi` and the `enable_tvm_ffi=True` flow. |

So the hard minimum for the three APIs you called out is:

* **`nvidia-cutlass-dsl >= 4.3.2`**

Now the extra “missing” you cited from notes is **`FastDivmodDivisor`** (flash-attention.bak expects it, 4.1.0 doesn’t have it). Your own kernel-dev-log already says the working score_mod setup uses:

* **`nvidia-cutlass-dsl >= 4.3.3`** plus **`apache-tvm-ffi`**

Given that, the safest recommendation is:

* **Pin `nvidia-cutlass-dsl==4.3.3`** (or the lowest version you have confirmed contains `FastDivmodDivisor`), because it sits above the 4.3.2 threshold and matches what your repo notes say was used successfully.

One more compatibility landmine that matters for your mixed setup:

* CUTLASS DSL **4.3.0 removed `cutlass.utils.ampere_helpers`**. If any code path still imports that (common in older flash-attn CuTe glue), it will explode once you move to 4.3.x.

That removal is one of the reasons “pip flash-attn FA4 + cutlass-dsl 4.3.x” can be fragile unless you make sure you’re loading a flash-attention tree that has already moved off the old imports.

---

## 2) Integration strategy that works on B200 + B300 without destabilizing normal attention

You want two things simultaneously:

1. **Score_mod FA4** for KV-bias (Kernel B) via `_flash_attn_fwd(..., score_mod=...)` on SM103.
2. Keep “normal attention” (FA2 and, ideally, FA4 varlen) working and not randomly shadowed by a local checkout.

The central problem is that today your code sometimes turns `flash_attn` into a “Frankenpackage” by inserting a local `flash-attention/flash_attn` directory into `flash_attn.__path__`. That can be good (score_mod support) but also dangerous (silently shadowing the pip wheel’s modules).

### Recommendation: treat “score_mod CuTe” as an opt-in overlay, not a global default

Concretely:

* Keep your default runtime using the **pip wheel** for `flash_attn` unless explicitly enabling score_mod CuTe.
* When score_mod is enabled, allow a **controlled overlay** of the local `flash-attention.bak` tree, but only when needed, and only in one place.

This matches your backend design already:

* B200 default can stay `triton`, opt-in `fa4`
* B300 default can stay `flash` (segment-combine) or `triton`, opt-in `fa4`

### Two dependency “tracks” (this is the cleanest way to avoid constant breakage)

**Track A (production / stable):**

* `flash-attn==2.8.3` (your pinned wheels)
* `nvidia-cutlass-dsl==4.1.0` only if you need basic FA4 testing
* No score_mod CuTe expectation

**Track B (score_mod CuTe):**

* `flash-attn==2.8.3` still installed (for FA2 and for shared compiled bits)
* `nvidia-cutlass-dsl==4.3.3` (or >= 4.3.3)
* `apache-tvm-ffi`
* Local `flash-attention.bak` used as the CuTe implementation provider

Track B is what you use when you explicitly want `_flash_attn_fwd(..., score_mod=...)`.

This is the least risky way to keep normal runs stable while still enabling the aggressive path.

---

## 3) Concrete `pyproject.toml` changes

Add a *new* dependency group for score_mod rather than mutating `fa4` in-place (because `fa4` today is effectively “basic FA4 bring-up” pinned to 4.1.0).

```toml
[dependency-groups]
# Existing:
fa4 = [
  "cuda-python>=12.9.0; python_version >= '3.12'",
  "nvidia-cutlass-dsl==4.1.0; python_version >= '3.12'",
]

# New:
fa4_scoremod = [
  "cuda-python>=12.9.0; python_version >= '3.12'",
  "nvidia-cutlass-dsl==4.3.3; python_version >= '3.12'",
  "apache-tvm-ffi; python_version >= '3.12'",
]
```

If you want to be slightly more flexible (but less reproducible), use `>=4.3.3,<4.4` once you’ve tested 4.3.4 etc.

---

## 4) Concrete install + patch commands for B300 (SM103)

### 4.1 Environment prerequisites (you already have this right)

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
```

### 4.2 Install the score_mod track dependencies

From repo root:

```bash
uv sync --group fa4_scoremod
```

Or explicit:

```bash
uv pip install "nvidia-cutlass-dsl==4.3.3" "apache-tvm-ffi" "cuda-python>=12.9.0"
```

### 4.3 Patch CUTLASS DSL for SM103

You have two patch sources in notes:

* `impl_utils.py` arch allowlist patch (critical in some versions)
* tcgen05 and cpasync admissible arch lists

I would update your **Python** patcher (`scripts/patch_cutlass_sm103.py`) to also patch `impl_utils.py`, because that’s the one that tends to fail earliest and noisily.

#### Minimal patch extension idea (drop-in logic)

In `scripts/patch_cutlass_sm103.py`, add:

* Locate `cutlass_base / "impl_utils.py"`
* Insert the “allow sm_103/sm_103a whenever sm_100a/sm_100f allowed” logic (you already documented this in `B300-FA4-PATCHES.md`)

Then run:

```bash
uv run python scripts/patch_cutlass_sm103.py
```

(If you stick to the `.sh` patcher, run that instead, but I’d prefer one canonical patch script.)

### 4.4 Make sure the local repo exists

You already have `flash-attention.bak/`. Keep it. Do not rely on the `flash-attention` symlink existing in production runs.

---

## 5) Code patches to keep normal FA2/FA4 stable while enabling score_mod CuTe

### Patch 1: Gate `flash_attn.__path__` injection in `attention.py`

Right now `wan2_1/modules/attention.py` always does `_extend_flash_attn_path()` at import time, which is how you get shadowing surprises.

Make it opt-in:

* Only extend the path when you are explicitly trying to use score_mod CuTe.
* Also search `flash-attention.bak` (not only `flash-attention`).

Example patch sketch:

```py
# src/scope/core/pipelines/wan2_1/modules/attention.py

def _extend_flash_attn_path() -> None:
    use_local = (
        os.getenv("SCOPE_FLASH_ATTN_LOCAL", "0") == "1"
        or os.getenv("SCOPE_KV_BIAS_BACKEND", "").lower() == "fa4"
    )
    if not use_local:
        return

    try:
        base_path = Path(__file__).resolve()
    except Exception:
        return

    for parent in base_path.parents:
        for repo_dir in ("flash-attention.bak", "flash-attention"):
            candidate = parent / repo_dir / "flash_attn"
            if candidate.is_dir():
                candidate_str = str(candidate)
                if candidate_str not in _flash_attn.__path__:
                    _flash_attn.__path__.insert(0, candidate_str)
                return
```

Result:

* Normal runs do not get shadowed.
* When you set `SCOPE_KV_BIAS_BACKEND=fa4` (or `SCOPE_FLASH_ATTN_LOCAL=1`) you get the local CuTe stack.

### Patch 2: Make KV-bias FA4 path explicitly request local CuTe

In `krea_realtime_video/modules/causal_model.py` you already do `_extend_flash_attn_path_for_score_mod()` and check `score_mod` in signature. Keep that, but align it with the same gating behavior so you do not accidentally inject local code when you are not using FA4 score_mod.

The most robust pattern is:

* If backend is `fa4`, do the injection and import `_flash_attn_fwd`
* Otherwise do not touch `flash_attn.__path__`

### Patch 3: On B300, never rely on `torch.compile(flex_attention)` in the score_mod environment

You already introduced `DISABLE_FLEX_ATTENTION_COMPILE` and SM103 guards. For the “score_mod track” on B300, I’d hard-default to eager flex_attention (or avoid flex_attention entirely by using `fa4`, `flash`, `triton` backends).

Concrete run flag:

```bash
export DISABLE_FLEX_ATTENTION_COMPILE=1
```

This is not about performance, it’s about preventing the `cutlass` naming conflict from killing inductor paths on SM103.

---

## 6) How to run and verify `_flash_attn_fwd(..., score_mod=...)` on B300

### One-shot correctness + perf harness (you already have it)

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export DISABLE_FLEX_ATTENTION_COMPILE=1
export SCOPE_KV_BIAS_BACKEND=fa4
export SCOPE_FLASH_ATTN_LOCAL=1   # optional if you key off KV_BIAS_BACKEND

uv run python scripts/test_fa4_kv_bias.py
```

Success criteria:

* The script does not exit with “_flash_attn_fwd lacks score_mod”
* Basic test passes
* Flex comparison passes (if available)
* Real-shape benchmark runs

---

## 7) If you still hit `FastDivmodDivisor` errors

If after moving to `nvidia-cutlass-dsl==4.3.3` you still see `FastDivmodDivisor` missing, there are only three plausible realities:

1. The symbol is in **a later 4.3.x** (so bump to 4.3.4/4.3.5 if they exist in your package index).
2. The symbol name changed (example: split into `FastDivmod` + `FastDivmodU32` style helpers).
3. Your local flash-attention.bak is at a commit that expects an even newer CUTLASS DSL snapshot than the PyPI `nvidia-cutlass-dsl` provides.

In that case, the high-leverage fix is to **stop guessing and bind the pair**:

* Pick a specific flash-attention commit that you know works on B200 for score_mod.
* Pin a specific `nvidia-cutlass-dsl` version that that commit works with.
* Treat them as a matched pair in the `fa4_scoremod` group.

You already have a strong hint in `kernel-dev-log.md` that the known-good score_mod setup used `>=4.3.3` and `apache-tvm-ffi`. If `FastDivmodDivisor` is still missing at 4.3.3, bumping within the same minor series is the first move.

---

## 8) The “frontier” option: making cutlass-dsl coexist with Inductor on SM103

If you want one environment that does everything (score_mod CuTe plus compiled flex_attention plus any Inductor paths), the real blocker is the **top-level module name `cutlass`**.

The only credible ways around that, without upstream changes, are:

1. **Disable all Inductor paths that import `cutlass`** (most pragmatic, what you are already converging on for B300).
2. **Import hook shim**: a custom `MetaPathFinder` that routes `import cutlass` differently depending on importer (torch._inductor vs flash-attention). This is doable but brittle and hard to test.
3. **Process isolation**: run score_mod attention in a helper process with the CuTe stack, keep main process clean for Inductor. GPU tensor handoff is possible but nontrivial and may erase the latency win.

Given your current architecture, (1) is the best “get it working now” path, and (2) is the most interesting experiment if you want a single-env solution later.

---

## Bottom line

* **Minimum cutlass-dsl for your missing APIs**: **4.3.2** (because `enable_tvm_ffi` lands there).
* **Practical pin for `flash-attention.bak` score_mod**: **4.3.3** (matches your own kernel-dev-log) plus `apache-tvm-ffi`.
* **Make local flash-attention CuTe an opt-in overlay** (gated path injection) so you don’t randomly break normal FA2/FA4.
* **On B300**, pair that with your SM103 patching and disable flex_attention compilation in the score_mod environment.

If you want, I can write the exact diffs for:

* `wan2_1/modules/attention.py` gating logic
* `causal_model.py` “only inject local CuTe when backend=fa4”
* `scripts/patch_cutlass_sm103.py` extended to patch `impl_utils.py` too

…in the same style as your repo (minimal, trip-breaker friendly, and safe defaults).
