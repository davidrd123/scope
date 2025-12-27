# Numerics & Precision in This Stack (KV-bias, LSE, FP8)

> **Explainer #16** — A practical “numerics map” for the realtime pipeline: where scaling happens, what dtype you’re actually in, why LSE exists, and why FP8 can win or lose depending on conversion overhead.
> **Updated:** 2025-12-26

---

## TL;DR

- **B300 quality gate:** FP8 quantization currently produces garbage output (gray/noise) on B300/SM103. Use BF16 (`--quantization none`) for real runs and treat FP8 as perf-only debugging.
- **KV-bias is applied as an additive log-bias to attention scores**: `log_bias = log(kv_cache_attention_bias)`. This is stable and composeable with LSE/softmax.
- **FA4 does softmax in base-2 (`exp2`) for speed**, but the meaning is the same: additive bias in score space becomes a multiplicative factor in probability space.
- **FP8 is not “free speed”**: if you see lots of dtype conversions (`to/copy_`) or `aten::_scaled_mm` overhead, FP8 can lose end-to-end even if matmuls are faster.
- **Bias edge case:** `kv_cache_attention_bias <= 0` is invalid (it becomes `log(<=0)`); `1.0` disables the bias path entirely.

---

## 1) The Three “Numerics Layers” You Care About

### Layer A — Model execution dtype (bf16/fp16/fp8)

In this repo, the diffusion transformer typically runs in:

- **bf16** (default) for weights/activations, or
- **FP8 (E4M3FN)** when enabled via the pipeline’s quantization option.

Relevant wiring:
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py` (quantization selection + torchao FP8 config)

### Layer B — Attention math dtype (softmax stability)

Even when Q/K/V are bf16/fp16:

- The softmax running stats (row max / row sum) are generally treated with **fp32-like stability needs**.
- LSE (`logsumexp`) is conceptually a float32 quantity because it’s used to combine segments stably and (in training) for backward.

Explainers that go deeper:
- Online softmax mechanics: [`07-online-softmax.md`](07-online-softmax.md)
- LSE combine: [`11-splitk-and-segment-combine.md`](11-splitk-and-segment-combine.md)

### Layer C — “Glue” numerics (casts, layout conversions, scaling)

This is the “death by a thousand cuts” category:

- `transpose(...).contiguous()` for a kernel interface
- dtype casts at attention boundaries (bf16↔fp16↔fp8)
- padding to alignment (e.g. flex_attention alignment to 128)
- “scaled mm” patterns (FP8 scaling + conversion)

These often dominate the *difference* between “FP8 wins” and “FP8 loses”.

---

## 2) KV-bias: What It Means Mathematically (and Why We Use `log(...)`)

In the biased KV-cache path we want certain cached tokens to contribute *less*.

The simplest way to do that is to add a constant `c` to the attention logits for those tokens:

```
P(j) = softmax(S_j)               # baseline
P'(j) = softmax(S_j + c)          # biased scores for some j
```

If we choose:

```
c = log(bias)    where 0 < bias <= 1
```

then for affected positions:

```
exp(S_j + log(bias)) = exp(S_j) * bias
```

So the bias directly acts as a multiplicative downweight in probability space.

### Where it’s implemented in this repo

Inside:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

The code computes:

```
log_bias = log(kv_cache_attention_bias)
```

and applies it only to a slice of the KV cache (older frames) while excluding:

- the first frame (stability / anchoring), and
- the current block (so you don’t punish “what I just saw”).

---

## 3) FA4 Numerics: `exp2` Is a Performance Trick, Not a Different Algorithm

FA4/CuTe often uses `exp2` internally because GPUs have very fast base-2 exponent approximations.

The identity is:

```
exp(x) = exp2(x * log2(e))
```

So implementations often maintain:

- scores in a “normal” scale (what you expect), and
- an internal `scale_log2` factor that converts them to base-2 exponent space.

### Important for score_mod / KV-bias

In our mental model (and in practice), `score_mod` in FA4 sees **scaled scores** (the conventional “QK^T / sqrt(d)” domain), and then the kernel converts to exp2 space.

That means adding a **natural-log** bias (`log(bias)`) is consistent across:

- FA4 score_mod (CuTe)
- FlashAttention segment-combine (LSE math is in natural log)
- Triton Kernel B (also consumes a natural-log bias constant)
- flex_attention score_mod (PyTorch softmax uses exp)

---

## 4) FP8 in This Repo: What It Changes (and What It Doesn’t)

### How FP8 is enabled

The realtime pipeline can quantize the diffusion model using torchao:

- `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
- quantization option: `Quantization.FP8_E4M3FN`

This config is “dynamic activation + FP8 weight”, which typically means:

- matmuls change (often to `aten::_scaled_mm`),
- extra scale tensors appear,
- and conversions/casts may be inserted at boundaries.

### B300 / SM103 status (quality gate)

On B300, FP8 output is currently unusable (gray/noise). For the current truth and upstream pointers, see [`session-state.md`](../b300/session-state.md). Treat any FP8 runs as perf-only debugging until quality is verified.

### Why FP8 sometimes loses

FP8 can underperform if:

- conversion kernels (`aten::to`, `aten::copy_`, layout transposes) dominate,
- the workload is not actually matmul-bound (e.g. decode/conv dominates),
- or compilation/autotuning changes the schedule and introduces overhead.

This is why we treat FP8 as a **perf-only experiment knob** unless output quality is validated.

---

## 5) Practical Checklist (So You Don’t Gaslight Yourself)

When you’re comparing numerics/perf experiments:

1) **Fix resolution/settings** (canonical `320x576`, stable settings).
2) **Pick one baseline dtype** (bf16 or fp8) and hold it constant for a sequence of experiments.
3) **Keep KV-bias constant** (`0.3` or `1.0`) unless your experiment is *about* bias.
4) **Always look for “glue” regressions**:
   - more `to/copy_/transpose/contiguous` work
   - padding to alignment for fallbacks
5) **Validate you’re on the backend you think** (avoid silent fallbacks; see #18).

---

## 6) Common Pitfalls

- **Setting `kv_cache_attention_bias <= 0`**: invalid (log blows up). Keep it in `(0, 1]`.
- **Assuming “bias != 1.0 means FA4 runs”**: the bias path has its own backend selection and can fall back to slower kernels.
- **Mixing module sources on SM103**: “FA4 varlen + vendored score_mod CuTe” can trip runtime DSL mismatches unless explicitly gated (see #12 and #17).

---

## References

- KV-bias implementation: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- Pipeline quantization wiring: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
- Online softmax + LSE: [`07-online-softmax.md`](07-online-softmax.md)
- Segment-combine math: [`11-splitk-and-segment-combine.md`](11-splitk-and-segment-combine.md)
- SM103 hazards: [`12-sm103-notes.md`](12-sm103-notes.md)
