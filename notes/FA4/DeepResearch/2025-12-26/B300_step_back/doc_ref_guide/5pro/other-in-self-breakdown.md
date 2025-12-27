# other_in_self breakdown (stack-attributed)

Goal: make "other_in_self is the majority" actionable.

This doc should answer:
- What exact ops are inside other_in_self
- Which ones are worth killing first
- Which ones are memory bound vs. launch bound vs. compute bound

---

## A. Profile setup

### A1. Minimal reproducible script

Record:
- commit hash:
- model config:
- batch, sequence length, heads, head dim:
- dtype:
- GPU (B300 SM103 or other):
- driver, CUDA toolkit, PyTorch version:

### A2. Mark ranges

If you do nothing else, add explicit ranges around:
- qkv_proj
- rope_apply
- kv_cache_write_k
- kv_cache_write_v
- layout_copies
- attention_call (FA2/FA4)
- residual / output proj (if in the same region)

Preferred: NVTX or `torch.profiler.record_function`.

---

## B. Stack-attributed time table

Fill this from a run that includes stacks (PyTorch profiler with stacks, or Nsight Systems with NVTX).

All times should be end-to-end GPU time, not CPU wall time.

| Sub-op | time (ms) | % of self_attn | evidence (kernel names / stack frames) | notes |
|--------|----------:|---------------:|----------------------------------------|-------|
| qkv_proj |  |  |  |  |
| rope_apply |  |  |  |  |
| k_cache_write |  |  |  |  |
| v_cache_write |  |  |  |  |
| layout_copies |  |  |  |  |
| fa_attention |  |  |  |  |
| other |  |  |  |  |
| total self_attn |  | 100% |  |  |

---

## C. "layout_copies" drill-down

List every copy or transpose, with shapes and why it exists.

| Copy | src layout | dst layout | bytes | time (us) | why needed | can we delete? |
|------|-----------|------------|------:|----------:|------------|----------------|
| copy_0 |  |  |  |  |  |  |
| copy_1 |  |  |  |  |  |  |

If a copy is caused by FA layout mismatch, explicitly reference the exact mismatch here.

---

## D. Interpretation

For each major contributor, write:
- Is it bandwidth bound, compute bound, or launch bound?
- Is it a "must-do" op (algorithmic) or a "format tax" (avoidable)?
- What is the first experiment that could delete it?

---

## E. Next experiment candidates

Rank 3 candidates by impact and risk.

| Rank | candidate | expected win | risk | prerequisite docs/tests |
|------|-----------|--------------|------|-------------------------|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

