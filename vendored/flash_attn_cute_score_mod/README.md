## Vendored `flash_attn.cute` (score_mod-capable)

This directory vendors the Python CuTe-DSL implementation of FlashAttention used for experimenting with **FA4/CuTe `score_mod`** (KV-cache bias) on Blackwell GPUs (B200/SM100 and B300/SM103).

Why this exists:
- The upstream `flash-attn` wheel used by this repo (`flash-attn==2.8.3`) does **not** expose the newer `flash_attn.cute.interface._flash_attn_fwd(..., score_mod=...)` API we need.
- RepoPrompt (and other tooling) only sees files in the Git repo; our previous local clone (`flash-attention.bak/`) was not available in GitHub.

Upstream provenance:
- Source repository: `https://github.com/Dao-AILab/flash-attention`
- Commit: `eacbc560be4811b40dee21c4449ab226d40a2edc`
- We vendor only `flash_attn/cute/` Python sources (no CUDA extension build products).

License:
- See `vendored/flash_attn_cute_score_mod/LICENSE`

Notes:
- This code is currently under active compatibility work with `nvidia-cutlass-dsl` versions used in this repo.
