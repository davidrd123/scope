# Guide to CuTe DSL for Custom FlashAttention Score Modifiers

> **Disclaimer:** This is an LLM-generated document (ChatGPT DeepResearch). While it includes
> citations to GitHub sources and NVIDIA docs, recommendations should be verified against the
> actual codebase before implementation. The CuTe DSL API may have changed since this was generated.
>
> **Quality Assessment:** ✅ Well-sourced with GitHub citations. Good reference material for
> score_mod patterns and TensorSSA semantics.

Source: [ChatGPT](https://chatgpt.com/g/g-p-6943a1900f6481918b389cc8e6c966ce/c/694a5126-562c-8330-b376-f82a9adc88a6)
Extracted: 2025-12-24T21:02:46.862Z
🚀 Exported with [Save my Chatbot](https://save.hugocolin.com)!

---

## User

You gather together a complete and comprehensive guide to Cute DSL to be used by coding agents.

## ChatGPT

To create a comprehensive guide to Cute DSL for coding agents, could you please clarify a few things?

What is the specific domain or purpose of Cute DSL (e.g., UI generation, testing, hardware design)?

Do you want code examples and best practices included?

Should this guide target a particular framework or environment?

Is the guide intended to teach from scratch or serve as a quick reference?

Once I have this info, I can begin gathering the most relevant and useful material.

## User

Query for DeepResearch:

 CUTE DSL is NVIDIA's CUDA Templates for Linear Algebra - a Python DSL for writing high-performance GPU kernels, specifically for attention and matrix operations. It's part of the CUTLASS library ecosystem.

 Context:
 - We're using CUTE to write custom score_mod functions for FlashAttention 4
 - Target: Blackwell GPUs (SM100, B200/B300)
 - Our use case: piecewise-constant attention bias for KV-cache sampling

 What we need:

 1. CUTE DSL syntax guide - the @cute.jit decorator, cute.where, cute.full_like, operators, tensor types (TensorSSA)
 2. score_mod function signature and semantics:
 def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 - What is tSrS_ssa? (attention scores tensor)
 - What are the index tensors? (shapes, dtypes)
 - How do aux_tensors work? (runtime parameters without recompile)
 3. Examples of score_mod patterns:
 - Causal masking
 - Sliding window
 - ALiBi (learned position bias)
 - Block-diagonal attention
 - Custom bias regions (our use case)
 4. Integration with FlashAttention 4:
 - flash_attn_func(..., score_mod=..., aux_tensors=...)
 - Varlen support
 - Forward vs backward considerations
 5. Debugging/iteration tips - how to test CUTE kernels, common pitfalls

 Key sources to check:
 - NVIDIA CUTLASS GitHub (cutlass-dsl package)
 - FlashAttention repo's flash_attn/cute/ directory
 - Any CUTE tutorials or blog posts from NVIDIA
 - Triton-to-CUTE migration guides (if they exist)

 Output format: A practical guide with code examples that a coding agent can use to write CUTE score_mod functions.

## ChatGPT

Understood. I’ll gather a comprehensive, code-rich guide to NVIDIA's CUTE DSL focused on writing `score_mod` functions for FlashAttention 4 on Blackwell GPUs. This will include syntax, function semantics, bias pattern examples, integration with FlashAttention, and debugging tips.

I’ll start the research now and let you know as soon as the guide is ready. Feel free to ask or add anything while I work on it.

## User

Key repos to reference:
 - https://github.com/NVIDIA/cutlass - specifically python/cutlass/cute/ and examples/cute/
 - Our local flash-attention.bak/flash_attn/cute/ has working score_mod examples
 - Installed package at .venv/.../nvidia_cutlass_dsl/python_packages/cutlass/cute/

## ChatGPT

Got it – I’ve updated the research to include those repositories and local sources.

## User

├── cute/tutorial/blackwell/ # C++ SM100 tutorials
 ├── python/CuTeDSL/blackwell/
 │ ├── fmha.py (133KB) # Flash Attention for Blackwell
 │ ├── fmha_bwd.py (131KB) # FMHA backward
 │ ├── mla.py (205KB) # Multi-head Latent Attention
 │ └── dense_gemm.py (68KB) # GEMM examples
 └── python/CuTeDSL/utils/
 └── fmha_helpers.py # FMHA utilities

 For your DeepResearch query, update to include:

 Local reference code:
 - vendored/cutlass-cute/python/CuTeDSL/blackwell/fmha.py - Full FMHA implementation
 - vendored/cutlass-cute/python/CuTeDSL/utils/fmha_helpers.py - Helper utilities
 - flash-attention.bak/tests/cute/score_mod_definitions.py - score_mod examples
 - flash-attention.bak/flash_attn/cute/interface.py - FA4 CUTE interface

## ChatGPT

Got it – I’ve updated the research with these additional local reference code paths.

## ChatGPT

# Guide to CuTe DSL for Custom FlashAttention Score Modifiers

## CuTe DSL Syntax Basics

**CuTe DSL Overview:** CuTe (CUDA Templates for Linear Algebra) is a Python DSL in the CUTLASS 4.x library that enables writing GPU kernels with high-level Python code. Functions decorated with `@cute.jit` are compiled into optimized device code (MLIR IR) that can be invoked from Python or from other CuTe DSL functions[docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#:~:text=%60%40jit%60). CuTe DSL uses native types (e.g. `cutlass.Float16`, `cutlass.Int32`) and provides Python operators and utilities that mimic NumPy/PyTorch semantics but execute at compile-time or on the GPU.

**TensorSSA:** In CuTe DSL, arithmetic on tensor values is done with `TensorSSA` objects – immutable, thread-local tensor values in SSA form (Static Single Assignment) that carry shape and data type metadata[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5787-L5795). You can think of a `TensorSSA` as a vector register containing one or more elements of a given shape and type. Standard Python arithmetic and comparison operators are overloaded to work on `TensorSSA` (with broadcasting rules similar to NumPy). For example, if `q_idx` and `kv_idx` are `TensorSSA` of type Int32, then `q_idx - kv_idx` produces a new `TensorSSA` with the difference for each element, and `q_idx >= kv_idx` yields a Boolean `TensorSSA` mask[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5892-L5900)[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5900-L5908). These boolean tensors can then be used for masking via `cute.where`.

**Key DSL Functions:** CuTe provides utilities like `cute.where(cond, x, y)` to select elements from `x` or `y` based on a boolean tensor `cond`[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L6648-L6656), and `cute.full_like(t, value)` to produce a tensor filled with a given value, matching shape of tensor `t`[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L6569-L6578). These are analogous to `torch.where` and `torch.full_like`. For example, `cute.full_like(tSrS_ssa, float("-inf"))` returns a tensor shaped like `tSrS_ssa` with all entries set to negative infinity (useful for masking out scores)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L18-L22). CuTe also exposes math operations (e.g. `cute.math.exp2` for exponent base 2) and type conversion via `.to(new_dtype)` on tensors.

**@cute.jit vs @cute.kernel:** The CuTe DSL defines two decorators: `@jit` for JIT-compiled functions and `@kernel` for full GPU kernels[docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#:~:text=1.%20%60%40jit%60%20%E2%80%94%20Host,functions)[docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html#:~:text=%60%40kernel%60). In practice, when writing FlashAttention score modifiers, we use `@cute.jit` since these functions will be inlined into the larger attention kernel rather than launched on their own. You do not need to manually specify grid or block sizes for `@cute.jit` functions – they become device functions called from the main FlashAttention kernel. Simply decorate the Python function with `@cute.jit` and use CuTe operations inside; the DSL will compile it to device code when FlashAttention is invoked.

## Understanding the `score_mod` Function Signature

In FlashAttention v4 with CuTe DSL, you can supply a custom scoring modification function (and its backward counterpart) to alter the attention scores before softmax. The function signature is:

python

Copy code

@cute.jit
def score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 ...

**Parameters:**

**`tSrS_ssa`** – A `TensorSSA` containing the raw attention score(s) for a small tile of positions. This is the value you will modify. Think of it as the "Score" tensor computed from the Q·K^T product for one or a few elements. If the kernel processes multiple score elements per thread, `tSrS_ssa` may be a vector (e.g. length 1 or 2) of those scores in registers. For example, on Blackwell GPUs with certain tile sizes, each thread might handle 2 scores at a time (`vec_size=2` when no aux tensors)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134). If aux tensors are used (see below), the implementation often falls back to `vec_size=1`[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134), meaning one score at a time.

**`b_idx, h_idx`** – The **batch index** and **head index** (within the batch) for the current score(s). These are typically `TensorSSA` of type Int32 (often scalar shape). Use these if your modification needs to vary per batch or per attention head. For instance, you might use `b_idx` to index a batch-specific bias vector[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L80-L88), or `h_idx` to apply a head-dependent scaling (as in ALiBi where bias depends on head index[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L44-L52)).

**`q_idx, kv_idx`** – The **query token index** and **key/value token index** for the score(s) being computed. These are crucial for position-based masking or biases. They are Int32 indices relative to the start of the current processing block (for fixed-length inputs, this is the absolute position in the sequence; for variable-length, see `seqlen_info` below). For example, to implement causal masking you compare `q_idx` and `kv_idx` and mask out where `kv_idx > q_idx`[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L18-L22). These indices can be used in arithmetic: e.g. `diff = q_idx - kv_idx` yields the relative position offset, which you can take absolute value of, etc.[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L24-L29).

**`seqlen_info`** – A struct carrying sequence length metadata, used mainly for **variable-length sequences** (when different sequences in a batch have different lengths or when using packed sequences). It typically contains offsets like `offset_q` and `offset_k` that convert local indices to global indices[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L124-L132). In a multi-batch varlen scenario, `q_idx` and `kv_idx` reset to 0 at the start of each sequence or each block; adding `seqlen_info.offset_q` (or `offset_k`) yields the **global index** in the entire batch’s padded sequence. If you are not dealing with packed sequences or paged attention, you might not need to use `seqlen_info` at all (it could be ignored or None). However, if implementing global attention biases (e.g. absolute position biases across the whole batch sequence), use these offsets. For example, FlashAttention provides `score_mod_global_kv_bias` which uses `kv_idx_global = kv_idx + seqlen_info.offset_k` to index into a global bias table[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L124-L132).

**`aux_tensors`** – A list of auxiliary tensors for any additional data your score modification needs (passed at runtime without requiring kernel recompilation). This is how you supply **runtime parameters** such as learned bias vectors or masks. The CuTe DSL treats `aux_tensors` as an optional list of global memory buffers that you can index inside the `score_mod` function. For instance, you might pass in a tensor of length B (batch biases) or length L (position biases) in the `aux_tensors` list. Inside `score_mod`, you can read from these by first creating a fragment and storing the index, then loading from the global tensor. For example, to add a per-batch bias from an aux tensor, you can do:

python

Copy code

batch_bias = aux_tensors[0] # global tensor of shape [batch_size]
dtype = batch_bias.element_type # e.g. cutlass.Float16
b_frag = cute.make_fragment(1, cutlass.Int32)
b_frag.store(b_idx) # store batch index into fragment
bias_frag = cute.make_fragment(1, dtype)
bias_frag[0] = batch_bias[b_frag[0]] # load bias for this batch index
bias_val = (bias_frag.load()).to(cutlass.Float32)
return tSrS_ssa + bias_val # add the bias to the scores

[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L80-L88). In this snippet, `cute.make_fragment(1, T)` creates a register fragment of length 1. We store the index in an Int32 fragment, then use it to index the `batch_bias` tensor, storing the result in a fragment of the bias’s dtype. Finally, we load that fragment (getting a TensorSSA scalar) and cast to Float32 for accumulation before adding to `tSrS_ssa`. This pattern is used because direct indexing of a global memory tensor with a `TensorSSA` index isn’t allowed; we must go through a fragment (which acts like a register). If you have multiple aux tensors, they will appear as `aux_tensors[0], aux_tensors[1]`, etc. You can pass any number of aux tensors (FlashAttention threads them through as a list and marks the kernel to use `vec_size=1` if any aux is present[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134)).

**Return value:** The `score_mod` function should return a modified score tensor (same shape as `tSrS_ssa`). Typically you will either add biases to `tSrS_ssa` or replace certain elements with $-\infty$ (for masking) or other computed values. For correctness, ensure you **do not change the shape** or data type of the score tensor (aside from upcasting to float for accumulation if needed). Also, avoid side effects – treat the inputs as read-only except for constructing the returned tensor.

## Examples of Common `score_mod` Patterns

Below are several typical attention score modifications and how to implement them in the CuTe DSL. Each example is a self-contained `@cute.jit` function that could be passed as `score_mod`:

**Causal Masking (no attending to future tokens):** Set scores for any key position greater than the query position to $-\infty$. You can do this by comparing indices and using `cute.where` to select either the original score or a large negative value. For example:

python

Copy code

@cute.jit
def score_mod_causal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 mask = operator.ge(q_idx, kv_idx) # True where q_idx >= kv_idx
 return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L18-L22). Here `operator.ge` produces a Boolean mask (TensorSSA) of the same shape, and we select `tSrS_ssa` when the mask is True, or $-\infty$ otherwise. This effectively zeroes out attention to future positions after softmax (since $\exp(-\infty) \approx 0$). Note that `cute.full_like(tSrS_ssa, float("-inf"))` will create a tensor of negative infinities with the appropriate shape and dtype.

**Sliding Window (local attention):** Only allow attending to tokens within a certain window of the query (e.g. +/- 256 positions), mask out distant positions. This can be done by checking the absolute difference `|q_idx - kv_idx|`. For instance, to enforce a window of 256 on each side:

python

Copy code

@cute.jit
def score_mod_sliding_window(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 diff = q_idx - kv_idx
 abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype)
 mask = operator.le(abs_diff, cute.full_like(abs_diff, 256))
 return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L57-L62). We compute `abs_diff` by invoking the MLIR math absolute value (`mlir_math.absi`) on the difference. The mask is True for pairs within 256 positions, False otherwise, and we mask out scores outside that range. (CuTe’s broadcasting and vectorized ops handle that `abs_diff` could be a vector.) This approach is flexible – by changing the constant or making it an aux parameter, you can adjust window size.

**ALiBi (Position-Dependent Bias per Head):** *ALiBi* (Attention Linear Bias) is a strategy where each attention head has a fixed slope by which it linearly penalizes farther keys. In practice, this means subtracting a term $s_h \cdot |q - k|$ from the score, where $s_h$ is a head-specific slope. In CuTe DSL, you can compute the slope based on `h_idx` and apply the bias. For example:

python

Copy code

@cute.jit
def score_mod_alibi(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 score = tSrS_ssa.to(cutlass.Float32) # upcast scores to float for accumulation
 # Compute slope = 2^(- (h_idx+1) / 8). Here 8 is an example factor controlling slope differences.
 slope_exp = (h_idx + cute.full_like(h_idx, 1)) * cute.full_like(h_idx, -8)
 slope = cute.math.exp2(slope_exp.to(cutlass.Float32) * cute.full_like(score, 0.125))
 # Apply bias = slope * |q_idx - kv_idx|
 diff = q_idx - kv_idx
 abs_diff = cute.TensorSSA(mlir_math.absi(diff), diff.shape, diff.dtype).to(cutlass.Float32)
 return score - slope * abs_diff

This example follows the FlashAttention implementation[GitHub](https://github.com/Jackmin801/flash-attention/blob/caa34bdef07079c1bc721721131d996bfa4e2317/tests/cute/test_score_mod.py#L66-L74)[GitHub](https://github.com/Jackmin801/flash-attention/blob/caa34bdef07079c1bc721721131d996bfa4e2317/tests/cute/test_score_mod.py#L76-L84): it calculates `slope = 2^{-(h+1)/8}` (the constants 0.125 etc. implement that) and then subtracts `slope * |q - k|` from the score. We use `.to(Float32)` to avoid overflow/precision issues since the biases might be fractional. The result is that higher heads (larger `h_idx`) get a smaller slope (hence smaller penalty for distance) while lower heads heavily penalize distant positions.

**Block-Diagonal Mask (e.g. for sparse patterns or group-by-group attention):** Only allow attention within certain blocks of the sequence. For instance, if queries and keys are divided into blocks of size 64 that should attend only within the same block, you can mask across blocks. Implementation: compute block indices by integer division and compare:

python

Copy code

@cute.jit
def score_mod_block_diagonal(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 q_block = q_idx // 64
 kv_block = kv_idx // 64
 mask = operator.eq(q_block, kv_block)
 return cute.where(mask, tSrS_ssa, cute.full_like(tSrS_ssa, float("-inf")))

[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L65-L70). This yields $-\infty$ for any query-key pair not in the same 64-length block. You can generalize this to arbitrary block sizes or patterns. This pattern is useful for block-sparse attention or grouping heads into segments.

**Custom Piecewise-Constant Bias (use case – KV cache sampling):** If you want to *add a bias* that changes at certain positions (rather than a mask), you can use a condition to select between different bias values. One approach is to predefine the bias values and thresholds and pass them via `aux_tensors`. For example, suppose you want to add bias `B1` to tokens in the first region of the cache and `B2` to the rest. You could pass a tensor like `region_bias = [B1, B2]` as `aux_tensors[0]`, and a cutoff index as an integer constant or aux as well. Then:

python

Copy code

@cute.jit
def score_mod_two_region_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
 kv_idx_global = kv_idx + seqlen_info.offset_k # global position if needed
 cutoff = cute.full_like(kv_idx_global, CUT_OFF) # CUT_OFF is an int or from aux_tensors
 mask_segment = operator.lt(kv_idx_global, cutoff)
 bias_vals = aux_tensors[0] # tensor of [B1, B2]
 # Select B1 or B2 based on mask
 frag = cute.make_fragment(1, cutlass.Int8) # or Int32 if index larger
 frag.store(mask_segment.to(cutlass.Int8)) # store bool as 0/1
 bias_frag = cute.make_fragment(1, bias_vals.element_type)
 bias_frag[0] = bias_vals[frag[0]]
 bias_val = bias_frag.load().to(cutlass.Float32)
 return tSrS_ssa + bias_val

In this pseudocode, we map the boolean condition to 0 or 1 and index into `bias_vals` to pick the corresponding bias. The result is that for keys before the cutoff, `B1` is added; for keys after, `B2` is added. You can extend this idea to multiple regions (though at some point a lookup table approach might be clearer).

Another approach is to directly encode piecewise logic with nested `where` or logical operations. For example, consider combining a causal mask and a range-based mask: FlashAttention’s “stress test” example shows how to compose conditions[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L275-L284). They compute `is_causal = (q_idx >= kv_idx)` and `is_nearby = (|q_idx_global - kv_idx_global| <= 512)`, then `both_conditions = is_causal & is_nearby`. Finally, they do `cute.where(both_conditions, tSrS_ssa + bias_val, cute.full_like(tSrS_ssa, float("-inf")))`[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/tests/cute/score_mod_definitions.py#L275-L284). This adds a bias (say `bias_val`) only to those positions that satisfy both conditions, and masks out everything else. Such combinations can achieve complex piecewise attention patterns.

**Additional Notes:** When implementing your custom patterns, ensure that any constants (like `float("-inf")` or small integers) are of the right type or wrapped with `cute.full_like` on a tensor so that the DSL can infer the correct shapes and dtypes. If you find yourself needing math operations not directly provided, CuTe DSL allows you to use MLIR intrinsics (as we did with `mlir_math.absi`) or you can break computations into smaller steps that use provided ops (e.g. use `**2` for squaring, etc.). Always test the function on small inputs (see tips below) to verify it behaves as intended.

## Using Aux Tensors for Dynamic Biases

The examples above like *batch bias* and *dual buffer bias* illustrate how to use `aux_tensors`. To recap:

**Passing aux_tensors:** In the FlashAttention API, you provide `aux_tensors` as a list of PyTorch CUDA tensors when calling the attention function. The library will convert them to CuTe compatible tensor objects internally[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L32-L40). The number of aux tensors is passed to the kernel compile as a template parameter (so it knows how many to expect)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L32-L40). If `aux_tensors` is not None, FlashAttention ensures the kernel is compiled with `has_aux_tensors=True` and often adjusts vectorization (processing one element at a time)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L128-L134) to simplify memory access patterns.

**Accessing aux data in CuTe DSL:** Within `score_mod`, you index `aux_tensors` like a Python list. Each element is a CuTe `Tensor` (not `TensorSSA`) representing a global memory array. You cannot directly do arithmetic on a `cute.Tensor` (which is essentially a pointer or memory view); instead you load values from it via a fragment as shown. One subtlety: if your aux tensor is multi-dimensional, you need to compute a 1D index or use a multi-dimensional coordinate. For example, if you passed a 2D bias tensor of shape [batch, head], you could compute an index or use two-stage fragment loads. Often it’s simpler to flatten things or pass separate arrays for separate dimensions.

**No recompilation for aux changes:** The benefit of aux tensors is that you can change their contents or swap them out between runs without recompiling the kernel, as long as the shapes/types remain the same. This makes it practical to use learned biases or prompt-dependent masks. Keep in mind that the kernel does not (by default) compute gradients with respect to aux tensors (they are treated as external data). If your aux values are learnable and you need gradients for them, that’s outside of the scope of `score_mod` (you’d have to handle that manually or via other means).

## Integration with FlashAttention 4

**Using `score_mod` in Forward Pass:** FlashAttention v4 provides a function (or module) to execute attention with optional custom modifications. For *fixed-length* sequences, you can call `flash_attn_func` (which internally uses `FlashAttnFunc.apply` for autograd) and for *variable-length* (or to use advanced options) there is `flash_attn_varlen_func`[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L1369-L1377)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L1379-L1387). The varlen version explicitly supports `score_mod` and `aux_tensors` arguments[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L1384-L1392)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L1399-L1407). In practice, you can use `flash_attn_varlen_func` even for uniform-length batches by just passing `score_mod` and leaving `cu_seqlens_q`/`cu_seqlens_k` as None. For example:

python

Copy code

output = flash_attn_varlen_func(
 q, k, v,
 cu_seqlens_q=None, cu_seqlens_k=None,
 score_mod=my_score_mod_func,
 aux_tensors=[aux1, aux2]
)

Here `q, k, v` are your input tensors (shape `[batch, seqlen, nheads, head_dim]` for standard MHA format in FlashAttn). The `aux_tensors` list can contain any additional CUDA tensors needed (must be list even if one tensor). Under the hood, FlashAttention will partition the work by GPU architecture (e.g. SM80 vs SM100) and JIT-compile a kernel with your `score_mod` inlined. **Important:** If you supply a `score_mod`, you *must also supply a `score_mod_bwd` when calling backward (training mode)*[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L56-L64). The FlashAttn API typically requires you to register both forward and backward modifications, since the backward pass needs to know how to propagate gradients through your custom operation. In the forward call (like above) you usually don’t pass `score_mod_bwd` – instead, the library caches the forward `score_mod` and will retrieve the backward version from it (e.g., by naming convention or a paired registration in the code). Check FlashAttention’s docs or code for how to set `score_mod_bwd` (often you attach it to the same function or provide as separate argument in a backward call). The bottom line: **if you are training (computing gradients), implement the gradient logic or at least provide an identity backward function.**

**Backward Function Considerations:** The backward (`score_mod_bwd`) should have a signature like `(dScore_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors)` and produce the modified gradient to pass back into the softmax. If your forward `score_mod` simply adds a bias or mask (without dependency on the original score values except to replace them), then the backward typically either passes the gradient through unchanged or zeros it out where scores were masked. For example, for a mask that sets scores to $-\infty$, the backward should zero out gradients for those positions (since a masked score produces no output influence). For an additive bias, the gradient w.r.t. score is just 1 (so you can return `dScore` unchanged, and there’s no gradient w.r.t. the bias constant unless you explicitly want to accumulate it externally). FlashAttention currently **only supports custom score_mod backward on Blackwell (SM100) GPUs**[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L62-L65) and requires no variable-length support in the backward yet (as of the time of writing, `varlen + score_mod` backward is not implemented)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L60-L64). This means if you use these features in training on older GPUs or with packed sequences, you might hit a `NotImplementedError`. As a workaround, you can restrict training to SM100 or disable `score_mod` during backward for now.

**Other integration points:** The `score_mod` is applied inside the softmax calculation of FlashAttention. It works alongside other FlashAttn features like `mask_mod` (used for custom dropout or other masks) and block-sparse attention. If you use block-sparse mode or Grouped-QA (GQA) packing, be aware that the indices seen by `score_mod` might be transformed (FlashAttn handles this by adjusting `q_idx` for packed heads, as seen in `apply_score_mod_inner` where it computes a logical `head_idx` for PackGQA cases[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/softmax.py#L396-L404)). In most cases, you don’t need to worry about this unless you enable `pack_gqa=True`. Also, if using FlashAttention’s built-in options like `causal=True` or `window_size=(L,R)` simultaneously with a `score_mod`, make sure they’re not duplicating work – e.g., if you set `causal=True` **and** your `score_mod` also applies a causal mask, you might be masking twice (this could be benign but unnecessary). Typically, you would use either the built-in flags or your custom `score_mod`, not both for the same effect. FlashAttention will raise an error if you try to combine certain options (for instance, `softcap and score_mod cannot be used together`[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L24-L28)).

**Performance:** The CuTe JIT will compile a specialized kernel for your `score_mod`. Compilation can take some time on the first run (similar to JIT compiling a Triton kernel), but subsequent calls with the same signature are cached. Ensure you pass the same Python function object to avoid recompiling (the DSL uses a hash of the function’s code for caching[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/interface.py#L20-L28)). In terms of runtime, simple operations like those shown (elementwise comparisons, additions) are cheap relative to the cost of the attention GEMM and softmax. However, more complex logic in `score_mod` (especially if it involves divergent branches or many memory accesses) could impact performance. Use the provided examples as a guide for the kind of operations that are reasonable.

## Tips for Debugging and Development

**1. Test on Small Cases:** Before relying on your custom `score_mod` in a full training run, test it on small, known inputs. FlashAttention’s own test suite does this by comparing the output of attention with a custom `score_mod` to a reference implementation in pure PyTorch[GitHub](https://github.com/Jackmin801/flash-attention/blob/caa34bdef07079c1bc721721131d996bfa4e2317/tests/cute/test_score_mod.py#L222-L230). For example, to verify `score_mod_causal`, they compare against a PyTorch attention where scores are masked with `-inf` for illegal positions[GitHub](https://github.com/Jackmin801/flash-attention/blob/caa34bdef07079c1bc721721131d996bfa4e2317/tests/cute/test_score_mod.py#L166-L174). You can mimic this approach: manually compute attention scores and apply your bias/mask in numpy/torch, run FlashAttention with your `score_mod` on the same inputs, and assert that outputs (and gradients, if applicable) match. This will catch many errors (off-by-one indices, dtype mismatches, etc.).

**2. Use CuTe’s Debug Features:** CuTe DSL provides logging and IR inspection that can be very helpful. By setting environment variables, you can make the DSL print out the MLIR IR or log messages during compilation[docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/debugging.html#:~:text=For%20users%20familiar%20with%20MLIR,IR%20is%20generated%20as%20expected). For instance, set `CUTE_DSL_PRINT_IR=1` to dump the generated IR (which shows how your Python code is lowered to loops and instructions), and `CUTE_DSL_LOG_LEVEL=20` (Info) or `10` (Debug) for more verbose logs[docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/debugging.html#:~:text=,export%20CUTE_DSL_LOG_TO_CONSOLE%3D1). Understanding the IR can confirm whether your conditions and arithmetic are being optimized correctly. Keep in mind the IR is in MLIR format, which has a learning curve, but even a high-level glimpse can be useful.

**3. Printing from the Kernel:** You can instrument your `score_mod` with prints to debug values at runtime. CuTe supports two ways to print: Python’s `print()`, which executes at *compile time* (useful for printing static info like shapes), and `cute.printf()`, which inserts a device-side print in the generated GPU code[docs.nvidia.com](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/debugging.html#:~:text=Using%20Python%E2%80%99s%20,printf). For example, you could do:

python

Copy code

cute.printf("q_idx=%d kv_idx=%d -> mask=%d\n", q_idx, kv_idx, mask)

inside your function to print those values during kernel execution (this will significantly slow down the kernel, but is useful for debugging). The format strings and arguments work similarly to CUDA `printf`. Remember to remove or disable these once things work.

**4. Common Pitfalls:** A few things to watch out for:

**Shape mismatches:** All returned tensors must match the shape of the input score tensor. If you do broadcasting (e.g. comparing a scalar to a vector), ensure the result is broadcast back to the vector shape. CuTe’s `operator` overloads handle most of this automatically via broadcast rules[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5914-L5922)[GitHub](https://github.com/sampan26/cute_bits/blob/5e062f65fa9bbe1b349404a8bcdcebbdbf79ee0f/cutlass/python/CuTeDSL/cutlass/cute/core.py#L5923-L5931).

**Dtype conversions:** If you create constants (like `0` or `0.0`) or use Python math in your function, ensure the values are cast to CuTe numeric types. Using `cute.full_like` with a Python scalar is a good way to do this, as it infers type from the tensor. Also, for comparisons, CuTe will yield Boolean tensors that are a special type – you usually don’t need to cast those (except when storing into fragments, where we cast to an Int type as shown).

**Memory access:** When using aux tensors, be careful with indices. If an index might go out of bounds of your aux tensor, you must clamp it or ensure masking prevents using those values. FlashAttention internally wraps indices for safety when aux tensors are used with varlen (they use a fast div/mod to wrap indices to valid ranges)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/softmax.py#L405-L414)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/softmax.py#L416-L424). But it’s best to ensure you don’t intentionally read invalid positions.

**Randomness and mask_mod:** If you also use FlashAttention’s `mask_mod` (for example, for dropout or other token-level masks), ensure it’s orthogonal to `score_mod`. The two mods are applied in different places (mask_mod can zero out entire rows for unused tokens, etc.), and combining them can be done but requires understanding the internals.

**Performance tuning:** If your custom modifications are complex and you want to optimize, consider if you can precompute parts outside the kernel (e.g. prepare a bias tensor per position in advance). Also, note that Blackwell GPUs allow larger registers and more complex kernels – if targeting older GPUs, simpler might be better.

**5. Learn from Examples:** NVIDIA’s CUTLASS repository includes full examples of FlashAttention written in CuTe DSL (see `examples/77_blackwell_fmha` in CUTLASS 4)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L12-L20). The FlashAttention v4 code itself is a great reference – for instance, the `flash_attn/cute/flash_fwd_sm100.py` file is essentially an FMHA implementation using CuTe, and it references how the score modification is integrated[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L2-L10)[GitHub](https://github.com/Dao-AILab/flash-attention/blob/ceb4110f7e432921c2efbd348ccf85685b2c9560/flash_attn/cute/flash_fwd_sm100.py#L34-L42). Studying these can give insight into advanced usage (like persistent kernels, cluster scheduling, etc.) beyond the scope of a simple `score_mod`. However, for most practical purposes, you can achieve your custom attention bias or masking by writing a concise Python function as we’ve outlined, and trust the CuTe DSL to handle the heavy lifting of optimization and GPU code generation.

## User

Check the GitHub repo davidrd123/scope, the feature/stream-recording branch to see the work that I've done so far and integrate with what we've been talking about. notes/FA4 folder has notes that point to files and profiling I did

## ChatGPT

To integrate the work you've done in the davidrd123/scope repository, could you clarify what kind of integration or synthesis you're looking for? For example:

Should I extract and summarize relevant `score_mod` development and profiling notes from `notes/FA4`?

Are there specific CuTe or FlashAttention files or implementations in that repo you'd like compared or merged into the guide we’re creating?

Would you like suggestions or a plan for improving or restructuring the existing code based on best practices?

Let me know which of these (or others) you'd like prioritized.

## User

Yes, to 1, to 2, there should be vendored files there for FlashAttention4, and definitely for 3, yes, suggestions are planned for improving or restructuring the existing code.

## ChatGPT

Great, I’ll review the davidrd123/scope GitHub repository under the feature/stream-recording branch, focusing on:

Extracting and summarizing your profiling notes and code references from the `notes/FA4` folder.

Reviewing relevant FlashAttention 4 and CuTe files you've vendored or modified.

Providing specific suggestions and a plan to improve or restructure the current codebase for your custom `score_mod` development.

I’ll begin the analysis and get back to you with a synthesis and concrete recommendations.

## User

![](https://chatgpt.com/backend-api/estuary/content?id=file_0000000088a0722fa08b2dbbcddf42a1&ts=490725&p=fs&cid=1&sig=f1f4816d4410722308485ed3f0b360863ed0abfdb244ab3509700e214416e461&v=0)

I'd like you to go through all the links here as potential performance optimization resources or approaches, and give me a comprehensive integrated report of all the most promising techniques. Essentially, an all-in-one resource that I can drop into my project and have my coding agents look at the code in more detail to estimate which we could make use of.

## ChatGPT

Got it. To produce the most useful integrated report for you, could you confirm a few details:

What specific components or kernels in your project are you aiming to optimize (e.g. FlashAttention, score_mod, kernel tiling, MHA, etc.)?

Are you targeting specific GPU architectures (e.g. Hopper, Blackwell, A100)?

Are you looking only for techniques applicable to custom CUDA/CuTe kernels, or also TorchScript, TorchDynamo, or Helion-level optimizations?

Do you want this report to prioritize techniques that are ready to drop in, or include more experimental or research-stage tools as well?

Once I have these answers, I can extract and synthesize the most relevant performance strategies from all the resources you linked.

## User

We're targeting Blackwell. Specifically, techniques applicable to COSMICUDA acute kernels. Also just understanding what these other optimizations look like. So I'm looking for a document that's comprehensive, unpacks all the material in those blogs, which can then serve as a reference in my code base for other coding agents with direct access to our priorities and what we're working on to evaluate it there. So I would just assume that we might want to be open to optimizing everything until we kind of like close off things that are less relevant. Yeah, so techniques that we're ready to drop in. Yeah, even research stage tools, right, because that might be an interesting little side project like, you know, that the kernel agent, you know, it's just interesting. Want to know kind of what the state of affairs is with that, how ready it is to use. Yeah, just so then unpacking all the information in those links. So I got it in a great markdown format that can be perused at leisure by other agents.

## ChatGPT

Understood. I’ll create a comprehensive, markdown-formatted reference document that unpacks all the blog posts and repositories you linked. This will include performance optimization techniques relevant to COSMICUDA acute kernels and Blackwell GPUs, along with an assessment of maturity (drop-in vs research-stage), potential applicability, and integration ideas for your project.

I’ll notify you as soon as it’s ready. Feel free to keep building in the meantime.

