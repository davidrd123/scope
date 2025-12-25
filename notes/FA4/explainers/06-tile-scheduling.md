# Tile Scheduling and Pipelining in FA4/CUTE

> **What this is:** An explainer for how FlashAttention 4 schedules work across tiles (blocks) and manages multi-stage pipelines to overlap loading with computation.
> **Context:** After understanding TMA memory loading (explainer #5), this explainer covers how FA4 decides *which* tiles to process and in *what order* to maximize L2 cache hit rates and hide memory latency.
> **Updated:** 2025-12-25

---

## Overview

### The Scheduling Problem

In attention, we have work along four axes:
1. **Batch** - Independent sequences
2. **Head** - Attention heads (can share K/V in GQA/MQA)
3. **M blocks** - Query tiles (along sequence dimension)
4. **N blocks** - Key/Value tiles (along sequence dimension)

A naive approach would launch one CUDA block per (batch, head, m_block) combination. But:
- Different tiles have different amounts of work (causal masking makes early M blocks process fewer N blocks)
- K/V tensors are large and expensive to reload from HBM
- We want to maximize L2 cache hits for K/V

### The Pipeline Problem

Even with good scheduling, each tile involves:
1. Load Q, K, V from global memory
2. Compute QK^T
3. Apply softmax
4. Compute O = softmax × V
5. Store O

If we did these sequentially, we'd waste cycles waiting for memory. **Pipelining** overlaps these stages:
- While computing tile N, load tile N+1
- While storing tile N-1, compute tile N

### What This Explainer Covers

1. The different tile scheduler types and when to use each
2. The LPT (Longest Processing Time first) strategy
3. L2 cache optimization for K/V reuse
4. Pipeline state management with mbarriers
5. How these connect to the warp-specialized kernel

---

## Tile Scheduler Types

FA4/CUTE provides several scheduler implementations in `tile_scheduler.py`:

### 1. SingleTileScheduler

The simplest scheduler - one CUDA block per tile:

```python
# tile_scheduler.py lines 76-166
class SingleTileScheduler:
    def get_grid_shape(params):
        return (
            round_up(num_block, cluster_shape_mn[0]),  # M blocks (rounded for clusters)
            num_head * num_splits,  # heads × splits
            num_batch,        # batch dimension
        )

    def get_current_work(self):
        block_idx, head_idx, batch_idx = self._blk_coord
        if is_split_kv:
            head_idx, split_idx = num_splits_divmod.divmod(head_idx)
        else:
            split_idx = 0
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            is_valid=self._is_first_block,  # becomes false after advance_to_next_work()
        )
```

Notes:
- This scheduler is "single tile" by construction: `advance_to_next_work()` just invalidates the tile.
- If `is_split_kv=True`, the grid's `y` dimension is `(head × split)` and `get_current_work()` divmods that into `(head_idx, split_idx)`.

**When used:** Non-causal, non-local attention when `is_persistent=False` (and not varlen).

### 2. StaticPersistentTileScheduler

Persistent kernel - each CUDA block processes multiple tiles:

```python
# tile_scheduler.py lines 169-250
class StaticPersistentTileScheduler:
    def get_grid_shape(params):
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (min(sm_count, total_blocks), 1, 1)

    def advance_to_next_work(self):
        # Jump by grid size to get next tile
        self._tile_idx += cute.arch.grid_dim()[0]
```

**Key insight:** Instead of launching `total_blocks` CUDA blocks, launch `sm_count` blocks that each process multiple tiles. This amortizes kernel launch overhead.

**When used:** Non-causal, non-local attention with `is_persistent=True`.

### 3. SingleTileLPTScheduler

The most sophisticated scheduler - optimizes for L2 cache and load balancing:

```python
# tile_scheduler.py lines 253-376
class SingleTileLPTScheduler:
    @staticmethod
    def create(args):
        # Estimate how many (head × batch) KV "units" fit in L2
        size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
        size_one_head = size_one_kv_head
        size_l2 = 50 * 1024 * 1024  # heuristic constant (bytes); code comment says "40 MB" but value is 50 MB

        # Round swizzle to a power of 2 (with a safe 1-head fallback)
        log2_floor = lambda n: 31 - clz(n)
        swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
```

Notes:
- This is still "single tile": each CTA gets exactly one `(m_block, head, batch, split)` work item; the scheduler mainly defines the *mapping/order* of CTA indices → tiles.
- For varlen, the kernel uses `SingleTileVarlenScheduler` instead, which can apply an LPT-like mapping when `lpt=True`.

**When used:** Non-varlen causal or local attention.

### 4. SingleTileVarlenScheduler

For variable-length sequences (packed batches):

```python
# tile_scheduler.py lines 502-716
class SingleTileVarlenScheduler:
    def _get_num_m_blocks(self, lane, bidb_start):
        # Different sequences have different lengths
        seqlen = params.mSeqUsedQ[batch_idx]
        return ceil_div(seqlen, tile_shape_m)
```

**When used:** When `mCuSeqlensQ` or `mSeqUsedQ` is provided.

---

## LPT Scheduling: The Key Optimization

### The Load Imbalance Problem

In **causal attention**, early M blocks have less work:

```
M block 0: Processes N blocks [0]           (1 block)
M block 1: Processes N blocks [0, 1]        (2 blocks)
M block 2: Processes N blocks [0, 1, 2]     (3 blocks)
...
M block N: Processes N blocks [0, 1, ..., N] (N+1 blocks)
```

If we schedule M blocks in order, early CUDA blocks finish quickly and sit idle while later blocks are still computing.

### LPT Solution: Process Longest First

LPT (Longest Processing Time) reverses the order:

```python
# tile_scheduler.py line 347
block = params.num_block - 1 - block  # Flip the block order
```

```
Processing order:
1. M block N   (most work, starts first)
2. M block N-1
3. ...
4. M block 0   (least work, finishes quickly)
```

All CUDA blocks finish around the same time, maximizing utilization.

---

## L2 Cache Optimization: The Swizzle

### The Problem

K and V tensors are accessed by all M blocks processing the same head. If each CUDA block loads K/V independently:

```
Block A (head 0, m_block 0): Load K[head 0], V[head 0] from HBM
Block B (head 0, m_block 1): Load K[head 0], V[head 0] from HBM  ← Wasteful!
Block C (head 1, m_block 0): Load K[head 1], V[head 1] from HBM
...
```

### The Solution: Swizzle for L2 Locality

The scheduler groups work to maximize L2 reuse:

```python
# tile_scheduler.py lines 273-283
size_l2 = 50 * 1024 * 1024  # heuristic constant (value is 50 MB)

# How many KV head-units can fit in L2?
swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
```

**Swizzle** is the size of a "section" in units of `(head × batch)` work items (often abbreviated `hb` in the scheduler code). A section is picked so the KV working-set for those `hb` items is likely to fit in L2.

```
With swizzle=4:

Within a section, the ordering is (block-major, head-minor):
  for m_block in [last .. first] (LPT):
    for hb in [0..3] (swizzle items):
      process (m_block, hb)
```

### The Coordinate Calculation

```python
# tile_scheduler.py lines 333-350
def get_current_work(self):
    # Which section (L2 group) are we in?
    bidhb, l2_mod = l2_major_divmod.divmod(self._tile_idx)

    # Within section: (block, head_residual)
    block, bidhb_residual = l2_minor_divmod.divmod(l2_mod)

    # Actual batch and head indices
    bidhb_actual = bidhb * swizzle + bidhb_residual
    batch_idx, head_idx = num_head_divmod.divmod(bidhb_actual)

    # LPT: reverse block order
    block = num_block - 1 - block
```

**Result (intended):** consecutive CTAs in a section touch a small group of `hb` items, so if the hardware schedules nearby CTAs together, KV for that section has a better chance to stay resident in L2 across nearby `m_block`s.

---

## Pipeline State Management

### The Multi-Stage Pipeline

FA4 overlaps multiple operations:

```python
# flash_fwd_sm100.py lines 1137-1138
kv_producer_state = cutlass.pipeline.make_pipeline_state(
    cutlass.pipeline.PipelineUserType.Producer, self.kv_stage
)  # 3-4 stages
```

KV staging is a ring buffer of `kv_stage` slots:
- Each KV load uses `producer_state.index` to pick a slot and `producer_state.phase` to disambiguate wrap-around.
- The producer advances the *same* state after every K *and* V load, so K/V interleave across the ring.

### Pipeline State (`index`, `phase`)

The SM100 kernel uses `cutlass.pipeline` state objects that expose the same core API: `.index`, `.phase`, `.advance()`, `.clone()`.
The file `vendored/flash_attn_cute_score_mod/flash_attn/cute/pipeline.py` contains a small reference implementation (`PipelineStateSimple`) that shows the *idea* of the encoding:

```python
# pipeline.py lines 49-118
class PipelineStateSimple:
    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index  # Encodes both index and phase

    @property
    def index(self) -> Int32:
        return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        return self._phase_index // self._stages

    def advance(self):
        self._phase_index += 1
```

**Index:** Which buffer slot (0 to stages-1)
**Phase:** "epoch" counter; barrier protocols typically only care about phase parity, but FA4 often passes the full `phase` value (see note in `pipeline.py`).

### Producer/Consumer Creation

```python
# pipeline.py lines 121-131
def make_pipeline_state(type: PipelineUserType, stages: int):
    if type is PipelineUserType.Producer:
        # Producer starts with flipped phase (buffer empty)
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        # Consumer starts at phase 0
        return PipelineStateSimple(stages, Int32(0))
```

The phase offset ensures producers wait for consumers to finish before reusing a slot.

**Important nuance:** the SM100 kernel typically calls `cutlass.pipeline.make_pipeline_state(...)` (not the `make_pipeline_state(...)` in `flash_attn.cute.pipeline`). `PipelineStateSimple` is useful as a mental model because it matches the observable behavior (`index = phase_index % stages`, `phase = phase_index // stages`), but don’t assume that exact class is used in the compiled kernel.

---

## The Loading Sequence

### Prologue: Fill the Pipeline

Before entering the main loop, we need to "prime" the pipeline:

```python
# flash_fwd_sm100.py lines 1249-1279
# Load first Q tile
load_Q(block=m_block*2 + 0, stage=0)  # Q0

# Load first K/V tile: start from the highest n_block and count down.
# (This is an inner-loop ordering choice; LPT in this kernel refers to the *m_block* mapping in the tile scheduler.)
n_block_first = n_block_max - 1 if n_block_max > 0 else 0
load_K(block=n_block_first, producer_state=kv_producer_state)  # K0
kv_producer_state.advance()

# Load second Q tile (if double-buffered)
if self.q_stage == 2:
    load_Q(block=m_block*2 + 1, stage=1)  # Q1

# Load first V tile
load_V(block=n_block_first, producer_state=kv_producer_state)  # V0
kv_producer_state.advance()

# Main loop: load remaining tiles
for i in range(n_block_max - 1 - n_block_min):
    n_block = n_block_max - 2 - i  # Descending n_block
    load_K(block=n_block, producer_state=kv_producer_state)
    kv_producer_state.advance()
    load_V(block=n_block, producer_state=kv_producer_state)
    kv_producer_state.advance()
```

### Interleaved K/V Loading

Note: K and V are loaded alternately:
- K0, V0, K1, V1, K2, V2, ...

This matches the kernel's structure:
- QK^T uses K, then later (after softmax) P×V uses V.
- On SM100, `sV` is a view on the same underlying shared-memory allocation as `sK`, so K/V staging is managed carefully via the KV pipeline stage/phase bookkeeping.

---

## Pipeline Synchronization

### Producer Side (Load Warp, KV path)

In the SM100 forward kernel, KV loads manage the pipeline mbarriers directly (instead of calling a `producer_acquire()` helper) because K and V may have different `tx_count`:

```python
# flash_fwd_sm100.py (see load_KV)
stage, phase = producer_state.index, producer_state.phase
mbarrier_wait(empty[stage], phase)
elect_one:
  mbarrier_arrive_and_expect_tx(full[stage], tma_copy_bytes[K_or_V])
tma_copy(..., tma_bar_ptr=full[stage])
```

### Consumer Side (Compute Warps)

```python
# flash_fwd_sm100.py (in mma loop)
pipeline_kv.consumer_wait(mma_kv_consumer_state)
...  # read sK/sV at mma_kv_consumer_state.index, do MMA/softmax steps
pipeline_kv.consumer_release(mma_kv_consumer_state)
mma_kv_consumer_state.advance()
```

### Blackwell-Specific: UMMA Pipelines

In `flash_fwd_sm100.py`, the KV pipeline is initialized via `make_and_init_load_kv_pipeline()`:
- `use_tma_KV=True` → `cutlass.pipeline.PipelineTmaUmma` (TMA producer, tcgen05 consumer)
- `use_tma_KV=False` → `cutlass.pipeline.PipelineAsyncUmma` (cp.async producer, tcgen05 consumer)

---

## Warp Specialization + Scheduling

### The Full Picture

```python
# flash_fwd_sm100.py lines 947-1019
# Each warp type has its own scheduler instance
TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

# LOAD WARP
if warp_idx in self.load_warp_ids:
    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()
    while work_tile.is_valid_tile:
        # Issue TMA loads
        load_Q(...); load_K(...); load_V(...)
        # Advance to next tile
        tile_scheduler.advance_to_next_work()
        work_tile = tile_scheduler.get_current_work()

# MMA WARP
if warp_idx == self.mma_warp_id:
    tile_scheduler = TileSchedulerCls()
    work_tile = tile_scheduler.initial_work_tile_info()
    while work_tile.is_valid_tile:
        # Wait for data, compute MMA
        pipeline_kv.consumer_wait(...)
        mma(sQ, sK, acc)
        # Advance
        tile_scheduler.advance_to_next_work()
        work_tile = tile_scheduler.get_current_work()

# SOFTMAX WARPS
# (similar pattern)

# EPILOGUE WARP
# (similar pattern)
```

Each warp type independently tracks its position through the tiles, synchronized only by mbarriers.

---

## Key Design Decisions

### 1. Why Power-of-2 Swizzle?

```python
# tile_scheduler.py line 283
swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
```

In practice the scheduler uses `FastDivmod` for div/mod operations; choosing a power-of-2 `swizzle` makes those operations cheaper (often lowering to shifts/bit ops).

### 2. Why Separate Schedulers per Warp?

Each warp type creates its own scheduler instance. This:
- Avoids cross-warp synchronization on tile state
- Lets each warp progress independently
- mbarriers handle the actual synchronization

### 3. Why FastDivmod?

```python
# tile_scheduler.py uses FastDivmod throughout
num_block_divmod = FastDivmod.create(args.num_block)
```

Integer division is slow on GPU. FastDivmod precomputes multiplier/shift for fast division:
```python
# Instead of: quotient = x // divisor
# FastDivmod does: quotient = (x * magic_mult) >> shift
```

### 4. Residual Section Handling

```python
# tile_scheduler.py lines 339-343
if bidhb < params.num_hb_quotient:
    block, bidhb_residual = l2_minor_divmod.divmod(l2_mod)
else:
    block, bidhb_residual = l2_minor_residual_divmod.divmod(l2_mod)
```

When `(num_head × num_batch)` isn't divisible by swizzle, the last "residual" section is smaller. Special handling avoids out-of-bounds access.

---

## Performance Impact

### L2 Hit Rate

Illustrative (not measured):
```
K/V accesses:
  Head 0: L2 HIT  (already cached from previous M block)
  Head 0: L2 HIT
  Head 0: L2 HIT
  Head 1: L2 MISS (new head, load from HBM)
  Head 1: L2 HIT
  ...
```

Illustrative (not measured):
```
K/V accesses:
  Head 0, M0: L2 MISS
  Head 1, M0: L2 MISS (evicted Head 0)
  Head 2, M0: L2 MISS (evicted Head 1)
  Head 0, M1: L2 MISS (evicted, need to reload!)
  ...
```

### Load Balancing

LPT scheduling:
```
SM 0: M_block=N   (100% utilization until near end)
SM 1: M_block=N-1 (100% utilization until near end)
...
All SMs finish together
```

FIFO scheduling:
```
SM 0: M_block=0   (done quickly, idle)
SM 1: M_block=1   (done quickly, idle)
...
SM N: M_block=N   (still working, others idle)
```

---

## Questions for Further Investigation

1. **Optimal swizzle size:**
   - Is 50MB the right L2 budget assumption?
   - How does it vary across GPU generations?

2. **Persistent vs single-tile tradeoff:**
   - When does persistent scheduling help?
   - Launch overhead vs scheduling flexibility

3. **Varlen complexity:**
   - The `SingleTileVarlenScheduler` uses warp-level prefix sum
   - Could this be simplified?

4. **Split-K integration:**
   - How does `num_splits` interact with scheduling?
   - Is there optimal split strategy per tile?

---

## References

### Code Files

| File | Description |
|------|-------------|
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/tile_scheduler.py` | All tile scheduler implementations |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/pipeline.py` | Pipeline state and synchronization helpers |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py` | SM100 forward kernel, shows scheduler + KV/Q pipelines |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/fast_math.py` | `FastDivmod` implementation |

### Related Explainers

| # | Topic | Relevance |
|---|-------|-----------|
| 5 | [TMA and Memory Loading](05-tma-memory-loading.md) | How tiles are actually loaded |
| 3 | [score_mod](03-score-mod.md) | What happens after tiles are loaded |

### Concepts to Explore Next

- **Explainer #7: Online Softmax** - How softmax is computed incrementally as tiles arrive
- **Explainer #1: How FA4 Attention Works** - Synthesis of all the pieces
