# Wan 2.1 14B T2V Base Model - Observations & Research Plan

**Last updated**: 2025-12-24  
**Status**: Architecture + upstream model availability verified; integration work pending  
**Purpose**: Document what we know from *this* codebase, confirm upstream facts, and keep a concrete “next actions” list.

---

## 0. What we verified (and what changed vs earlier notes)

### Verified upstream architecture (from `config.json`)
- **Wan2.1-T2V-14B**: `dim=5120`, `num_layers=40`, `num_heads=40`, `ffn_dim=13824` → `head_dim=5120/40=128`
- **Wan2.1-T2V-1.3B**: `dim=1536`, `num_layers=30`, `num_heads=12`, `ffn_dim=8960` → `head_dim=1536/12=128`

**Implication:** Both models use `head_dim=128`. The “weird head/channel config” comment in our causal pipelines is *not* about `dim % num_heads != 0` — it likely refers to kernel/autotune behavior for `head_dim=128` (see §2.3).

### Verified upstream VAE availability
Both base repos (`Wan-AI/Wan2.1-T2V-1.3B` and `Wan-AI/Wan2.1-T2V-14B`) include a `Wan2.1_VAE.pth` (~508MB).

**Codebase implication:** `pipeline_artifacts.py` currently downloads `Wan2.1_VAE.pth` only under the **1.3B** folder, but `WanVAEWrapper(model_name="Wan2.1-T2V-14B")` will *default* to looking under the **14B** folder unless `vae_path` is set. See §4 for mitigation options.

### Verified upstream VACE availability for 14B
VACE is no longer a “maybe”:
- Official full checkpoint repo exists: `Wan-AI/Wan2.1-VACE-14B`
- “VACE module only” weights exist: `Wan2_1-VACE_module_14B_*.safetensors` in `Kijai/WanVideo_comfy` (bf16/fp16 variants)

**Implication:** 14B VACE integration is now “engineering + VRAM + quality evaluation”, not “blocked on missing weights”.

---

## 1. Current State Summary (in this repo)

### 1.1 Pipeline Model Usage

| Pipeline | Base Model | Checkpoint | Notes |
|----------|------------|------------|------|
| **Krea Realtime** | Wan2.1-T2V-14B | `krea-realtime-video-14b.safetensors` | CausVid distillation |
| **LongLive** | Wan2.1-T2V-1.3B | `longlive_base.pt` + `lora.pt` | NVlabs distillation |
| **StreamDiffusionV2** | Wan2.1-T2V-1.3B | `wan_causal_dmd_v2v/model.pt` | DMD distillation |
| **Reward Forcing** | Wan2.1-T2V-1.3B | `rewardforcing.pt` | Reward-guided finetune |

### 1.2 Key observation (still true)
- **Krea is the only 14B realtime/distilled pipeline** in this codebase.
- All other shipped realtime pipelines use **1.3B** distilled checkpoints.

---

## 2. Architecture differences (verified) and what matters for integration

### 2.1 “Defaults” in code are not the real model
In our causal model implementations (e.g. `longlive/modules/causal_model.py`) the constructor defaults are **not** the upstream sizes:

```python
def __init__(..., dim=2048, ffn_dim=8192, num_heads=16, num_layers=32, ...):
    ...
```

But `WanDiffusionWrapper.__init__` loads `config.json` from `generator_path` and then filters kwargs by the model’s `__init__` signature, so in practice **the HF `config.json` overrides those defaults**.

**Actionable check:** if something feels inconsistent (“why is dim 2048?”), print the loaded config dict inside `WanDiffusionWrapper` after `json.load`.

### 2.2 Verified upstream sizes (quick reference)

| Model | dim | num_layers | num_heads | head_dim | ffn_dim |
|------:|----:|-----------:|----------:|---------:|--------:|
| Wan2.1-T2V-1.3B | 1536 | 30 | 12 | 128 | 8960 |
| Wan2.1-T2V-14B  | 5120 | 40 | 40 | 128 | 13824 |

**Why this matters:**
- Any adapter weights (LoRA, VACE modules, etc.) are **not** portable across 1.3B ↔ 14B because weight shapes depend on `dim`, `num_layers`, etc.
- Anything that assumes a fixed layer count (e.g. choosing VACE injection layers, caching schedules) must adapt to 30 vs 40 layers.

### 2.3 About the “weird channel/head configuration” comment
We have this comment in the 1.3B causal pipelines:

```python
# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
```

Given both 1.3B and 14B use `head_dim=128`, this is likely about **flex_attention compiled kernel resource usage for D=128** rather than a literal “invalid” shape.

Practical implication:
- If you remove `mode="max-autotune-no-cudagraphs"` in the 1.3B pipelines, you may regress to compile/runtime failures depending on GPU + torch build.
- For 14B pipelines, we should validate whether the same kernel constraints apply *if/when* we enable the same compiled attention path.

---

## 3. VACE compatibility: what is true now

### 3.1 How VACE is loaded in this repo
`VACEEnabledPipeline._init_vace()` does:
1. Create a new `CausalWanModel` with additional VACE blocks enabled.
2. Load *only* VACE-specific weights from `vace_path` via `load_vace_weights_only()` (filters keys like `"vace_blocks."`, `"vace_patch_embedding."`).

This means we can point `vace_path` either at:
- a “VACE module only” checkpoint (preferred: smaller), **or**
- a full VACE model checkpoint (bigger) as long as it contains the expected VACE keys.

### 3.2 What we can do for 14B now (concrete)
Because 14B-specific VACE weights exist, the next technical steps are straightforward:

- Add a new artifact entry for the 14B module (or full VACE) download.
- Make (or extend) a 14B pipeline that inherits `VACEEnabledPipeline` (Krea currently does not).
- Pass `vace_path` to that pipeline.

**Expected shape sanity check (quick):**
- 1.3B VACE patch embedding conv weight should be shaped like `(1536, 96, 1, 2, 2)`  
- 14B VACE patch embedding conv weight should be shaped like `(5120, 96, 1, 2, 2)`

If these don’t match, the checkpoint is not the right family.

### 3.3 Open VACE questions (engineering, not “missing model”)
- Does the official `Wan-AI/Wan2.1-VACE-14B` release use the same key naming conventions as our `load_vace_weights_only()` filter?
- How much VRAM does 14B+VACE require at target resolutions?
- Can we combine Krea’s distilled checkpoint with VACE weights in a way that produces stable results? (Distilled checkpoints sometimes change internal block structure.)

---

## 4. Artifacts currently downloaded (and likely pitfalls)

From `pipeline_artifacts.py` (current state):

```python
# Krea downloads:
"Wan-AI/Wan2.1-T2V-14B" -> ["config.json"]  # config only
"krea/krea-realtime-video" -> ["krea-realtime-video-14b.safetensors"]

# Shared:
"Wan-AI/Wan2.1-T2V-1.3B" -> ["config.json", "Wan2.1_VAE.pth", "google"]
"Kijai/WanVideo_comfy" -> ["umt5-xxl-enc-fp8_e4m3fn.safetensors", "Wan2_1-VACE_module_1_3B_bf16.safetensors"]
```

### 4.1 VAE path mismatch risk for 14B
Because the 14B artifact doesn’t download `Wan2.1_VAE.pth`, any 14B pipeline that relies on the default VAE lookup will fail unless:
- we add `Wan2.1_VAE.pth` to the 14B artifact list, **or**
- we set `vae_path` explicitly to the 1.3B VAE file, **or**
- we maintain a shared VAE location and teach `WanVAEWrapper` to fall back.

### 4.2 VACE 14B artifact (recommended)
Given upstream availability, consider adding:
- `Kijai/WanVideo_comfy` -> `Wan2_1-VACE_module_14B_bf16.safetensors` (and/or fp16)
or optionally:
- `Wan-AI/Wan2.1-VACE-14B` -> relevant VACE checkpoint files (likely much larger)

---

## 5. Portability: what can and cannot be “just switched to 14B”

### 5.1 Portable (mostly config-level)
- The core loading path via `WanDiffusionWrapper` is *intended* to be architecture-driven by `config.json`.
- Common components (UMT5 encoder, shared VAE file) appear reusable across 1.3B/14B.

### 5.2 Not portable (checkpoint-level)
- LongLive / StreamDiffusionV2 / Reward Forcing are distilled/fine-tuned for **1.3B** weights.
- A “14B port” requires:
  - a 14B distilled checkpoint, or
  - retraining distillation/fine-tuning from 14B base.

---

## 6. Remaining external research gaps

1. **Are there public 14B distillations equivalent to LongLive / StreamDiffusion / Reward Forcing?**  
   (No obvious 14B equivalents found yet; likely needs training.)
2. **Krea internals vs VACE assumptions:** does the Krea distilled model preserve the same block/key structure our VACE wrapper expects?
3. **Compute + VRAM benchmarks** for:
   - 14B base
   - 14B distilled (Krea)
   - 14B + VACE (module)
4. **Resolution support details** (e.g. 720p) and which checkpoints are tuned for it.

---

## 7. Runbook: reproduce the “verified” facts locally

### 7.1 Dump the upstream sizes we actually load
After models are downloaded:

```bash
cat wan_models/Wan2.1-T2V-14B/config.json | jq '{dim, num_heads, num_layers, ffn_dim}'
cat wan_models/Wan2.1-T2V-1.3B/config.json | jq '{dim, num_heads, num_layers, ffn_dim}'
```

### 7.2 Confirm where the 14B pipeline is pulling its VAE from
```bash
ls -lh wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth
ls -lh wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

If the 14B one is missing, either copy/symlink the 1.3B file there, or set `vae_path` explicitly.

### 7.3 Sanity-check a VACE module’s expected shapes
```bash
python - <<'PY'
from safetensors.torch import load_file

# Edit this path to where your artifact downloader stored it:
path = "wan_models/WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors"

sd = load_file(path, device="cpu")
print("num_keys:", len(sd))

w = sd.get("vace_patch_embedding.weight")
print("vace_patch_embedding.weight:", None if w is None else (tuple(w.shape), w.dtype))

# Optional: show a few VACE-related keys
vace_keys = [k for k in sd.keys() if k.startswith("vace_") or k.startswith("vace.")]
print("sample_vace_keys:", vace_keys[:25])
PY
```

(We should keep a small helper script in-repo for safetensors introspection if we do this often.)

---

## 8. Summary

**What we now know for sure:**
- 14B vs 1.3B sizes are confirmed (and differ substantially).
- VACE exists for 14B upstream (official and “module-only” variants).
- VAE appears to be shared across repos, but our artifact layout may not match what 14B pipelines expect by default.

**Next integration milestone:**
- Add + wire **VACE 14B** downloads and a VACE-enabled 14B pipeline (or adapt Krea), then benchmark VRAM/speed/quality.

