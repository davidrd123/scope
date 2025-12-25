# B200 Session State (SM100)

## Navigation

- Map (timeline + “what goes where”): `notes/FA4/optimization-map.md`
- Shareable perf story: `notes/FA4/docs/kernel-optimization-guide.md`
- Where to record one-change trials: `notes/FA4/b200/experiments.md`
- Historical bring-up notes: `notes/FA4/b200/bringup-log.md`

---

## Current Status

**B200 (SM100) is the “happy path” for FA4/CuTe.** Most of the B300-specific runtime/toolchain workarounds are not needed.

At the canonical benchmark settings (`320x576`, 4 denoise steps, KV-bias `0.3`, quality-preserving), we generally expect **~`20 FPS`** class performance (exact number varies with quantization, compile, and backend choices).

---

## Repro / Benchmark

**End-to-end FPS (canonical):**
```bash
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3
```

**Get a breakdown (attention buckets):**
```bash
PROFILE_ATTENTION=1 \
uv run python scripts/profile_krea_pipeline_blocks.py \
  --height 320 --width 576 \
  --iters 6 --skip 2 \
  --kv-cache-attention-bias 0.3
```

**Notes:**
- `scripts/profile_b300_denoise_drilldown.sh` also works on B200 (name is historical); it writes JSON artifacts under `outputs/`.

---

## What To Record (so we can compare later)

When you try a change, capture it as a card in `notes/FA4/b200/experiments.md`:
- GPU, env, torch/cuda version
- key knobs (resolution/steps/bias, quantization, compile, backend env vars)
- command(s), baseline, result, artifacts, lesson

---

## Known Pitfalls

- If you’re comparing against B300 numbers, make sure you’re using the same benchmark settings (especially `320x576`, steps, and bias).
- If an FA4/CuTe call starts failing with a signature mismatch, it’s usually a version skew issue (installed `flash_attn` vs our vendored CuTe helpers). Record the exact error + versions in an experiment card.
