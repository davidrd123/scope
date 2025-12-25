# B200 Experiments Log (Small, Reproducible)

**Purpose:** Capture “one-change” experiments as tiny cards so we can learn fast without losing the thread.

**Why a single file?** It avoids repo clutter (no explosion of one-off files). If this grows too large, we can later split into dated files and add an index.

---

## Canonical benchmark settings (for comparable perf numbers)

- **Resolution:** `320x576`
- **Denoising steps:** `4`
- **KV-cache attention bias:** `0.3`
- **Quality-preserving:** avoid accuracy shortcuts (e.g. don’t use KV recompute skipping for “real” results)

Suggested measurement harness:
- `scripts/profile_krea_pipeline_blocks.py` (end-to-end FPS; optional block JSON)
- `PROFILE_ATTENTION=1` (attention bucket breakdown)

---

## Experiment Card Template (copy/paste)

### YYYY-MM-DD — <short title>

**Question:**  
<What are we trying to learn?>

**Hypothesis:**  
<What do we expect to happen, and why?>

**Change (one thing):**  
<Single config tweak / code change / backend swap>

**Benchmark config:**  
- GPU: B200 (SM100)  
- Env: <uv/.venv details>  
- torch / cuda: <e.g. 2.x.y+cu12x / 12.x>  
- Settings: `320x576`, steps=`4`, bias=`0.3`  
- Notes: <compile on/off, quantization, backend vars>

**Command(s):**
```bash
<exact command(s) used>
```

**Baseline:**  
<FPS + any relevant breakdown you’re comparing against>

**Result:**  
<FPS delta + anything surprising in the breakdown>

**Decision:**  
<Keep / revert / follow-up / not worth it>

**Artifacts (optional):**  
- `outputs/<...>.log`  
- `outputs/<...>.json`

**Lessons (write like you’re teaching “future you”):**  
- <What did we learn about the system / tooling / GPU behavior?>
- <What would you try next, and why?>

---

## Experiments

<!-- Add cards here -->
