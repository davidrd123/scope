# Fix: remove chunk padding by switching to a power‑of‑2 pair dimension (C=64) and a FlashAttn‑like ti...

> **Disclaimer:** This is an LLM-generated document (ChatGPT). The diagnosis and implementation
> suggestions should be verified against the actual kernel code in `triton_rope_fused.py`.
>
> **Quality Assessment:** ✅ Well-grounded. The diagnosis of C0/C1/C2 padding overhead (22/21/21 → 32/32/32)
> is specific to the actual kernel implementation. The register pressure analysis is reasonable.
>
> **Key Diagnosis:** The RoPE Step-2 regression was caused by power-of-2 padding inflating channel
> lanes by ~50% (64 pairs → 96 padded pairs), causing register pressure and lower occupancy on SM100.

Source: [ChatGPT](https://chatgpt.com/g/g-p-6943a1900f6481918b389cc8e6c966ce/c/694a13f8-aa70-8329-aa1a-458d0b85436d)
Extracted: 2025-12-24T21:03:10.420Z
🚀 Exported with [Save my Chatbot](https://save.hugocolin.com)!

---

## User

<file_map>
/Users/daviddickinson/Projects/Lora/scope
├── notes
│ ├── FA4
│ │ ├── DeepResearch
│ │ │ ├── 2025-12-22
│ │ │ │ ├── Phase3_01.md *
│ │ │ │ ├── MSU.md
│ │ │ │ └── wafer.md
│ │ │ ├── MSU_chat.md
│ │ │ ├── summary.md
│ │ │ └── wafer.md
│ │ ├── phase3-triton-rope-step2.md *
│ │ ├── phase3-triton-rope.md *
│ │ ├── b200-bringup-log.md
│ │ ├── b300-investigation.md
│ │ ├── kernel-dev-log.md
│ │ └── rope-optimization.md
│ ├── 5pro_chat_01.md
│ ├── 5pro_chat_02.md
│ ├── input-mode-fix.md
│ ├── krea_rt.md
│ ├── krea-offline-render-tuning.md
│ ├── offline-timeline-renderer-report.md
│ ├── offline-timeline-renderer.md
│ ├── render-timeline-guide.md
│ └── TODO-next-session.md
├── scripts
│ ├── bench_triton_rope.py * +
│ ├── test_rope_correctness.py * +
│ ├── bench_blockwise_attn.py +
│ ├── patch_cutlass_sm103.py +
│ ├── profile_attention.py +
│ └── triton_sdpa.py +
├── src
│ └── scope
│ ├── core
│ │ ├── kernels
│ │ │ ├── triton_rope_fused.py * +
│ │ │ ├── triton_rotary.py * +
│ │ │ ├── __init__.py +
│ │ │ └── triton_attention.py +
│ │ ├── pipelines
│ │ │ ├── krea_realtime_video
│ │ │ │ ├── modules
│ │ │ │ │ ├── causal_model.py * +
│ │ │ │ │ ├── model.py * +
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── vae_block3.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── prepare_context_frames.py +
│ │ │ │ │ └── recompute_kv_cache.py +
│ │ │ │ ├── components
│ │ │ │ │ └── __init__.py
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── __init__.py +
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ └── test.py +
│ │ │ ├── base
│ │ │ │ └── __init__.py
│ │ │ ├── longlive
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── prepare_recache_frames.py +
│ │ │ │ │ └── recache_frames.py +
│ │ │ │ ├── components
│ │ │ │ │ └── __init__.py
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── causal_model.py +
│ │ │ │ │ ├── model.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── __init__.py +
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ ├── test_vace.py +
│ │ │ │ └── test.py +
│ │ │ ├── passthrough
│ │ │ │ ├── __init__.py +
│ │ │ │ └── pipeline.py +
│ │ │ ├── reward_forcing
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ └── prepare_next.py +
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── causal_model.py +
│ │ │ │ │ └── model.py +
│ │ │ │ ├── __init__.py
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ └── test.py +
│ │ │ ├── streamdiffusionv2
│ │ │ │ ├── components
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── docs
│ │ │ │ │ └── usage.md
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── causal_model.py +
│ │ │ │ │ ├── model.py +
│ │ │ │ │ └── vae.py +
│ │ │ │ ├── __init__.py +
│ │ │ │ ├── model.yaml
│ │ │ │ ├── modular_blocks.py +
│ │ │ │ ├── pipeline.py +
│ │ │ │ ├── test_r2v.py +
│ │ │ │ └── test.py +
│ │ │ ├── wan2_1
│ │ │ │ ├── blocks
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── auto_prepare_latents.py +
│ │ │ │ │ ├── clean_kv_cache.py +
│ │ │ │ │ ├── decode.py +
│ │ │ │ │ ├── denoise.py +
│ │ │ │ │ ├── embedding_blending.py +
│ │ │ │ │ ├── noise_scale_controller.py +
│ │ │ │ │ ├── prepare_latents.py +
│ │ │ │ │ ├── prepare_next.py +
│ │ │ │ │ ├── prepare_video_latents.py +
│ │ │ │ │ ├── preprocess_video.py +
│ │ │ │ │ ├── set_timesteps.py +
│ │ │ │ │ ├── set_transformer_blocks_local_attn_size.py +
│ │ │ │ │ ├── setup_caches.py +
│ │ │ │ │ └── text_conditioning.py +
│ │ │ │ ├── components
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── generator.py +
│ │ │ │ │ ├── scheduler.py +
│ │ │ │ │ └── text_encoder.py +
│ │ │ │ ├── lora
│ │ │ │ │ ├── strategies
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ ├── module_targeted_lora.py +
│ │ │ │ │ │ ├── peft_lora.py +
│ │ │ │ │ │ └── permanent_merge_lora.py +
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── manager.py +
│ │ │ │ │ ├── mixin.py +
│ │ │ │ │ └── utils.py +
│ │ │ │ ├── modules
│ │ │ │ │ ├── __init__.py
│ │ │ │ │ ├── attention.py +
│ │ │ │ │ ├── clip.py +
│ │ │ │ │ ├── sage.py +
│ │ │ │ │ ├── t5.py +
│ │ │ │ │ ├── tokenizers.py +
│ │ │ │ │ └── xlm_roberta.py +
│ │ │ │ ├── vace
│ │ │ │ │ ├── blocks
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ └── vace_encoding.py +
│ │ │ │ │ ├── models
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ ├── attention_blocks.py +
│ │ │ │ │ │ └── causal_vace_model.py +
│ │ │ │ │ ├── utils
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ ├── encoding.py +
│ │ │ │ │ │ └── weight_loader.py +
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── design.md
│ │ │ │ │ └── mixin.py +
│ │ │ │ ├── vae
│ │ │ │ │ ├── modules
│ │ │ │ │ │ ├── __init__.py +
│ │ │ │ │ │ └── vae.py +
│ │ │ │ │ ├── __init__.py +
│ │ │ │ │ ├── constants.py +
│ │ │ │ │ └── wan.py +
│ │ │ │ ├── __init__.py
│ │ │ │ └── utils.py +
│ │ │ ├── __init__.py +
│ │ │ ├── blending.py +
│ │ │ ├── components.py +
│ │ │ ├── defaults.py +
│ │ │ ├── helpers.py +
│ │ │ ├── interface.py +
│ │ │ ├── memory.py +
│ │ │ ├── process.py +
│ │ │ ├── registry.py +
│ │ │ ├── schema.py +
│ │ │ ├── utils.py +
│ │ │ └── video.py +
│ │ ├── __init__.py
│ │ └── config.py +
│ ├── cli
│ │ ├── __init__.py
│ │ └── render_timeline.py +
│ ├── server
│ │ ├── spout
│ │ │ ├── __init__.py +
│ │ │ ├── receiver.py +
│ │ │ └── sender.py +
│ │ ├── __init__.py
│ │ ├── app.py +
│ │ ├── artifacts.py +
│ │ ├── build.py +
│ │ ├── credentials.py +
│ │ ├── download_models.py +
│ │ ├── download_progress_manager.py +
│ │ ├── frame_processor.py +
│ │ ├── logs_config.py +
│ │ ├── models_config.py +
│ │ ├── pipeline_artifacts.py +
│ │ ├── pipeline_manager.py +
│ │ ├── schema.py +
│ │ ├── tracks.py +
│ │ └── webrtc.py +
│ └── __init__.py
├── .github
│ ├── ISSUE_TEMPLATE
│ │ └── bug-report.yml
│ └── workflows
│ ├── build-electron.yml
│ ├── docker-build.yml
│ ├── lint.yml
│ └── test.yml
├── app
│ ├── assets
│ │ ├── icon.ico
│ │ ├── icon.png
│ │ └── tray-icon.png
│ ├── build
│ │ ├── linux
│ │ │ ├── after-install.sh
│ │ │ ├── after-remove.sh
│ │ │ └── daydream-scope.desktop
│ │ ├── mac
│ │ │ └── entitlements.mac.plist
│ │ ├── win
│ │ │ └── installer.nsh
│ │ ├── afterPack.js +
│ │ └── update-latest-yml.js +
│ ├── src
│ │ ├── components
│ │ │ ├── ErrorBoundary.tsx +
│ │ │ ├── LogViewer.html
│ │ │ ├── LogViewer.js
│ │ │ ├── ServerLoading.tsx +
│ │ │ └── Setup.tsx +
│ │ ├── services
│ │ │ ├── electronApp.ts +
│ │ │ ├── pythonProcess.ts +
│ │ │ └── setup.ts +
│ │ ├── types
│ │ │ ├── ipc.ts +
│ │ │ └── services.ts +
│ │ ├── utils
│ │ │ ├── config.test.ts +
│ │ │ ├── config.ts +
│ │ │ ├── logger.ts +
│ │ │ └── port.ts +
│ │ ├── index.css
│ │ ├── main.ts +
│ │ ├── preload.ts +
│ │ └── renderer.tsx +
│ ├── .gitignore
│ ├── dev-app-update.yml
│ ├── electron-builder.yml
│ ├── index.html
│ ├── package-lock.json
│ ├── package.json
│ ├── postcss.config.js +
│ ├── tsconfig.json
│ ├── tsconfig.node.json
│ ├── vite.config.ts +
│ ├── vite.main.config.ts +
│ ├── vite.preload.config.ts +
│ ├── vite.renderer.config.ts +
│ └── vitest.config.ts +
├── docs
│ ├── contributing.md
│ ├── lora.md
│ ├── server.md
│ ├── spout.md
│ └── vace.md
├── frontend
│ ├── assets
│ │ └── wireframe.png
│ ├── public
│ │ ├── assets
│ │ │ ├── discord-symbol-white.svg
│ │ │ └── github-mark-white.svg
│ │ ├── icon.svg
│ │ └── vite.svg
│ ├── src
│ │ ├── assets
│ │ │ └── react.svg
│ │ ├── components
│ │ │ ├── shared
│ │ │ │ ├── PromptField.tsx +
│ │ │ │ ├── TemporalTransitionControls.tsx +
│ │ │ │ └── WeightSlider.tsx +
│ │ │ ├── ui
│ │ │ │ ├── badge.tsx +
│ │ │ │ ├── button.tsx +
│ │ │ │ ├── card.tsx +
│ │ │ │ ├── debounced-slider.tsx +
│ │ │ │ ├── dialog.tsx +
│ │ │ │ ├── file-picker.tsx +
│ │ │ │ ├── input.tsx +
│ │ │ │ ├── label-with-tooltip.tsx +
│ │ │ │ ├── play-overlay.tsx +
│ │ │ │ ├── progress.tsx +
│ │ │ │ ├── select.tsx +
│ │ │ │ ├── slider-with-input.tsx +
│ │ │ │ ├── slider.tsx +
│ │ │ │ ├── sonner.tsx +
│ │ │ │ ├── spinner.tsx +
│ │ │ │ ├── textarea.tsx +
│ │ │ │ ├── toggle-group.tsx +
│ │ │ │ ├── toggle.tsx +
│ │ │ │ └── tooltip.tsx +
│ │ │ ├── DenoisingStepsSlider.tsx +
│ │ │ ├── DownloadDialog.tsx +
│ │ │ ├── Header.tsx +
│ │ │ ├── ImageManager.tsx +
│ │ │ ├── InputAndControlsPanel.tsx +
│ │ │ ├── LoRAManager.tsx +
│ │ │ ├── MediaPicker.tsx +
│ │ │ ├── PromptInput.tsx +
│ │ │ ├── PromptInputWithTimeline.tsx +
│ │ │ ├── PromptTimeline.tsx +
│ │ │ ├── ReportBugDialog.tsx +
│ │ │ ├── SettingsPanel.tsx +
│ │ │ ├── StatusBar.tsx +
│ │ │ ├── TimelinePromptEditor.tsx +
│ │ │ └── VideoOutput.tsx +
│ │ ├── data
│ │ │ ├── parameterMetadata.ts +
│ │ │ └── pipelines.ts +
│ │ ├── hooks
│ │ │ ├── useLocalSliderValue.ts +
│ │ │ ├── useLocalVideo.ts +
│ │ │ ├── usePipeline.ts +
│ │ │ ├── usePromptManager.ts +
│ │ │ ├── useStreamRecorder.ts +
│ │ │ ├── useStreamState.ts +
│ │ │ ├── useTimelinePlayback.ts +
│ │ │ ├── useVideoSource.ts +
│ │ │ ├── useWebRTC.ts +
│ │ │ └── useWebRTCStats.ts +
│ │ ├── lib
│ │ │ ├── api.ts +
│ │ │ └── utils.ts +
│ │ ├── pages
│ │ │ └── StreamPage.tsx +
│ │ ├── types
│ │ │ └── index.ts +
│ │ ├── utils
│ │ │ ├── loraHelpers.ts +
│ │ │ ├── promptColors.ts +
│ │ │ ├── promptWeights.ts +
│ │ │ └── timelinePromptSubmission.ts +
│ │ ├── App.css
│ │ ├── App.tsx +
│ │ ├── index.css
│ │ ├── main.tsx +
│ │ └── vite-env.d.ts
│ ├── .gitignore
│ ├── .prettierrc
│ ├── components.json
│ ├── eslint.config.js +
│ ├── index.html
│ ├── package-lock.json
│ ├── package.json
│ ├── postcss.config.js
│ ├── README.md
│ ├── tailwind.config.js
│ ├── tsconfig.app.json
│ ├── tsconfig.json
│ ├── tsconfig.node.json
│ └── vite.config.ts +
├── tests
│ ├── __init__.py
│ ├── test_logs_config.py +
│ └── test_render_timeline_cli.py +
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── build.sh
├── CLAUDE.md
├── Dockerfile
├── LICENSE.md
├── publish.sh
├── pyproject.toml
├── README.md
└── uv.lock

(* denotes selected files)
(+ denotes code-map available)
</file_map>
<file_contents>
File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/phase3-triton-rope-step2.md

```
md
# Phase 3 – Step 2 (Triton RoPE "3‑Way Lookup Fused") Spec

## Status (2025-12-23)

**IMPLEMENTED BUT REGRESSED**: Kernel works correctly but causes 20 → 17.8 FPS regression.

- **Kernel:** `src/scope/core/kernels/triton_rope_fused.py`
- **Integration:** Fast path in `rope_apply()` and `causal_rope_apply()`
- **Disable flag:** `SCOPE_DISABLE_TRITON_ROPE_FUSED=1` (use this until fixed)

**Root cause:** Power-of-2 padding. Triton `tl.arange()` requires power-of-2 ranges:
- Actual: C0=22, C1=21, C2=21 (64 pairs)
- Padded: 32, 32, 32 (96 pairs) = **50% more work**

**Fix needed:** Channel tiling or different approach to avoid padding overhead.

---

## Goal

Implement a **single Triton kernel** that applies RoPE for Krea Realtime Video by:

- Computing `(f_idx, h_idx, w_idx)` **inside the kernel** (no Python loops for indexing)
- Loading RoPE angles from **three small per-axis tables** (time/height/width)
- Rotating `x[..., 2*j:2*j+2]` in-place or out-of-place **without materializing `(seq_len, C)` cos/sin**
- Preserving tail (`t >= seq_len`) without cloning the whole tensor

Primary target: **B200 (SM100)**, steady-state shapes: `B=1`, `H=16`, `D=128`, `seq_len=f*h*w` with `L` padded to a multiple of 128.

## Current Baseline (what Step 2 replaces)

`rope_apply()` / `causal_rope_apply()` today:

- Split `freqs` into 3 chunks.
- Build `freqs_i` via `expand + cat + reshape` → then `cos/sin = freqs_i.real/imag`.
- Cache `(cos, sin)` in `_ROPE_CACHE` (Phase 2.5).
- Apply rotation (PyTorch) or call `triton_apply_rotary` (Phase 3 Step 0).
- If `seq_len 0 for `causal_rope_apply`)

### Internal cached tables (per device + freqs identity)

Instead of building `(seq_len, C)` cos/sin, build (and cache) these **float32** tables:

- `cos_f, sin_f`: `[max_seq_len, C0]` (time axis; rows indexed by `f_idx`)
- `cos_h, sin_h`: `[h, C1]` (height axis; rows indexed by `h_idx`)
- `cos_w, sin_w`: `[w, C2]` (width axis; rows indexed by `w_idx`)

Notes:
- Height/width tables do **not** need `max_seq_len` rows; slice to `[:h]`, `[:w]`.
- Cache key should include `freqs.data_ptr()`, device index, and chunk sizes (and/or `freqs.shape`).

### Output semantics

Provide three modes:

1) **In-place (preferred for inference):**
 - kernel writes `x[..., t _ROPE_CACHE_MAX:
 old = _ROPE_CACHE_ORDER.pop(0)
 _ROPE_CACHE.pop(old, None)

def get_rope_cos_sin(freqs, f, h, w, start_frame, dtype, device, c):
 """
 Get cached (cos, sin) for RoPE, building on cache miss.

 Args:
 freqs: Complex tensor tuple from freqs.split() - (freqs[0], freqs[1], freqs[2])
 f, h, w: Grid dimensions
 start_frame: Starting frame index (0 for rope_apply)
 dtype: Target dtype (bf16/fp16)
 device: Target device
 c: x.size(3)//2 after freq split

 Returns:
 (cos, sin): Tensors of shape (seq_len, 1, c)
 """
 key = _rope_cache_key(device, dtype, f, h, w, start_frame, c)
 cached = _rope_cache_get(key)
 if cached is not None:
 return cached

 # Cache miss - build freqs_i
 seq_len = f * h * w
 freqs_i = torch.cat(
 [
 freqs[0][start_frame : start_frame + f]
 .view(f, 1, 1, -1)
 .expand(f, h, w, -1),
 freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
 freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
 ],
 dim=-1,
 ).reshape(seq_len, 1, -1)

 # Extract cos/sin and convert to target dtype
 cos = freqs_i.real.to(device=device, dtype=dtype)
 sin = freqs_i.imag.to(device=device, dtype=dtype)

 _rope_cache_put(key, (cos, sin))
 return cos, sin

# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
 """Optimized RoPE using cached sin/cos (no float64, no complex math).

 Uses Triton fused 3-way kernel when available (Step 2).
 Falls back to Triton rotary (Step 0) or PyTorch.
 """
 # --- Fused Triton 3-way RoPE fast path (D=128 only) ---
 if USE_TRITON_ROPE_FUSED and x.is_cuda and x.ndim == 4 and x.shape[-1] == 128:
 try:
 return triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=0, inplace=None)
 except Exception:
 # Optional strict mode to surface kernel issues early:
 if os.environ.get("SCOPE_TRITON_ROPE_FUSED_STRICT", "0") == "1":
 raise
 # Otherwise fall back to existing implementation
 pass

 n, c = x.size(2), x.size(3) // 2

 # split freqs (freqs is complex: real=cos, imag=sin)
 freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

 if x.size(0) == 1:
 if isinstance(grid_sizes, torch.Tensor):
 f, h, w = (int(v) for v in grid_sizes[0].tolist())
 else:
 f, h, w = (int(v) for v in grid_sizes[0])
 seq_len = f * h * w

 cos, sin = get_rope_cos_sin(freqs_split, f, h, w, 0, x.dtype, x.device, c)
 cos_fa = cos.squeeze(1)
 sin_fa = sin.squeeze(1)

 if USE_TRITON_ROTARY:
 if seq_len == x.shape[1]:
 return triton_apply_rotary(x, cos_fa, sin_fa, interleaved=True)
 out = x.clone()
 out[:, :seq_len] = triton_apply_rotary(
 x[:, :seq_len], cos_fa, sin_fa, interleaved=True
 )
 return out

 x_i = x[0, :seq_len].reshape(seq_len, n, -1, 2)
 x0, x1 = x_i[..., 0], x_i[..., 1]
 x0_new = x0 * cos - x1 * sin
 x1_new = x0 * sin + x1 * cos
 x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(2).unsqueeze(0)
 if seq_len == x.shape[1]:
 return x_i
 out = x.clone()
 out[:, :seq_len] = x_i
 return out

 out = x.clone()
 for i, (f, h, w) in enumerate(grid_sizes.tolist()):
 seq_len = f * h * w
 cos, sin = get_rope_cos_sin(freqs_split, f, h, w, 0, x.dtype, x.device, c)

 if USE_TRITON_ROTARY:
 cos_fa = cos.squeeze(1)
 sin_fa = sin.squeeze(1)
 out[i : i + 1, :seq_len] = triton_apply_rotary(
 x[i : i + 1, :seq_len], cos_fa, sin_fa, interleaved=True
 )
 continue

 x_i = x[i, :seq_len].reshape(seq_len, n, -1, 2)
 x0, x1 = x_i[..., 0], x_i[..., 1]
 x0_new = x0 * cos - x1 * sin
 x1_new = x0 * sin + x1 * cos
 out[i, :seq_len] = torch.stack([x0_new, x1_new], dim=-1).flatten(2)
 return out

class WanRMSNorm(nn.Module):

 def __init__(self, dim, eps=1e-5):
 super().__init__()
 self.dim = dim
 self.eps = eps
 self.weight = nn.Parameter(torch.ones(dim))

 def forward(self, x):
 r"""
 Args:
 x(Tensor): Shape [B, L, C]
 """
 return self._norm(x.float()).type_as(x) * self.weight

 def _norm(self, x):
 return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class WanLayerNorm(nn.LayerNorm):

 def __init__(self, dim, eps=1e-6, elementwise_affine=False):
 super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

 def forward(self, x):
 r"""
 Args:
 x(Tensor): Shape [B, L, C]
 """
 return super().forward(x).type_as(x)

class WanSelfAttention(nn.Module):

 def __init__(self,
 dim,
 num_heads,
 window_size=(-1, -1),
 qk_norm=True,
 eps=1e-6):
 assert dim % num_heads == 0
 super().__init__()
 self.dim = dim
 self.num_heads = num_heads
 self.head_dim = dim // num_heads
 self.window_size = window_size
 self.qk_norm = qk_norm
 self.eps = eps

 # layers
 self.q = nn.Linear(dim, dim)
 self.k = nn.Linear(dim, dim)
 self.v = nn.Linear(dim, dim)
 self.o = nn.Linear(dim, dim)
 self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
 self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

 def forward(self, x, seq_lens, grid_sizes, freqs):
 r"""
 Args:
 x(Tensor): Shape [B, L, num_heads, C / num_heads]
 seq_lens(Tensor): Shape [B]
 grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
 freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
 """
 b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

 # query, key, value function
 def qkv_fn(x):
 q = self.norm_q(self.q(x)).view(b, s, n, d)
 k = self.norm_k(self.k(x)).view(b, s, n, d)
 v = self.v(x).view(b, s, n, d)
 return q, k, v

 q, k, v = qkv_fn(x)

 if SAGEATTN_AVAILABLE:
 # print("Using sageattention in crossattn")
 og_dtype = q.dtype
 q = q.transpose(1, 2).to(dtype)
 k = k.transpose(1, 2).to(dtype)
 v = v.transpose(1, 2).to(dtype)
 x = sageattn_func(
 q=rope_apply(q, grid_sizes, freqs),
 k=rope_apply(k, grid_sizes, freqs),
 v=v,
 )
 x = x.transpose(1, 2).contiguous().to(og_dtype)
 else:
 x = flash_attention(
 q=rope_apply(q, grid_sizes, freqs),
 k=rope_apply(k, grid_sizes, freqs),
 v=v,
 k_lens=seq_lens,
 window_size=self.window_size)

 # output
 x = x.flatten(2)
 x = self.o(x)
 return x

class WanT2VCrossAttention(WanSelfAttention):

 def forward(self, x, context, context_lens, crossattn_cache=None):
 r"""
 Args:
 x(Tensor): Shape [B, L1, C]
 context(Tensor): Shape [B, L2, C]
 context_lens(Tensor): Shape [B]
 crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
 """
 b, n, d = x.size(0), self.num_heads, self.head_dim

 # compute query, key, value
 q = self.norm_q(self.q(x)).view(b, -1, n, d)

 if crossattn_cache is not None:
 if not crossattn_cache["is_init"]:
 crossattn_cache["is_init"] = True
 k = self.norm_k(self.k(context)).view(b, -1, n, d)
 v = self.v(context).view(b, -1, n, d)
 crossattn_cache["k"] = k
 crossattn_cache["v"] = v
 else:
 k = crossattn_cache["k"]
 v = crossattn_cache["v"]
 else:
 k = self.norm_k(self.k(context)).view(b, -1, n, d)
 v = self.v(context).view(b, -1, n, d)

 # compute attention
 if SAGEATTN_AVAILABLE:
 # print("Using sageattention in crossattn")
 dtype = torch.bfloat16
 og_dtype = q.dtype
 q = q.transpose(1, 2).to(dtype)
 k = k.transpose(1, 2).to(dtype)
 v = v.transpose(1, 2).to(dtype)
 x = sageattn_func(
 q=q,
 k=k,
 v=v,
 )
 x = x.transpose(1, 2).contiguous().to(og_dtype)
 elif FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
 x = flash_attention(q, k, v, k_lens=context_lens)
 else:
 dtype = torch.bfloat16
 q = q.transpose(1, 2).to(dtype)
 k = k.transpose(1, 2).to(dtype)
 v = v.transpose(1, 2).to(dtype)

 x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
 x = x.transpose(1, 2).contiguous()

 # output
 x = x.flatten(2)
 x = self.o(x)
 return x

class WanGanCrossAttention(WanSelfAttention):

 def forward(self, x, context, crossattn_cache=None):
 r"""
 Args:
 x(Tensor): Shape [B, L1, C]
 context(Tensor): Shape [B, L2, C]
 context_lens(Tensor): Shape [B]
 crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
 """
 b, n, d = x.size(0), self.num_heads, self.head_dim

 # compute query, key, value
 qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

 kk = self.norm_k(self.k(x)).view(b, -1, n, d)
 vv = self.v(x).view(b, -1, n, d)

 # compute attention
 x = flash_attention(qq, kk, vv)

 # output
 x = x.flatten(2)
 x = self.o(x)
 return x

class WanI2VCrossAttention(WanSelfAttention):

 def __init__(self,
 dim,
 num_heads,
 window_size=(-1, -1),
 qk_norm=True,
 eps=1e-6):
 super().__init__(dim, num_heads, window_size, qk_norm, eps)

 self.k_img = nn.Linear(dim, dim)
 self.v_img = nn.Linear(dim, dim)
 # self.alpha = nn.Parameter(torch.zeros((1, )))
 self.norm_k_img = WanRMSNorm(
 dim, eps=eps) if qk_norm else nn.Identity()

 def forward(self, x, context, context_lens):
 r"""
 Args:
 x(Tensor): Shape [B, L1, C]
 context(Tensor): Shape [B, L2, C]
 context_lens(Tensor): Shape [B]
 """
 context_img = context[:, :257]
 context = context[:, 257:]
 b, n, d = x.size(0), self.num_heads, self.head_dim

 # compute query, key, value
 q = self.norm_q(self.q(x)).view(b, -1, n, d)
 k = self.norm_k(self.k(context)).view(b, -1, n, d)
 v = self.v(context).view(b, -1, n, d)
 k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
 v_img = self.v_img(context_img).view(b, -1, n, d)
 img_x = flash_attention(q, k_img, v_img, k_lens=None)
 # compute attention
 x = flash_attention(q, k, v, k_lens=context_lens)

 # output
 x = x.flatten(2)
 img_x = img_x.flatten(2)
 x = x + img_x
 x = self.o(x)
 return x

WAN_CROSSATTENTION_CLASSES = {
 't2v_cross_attn': WanT2VCrossAttention,
 'i2v_cross_attn': WanI2VCrossAttention,
}

class WanAttentionBlock(nn.Module):

 def __init__(self,
 cross_attn_type,
 dim,
 ffn_dim,
 num_heads,
 window_size=(-1, -1),
 qk_norm=True,
 cross_attn_norm=False,
 eps=1e-6):
 super().__init__()
 self.dim = dim
 self.ffn_dim = ffn_dim
 self.num_heads = num_heads
 self.window_size = window_size
 self.qk_norm = qk_norm
 self.cross_attn_norm = cross_attn_norm
 self.eps = eps

 # layers
 self.norm1 = WanLayerNorm(dim, eps)
 self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
 eps)
 self.norm3 = WanLayerNorm(
 dim, eps,
 elementwise_affine=True) if cross_attn_norm else nn.Identity()
 self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
 num_heads,
 (-1, -1),
 qk_norm,
 eps)
 self.norm2 = WanLayerNorm(dim, eps)
 self.ffn = nn.Sequential(
 nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
 nn.Linear(ffn_dim, dim))

 # modulation
 self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

 def forward(
 self,
 x,
 e,
 seq_lens,
 grid_sizes,
 freqs,
 context,
 context_lens,
 ):
 r"""
 Args:
 x(Tensor): Shape [B, L, C]
 e(Tensor): Shape [B, 6, C]
 seq_lens(Tensor): Shape [B], length of each sequence in batch
 grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
 freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
 """
 # assert e.dtype == torch.float32
 # with amp.autocast(dtype=torch.float32):
 e = (self.modulation + e).chunk(6, dim=1)
 # assert e[0].dtype == torch.float32

 # self-attention
 y = self.self_attn(
 self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
 freqs)
 # with amp.autocast(dtype=torch.float32):
 x = x + y * e[2]

 # cross-attention & ffn function
 def cross_attn_ffn(x, context, context_lens, e):
 x = x + self.cross_attn(self.norm3(x), context, context_lens)
 y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
 # with amp.autocast(dtype=torch.float32):
 x = x + y * e[5]
 return x

 x = cross_attn_ffn(x, context, context_lens, e)
 return x

class GanAttentionBlock(nn.Module):

 def __init__(self,
 dim=1536,
 ffn_dim=8192,
 num_heads=12,
 window_size=(-1, -1),
 qk_norm=True,
 cross_attn_norm=True,
 eps=1e-6):
 super().__init__()
 self.dim = dim
 self.ffn_dim = ffn_dim
 self.num_heads = num_heads
 self.window_size = window_size
 self.qk_norm = qk_norm
 self.cross_attn_norm = cross_attn_norm
 self.eps = eps

 # layers
 # self.norm1 = WanLayerNorm(dim, eps)
 # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
 # eps)
 self.norm3 = WanLayerNorm(
 dim, eps,
 elementwise_affine=True) if cross_attn_norm else nn.Identity()

 self.norm2 = WanLayerNorm(dim, eps)
 self.ffn = nn.Sequential(
 nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
 nn.Linear(ffn_dim, dim))

 self.cross_attn = WanGanCrossAttention(dim, num_heads,
 (-1, -1),
 qk_norm,
 eps)

 # modulation
 # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

 def forward(
 self,
 x,
 context,
 # seq_lens,
 # grid_sizes,
 # freqs,
 # context,
 # context_lens,
 ):
 r"""
 Args:
 x(Tensor): Shape [B, L, C]
 e(Tensor): Shape [B, 6, C]
 seq_lens(Tensor): Shape [B], length of each sequence in batch
 grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
 freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
 """
 # assert e.dtype == torch.float32
 # with amp.autocast(dtype=torch.float32):
 # e = (self.modulation + e).chunk(6, dim=1)
 # assert e[0].dtype == torch.float32

 # # self-attention
 # y = self.self_attn(
 # self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
 # freqs)
 # # with amp.autocast(dtype=torch.float32):
 # x = x + y * e[2]

 # cross-attention & ffn function
 def cross_attn_ffn(x, context):
 token = context + self.cross_attn(self.norm3(x), context)
 y = self.ffn(self.norm2(token)) + token # * (1 + e[4]) + e[3])
 # with amp.autocast(dtype=torch.float32):
 # x = x + y * e[5]
 return y

 x = cross_attn_ffn(x, context)
 return x

class Head(nn.Module):

 def __init__(self, dim, out_dim, patch_size, eps=1e-6):
 super().__init__()
 self.dim = dim
 self.out_dim = out_dim
 self.patch_size = patch_size
 self.eps = eps

 # layers
 out_dim = math.prod(patch_size) * out_dim
 self.norm = WanLayerNorm(dim, eps)
 self.head = nn.Linear(dim, out_dim)

 # modulation
 self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

 def forward(self, x, e):
 r"""
 Args:
 x(Tensor): Shape [B, L1, C]
 e(Tensor): Shape [B, C]
 """
 # assert e.dtype == torch.float32
 # with amp.autocast(dtype=torch.float32):
 e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
 x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
 return x

class MLPProj(torch.nn.Module):

 def __init__(self, in_dim, out_dim):
 super().__init__()

 self.proj = torch.nn.Sequential(
 torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
 torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
 torch.nn.LayerNorm(out_dim))

 def forward(self, image_embeds):
 clip_extra_context_tokens = self.proj(image_embeds)
 return clip_extra_context_tokens

class RegisterTokens(nn.Module):
 def __init__(self, num_registers: int, dim: int):
 super().__init__()
 self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
 self.rms_norm = WanRMSNorm(dim, eps=1e-6)

 def forward(self):
 return self.rms_norm(self.register_tokens)

 def reset_parameters(self):
 nn.init.normal_(self.register_tokens, std=0.02)

class WanModel(ModelMixin, ConfigMixin):
 r"""
 Wan diffusion backbone supporting both text-to-video and image-to-video.
 """

 ignore_for_config = [
 'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
 ]
 _no_split_modules = ['WanAttentionBlock']
 _supports_gradient_checkpointing = True

 @register_to_config
 def __init__(self,
 model_type='t2v',
 patch_size=(1, 2, 2),
 text_len=512,
 in_dim=16,
 dim=2048,
 ffn_dim=8192,
 freq_dim=256,
 text_dim=4096,
 out_dim=16,
 num_heads=16,
 num_layers=32,
 window_size=(-1, -1),
 qk_norm=True,
 cross_attn_norm=True,
 eps=1e-6):
 r"""
 Initialize the diffusion model backbone.

 Args:
 model_type (`str`, *optional*, defaults to 't2v'):
 Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
 patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
 3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
 text_len (`int`, *optional*, defaults to 512):
 Fixed length for text embeddings
 in_dim (`int`, *optional*, defaults to 16):
 Input video channels (C_in)
 dim (`int`, *optional*, defaults to 2048):
 Hidden dimension of the transformer
 ffn_dim (`int`, *optional*, defaults to 8192):
 Intermediate dimension in feed-forward network
 freq_dim (`int`, *optional*, defaults to 256):
 Dimension for sinusoidal time embeddings
 text_dim (`int`, *optional*, defaults to 4096):
 Input dimension for text embeddings
 out_dim (`int`, *optional*, defaults to 16):
 Output video channels (C_out)
 num_heads (`int`, *optional*, defaults to 16):
 Number of attention heads
 num_layers (`int`, *optional*, defaults to 32):
 Number of transformer blocks
 window_size (`tuple`, *optional*, defaults to (-1, -1)):
 Window size for local attention (-1 indicates global attention)
 qk_norm (`bool`, *optional*, defaults to True):
 Enable query/key normalization
 cross_attn_norm (`bool`, *optional*, defaults to False):
 Enable cross-attention normalization
 eps (`float`, *optional*, defaults to 1e-6):
 Epsilon value for normalization layers
 """

 super().__init__()

 assert model_type in ['t2v', 'i2v']
 self.model_type = model_type

 self.patch_size = patch_size
 self.text_len = text_len
 self.in_dim = in_dim
 self.dim = dim
 self.ffn_dim = ffn_dim
 self.freq_dim = freq_dim
 self.text_dim = text_dim
 self.out_dim = out_dim
 self.num_heads = num_heads
 self.num_layers = num_layers
 self.window_size = window_size
 self.qk_norm = qk_norm
 self.cross_attn_norm = cross_attn_norm
 self.eps = eps
 self.local_attn_size = 21

 # embeddings
 self.patch_embedding = nn.Conv3d(
 in_dim, dim, kernel_size=patch_size, stride=patch_size)
 self.text_embedding = nn.Sequential(
 nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
 nn.Linear(dim, dim))

 self.time_embedding = nn.Sequential(
 nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
 self.time_projection = nn.Sequential(
 nn.SiLU(), nn.Linear(dim, dim * 6))

 # blocks
 cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
 self.blocks = nn.ModuleList([
 WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
 window_size, qk_norm, cross_attn_norm, eps)
 for _ in range(num_layers)
 ])

 # head
 self.head = Head(dim, out_dim, patch_size, eps)

 # buffers (don't use register_buffer otherwise dtype will be changed in to())
 assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
 d = dim // num_heads
 self.freqs = torch.cat([
 # rope_params(1024, d - 4 * (d // 6)),
 rope_params_riflex(1024, d - 4 * (d // 6), ),
 rope_params(1024, 2 * (d // 6)),
 rope_params(1024, 2 * (d // 6))
 ],
 dim=1)

 if model_type == 'i2v':
 self.img_emb = MLPProj(1280, dim)

 # initialize weights
 self.init_weights()

 self.gradient_checkpointing = False

 def _set_gradient_checkpointing(self, module, value=False):
 self.gradient_checkpointing = value

 def forward(
 self,
 *args,
 **kwargs
 ):
 # if kwargs.get('classify_mode', False) is True:
 # kwargs.pop('classify_mode')
 # return self._forward_classify(*args, **kwargs)
 # else:
 return self._forward(*args, **kwargs)

 def _forward(
 self,
 x,
 t,
 context,
 seq_len,
 classify_mode=False,
 concat_time_embeddings=False,
 register_tokens=None,
 cls_pred_branch=None,
 gan_ca_blocks=None,
 clip_fea=None,
 y=None,
 ):
 r"""
 Forward pass through the diffusion model

 Args:
 x (List[Tensor]):
 List of input video tensors, each with shape [C_in, F, H, W]
 t (Tensor):
 Diffusion timesteps tensor of shape [B]
 context (List[Tensor]):
 List of text embeddings each with shape [L, C]
 seq_len (`int`):
 Maximum sequence length for positional encoding
 clip_fea (Tensor, *optional*):
 CLIP image features for image-to-video mode
 y (List[Tensor], *optional*):
 Conditional video inputs for image-to-video mode, same shape as x

 Returns:
 List[Tensor]:
 List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
 """
 if self.model_type == 'i2v':
 assert clip_fea is not None and y is not None
 # params
 device = self.patch_embedding.weight.device
 if self.freqs.device != device:
 self.freqs = self.freqs.to(device)

 if y is not None:
 x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

 # embeddings
 x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
 grid_sizes = torch.stack(
 [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
 x = [u.flatten(2).transpose(1, 2) for u in x]
 seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
 assert seq_lens.max() b n d", b=x.shape[0])
 # x = torch.cat([registers, x], dim=1)

 gan_idx = 0
 for ii, block in enumerate(self.blocks):
 if torch.is_grad_enabled() and self.gradient_checkpointing:
 x = torch.utils.checkpoint.checkpoint(
 create_custom_forward(block),
 x, **kwargs,
 use_reentrant=False,
 )
 else:
 x = block(x, **kwargs)

 if classify_mode and ii in [13, 21, 29]:
 gan_token = registers[:, gan_idx: gan_idx + 1]
 final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
 gan_idx += 1

 if classify_mode:
 final_x = torch.cat(final_x, dim=1)
 if concat_time_embeddings:
 final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
 else:
 final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

 # head
 x = self.head(x, e)

 # unpatchify
 x = self.unpatchify(x, grid_sizes)

 if classify_mode:
 return torch.stack(x), final_x

 return torch.stack(x)

 def _forward_classify(
 self,
 x,
 t,
 context,
 seq_len,
 register_tokens,
 cls_pred_branch,
 clip_fea=None,
 y=None,
 ):
 r"""
 Feature extraction through the diffusion model

 Args:
 x (List[Tensor]):
 List of input video tensors, each with shape [C_in, F, H, W]
 t (Tensor):
 Diffusion timesteps tensor of shape [B]
 context (List[Tensor]):
 List of text embeddings each with shape [L, C]
 seq_len (`int`):
 Maximum sequence length for positional encoding
 clip_fea (Tensor, *optional*):
 CLIP image features for image-to-video mode
 y (List[Tensor], *optional*):
 Conditional video inputs for image-to-video mode, same shape as x

 Returns:
 List[Tensor]:
 List of video features with original input shapes [C_block, F, H / 8, W / 8]
 """
 if self.model_type == 'i2v':
 assert clip_fea is not None and y is not None
 # params
 device = self.patch_embedding.weight.device
 if self.freqs.device != device:
 self.freqs = self.freqs.to(device)

 if y is not None:
 x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

 # embeddings
 x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
 grid_sizes = torch.stack(
 [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
 x = [u.flatten(2).transpose(1, 2) for u in x]
 seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
 assert seq_lens.max() cfphqwr', u)
 u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
 out.append(u)
 return out

 def init_weights(self):
 r"""
 Initialize model parameters using Xavier initialization.
 """

 # basic init
 for m in self.modules():
 if isinstance(m, nn.Linear):
 nn.init.xavier_uniform_(m.weight)
 if m.bias is not None:
 nn.init.zeros_(m.bias)

 # init embeddings
 nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
 for m in self.text_embedding.modules():
 if isinstance(m, nn.Linear):
 nn.init.normal_(m.weight, std=.02)
 for m in self.time_embedding.modules():
 if isinstance(m, nn.Linear):
 nn.init.normal_(m.weight, std=.02)

 # init output layer
 nn.init.zeros_(self.head.head.weight)
```

File: /Users/daviddickinson/Projects/Lora/scope/notes/FA4/DeepResearch/2025-12-22/Phase3_01.md

```
md
Got it - I've updated the plan to focus on drafting the fused kernel, wrapper, and feature flag integration only.

Below is a concrete “Option 1” drop-in: **new fused Triton kernel module + wrapper + feature-flag integration** for **D=128 fast-path only**, with:

* **3-way axis lookup** (time/height/width) inside the kernel (no `(seq_len, C)` materialization)
* **module-scope axis-table cache** (shared between model + causal model)
* **in-place rotation** when `not torch.is_grad_enabled()` **and** `seq_len == L`
* **tail preservation** via `out = x.clone()` only when needed (`seq_len x.shape[1]` and raises → you fall back. (Today SAGEATTN is false, but this prevents silent corruption later.)
* **Tail preservation**:

 * `seq_len == L`: no clone, out is fully written
 * `seq_len argparse.Namespace:
 parser = argparse.ArgumentParser(description="Benchmark Triton RoPE vs PyTorch fallback.")
 parser.add_argument("--f", type=int, default=3, help="Frames (F)")
 parser.add_argument("--h", type=int, default=30, help="Height tokens (H)")
 parser.add_argument("--w", type=int, default=52, help="Width tokens (W)")
 parser.add_argument("--heads", type=int, default=16, help="Num heads")
 parser.add_argument("--head-dim", type=int, default=128, help="Head dim (D)")
 parser.add_argument("--start-frame", type=int, default=0, help="Start frame (causal)")
 parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Input dtype")
 parser.add_argument("--pad", type=int, default=0, help="Extra tail tokens (right pad).")
 parser.add_argument(
 "--pad-to-multiple",
 type=int,
 default=0,
 help="Right-pad so (seq_len+pad) is a multiple of this value (added on top of --pad).",
 )
 parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
 parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
 parser.add_argument(
 "--disable-triton-rotary",
 action="store_true",
 help="Set SCOPE_DISABLE_TRITON_ROTARY=1 (must happen before imports).",
 )
 return parser.parse_args()

def build_freqs(max_seq_len: int, head_dim: int, device: torch.device, rope_params_fn) -> torch.Tensor:
 """Build freqs like CausalWanModel (rope_params only)."""
 d = head_dim
 c0_dim = d - 4 * (d // 6)
 c1_dim = 2 * (d // 6)
 c2_dim = 2 * (d // 6)
 freqs = torch.cat(
 [
 rope_params_fn(max_seq_len, c0_dim),
 rope_params_fn(max_seq_len, c1_dim),
 rope_params_fn(max_seq_len, c2_dim),
 ],
 dim=1,
 )
 return freqs.to(device)

def rope_torch(x_slice: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
 """Reference PyTorch RoPE using pre-materialized cos/sin."""
 seq_len, n, d = x_slice.shape[1], x_slice.shape[2], x_slice.shape[3]
 c = d // 2
 if cos.dim() == 2:
 cos = cos.unsqueeze(1)
 if sin.dim() == 2:
 sin = sin.unsqueeze(1)
 x_i = x_slice.reshape(1, seq_len, n, c, 2)
 x0, x1 = x_i[..., 0], x_i[..., 1]
 x0_new = x0 * cos - x1 * sin
 x1_new = x0 * sin + x1 * cos
 x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(3)
 return x_i

def main() -> None:
 args = parse_args()
 if not torch.cuda.is_available():
 raise SystemExit("CUDA is required for this benchmark.")

 if args.disable_triton_rotary:
 os.environ["SCOPE_DISABLE_TRITON_ROTARY"] = "1"

 from scope.core.pipelines.krea_realtime_video.modules.model import (
 get_rope_cos_sin,
 rope_apply,
 rope_params,
 USE_TRITON_ROTARY,
 )
 from scope.core.pipelines.krea_realtime_video.modules.causal_model import (
 causal_rope_apply,
 )

 try:
 from scope.core.kernels.triton_rotary import apply_rotary as triton_apply_rotary
 TRITON_AVAILABLE = True
 except ImportError:
 triton_apply_rotary = None
 TRITON_AVAILABLE = False

 # Optional future kernel (Step 2) – harness should work the moment it lands.
 try:
 from scope.core.kernels.triton_rope_fused import rope_fused_3way as triton_rope_fused_3way
 TRITON_FUSED_AVAILABLE = True
 except Exception:
 triton_rope_fused_3way = None
 TRITON_FUSED_AVAILABLE = False

 dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
 device = torch.device("cuda")

 f, h, w = args.f, args.h, args.w
 seq_len = f * h * w
 pad_to_multiple = int(args.pad_to_multiple)
 pad_mult = 0
 if pad_to_multiple > 0:
 pad_mult = math.ceil(seq_len / pad_to_multiple) * pad_to_multiple - seq_len
 b, n, d = 1, args.heads, args.head_dim
 c = d // 2

 freqs = build_freqs(1024, d, device, rope_params)
 freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
 cos, sin = get_rope_cos_sin(
 freqs_split, f, h, w, args.start_frame, dtype, device, c
 )
 cos_fa = cos.squeeze(1).contiguous()
 sin_fa = sin.squeeze(1).contiguous()

 pad_total = int(args.pad) + int(pad_mult)
 x = torch.randn(b, seq_len + pad_total, n, d, device=device, dtype=dtype)
 grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long, device="cpu")

 print(f"USE_TRITON_ROTARY={USE_TRITON_ROTARY}")
 print(
 f"shape: B={b} L={seq_len}+{pad_total} (pad={args.pad} pad_mult={pad_mult}) H={n} D={d} dtype={args.dtype}"
 )
 print(f"TRITON_AVAILABLE={TRITON_AVAILABLE} TRITON_FUSED_AVAILABLE={TRITON_FUSED_AVAILABLE}")

 # Correctness check
 torch_out = rope_torch(x[:, :seq_len], cos, sin)
 if TRITON_AVAILABLE:
 triton_out = triton_apply_rotary(x[:, :seq_len], cos_fa, sin_fa, interleaved=True)
 max_err = (triton_out - torch_out).abs().max().item()
 mean_err = (triton_out - torch_out).abs().mean().item()
 print(f"Correctness: max_err={max_err:.6f} mean_err={mean_err:.6f}")
 else:
 print("Triton rotary not available; skipping correctness against Triton.")

 # Benchmark
 def bench(fn, label: str) -> float:
 for _ in range(args.warmup):
 _ = fn()
 torch.cuda.synchronize()
 torch.cuda.nvtx.range_push(label)
 start = time.perf_counter()
 for _ in range(args.iters):
 _ = fn()
 torch.cuda.synchronize()
 ms = (time.perf_counter() - start) / args.iters * 1000.0
 torch.cuda.nvtx.range_pop()
 print(f"{label}: {ms:.3f} ms")
 return ms

 # Kernel-only baselines (prefix only)
 bench(lambda: rope_torch(x[:, :seq_len], cos, sin), "PyTorch (prefix only)")
 if TRITON_AVAILABLE:
 bench(
 lambda: triton_apply_rotary(x[:, :seq_len], cos_fa, sin_fa, interleaved=True),
 "Triton rotary (prefix only)",
 )
 else:
 print("Triton rotary not available; skipping Triton benchmark.")

 # End-to-end wrappers (include cache + tail handling)
 bench(lambda: rope_apply(x, grid_sizes, freqs), "rope_apply (end-to-end)")
 bench(
 lambda: causal_rope_apply(x, grid_sizes, freqs, start_frame=args.start_frame),
 f"causal_rope_apply(sf={args.start_frame}) (end-to-end)",
 )

 # Future: Step 2 fused wrapper direct call (if/when it exists)
 if TRITON_FUSED_AVAILABLE and triton_rope_fused_3way is not None:
 bench(
 lambda: triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=args.start_frame, inplace=None),
 "triton_rope_fused_3way (direct)",
 )
 else:
 print("Triton fused RoPE wrapper not available; skipping Step-2 benchmark.")

if __name__ == "__main__":
 main()
```

File: /Users/daviddickinson/Projects/Lora/scope/src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py

```
py
# Modified from https://github.com/krea-ai/realtime-video
import atexit
import functools
import logging
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.attention.flex_attention import (
 BlockMask,
 create_block_mask,
 flex_attention,
)

from scope.core.pipelines.wan2_1.modules.attention import attention

logger = logging.getLogger(__name__)

# Triton Kernel B: 10.7% faster than flex_attention for KV-cache bias path
# Set USE_TRITON_KERNEL_B=1 to enable
USE_TRITON_KERNEL_B = os.getenv("USE_TRITON_KERNEL_B", "1") == "1"
_triton_kernel_b = None

if USE_TRITON_KERNEL_B:
 try:
 from scope.core.kernels import triton_kernel_b as _triton_kernel_b
 logger.info("Triton Kernel B enabled for KV-cache attention bias")
 except ImportError as e:
 logger.warning(f"Triton Kernel B not available: {e}")
 USE_TRITON_KERNEL_B = False

# Simple profiler for timing operations
# Note: Profiling is incompatible with torch.compile - disable if compile is enabled
_PROFILE_ENABLED = os.getenv("PROFILE_ATTENTION", "0") == "1"
_profile_times = defaultdict(float)
_profile_counts = defaultdict(int)
_profile_call_count = 0

def _should_profile():
 """Check if profiling should run (disabled during torch.compile tracing)."""
 if not _PROFILE_ENABLED:
 return False
 # torch.compiler.is_compiling() returns True during dynamo tracing
 if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
 if torch.compiler.is_compiling():
 return False
 return True

def profile_report():
 """Print profiling report."""
 if not _profile_times:
 return
 total = sum(_profile_times.values())
 logger.info("=== Attention Profiling Report ===")
 for name, time_ms in sorted(_profile_times.items(), key=lambda x: -x[1]):
 count = _profile_counts[name]
 pct = 100 * time_ms / total if total > 0 else 0
 logger.info(f" {name}: {time_ms:.1f}ms ({pct:.1f}%) [{count} calls, {time_ms/count:.2f}ms/call]")
 logger.info(f" TOTAL: {total:.1f}ms")

 # Compute p_bias vs p_recompute
 bias_time = _profile_times.get("self_attn_kv_bias", 0)
 recompute_time = _profile_times.get("self_attn_block_mask", 0)
 plain_time = _profile_times.get("self_attn_kv_plain", 0)
 attn_total = bias_time + recompute_time + plain_time
 if attn_total > 0:
 logger.info("=== p_bias vs p_recompute ===")
 logger.info(f" p_bias (Kernel B): {100 * bias_time / attn_total:.1f}%")
 logger.info(f" p_recompute (Kernel A): {100 * recompute_time / attn_total:.1f}%")
 logger.info(f" p_plain (FA path): {100 * plain_time / attn_total:.1f}%")

# Register atexit handler to print profiling report on exit
atexit.register(profile_report)

class _ProfileBlock:
 """Context manager for profiling a code block."""
 __slots__ = ('name', 'start_event', 'end_event')

 def __init__(self, name: str):
 self.name = name
 self.start_event = None
 self.end_event = None

 def __enter__(self):
 if _should_profile():
 self.start_event = torch.cuda.Event(enable_timing=True)
 self.end_event = torch.cuda.Event(enable_timing=True)
 self.start_event.record()
 return self

 def __exit__(self, *args):
 if self.start_event is not None:
 self.end_event.record()
 torch.cuda.synchronize()
 elapsed_ms = self.start_event.elapsed_time(self.end_event)
 _profile_times[self.name] += elapsed_ms
 _profile_counts[self.name] += 1

from .model import (
 USE_TRITON_ROTARY,
 USE_TRITON_ROPE_FUSED,
 WAN_CROSSATTENTION_CLASSES,
 MLPProj,
 WanLayerNorm,
 WanRMSNorm,
 get_rope_cos_sin,
 rope_apply,
 rope_params,
 sinusoidal_embedding_1d,
 triton_apply_rotary,
 triton_rope_fused_3way,
)

flex_attention = torch.compile(
 flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
)

# Constants for flex_attention operations
FLEX_ATTENTION_ALIGNMENT = 128
KV_CACHE_ATTENTION_BIAS_DISABLED = 1.0

def _pad_tensor_for_flex_attention(
 tensor: torch.Tensor, target_length: int, pad_dim: int = 1
) -> torch.Tensor:
 """Pad tensor to target_length along pad_dim. Returns original if no padding needed."""
 current_length = tensor.shape[pad_dim]
 padded_length = target_length - current_length

 if padded_length = local_window_start)) | (
 q_indices == kv_indices
 )

 if dtype == torch.bool:
 return mask
 elif dtype == torch.float32 or dtype == torch.float16:
 # Convert to additive mask (0.0 for attend, -inf for mask)
 return mask.float() * 0.0 + (~mask).float() * float("-inf")
 else:
 raise ValueError(f"Unsupported dtype: {dtype}")

@functools.lru_cache(maxsize=32)
def get_block_mask(
 device: str,
 num_frames: int = 21,
 frame_seqlen: int = 1560,
 num_frame_per_block=3,
 local_attn_size=-1,
):
 total_length = num_frames * frame_seqlen

 # we do right padding to get to a multiple of 128
 padded_length = math.ceil(total_length / 128) * 128 - total_length

 ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

 # Block-wise causal mask will attend to all elements that are before the end of the current chunk
 frame_indices = torch.arange(
 start=0,
 end=total_length,
 step=frame_seqlen * num_frame_per_block,
 device=device,
 )

 for tmp in frame_indices:
 ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
 tmp + frame_seqlen * num_frame_per_block
 )

 def attention_mask(b, h, q_idx, kv_idx):
 if local_attn_size == -1:
 return (kv_idx = (ends[q_idx] - local_attn_size * frame_seqlen))
 ) | (q_idx == kv_idx)

 block_mask = create_block_mask(
 attention_mask,
 B=None,
 H=None,
 Q_LEN=total_length + padded_length,
 KV_LEN=total_length + padded_length,
 _compile=False,
 device=device,
 )
 return block_mask

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
 """Optimized RoPE using cached sin/cos (no float64, no complex math).

 Uses Triton fused 3-way kernel when available (Step 2).
 Falls back to Triton rotary (Step 0) or PyTorch.
 """
 # --- Fused Triton 3-way RoPE fast path (D=128 only) ---
 if USE_TRITON_ROPE_FUSED and x.is_cuda and x.ndim == 4 and x.shape[-1] == 128:
 try:
 return triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=int(start_frame), inplace=None)
 except Exception:
 if os.environ.get("SCOPE_TRITON_ROPE_FUSED_STRICT", "0") == "1":
 raise
 pass

 n, c = x.size(2), x.size(3) // 2

 # split freqs (freqs is complex: real=cos, imag=sin)
 freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

 if x.size(0) == 1:
 if isinstance(grid_sizes, torch.Tensor):
 f, h, w = (int(v) for v in grid_sizes[0].tolist())
 else:
 f, h, w = (int(v) for v in grid_sizes[0])
 seq_len = f * h * w

 cos, sin = get_rope_cos_sin(
 freqs_split, f, h, w, start_frame, x.dtype, x.device, c
 )
 cos_fa = cos.squeeze(1)
 sin_fa = sin.squeeze(1)

 if USE_TRITON_ROTARY:
 if seq_len == x.shape[1]:
 return triton_apply_rotary(x, cos_fa, sin_fa, interleaved=True)
 out = x.clone()
 out[:, :seq_len] = triton_apply_rotary(
 x[:, :seq_len], cos_fa, sin_fa, interleaved=True
 )
 return out

 x_i = x[0, :seq_len].reshape(seq_len, n, -1, 2)
 x0, x1 = x_i[..., 0], x_i[..., 1]
 x0_new = x0 * cos - x1 * sin
 x1_new = x0 * sin + x1 * cos
 x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(2).unsqueeze(0)
 if seq_len == x.shape[1]:
 return x_i
 out = x.clone()
 out[:, :seq_len] = x_i
 return out

 out = x.clone()
 for i, (f, h, w) in enumerate(grid_sizes.tolist()):
 seq_len = f * h * w
 cos, sin = get_rope_cos_sin(
 freqs_split, f, h, w, start_frame, x.dtype, x.device, c
 )

 if USE_TRITON_ROTARY:
 cos_fa = cos.squeeze(1)
 sin_fa = sin.squeeze(1)
 out[i : i + 1, :seq_len] = triton_apply_rotary(
 x[i : i + 1, :seq_len], cos_fa, sin_fa, interleaved=True
 )
 continue

 x_i = x[i, :seq_len].reshape(seq_len, n, -1, 2)
 x0, x1 = x_i[..., 0], x_i[..., 1]
 x0_new = x0 * cos - x1 * sin
 x1_new = x0 * sin + x1 * cos
 out[i, :seq_len] = torch.stack([x0_new, x1_new], dim=-1).flatten(2)
 return out

class CausalWanSelfAttention(nn.Module):
 def __init__(
 self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6
 ):
 assert dim % num_heads == 0
 super().__init__()
 self.dim = dim
 self.num_heads = num_heads
 self.head_dim = dim // num_heads
 self.local_attn_size = local_attn_size
 self.sink_size = sink_size
 self.qk_norm = qk_norm
 self.eps = eps
 self.frame_seq_length = 1560 # Default value, can be updated dynamically
 self.max_attention_size = (
 32760 if local_attn_size == -1 else local_attn_size * self.frame_seq_length
 )
 self.fused_projections = False

 # layers
 self.q = nn.Linear(dim, dim)
 self.k = nn.Linear(dim, dim)
 self.v = nn.Linear(dim, dim)
 self.o = nn.Linear(dim, dim)
 self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
 self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

 @torch.no_grad()
 def fuse_projections(self):
 # if not self.is_cross_attention:
 if self.fused_projections:
 return
 concatenated_weights = torch.cat(
 [self.q.weight.data, self.k.weight.data, self.v.weight.data]
 )
 concatenated_bias = torch.cat(
 [self.q.bias.data, self.k.bias.data, self.v.bias.data]
 )
 out_features, in_features = concatenated_weights.shape
 with torch.device("meta"):
 self.to_qkv = torch.nn.Linear(in_features, out_features, bias=True)
 self.to_qkv.load_state_dict(
 {"weight": concatenated_weights, "bias": concatenated_bias},
 strict=True,
 assign=True,
 )
 self.fused_projections = True

 def forward(
 self,
 x,
 seq_lens,
 grid_sizes,
 freqs,
 block_mask,
 kv_cache=None,
 current_start=0,
 cache_start=None,
 kv_cache_attention_bias=1.0,
 ):
 r"""
 Args:
 x(Tensor): Shape [B, L, num_heads, C / num_heads]
 seq_lens(Tensor): Shape [B]
 grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
 freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
 block_mask (BlockMask)
 """
 b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
 if cache_start is None:
 cache_start = current_start

 # query, key, value function
 # @torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
 def qkv_fn(x):
 if self.fused_projections:
 # print("Using fused projections")
 q, k, v = self.to_qkv(x).chunk(3, dim=-1)
 q = self.norm_q(q).view(b, s, n, d)
 k = self.norm_k(k).view(b, s, n, d)
 v = v.view(b, s, n, d)
 else:
 q = self.norm_q(self.q(x)).view(b, s, n, d)
 k = self.norm_k(self.k(x)).view(b, s, n, d)
 v = self.v(x).view(b, s, n, d)
 return q, k, v

 with _ProfileBlock("qkv_projection"):
 q, k, v = qkv_fn(x)

 if kv_cache is None or block_mask is not None:
 # if it is teacher forcing training?
 # is_tf = (s == seq_lens[0].item() * 2)
 is_tf = False
 if is_tf:
 print("Teacher forcing training")
 q_chunk = torch.chunk(q, 2, dim=1)
 k_chunk = torch.chunk(k, 2, dim=1)
 roped_query = []
 roped_key = []
 # rope should be same for clean and noisy parts
 for ii in range(2):
 rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
 rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
 roped_query.append(rq)
 roped_key.append(rk)

 roped_query = torch.cat(roped_query, dim=1)
 roped_key = torch.cat(roped_key, dim=1)

 padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
 padded_roped_query = torch.cat(
 [
 roped_query,
 torch.zeros(
 [q.shape[0], padded_length, q.shape[2], q.shape[3]],
 device=q.device,
 dtype=v.dtype,
 ),
 ],
 dim=1,
 )

 padded_roped_key = torch.cat(
 [
 roped_key,
 torch.zeros(
 [k.shape[0], padded_length, k.shape[2], k.shape[3]],
 device=k.device,
 dtype=v.dtype,
 ),
 ],
 dim=1,
 )

 padded_v = torch.cat(
 [
 v,
 torch.zeros(
 [v.shape[0], padded_length, v.shape[2], v.shape[3]],
 device=v.device,
 dtype=v.dtype,
 ),
 ],
 dim=1,
 )

 with _ProfileBlock("self_attn_block_mask"):
 attn_out = flex_attention(
 query=padded_roped_query.transpose(2, 1),
 key=padded_roped_key.transpose(2, 1),
 value=padded_v.transpose(2, 1),
 block_mask=block_mask,
 )
 if padded_length > 0:
 attn_out = attn_out[:, :, :-padded_length]
 x = attn_out.transpose(2, 1)

 else:
 roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
 roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

 local_end_index = roped_key.shape[1]
 kv_cache["k"][:, :local_end_index] = roped_key
 kv_cache["v"][:, :local_end_index] = v

 kv_cache["global_end_index"] = local_end_index
 kv_cache["local_end_index"] = local_end_index

 padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
 padded_roped_query = torch.cat(
 [
 roped_query,
 torch.zeros(
 [q.shape[0], padded_length, q.shape[2], q.shape[3]],
 device=q.device,
 dtype=v.dtype,
 ),
 ],
 dim=1,
 )

 padded_roped_key = torch.cat(
 [
 roped_key,
 torch.zeros(
 [k.shape[0], padded_length, k.shape[2], k.shape[3]],
 device=k.device,
 dtype=v.dtype,
 ),
 ],
 dim=1,
 )
 # print("shape of padded_roped_query", padded_roped_query.shape)
 # print("shape of padded_roped_key", padded_roped_key.shape)

 padded_v = torch.cat(
 [
 v,
 torch.zeros(
 [v.shape[0], padded_length, v.shape[2], v.shape[3]],
 device=v.device,
 dtype=v.dtype,
 ),
 ],
 dim=1,
 )

 with _ProfileBlock("self_attn_block_mask"):
 attn_out = flex_attention(
 query=padded_roped_query.transpose(2, 1).contiguous(),
 key=padded_roped_key.transpose(2, 1).contiguous(),
 value=padded_v.transpose(2, 1).contiguous(),
 block_mask=block_mask,
 kernel_options={
 "BLOCKS_ARE_CONTIGUOUS": True,
 },
 )
 if padded_length > 0:
 attn_out = attn_out[:, :, :-padded_length]
 x = attn_out.transpose(2, 1)
 else:
 # frame_seqlen = math.prod(grid_sizes[0][1:]).item() # torch compile doesn't like this
 frame_seqlen = self.frame_seq_length
 current_start_frame = current_start // frame_seqlen
 with _ProfileBlock("rope_apply"):
 roped_query = causal_rope_apply(
 q, grid_sizes, freqs, start_frame=current_start_frame
 ).type_as(v)
 roped_key = causal_rope_apply(
 k, grid_sizes, freqs, start_frame=current_start_frame
 ).type_as(v)

 current_end = current_start + roped_query.shape[1]
 sink_tokens = self.sink_size * frame_seqlen
 # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
 kv_cache_size = kv_cache["k"].shape[1]
 num_new_tokens = roped_query.shape[1]
 if (
 self.local_attn_size != -1
 and (current_end > kv_cache["global_end_index"])
 and (num_new_tokens + kv_cache["local_end_index"] > kv_cache_size)
 ):
 with _ProfileBlock("cache_eviction"):
 # Calculate the number of new tokens added in this step
 # Shift existing cache content left to discard oldest tokens
 # Clone the source slice to avoid overlapping memory error
 num_evicted_tokens = (
 num_new_tokens + kv_cache["local_end_index"] - kv_cache_size
 )
 num_rolled_tokens = (
 kv_cache["local_end_index"] - num_evicted_tokens - sink_tokens
 )
 kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
 kv_cache["k"][
 :,
 sink_tokens + num_evicted_tokens : sink_tokens
 + num_evicted_tokens
 + num_rolled_tokens,
 ].clone()
 )
 kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
 kv_cache["v"][
 :,
 sink_tokens + num_evicted_tokens : sink_tokens
 + num_evicted_tokens
 + num_rolled_tokens,
 ].clone()
 )
 # Insert the new keys/values at the end
 local_end_index = (
 kv_cache["local_end_index"]
 + current_end
 - kv_cache["global_end_index"]
 - num_evicted_tokens
 )
 local_start_index = local_end_index - num_new_tokens
 kv_cache["k"][:, local_start_index:local_end_index] = roped_key
 kv_cache["v"][:, local_start_index:local_end_index] = v
 else:
 with _ProfileBlock("cache_update"):
 # Assign new keys/values directly up to current_end
 local_end_index = (
 kv_cache["local_end_index"]
 + current_end
 - kv_cache["global_end_index"]
 )
 local_start_index = local_end_index - num_new_tokens
 kv_cache["k"][:, local_start_index:local_end_index] = roped_key
 kv_cache["v"][:, local_start_index:local_end_index] = v

 kv_start_idx = max(0, local_end_index - self.max_attention_size)
 cached_k = kv_cache["k"][:, kv_start_idx:local_end_index]
 cached_v = kv_cache["v"][:, kv_start_idx:local_end_index]

 if kv_cache_attention_bias != KV_CACHE_ATTENTION_BIAS_DISABLED:
 # Use flex_attention with bias to mitigate error accumulation in past frames
 # log_scale in (0, 1]: smaller values = less attention to past frame tokens
 log_scale = math.log(kv_cache_attention_bias)

 # Exclude first frame and current block from bias
 cache_len = local_end_index - kv_start_idx
 cache_current_block_start = (
 cache_len - frame_seqlen * self.num_frame_per_block
 )

 q_len = roped_query.shape[1]

 if USE_TRITON_KERNEL_B and _triton_kernel_b is not None:
 # Triton Kernel B: 10.7% faster, no padding needed
 # Input: (B, L, H, D) -> (B, H, L, D)
 with _ProfileBlock("transpose_contiguous"):
 Q_t = roped_query.transpose(2, 1).contiguous()
 K_t = cached_k.transpose(2, 1).contiguous()
 V_t = cached_v.transpose(2, 1).contiguous()
 with _ProfileBlock("self_attn_kv_bias"):
 x = _triton_kernel_b(
 Q=Q_t,
 K=K_t,
 V=V_t,
 frame_seqlen=frame_seqlen,
 current_block_start=cache_current_block_start,
 log_bias=log_scale,
 ).transpose(2, 1) # (B, H, L, D) -> (B, L, H, D)
 else:
 # Fallback: flex_attention with padding
 kv_len = cached_k.shape[1]
 target_padded_length = (
 math.ceil(max(q_len, kv_len) / FLEX_ATTENTION_ALIGNMENT)
 * FLEX_ATTENTION_ALIGNMENT
 )

 padded_roped_query = _pad_tensor_for_flex_attention(
 roped_query, target_padded_length, pad_dim=1
 )
 padded_k = _pad_tensor_for_flex_attention(
 cached_k, target_padded_length, pad_dim=1
 )
 padded_v = _pad_tensor_for_flex_attention(
 cached_v, target_padded_length, pad_dim=1
 )

 # Convert scalars to tensors to avoid ShapeAsConstantBuffer dtype issues during compilation
 # This is critical when using torch.compile with flex_attention
 frame_seqlen_tensor = torch.as_tensor(
 frame_seqlen, dtype=torch.int32, device=roped_query.device
 )
 cache_current_block_start_tensor = torch.as_tensor(
 cache_current_block_start, dtype=torch.int32, device=roped_query.device
 ).squeeze()
 log_scale_tensor = torch.as_tensor(
 log_scale, dtype=roped_query.dtype, device=roped_query.device
 )

 def score_mod(score, b_idx, h_idx, q_idx, kv_idx):
 # Apply bias only to past frames (exclude first frame and current block)
 return torch.where(
 (kv_idx >= frame_seqlen_tensor)
 & (kv_idx BlockMask:
 """
 we will divide the token sequence into the following format
 [1 latent frame] [1 latent frame] ... [1 latent frame]
 We use flexattention to construct the attention mask
 """
 block_mask = get_block_mask(
 str(device), num_frames, frame_seqlen, num_frame_per_block, local_attn_size
 )
 return block_mask

 @staticmethod
 def _prepare_teacher_forcing_mask(
 device: torch.device | str,
 num_frames: int = 21,
 frame_seqlen: int = 1560,
 num_frame_per_block=1,
 ) -> BlockMask:
 """
 we will divide the token sequence into the following format
 [1 latent frame] [1 latent frame] ... [1 latent frame]
 We use flexattention to construct the attention mask
 """
 # debug
 DEBUG = False
 if DEBUG:
 num_frames = 9
 frame_seqlen = 256

 total_length = num_frames * frame_seqlen * 2

 # we do right padding to get to a multiple of 128
 padded_length = math.ceil(total_length / 128) * 128 - total_length

 clean_ends = num_frames * frame_seqlen
 # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
 context_ends = torch.zeros(
 total_length + padded_length, device=device, dtype=torch.long
 )
 # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
 noise_context_starts = torch.zeros(
 total_length + padded_length, device=device, dtype=torch.long
 )
 noise_context_ends = torch.zeros(
 total_length + padded_length, device=device, dtype=torch.long
 )
 noise_noise_starts = torch.zeros(
 total_length + padded_length, device=device, dtype=torch.long
 )
 noise_noise_ends = torch.zeros(
 total_length + padded_length, device=device, dtype=torch.long
 )

 # Block-wise causal mask will attend to all elements that are before the end of the current chunk
 attention_block_size = frame_seqlen * num_frame_per_block
 frame_indices = torch.arange(
 start=0,
 end=num_frames * frame_seqlen,
 step=attention_block_size,
 device=device,
 dtype=torch.long,
 )

 # attention for clean context frames
 for start in frame_indices:
 context_ends[start : start + attention_block_size] = (
 start + attention_block_size
 )

 noisy_image_start_list = torch.arange(
 num_frames * frame_seqlen,
 total_length,
 step=attention_block_size,
 device=device,
 dtype=torch.long,
 )
 noisy_image_end_list = noisy_image_start_list + attention_block_size

 # attention for noisy frames
 for block_index, (start, end) in enumerate(
 zip(noisy_image_start_list, noisy_image_end_list, strict=False)
 ):
 # attend to noisy tokens within the same block
 noise_noise_starts[start:end] = start
 noise_noise_ends[start:end] = end
 # attend to context tokens in previous blocks
 # noise_context_starts[start:end] = 0
 noise_context_ends[start:end] = block_index * attention_block_size

 def attention_mask(b, h, q_idx, kv_idx):
 # first design the mask for clean frames
 clean_mask = (q_idx = noise_noise_starts[q_idx]
 )
 C2 = (kv_idx = noise_context_starts[q_idx]
 )
 noise_mask = (q_idx >= clean_ends) & (C1 | C2)

 eye_mask = q_idx == kv_idx
 return eye_mask | clean_mask | noise_mask

 block_mask = create_block_mask(
 attention_mask,
 B=None,
 H=None,
 Q_LEN=total_length + padded_length,
 KV_LEN=total_length + padded_length,
 _compile=False,
 device=device,
 )

 if DEBUG:
 print(block_mask)
 import imageio
 import numpy as np
 from torch.nn.attention.flex_attention import create_mask

 mask = create_mask(
 attention_mask,
 B=None,
 H=None,
 Q_LEN=total_length + padded_length,
 KV_LEN=total_length + padded_length,
 device=device,
 )
 import cv2

 mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
 imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255.0 * mask))

 return block_mask

 @staticmethod
 def _prepare_blockwise_causal_attn_mask_i2v(
 device: torch.device | str,
 num_frames: int = 21,
 frame_seqlen: int = 1560,
 num_frame_per_block=4,
 local_attn_size=-1,
 ) -> BlockMask:
 """
 we will divide the token sequence into the following format
 [1 latent frame] [N latent frame] ... [N latent frame]
 The first frame is separated out to support I2V generation
 We use flexattention to construct the attention mask
 """
 total_length = num_frames * frame_seqlen

 # we do right padding to get to a multiple of 128
 padded_length = math.ceil(total_length / 128) * 128 - total_length

 ends = torch.zeros(
 total_length + padded_length, device=device, dtype=torch.long
 )

 # special handling for the first frame
 ends[:frame_seqlen] = frame_seqlen

 # Block-wise causal mask will attend to all elements that are before the end of the current chunk
 frame_indices = torch.arange(
 start=frame_seqlen,
 end=total_length,
 step=frame_seqlen * num_frame_per_block,
 device=device,
 )

 for idx, tmp in enumerate(frame_indices):
 ends[tmp : tmp + frame_seqlen * num_frame_per_block] = (
 tmp + frame_seqlen * num_frame_per_block
 )

 def attention_mask(b, h, q_idx, kv_idx):
 if local_attn_size == -1:
 return (kv_idx = (ends[q_idx] - local_attn_size * frame_seqlen))
 ) | (q_idx == kv_idx)

 block_mask = create_block_mask(
 attention_mask,
 B=None,
 H=None,
 Q_LEN=total_length + padded_length,
 KV_LEN=total_length + padded_length,
 _compile=False,
 device=device,
 )

 # if not dist.is_initialized() or dist.get_rank() == 0:
 # print(
 # f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
 # print(block_mask)

 # import imageio
 # import numpy as np
 # from torch.nn.attention.flex_attention import create_mask

 # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
 # padded_length, KV_LEN=total_length + padded_length, device=device)
 # import cv2
 # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
 # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

 return block_mask

 def _forward_inference(
 self,
 x,
 t,
 context,
 seq_len,
 clip_fea=None,
 y=None,
 kv_cache: dict = None,
 crossattn_cache: dict = None,
 current_start: int = 0,
 cache_start: int = 0,
 kv_cache_attention_bias: float = 1.0,
 ):
 r"""
 Run the diffusion model with kv caching.
 See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
 This function will be run for num_frame times.
 Process the latent frames one by one (1560 tokens each)

 Args:
 x (List[Tensor]):
 List of input video tensors, each with shape [C_in, F, H, W]
 t (Tensor):
 Diffusion timesteps tensor of shape [B]
 context (List[Tensor]):
 List of text embeddings each with shape [L, C]
 seq_len (`int`):
 Maximum sequence length for positional encoding
 clip_fea (Tensor, *optional*):
 CLIP image features for image-to-video mode
 y (List[Tensor], *optional*):
 Conditional video inputs for image-to-video mode, same shape as x

 Returns:
 List[Tensor]:
 List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
 """

 if self.model_type == "i2v":
 assert clip_fea is not None and y is not None
 # params
 device = self.patch_embedding.weight.device
 if self.freqs.device != device:
 self.freqs = self.freqs.to(device)

 if y is not None:
 x = [torch.cat([u, v], dim=0) for u, v in zip(x, y, strict=False)]

 # embeddings
 x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
 grid_sizes = torch.stack(
 [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
 )
 x = [u.flatten(2).transpose(1, 2) for u in x]
 seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
 assert seq_lens.max() cfphqwr", u)
 u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)])
 out.append(u)
 return out

 def init_weights(self):
 r"""
 Initialize model parameters using Xavier initialization.
 """

 # basic init
 for m in self.modules():
 if isinstance(m, nn.Linear):
 nn.init.xavier_uniform_(m.weight)
 if m.bias is not None:
 nn.init.zeros_(m.bias)

 # init embeddings
 nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
 for m in self.text_embedding.modules():
 if isinstance(m, nn.Linear):
 nn.init.normal_(m.weight, std=0.02)
 for m in self.time_embedding.modules():
 if isinstance(m, nn.Linear):
 nn.init.normal_(m.weight, std=0.02)

 # init output layer
 nn.init.zeros_(self.head.head.weight)
```

File: /Users/daviddickinson/Projects/Lora/scope/src/scope/core/kernels/triton_rope_fused.py

```
py
# src/scope/core/kernels/triton_rope_fused.py
# Copyright (c) 2025
# Triton fused 3-way RoPE (time/height/width) lookup + rotate.
#
# Fast-path target: D=128 (C=64 pairs).
# Avoids materializing (seq_len, C) cos/sin tables.

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Module-scope cache: per-axis tables keyed by (freqs_ptr, device_index, c0,c1,c2)
# This is intentionally module scope (like the existing _ROPE_CACHE).
# -----------------------------------------------------------------------------
_ROPE_AXIS_CACHE: Dict[
 Tuple[int, int, int, int, int],
 Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

def clear_rope_axis_cache() -> None:
 _ROPE_AXIS_CACHE.clear()

def get_rope_axis_tables(
 freqs: torch.Tensor,
 c0: int,
 c1: int,
 c2: int,
 device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
 """
 Split freqs[:, :C] into 3 semantic chunks and cache contiguous float32 cos/sin tables:
 - time chunk: [max_seq, c0]
 - height chunk:[max_seq, c1]
 - width chunk: [max_seq, c2]

 Returns:
 cos_f, sin_f, cos_h, sin_h, cos_w, sin_w (all float32, contiguous, on `device`)
 """
 if freqs.device != device:
 freqs = freqs.to(device)

 if freqs.dtype not in (torch.complex64, torch.complex128):
 raise TypeError(f"freqs must be complex, got {freqs.dtype}")

 if device.type != "cuda":
 raise RuntimeError("get_rope_axis_tables requires CUDA device for fused kernel path")

 dev_index = device.index if device.index is not None else 0
 key = (int(freqs.data_ptr()), int(dev_index), int(c0), int(c1), int(c2))
 cached = _ROPE_AXIS_CACHE.get(key)
 if cached is not None:
 return cached

 # Split into semantic chunks (time/height/width)
 f0, f1, f2 = freqs.split([c0, c1, c2], dim=1)

 # .real/.imag are views; contiguous() materializes dense float32 tables.
 cos_f = f0.real.contiguous().to(dtype=torch.float32, device=device)
 sin_f = f0.imag.contiguous().to(dtype=torch.float32, device=device)

 cos_h = f1.real.contiguous().to(dtype=torch.float32, device=device)
 sin_h = f1.imag.contiguous().to(dtype=torch.float32, device=device)

 cos_w = f2.real.contiguous().to(dtype=torch.float32, device=device)
 sin_w = f2.imag.contiguous().to(dtype=torch.float32, device=device)

 cached = (cos_f, sin_f, cos_h, sin_h, cos_w, sin_w)
 _ROPE_AXIS_CACHE[key] = cached
 return cached

# -----------------------------------------------------------------------------
# Triton kernel: grid = (B*H, ceil_div(seq_len, BLOCK_L))
# Applies 3-way RoPE lookup + rotate in 3 contiguous chunks.
# x is interleaved pairs: (2*j, 2*j+1)
# -----------------------------------------------------------------------------
@triton.jit
def rope_fused_3way_kernel(
 X_ptr,
 OUT_ptr,
 COS_F_ptr,
 SIN_F_ptr,
 COS_H_ptr,
 SIN_H_ptr,
 COS_W_ptr,
 SIN_W_ptr,
 # x strides: [B, L, H, D]
 stride_xb,
 stride_xl,
 stride_xh,
 stride_xd,
 # out strides: [B, L, H, D]
 stride_ob,
 stride_ol,
 stride_oh,
 stride_od,
 # cos/sin strides (all 2D): [max_seq, c_axis]
 stride_cf_l,
 stride_cf_c,
 stride_ch_l,
 stride_ch_c,
 stride_cw_l,
 stride_cw_c,
 # sizes / params
 L, # padded length (x.shape[1])
 HN, # num heads (x.shape[2])
 HW, # grid_h * grid_w
 W, # grid_w
 START_FRAME,
 SEQ_LEN, # f * grid_h * grid_w
 # meta - actual chunk sizes
 C0: tl.constexpr,
 C1: tl.constexpr,
 C2: tl.constexpr,
 # meta - padded power-of-2 sizes for arange
 C0_PAD: tl.constexpr,
 C1_PAD: tl.constexpr,
 C2_PAD: tl.constexpr,
 BLOCK_L: tl.constexpr,
):
 pid_bh = tl.program_id(0)
 pid_l = tl.program_id(1)

 b = pid_bh // HN
 h = pid_bh - b * HN

 offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
 mask_l = offs_l (f_idx, h_idx, w_idx) without modulo ops:
 # f_local = t // (H*W)
 # rem = t - f_local*(H*W)
 # h_idx = rem // W
 # w_idx = rem - h_idx*W
 t = offs_l
 f_local = t // HW
 rem = t - f_local * HW
 h_idx = rem // W
 w_idx = rem - h_idx * W

 f_idx = START_FRAME + f_local

 x_base = X_ptr + b * stride_xb + h * stride_xh
 o_base = OUT_ptr + b * stride_ob + h * stride_oh

 # ----------------------------
 # Chunk 0 (time): pairs [0:C0]
 # Use padded power-of-2 arange with mask
 # ----------------------------
 offs_c0 = tl.arange(0, C0_PAD)
 mask_c0 = offs_c0 Tuple[int, int, int]:
 """
 Returns (f, gh, gw). Only supports:
 - [3] tensor/list
 - [1,3] tensor
 - [B,3] tensor where all rows are identical
 """
 if isinstance(grid_sizes, (tuple, list)):
 if len(grid_sizes) != 3:
 raise ValueError(f"grid_sizes must be len==3, got {len(grid_sizes)}")
 f, gh, gw = grid_sizes
 return int(f), int(gh), int(gw)

 if not isinstance(grid_sizes, torch.Tensor):
 raise TypeError(f"grid_sizes must be Tensor or sequence, got {type(grid_sizes)}")

 if grid_sizes.ndim == 1:
 if grid_sizes.numel() != 3:
 raise ValueError(f"grid_sizes[ndim=1] must have 3 elements, got {grid_sizes.numel()}")
 f, gh, gw = grid_sizes.tolist()
 return int(f), int(gh), int(gw)

 if grid_sizes.ndim == 2:
 if grid_sizes.shape[1] != 3:
 raise ValueError(f"grid_sizes[ndim=2] must be [*,3], got {tuple(grid_sizes.shape)}")
 # B==1 fast path
 if grid_sizes.shape[0] == 1:
 f, gh, gw = grid_sizes[0].tolist()
 return int(f), int(gh), int(gw)

 # Only fuse when identical grid sizes across batch
 # (avoids per-sample Python loops, which killed earlier gains)
 first = grid_sizes[0]
 if not torch.all(grid_sizes == first):
 raise NotImplementedError("Fused RoPE only supports identical grid_sizes across batch")
 f, gh, gw = first.tolist()
 return int(f), int(gh), int(gw)

 raise ValueError(f"Unsupported grid_sizes.ndim={grid_sizes.ndim}")

def _next_power_of_2(n: int) -> int:
 """Return the smallest power of 2 >= n."""
 if n torch.Tensor:
 """
 Fused 3-way RoPE (time/height/width) for x in [B, L, H, D] interleaved-pair layout.

 Fast-path constraints:
 - CUDA only
 - D == 128 (so C == 64)
 - freqs is complex, shape [>=start_frame+f, 64]

 Tail handling (CRITICAL - avoids full tensor clone):
 - if seq_len L:
 # This is the SAGEATTN layout trap case if someone passed [B,H,L,D],
 # or simply inconsistent padding.
 raise ValueError(
 f"Expected x as [B,L,H,D] with L>=seq_len. Got L={L}, seq_len={seq_len}, shape={tuple(x.shape)}"
 )

 # Chunk sizes are semantic. Compute and assert.
 C = D // 2
 c1 = C // 3
 c2 = C // 3
 c0 = C - c1 - c2 # == C - 2*(C//3)

 if c0 != C - 2 * (C // 3):
 raise AssertionError("Chunk size formula mismatch")

 if (c0 + c1 + c2) != C:
 raise AssertionError(f"Chunk sizes don't sum to C: {c0}+{c1}+{c2} != {C}")

 # Bounds guard for causal offset (only time axis gets start_frame)
 start_frame = int(start_frame)
 if start_frame = 0, got {start_frame}")
 if (start_frame + f) > freqs.shape[0]:
 raise ValueError(
 f"start_frame+f out of bounds: start_frame={start_frame}, f={f}, freqs_len={freqs.shape[0]}"
 )
 if freqs.shape[1] != C:
 raise ValueError(f"freqs second dim must be C={C}, got {freqs.shape[1]}")

 # Default inplace policy (inference only, full tensor)
 if inplace is None:
 inplace = (not torch.is_grad_enabled()) and (seq_len == L)

 # Output allocation / tail preservation
 # CRITICAL FIX: Never clone full tensor. Use empty_like + tiny tail copy.
 if inplace:
 out = x
 else:
 out = torch.empty_like(x)
 # Kernel will write prefix; copy tail if needed
 if seq_len = 3.0

from typing import Optional, Union

import torch

import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
 OUT, # Pointers to matrices
 X,
 COS,
 SIN,
 CU_SEQLENS,
 SEQLEN_OFFSETS, # this could be int or a pointer
 # Matrix dimensions
 seqlen,
 nheads,
 seqlen_ro,
 # strides
 stride_out_batch,
 stride_out_seqlen,
 stride_out_nheads,
 stride_out_headdim,
 stride_x_batch,
 stride_x_seqlen,
 stride_x_nheads,
 stride_x_headdim,
 # Meta-parameters
 # We want ROTARY_DIM to be constexpr, otherwise the triton compiler doesn't know that
 # the mask is constant every 8 elements, and it will generate LDG.16 instead of LDG.128
 ROTARY_DIM: tl.constexpr,
 IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
 IS_VARLEN: tl.constexpr,
 INTERLEAVED: tl.constexpr,
 CONJUGATE: tl.constexpr,
 BLOCK_H: tl.constexpr,
 BLOCK_M: tl.constexpr,
):
 BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
 ROTARY_DIM_HALF = ROTARY_DIM // 2
 pid_head = tl.program_id(axis=0)
 pid_m = tl.program_id(axis=1)
 pid_batch = tl.program_id(axis=2)

 if not IS_VARLEN:
 X = X + pid_batch * stride_x_batch
 OUT = OUT + pid_batch * stride_out_batch
 else:
 start_idx = tl.load(CU_SEQLENS + pid_batch)
 seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
 X = X + start_idx * stride_x_seqlen
 OUT = OUT + start_idx * stride_out_seqlen

 if pid_m * BLOCK_M >= seqlen:
 return

 rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
 rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
 if not IS_SEQLEN_OFFSETS_TENSOR:
 rm_cs = rm + SEQLEN_OFFSETS
 else:
 rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

 rk_half = tl.arange(0, BLOCK_K // 2)
 COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
 SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
 mask_cs = (rm_cs[:, None] torch.Tensor:
 """
 Arguments:
 x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
 else (total_seqlen, nheads, headdim).
 cos: (seqlen_ro, rotary_dim / 2)
 sin: (seqlen_ro, rotary_dim / 2)
 seqlen_offsets: integer or integer tensor of size (batch,)
 cu_seqlens: (batch + 1,) or None
 max_seqlen: int
 Returns:
 y: (batch, seqlen, nheads, headdim)
 """
 is_varlen = cu_seqlens is not None
 if not is_varlen:
 batch, seqlen, nheads, headdim = x.shape
 else:
 assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
 total_seqlen, nheads, headdim = x.shape
 batch_p_1 = cu_seqlens.shape[0]
 batch = batch_p_1 - 1
 seqlen = max_seqlen
 seqlen_ro, rotary_dim = cos.shape
 assert sin.shape == cos.shape
 rotary_dim *= 2
 assert rotary_dim = seqlen, "seqlen_ro must be >= seqlen"

 cos, sin = cos.contiguous(), sin.contiguous()
 if isinstance(seqlen_offsets, torch.Tensor):
 assert seqlen_offsets.shape == (batch,)
 assert seqlen_offsets.dtype in [torch.int32, torch.int64]
 seqlen_offsets = seqlen_offsets.contiguous()
 else:
 assert seqlen_offsets + seqlen C=64 -> chunks 22/21/21
 - Current Python impl concatenates these chunks (not multiply).
- `start_frame` only offsets the time axis.
- Tail tokens (`t >= seq_len`) must be preserved; kernel should not write them.

## Incremental Plan

### Step 0 (optional baseline) - COMPLETE (2025-12-23)

**Status:** Integrated and working.

Used FlashAttention's Triton rotary kernel with pre-materialized cos/sin.

**Files changed:**
- Copied: `vendored/rope/flash_attn_triton_rotary.py` → `src/scope/core/kernels/triton_rotary.py`
- Modified: `rope_apply()` and `causal_rope_apply()` to use Triton when available
- Feature flag: `SCOPE_DISABLE_TRITON_ROTARY=1` to force PyTorch fallback

**Benchmark results:**

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| PyTorch rope_apply | 0.200 | Baseline |
| Direct Triton kernel | 0.101 | **1.98x faster** |
| Integrated rope_apply | 0.183 | Only 8.5% faster (overhead) |

**Overhead breakdown:**

| Step | Time (ms) |
|------|-----------|
| Prep (slice+squeeze) | 0.003 |
| Triton kernel | 0.101 |
| Post (squeeze+cat) | 0.025 |
| **Expected total** | 0.129 |
| **Actual integrated** | 0.183 |
| **Unexplained gap** | 0.054 |

The 0.054ms gap comes from Python loop overhead: `grid_sizes.tolist()`, cache lookup, `freqs.split()`, `torch.stack()`.

**Correctness:**
- FP32: Exact match (diff 0.
- Performance: Step 1 should at least match Phase 2.5; Step 2 should improve further without regressions.
- Stability: no layout assumptions broken (watch SAGEATTN path if enabled).
```

File: /Users/daviddickinson/Projects/Lora/scope/scripts/test_rope_correctness.py

```
py
#!/usr/bin/env python3
"""
Correctness checks for RoPE paths (PyTorch reference vs rope_apply/causal_rope_apply).

Examples:
 uv run python scripts/test_rope_correctness.py
 SCOPE_DISABLE_TRITON_ROTARY=1 uv run python scripts/test_rope_correctness.py
 uv run python scripts/test_rope_correctness.py --f 3 --h 30 --w 52 --pad 64
 uv run python scripts/test_rope_correctness.py --f 3 --h 30 --w 52 --pad-to-multiple 128
"""

import argparse
import math
import os
import sys

import torch

def parse_args() -> argparse.Namespace:
 parser = argparse.ArgumentParser(description="RoPE correctness checks.")
 parser.add_argument("--f", type=int, default=3, help="Frames (F)")
 parser.add_argument("--h", type=int, default=30, help="Height tokens (H)")
 parser.add_argument("--w", type=int, default=52, help="Width tokens (W)")
 parser.add_argument("--heads", type=int, default=16, help="Num heads")
 parser.add_argument("--head-dim", type=int, default=128, help="Head dim (D)")
 parser.add_argument("--start-frames", type=int, nargs="*", default=[0, 2], help="start_frame values to test")
 parser.add_argument("--pad", type=int, default=0, help="Extra tail tokens to preserve")
 parser.add_argument(
 "--pad-to-multiple",
 type=int,
 default=0,
 help="Right-pad L so (seq_len+pad) is a multiple of this value (added on top of --pad).",
 )
 parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Input dtype")
 parser.add_argument("--atol", type=float, default=None, help="Max error tolerance (abs)")
 parser.add_argument(
 "--disable-triton-rotary",
 action="store_true",
 help="Set SCOPE_DISABLE_TRITON_ROTARY=1 (must happen before imports).",
 )
 return parser.parse_args()

def main() -> None:
 args = parse_args()
 if not torch.cuda.is_available():
 raise SystemExit("CUDA is required for this test.")

 if args.disable_triton_rotary:
 os.environ["SCOPE_DISABLE_TRITON_ROTARY"] = "1"

 # Import after env var is set (caller can also set SCOPE_DISABLE_TRITON_ROTARY=1)
 from scope.core.pipelines.krea_realtime_video.modules.model import (
 get_rope_cos_sin,
 rope_apply,
 rope_params,
 USE_TRITON_ROTARY,
 )
 from scope.core.pipelines.krea_realtime_video.modules.causal_model import (
 causal_rope_apply,
 )

 dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
 device = torch.device("cuda")
 atol = args.atol
 if atol is None:
 atol = 5e-2 if dtype is torch.bfloat16 else 2e-2

 f, h, w = args.f, args.h, args.w
 seq_len = f * h * w
 pad_to_multiple = int(args.pad_to_multiple)
 pad_mult = 0
 if pad_to_multiple > 0:
 pad_mult = math.ceil(seq_len / pad_to_multiple) * pad_to_multiple - seq_len
 b, n, d = 1, args.heads, args.head_dim

 def build_freqs(max_seq_len: int, head_dim: int) -> torch.Tensor:
 """Match the model's freqs construction (rope_params only)."""
 dim = head_dim
 c0_dim = dim - 4 * (dim // 6)
 c1_dim = 2 * (dim // 6)
 c2_dim = 2 * (dim // 6)
 freqs_local = torch.cat(
 [
 rope_params(max_seq_len, c0_dim),
 rope_params(max_seq_len, c1_dim),
 rope_params(max_seq_len, c2_dim),
 ],
 dim=1,
 )
 return freqs_local.to(device)

 def rope_ref(
 x_tensor: torch.Tensor,
 freqs_tensor: torch.Tensor,
 f_val: int,
 h_val: int,
 w_val: int,
 start_frame: int,
 ) -> torch.Tensor:
 """Reference RoPE using cached cos/sin tables."""
 nheads, c = x_tensor.size(2), x_tensor.size(3) // 2
 freqs_split = freqs_tensor.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
 cos, sin = get_rope_cos_sin(
 freqs_split,
 f_val,
 h_val,
 w_val,
 start_frame,
 x_tensor.dtype,
 x_tensor.device,
 c,
 )

 seq_len_local = f_val * h_val * w_val
 x_i = x_tensor[:, :seq_len_local].reshape(
 x_tensor.size(0), seq_len_local, nheads, c, 2
 )
 x0, x1 = x_i[..., 0], x_i[..., 1]
 x0_new = x0 * cos - x1 * sin
 x1_new = x0 * sin + x1 * cos
 x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(3)
 return torch.cat([x_i, x_tensor[:, seq_len_local:]], dim=1)

 freqs = build_freqs(1024, d)
 grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)

 pad_total = int(args.pad) + int(pad_mult)
 x = torch.randn(b, seq_len + pad_total, n, d, device=device, dtype=dtype)

 print(f"USE_TRITON_ROTARY={USE_TRITON_ROTARY}")
 print(
 f"shape: B={b} L={seq_len}+{pad_total} (pad={args.pad} pad_mult={pad_mult}) H={n} D={d} dtype={args.dtype}"
 )

 # rope_apply (start_frame=0)
 ref = rope_ref(x, freqs, f, h, w, start_frame=0)
 out = rope_apply(x, grid_sizes, freqs)
 max_err = (out - ref).abs().max().item()
 mean_err = (out - ref).abs().mean().item()
 print(f"rope_apply: max_err={max_err:.6f} mean_err={mean_err:.6f}")
 if max_err > atol:
 print("rope_apply: FAIL")
 sys.exit(1)

 # causal_rope_apply for each start_frame
 for sf in args.start_frames:
 ref = rope_ref(x, freqs, f, h, w, start_frame=sf)
 out = causal_rope_apply(x, grid_sizes, freqs, start_frame=sf)
 max_err = (out - ref).abs().max().item()
 mean_err = (out - ref).abs().mean().item()
 print(f"causal_rope_apply(start_frame={sf}): max_err={max_err:.6f} mean_err={mean_err:.6f}")
 if max_err > atol:
 print("causal_rope_apply: FAIL")
 sys.exit(1)

 print("PASS")

if __name__ == "__main__":
 main()
```

</file_contents>
<user_instructions>
<taskname="Fix fused RoPE"/>
<task>
Diagnose and fix the Step-2 fused 3-way RoPE kernel regression on B200. The fused path is in `src/scope/core/kernels/triton_rope_fused.py` and is selected first in `rope_apply()` and `causal_rope_apply()` in the Krea realtime model modules. Confirm whether power-of-2 padded channel aranges are the dominant overhead, verify fused-path execution and runtime shapes, and propose an implementation that removes extra work while preserving correctness, tail handling, and fallback behavior.
</task>
<architecture>
- RoPE entrypoints live in `src/scope/core/pipelines/krea_realtime_video/modules/model.py` (rope_apply) and `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (causal_rope_apply). Both attempt fused kernel first, then fall back to Step-0 Triton rotary (`triton_rotary.apply_rotary`) or PyTorch.
- Fused kernel module `src/scope/core/kernels/triton_rope_fused.py` provides cached per-axis cos/sin tables (get_rope_axis_tables), a Triton kernel (rope_fused_3way_kernel), and the wrapper (rope_fused_3way) that handles padding/tail and launch params.
- Baseline Step-0 kernel is `src/scope/core/kernels/triton_rotary.py` (FlashAttention rotary). Validation harnesses are `scripts/test_rope_correctness.py` and `scripts/bench_triton_rope.py`.
- Design/spec context lives in `notes/FA4/phase3-triton-rope.md`, `notes/FA4/phase3-triton-rope-step2.md`, and original Step-2 proposal `notes/FA4/DeepResearch/2025-12-22/Phase3_01.md`.
</architecture>
<selected_context>
src/scope/core/kernels/triton_rope_fused.py: Fused 3-way RoPE kernel + wrapper. Key details: per-axis cache, kernel mapping t→(f,h,w), and padded arange lanes (C0_PAD/C1_PAD/C2_PAD) from _next_power_of_2 in wrapper.
src/scope/core/kernels/triton_rotary.py: Baseline Step-0 Triton rotary kernel used for fallback and comparison.
src/scope/core/pipelines/krea_realtime_video/modules/model.py: rope_apply(), env flags (SCOPE_DISABLE_TRITON_ROPE_FUSED, SCOPE_TRITON_ROPE_FUSED_STRICT), RoPE cache (get_rope_cos_sin), and fused fast path (D=128, CUDA, ndim=4).
src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py: causal_rope_apply() fast path with start_frame; attention flow uses rope_apply/causal_rope_apply; imports fused flags and triton wrappers from model.
scripts/bench_triton_rope.py: Bench harness comparing rope_apply/causal_rope_apply and direct fused calls; supports pad-to-multiple.
scripts/test_rope_correctness.py: Correctness harness vs reference for rope_apply/causal_rope_apply with padding and start_frame variations.
notes/FA4/phase3-triton-rope.md: Overall Phase 3 plan, Step-0 baseline numbers, Step-2 regression summary, env flags, and fix options.
notes/FA4/phase3-triton-rope-step2.md: Step-2 spec; explicitly calls out power-of-2 padding overhead (22/21/21→32/32/32) and tail-copy semantics.
notes/FA4/DeepResearch/2025-12-22/Phase3_01.md: Original Step-2 kernel + wrapper draft for reference.
</selected_context>
<relationships>
- model.py: rope_apply() → triton_rope_fused_3way (Step 2) → fallback to triton_rotary.apply_rotary (Step 0) → PyTorch RoPE.
- causal_model.py: causal_rope_apply() → triton_rope_fused_3way with start_frame → same fallback chain.
- triton_rope_fused.py: wrapper computes c0/c1/c2 and c*_pad via _next_power_of_2; kernel uses tl.arange(0, C*_PAD) with masks, creating extra work for D=128 (22/21/21 padded to 32/32/32).
- scripts/test_rope_correctness.py and scripts/bench_triton_rope.py call rope_apply/causal_rope_apply and optionally direct fused kernel for validation.
</relationships>
<ambiguities>
- Actual runtime path selection (fused vs fallback) depends on env flags, tensor layout, and shape; verify at call sites with D=128, CUDA, ndim=4 and L vs seq_len.
- Real runtime shapes/strides and start_frame behavior should be confirmed at rope_apply/causal_rope_apply boundaries (seq_len vs padded L, current_start_frame derivation).
</ambiguities>

</user_instructions>

## ChatGPT

### Diagnosis: why Step‑2 regressed even though microbench looked OK

Your current fused kernel in `src/scope/core/kernels/triton_rope_fused.py` is doing **extra channel work** because it pads each chunk’s channel arange to the next power‑of‑2:

Real chunk sizes at **D=128** (C=64 pairs): **C0/C1/C2 = 22/21/21**

Kernel uses: **C0_PAD/C1_PAD/C2_PAD = 32/32/32**

That is **96 pairs vs 64 pairs ⇒ +50% lanes** (and worse: +50% register footprint for the per‑program 2D tensors)

Even with masked loads/stores, Triton still builds and carries around tensors shaped like:

`BLOCK_L x C0_PAD` = `128 x 32` = **4096** elements per intermediate (x0, x1, cos, sin, y0, y1…)

repeated 3× (time/height/width)

That’s a *lot* of live state and very likely pushes you into **register pressure / spills / lower occupancy** on SM100, which is exactly the kind of thing that can show up as an end‑to‑end FPS regression even when a single‑kernel microbench looks “fine”.

Tail handling is **not** the culprit anymore (you already fixed clone → `empty_like + tiny tail copy`), so the remaining dominant risk is exactly the **padded channel lanes**.

## Confirm the fused path is actually executing (and with what shapes)

You already have the right hooks; you just need to run them in “no‑fallback” mode and print the real runtime shapes once.

### 1) Force fused path, crash on fallback

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
SCOPE_DISABLE_TRITON_ROTARY=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128

If that passes, you’ve confirmed:

fused import works

fused wrapper executes for your test shape

fused kernel is correct on padded L cases (tail preserved)

### 2) Benchmark fused directly vs rope_apply wrappers

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128

This prints `TRITON_FUSED_AVAILABLE` and includes:

`rope_apply (end-to-end)`

`causal_rope_apply (end-to-end)`

`triton_rope_fused_3way (direct)` (if available)

### 3) Add a “print once” debug to the wrapper (recommended)

Add to `rope_fused_3way()` (near the end, after computing `c0/c1/c2` and `c*_pad`):

py

Copy code

_DEBUG = os.environ.get("SCOPE_TRITON_ROPE_FUSED_DEBUG", "0") == "1"
global _ROPE_FUSED_DEBUG_PRINTED
try:
 _ROPE_FUSED_DEBUG_PRINTED
except NameError:
 _ROPE_FUSED_DEBUG_PRINTED = False

if _DEBUG and not _ROPE_FUSED_DEBUG_PRINTED:
 _ROPE_FUSED_DEBUG_PRINTED = True
 print(
 "[rope_fused_3way] ",
 f"B={B} L={L} HN={HN} D={D} | grid=(f={f},h={gh},w={gw}) seq_len={seq_len} start_frame={start_frame} ",
 f"| chunks c0/c1/c2={c0}/{c1}/{c2} pads={c0_pad}/{c1_pad}/{c2_pad} ",
 f"| block_l={block_l} warps={num_warps} stages={num_stages} inplace={inplace}",
 )

Then run:

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_DEBUG=1 \
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128

This will conclusively show the **real runtime (f,h,w), L vs seq_len**, and the **pad inflation**.

## Verify “padding is the dominant overhead” with NCU/NSYS (what to look for)

Once you have a capture on B200:

**Register count / spills**: the padded kernel should show notably higher `regs/thread` and possibly local memory traffic.

**Occupancy / waves per SM**: padded kernel should show lower occupancy / fewer active warps.

**Instruction count / math**: padded kernel executes ~1.5× more math on channels.

If you keep two variants (old padded vs fixed), this becomes an easy A/B.

# Fix: remove chunk padding by switching to a power‑of‑2 pair dimension (C=64) and a FlashAttn‑like tiling

Key observation: for **D=128**, the total number of pairs **C=64 is already power‑of‑two**.

So don’t build three separate aranges at sizes 22/21/21 (and then pad to 32/32/32). 

Instead:

Use `rk_half = tl.arange(0, 64)` (power‑of‑two, vectorizable)

Build a full per‑token cos/sin vector of length 64 by selecting from:

`cos_f[f_idx, :c0]`

`cos_h[h_idx, :c1]` into slots `[c0:c0+c1]`

`cos_w[w_idx, :c2]` into slots `[c0+c1:]`

Rotate using the same proven layout strategy as `triton_rotary.py`:

`BLOCK_M=8` tokens

`BLOCK_H=2` heads

`BLOCK_K=128` headdim

This eliminates the **96‑pair inflation** entirely and also fixes the **giant 128×32 intermediates** problem.

## Concrete implementation (drop‑in)

Below is a patch-style rewrite of the kernel + launch while preserving:

wrapper API

tail copy semantics

`inplace` semantics

fallback behavior (unchanged in callers)

### 1) Replace the current kernel with a v2 kernel

In `src/scope/core/kernels/triton_rope_fused.py`, add this new kernel (you can keep the old one around temporarily for A/B, but default to this):

py

Copy code

@triton.jit
def rope_fused_3way_kernel_v2(
 X_ptr,
 OUT_ptr,
 COS_F_ptr,
 SIN_F_ptr,
 COS_H_ptr,
 SIN_H_ptr,
 COS_W_ptr,
 SIN_W_ptr,
 # x strides: [B, L, H, D]
 stride_xb,
 stride_xl,
 stride_xh,
 stride_xd,
 # out strides: [B, L, H, D]
 stride_ob,
 stride_ol,
 stride_oh,
 stride_od,
 # cos/sin strides (2D): [max_seq, c_axis]
 stride_cf_l,
 stride_cf_c,
 stride_ch_l,
 stride_ch_c,
 stride_cw_l,
 stride_cw_c,
 # sizes / params
 SEQ_LEN, # f * gh * gw (active tokens)
 HN, # num heads
 HW, # gh * gw
 W, # gw
 START_FRAME,
 # meta
 C0: tl.constexpr,
 C1: tl.constexpr,
 C2: tl.constexpr,
 BLOCK_H: tl.constexpr,
 BLOCK_M: tl.constexpr,
):
 pid_h = tl.program_id(0) # head blocks
 pid_m = tl.program_id(1) # token blocks
 pid_b = tl.program_id(2) # batch

 rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
 rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
 mask_m = rm < SEQ_LEN
 mask_h = rh < HN

 # token -> (f_idx, h_idx, w_idx)
 t = rm
 f_local = t // HW
 rem = t - f_local * HW
 h_idx = rem // W
 w_idx = rem - h_idx * W
 f_idx = START_FRAME + f_local

 # Build cos/sin for all 64 pairs (C=64 is power-of-two for D=128)
 rk = tl.arange(0, 64)
 cos = tl.full((BLOCK_M, 64), 1.0, tl.float32)
 sin = tl.full((BLOCK_M, 64), 0.0, tl.float32)

 # time chunk [0:C0)
 m0 = rk < C0
 cos0 = tl.load(
 COS_F_ptr + f_idx[:, None] * stride_cf_l + rk[None, :] * stride_cf_c,
 mask=mask_m[:, None] & m0[None, :],
 other=1.0,
 ).to(tl.float32)
 sin0 = tl.load(
 SIN_F_ptr + f_idx[:, None] * stride_cf_l + rk[None, :] * stride_cf_c,
 mask=mask_m[:, None] & m0[None, :],
 other=0.0,
 ).to(tl.float32)
 cos = tl.where(m0[None, :], cos0, cos)
 sin = tl.where(m0[None, :], sin0, sin)

 # height chunk [C0:C0+C1)
 m1 = (rk >= C0) & (rk < (C0 + C1))
 idx1 = rk - C0
 idx1 = tl.where(m1, idx1, 0) # keep pointers sane for masked lanes
 cos1 = tl.load(
 COS_H_ptr + h_idx[:, None] * stride_ch_l + idx1[None, :] * stride_ch_c,
 mask=mask_m[:, None] & m1[None, :],
 other=1.0,
 ).to(tl.float32)
 sin1 = tl.load(
 SIN_H_ptr + h_idx[:, None] * stride_ch_l + idx1[None, :] * stride_ch_c,
 mask=mask_m[:, None] & m1[None, :],
 other=0.0,
 ).to(tl.float32)
 cos = tl.where(m1[None, :], cos1, cos)
 sin = tl.where(m1[None, :], sin1, sin)

 # width chunk [C0+C1:C0+C1+C2)
 m2 = rk >= (C0 + C1)
 idx2 = rk - (C0 + C1)
 idx2 = tl.where(m2, idx2, 0)
 cos2 = tl.load(
 COS_W_ptr + w_idx[:, None] * stride_cw_l + idx2[None, :] * stride_cw_c,
 mask=mask_m[:, None] & m2[None, :],
 other=1.0,
 ).to(tl.float32)
 sin2 = tl.load(
 SIN_W_ptr + w_idx[:, None] * stride_cw_l + idx2[None, :] * stride_cw_c,
 mask=mask_m[:, None] & m2[None, :],
 other=0.0,
 ).to(tl.float32)
 cos = tl.where(m2[None, :], cos2, cos)
 sin = tl.where(m2[None, :], sin2, sin)

 # Load X as [BLOCK_H, BLOCK_M, 128] (interleaved pairs)
 rk_d = tl.arange(0, 128)

 x_base = X_ptr + pid_b * stride_xb
 o_base = OUT_ptr + pid_b * stride_ob

 x_ptrs = x_base + rh[:, None, None] * stride_xh + rm[None, :, None] * stride_xl + rk_d[None, None, :] * stride_xd
 o_ptrs = o_base + rh[:, None, None] * stride_oh + rm[None, :, None] * stride_ol + rk_d[None, None, :] * stride_od

 mask = mask_h[:, None, None] & mask_m[None, :, None] # broadcast

 x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
 x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, 64, 2]))

 cos_b = cos[None, :, :]
 sin_b = sin[None, :, :]

 y0 = x0 * cos_b - x1 * sin_b
 y1 = x0 * sin_b + x1 * cos_b

 y = tl.reshape(tl.join(y0, y1), [BLOCK_H, BLOCK_M, 128])
 tl.store(o_ptrs, y, mask=mask)

### 2) Update the wrapper to launch v2 and delete the C*_PAD path

In `rope_fused_3way()`:

**Remove** `_next_power_of_2`, `c*_pad`, and the `C0_PAD/C1_PAD/C2_PAD` meta args

Change token tiling from “BLOCK_L” to “BLOCK_M”, and add `BLOCK_H`

Example wrapper launch section:

py

Copy code

 # --- Kernel launch params ---
 # Keep old env var name as an alias, but default to a FlashAttn-like token tile.
 if block_l is None:
 block_m = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_M",
 os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_L", "8")))
 else:
 block_m = int(block_l)

 block_h = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_H", "2"))

 if num_warps is None:
 num_warps = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_WARPS", "4"))
 if num_stages is None:
 num_stages = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_STAGES", "2"))

 hw = int(gh) * int(gw)

 grid = (
 triton.cdiv(HN, block_h),
 triton.cdiv(seq_len, block_m),
 B,
 )

 with torch.cuda.device(x.device.index):
 rope_fused_3way_kernel_v2[grid](
 x,
 out,
 cos_f,
 sin_f,
 cos_h,
 sin_h,
 cos_w,
 sin_w,
 x.stride(0), x.stride(1), x.stride(2), x.stride(3),
 out.stride(0), out.stride(1), out.stride(2), out.stride(3),
 cos_f.stride(0), cos_f.stride(1),
 cos_h.stride(0), cos_h.stride(1),
 cos_w.stride(0), cos_w.stride(1),
 seq_len,
 HN,
 hw,
 int(gw),
 start_frame,
 C0=c0, C1=c1, C2=c2,
 BLOCK_H=block_h,
 BLOCK_M=block_m,
 num_warps=num_warps,
 num_stages=num_stages,
 )

Everything else in the wrapper (bounds checks, tail copy, cache) stays the same.

## Why this fixes the regression

**No chunk padding** → you rotate **exactly 64 pairs**, not 96.

You avoid the **huge `[128, 32]` intermediates** entirely.

The headdim arange is **128** and the pair arange is **64** (both power‑of‑two), so you preserve the vectorization motivation without inflating work.

Tiling (`BLOCK_M`, `BLOCK_H`) matches the proven FlashAttention rotary kernel structure, which is typically very well‑behaved on modern NVIDIA SMs.

## Validation checklist

### Correctness

Run:

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128

Also test causal offsets:

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128 --start-frames 0 2 5

### Performance (micro + wrapper)

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128

Try quick tile tuning:

bash

Copy code

SCOPE_TRITON_ROPE_FUSED_BLOCK_M=8 uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
SCOPE_TRITON_ROPE_FUSED_BLOCK_M=16 uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
SCOPE_TRITON_ROPE_FUSED_BLOCK_M=32 uv run python scripts/bench_triton_rope.py --pad-to-multiple 128

### Pipeline confirmation

Run your end-to-end FPS test with:

bash

Copy code

unset SCOPE_DISABLE_TRITON_ROPE_FUSED
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
SCOPE_TRITON_ROPE_FUSED_DEBUG=1 \
<your normal run command>

You should see one debug line confirming `pads` are gone (or you can delete pad reporting entirely after the fix).

## If you want a minimal-risk A/B switch (optional but useful)

Keep the old padded kernel as `rope_fused_3way_kernel_padded` and add an env var:

`SCOPE_TRITON_ROPE_FUSED_IMPL=padded|v2` (default `v2`)

That makes it trivial to prove:

regression correlates with padded implementation

v2 removes it

### Summary

The current regression is overwhelmingly consistent with the **32/32/32 padding path** creating **50% extra lane work** plus **bad register pressure** from giant `[BLOCK_L, C_PAD]` tensors.

The fix is to stop doing per-chunk power‑of‑two padding and instead operate on the natural power‑of‑two dimensions you already have at D=128 (**pairs=64, headdim=128**), using FlashAttn‑style tiling.

The proposed kernel preserves correctness, tail handling, and fallback behavior, while removing the dominant extra work.

If you want, I can also suggest the exact Nsight Compute sections/counters to compare between `padded` and `v2` so you get a clean “this was the regression lever” proof in one capture.

## User

Check the GitHub repo davidrd123/scope, the feature/stream-recording branch to see the work that I've done so far and integrate with what we've been talking about. notes/FA4 folder has notes that point to files and profiling I did

## ChatGPT

To integrate the work you've done in the `feature/stream-recording` branch of the `davidrd123/scope` GitHub repo and align it with the debugging and optimization we've discussed, could you please clarify the following:

Which specific files or components in that branch are most relevant to the fused kernel optimization?

Are there specific files in `notes/FA4` that contain critical insights you'd like me to focus on?

Are you asking for a summary of your current profiling work to identify gaps or inefficiencies?

Should I examine the branch for correctness, performance integration, or something else (like validation coverage or pipeline integration)?

Let me know how you'd like this synthesis to be structured—e.g., as a report, integration checklist, or set of implementation suggestions.

## User

Well, that's what I want you to dig into.

## ChatGPT

Understood. I'll review the `feature/stream-recording` branch of the `davidrd123/scope` GitHub repository, focusing on how your current work—particularly under `notes/FA4`—connects to the fused RoPE kernel optimization discussion. I'll synthesize your profiling insights, implementation state, and integration progress, and identify where performance or correctness gaps may remain.

I'll get back to you shortly with a detailed analysis.

## User

![](https://chatgpt.com/backend-api/estuary/content?id=file_0000000039d8722f9dc61dba2091199f&ts=490725&p=fs&cid=1&sig=9ef7992fedce740aaaf477df8ee2be5e4fa3116f87b8bbef7eead04e8ab532cd&v=0)

I'd like you to go through all the links here as potential performance optimization resources or approaches, and give me a comprehensive integrated report of all the most promising techniques. Essentially, an all-in-one resource that I can drop into my project and have my coding agents look at the code in more detail to estimate which we could make use of.

## ChatGPT

Below is an integrated, agent oriented performance optimization report based on the links in your screenshot. I grouped everything into a practical playbook: what each resource is really about, what kinds of bottlenecks it addresses, and how to evaluate whether it is worth pulling into your codebase.

I was not able to fully retrieve the KernelFalcon blog post text or the TorchCodec post contents in this session due to tool limits, but the KernelAgent README provides a concrete summary of the KernelFalcon approach, and I included a TorchCodec evaluation plan based on the likely scope implied by the post title.

## Executive synthesis

Across all links, there are three big performance levers:

**Remove Python and framework overhead** by compiling and capturing graphs

torch.compile (Inductor) for steady state throughput

torch.export + AOTInductor (AOTI) for cold start and deployment (also helps reproducibility and scaling)

**Win attention and memory bandwidth** with specialized attention kernels

FlexAttention for “non standard” attention patterns and masking (block sparse, custom score modifications), especially when you need correctness plus flexibility in native PyTorch

Pair that with compile so the “flexibility tax” does not become death by overhead

**Go below the graph compiler when needed**

Warp specialization and other kernel level techniques for memory bound kernels and overlap of compute and memory

KernelAgent / KernelFalcon style “generate Triton kernels with verification” when the compiler cannot fuse what you want

A major theme in the Flux Fast and SAM2 posts is that “just compiling” is not enough. The big wins come from:

constraining shapes and data movement,

reducing precision where tolerable,

pushing preprocessing onto GPU,

and, for real deployments, addressing cold starts with AOT compilation.

## What each link gives you, distilled into actionable techniques

### 1) KernelFalcon + KernelAgent: automated kernel synthesis when compilers hit a wall

**What it is** 

KernelAgent is a multi agent system that takes PyTorch programs and synthesizes verified Triton kernels, then composes them back into an equivalent forward pass. It combines static analysis, LLM assisted refactoring to isolate fusable subgraphs, parallel kernel generation, strict runtime verification, and final composition into a kernel only implementation of the original forward. [GitHub](https://github.com/meta-pytorch/KernelAgent)
**Why it matters** 

This is a “last mile” tool for when:

torch.compile cannot fuse a subgraph you care about,

you need a very specific memory layout or epilogue fusion,

or you have a hotspot kernel pattern that repeats and is stable enough to invest in.

**How to use it safely in a product codebase** 

Treat it like an opt in backend for specific hotspots, not a general “compile everything” system:

Identify 1 to 3 kernels that dominate runtime in your stream path.

Build small, deterministic reproduction harnesses for each.

Run KernelAgent to generate candidate Triton kernels.

Keep strict runtime verification in CI for those kernels (the project is explicitly oriented around verification). [GitHub](https://github.com/meta-pytorch/KernelAgent)

**Agent tasks to assign**

“Find top 5 GPU kernels by time and bytes moved, isolate the smallest subgraph that produces each.”

“Attempt Inductor fusion first. If fusion fails, try KernelAgent on the subgraph.”

“Produce a kernel acceptance report: speedup, max error, supported shapes, compile time, maintenance risk.”

### 2) Helion: a deployment oriented optimization layer on top of PyTorch 2

**What it is** 

Helion is described as a lightweight compilation and optimization framework for deep learning inference built on PyTorch 2, with a Python API to flexibly compose multiple optimizations, and a focus on low overhead and serverless friendly deployment. [PyTorch](https://pytorch.org/blog/helion/)
**Where it fits** 

If your stream recording feature has:

elastic scaling needs,

“spin up and go” cold starts,

or multiple small models invoked per session,

Helion’s emphasis (composition + low overhead) is aligned with production constraints rather than only benchmark throughput. [PyTorch](https://pytorch.org/blog/helion/)

**Practical evaluation plan**

Measure cold start time to first inference (new process, cold GPU context if relevant).

Measure p50 and p99 latency under bursty load.

Compare:

eager

torch.compile only

torch.export + AOTI

Helion configuration if it wraps or simplifies the above

### 3) torch.compile + diffusers guide: the “how to actually get peak” workflow

**What it is** 

The diffusers post is a hands on guide focused on getting peak diffusion performance using torch.compile plus related tooling like torchao, including a workflow for correctness validation and an explanation of internals. [PyTorch](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
**What to extract into your project playbook** 

Even if you are not shipping diffusion, the transferable parts are:

a stepwise optimization methodology (change one thing, measure, validate correctness),

and a compiler first approach (compile the dominant modules, validate numerics, iterate). [PyTorch](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)

**How to apply the philosophy to your stream pipeline** 

Make an internal “performance recipe” doc that every agent follows:

lock shapes (or bucket them)

compile the hot path

eliminate CPU side overhead (data movement, preprocessing, Python loops)

only then do kernel level work

This prevents you from wasting time writing kernels for something that is actually CPU bound.

### 4) Flux Fast on H100: model specific wins, but also general lessons

**What it is** 

Flux Fast is a set of optimizations for Flux text to image models. The post highlights major improvements including large reductions in Inductor compilation overhead, removal of CPU overhead in the model, a substantial speedup via torch.compile, and a concrete throughput target (“55 images per minute”) on H100s. [PyTorch](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)
**Generalizable techniques** 

Even if Flux is not your workload, these are broadly applicable:

Reduce compilation overhead and make it predictable

Identify and eliminate CPU overhead inside a GPU dominated pipeline

Treat “compile time” and “steady state runtime” as two separate performance problems [PyTorch](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)

**What to ask your agents to do immediately**

“Measure compile overhead vs steady state runtime in the stream path.”

“Find CPU bound segments inside the GPU path (synchronizations, tensor conversions, per frame Python work).”

“Prototype a ‘warm pool’ strategy: compile once, reuse artifacts.”

### 5) FlexAttention and FlexAttention for inference: flexible attention you can still compile

**What it is** 

FlexAttention is a PyTorch native attention approach oriented around flexibility: it supports custom attention patterns via masking and score modifications, rather than forcing you into one fixed kernel pattern. [PyTorch](https://pytorch.org/blog/flexattention/)
The inference specific post focuses on applying FlexAttention in inference settings. [PyTorch](https://pytorch.org/blog/flexattention-for-inference/)
**When FlexAttention is a good fit** 

Use it when you need one of these:

block sparse or structured sparse attention patterns

streaming or chunked attention where the mask changes per step

custom score modifications (for example biases, locality constraints, gating)

a native PyTorch route that still plays well with compilation [PyTorch+1](https://pytorch.org/blog/flexattention/)

**Important strategic point** 

FlexAttention is not necessarily the fastest possible attention for the simplest “standard causal full attention” case. Its value is that it unlocks weird masks and score logic without instantly falling off a performance cliff, especially once compiled. [PyTorch](https://pytorch.org/blog/flexattention/)
**How this intersects with your FA4 notes** 

If your notes/FA4 work is about pushing maximum throughput for standard attention, keep doing that for the core case. Then treat FlexAttention as:

your escape hatch for any non standard streaming mask logic,

or for research variants where FA4 kernels are not directly usable.

Agents should benchmark both for your exact shapes and masks.

### 6) Warp specialization: kernel level overlap and reducing stalls

**What it is** 

Warp specialization is a GPU optimization technique that partitions warps into specialized roles (for example, some warps focus on memory movement while others focus on compute), which can help hide memory latency and improve utilization. [PyTorch](https://pytorch.org/blog/warp-specialization/)

**When it is worth the complexity**

Your profiles show kernels are memory bound, not compute bound

You see long scoreboard stalls or memory pipe saturation

You are already past the easy wins (compile, fusion, precision, data movement)

**Where to apply**

custom Triton kernels that stream tiles from global memory

preprocessing kernels (packing, interleaving, quantization transforms)

attention adjacent kernels (KV cache transforms, block mask materialization)

In practice, do not start here. Use it as a second phase once you have stable hotspots that cannot be fixed at the graph level.

### 7) Segment Anything 2 acceleration: the deployment grade latency playbook

**What it is** 

This post is explicitly focused on low latency and fast cold starts, using torch.compile and torch.export to create optimized deployments of SAM2, leveraging AOTInductor via torch.export, reduced precision, batched prompts, and GPU preprocessing. It reports up to 13x improvement in p90 execution latency and queue times compared to eager float32, and evaluates in an autoscaling deployment context. [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)
**Directly transferable techniques** 

This is the most “production relevant” link if your stream recording feature is latency sensitive:

**AOT compilation via torch.export + AOTI** for fast startup and scale out [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

**Reduced precision** with explicit accuracy checks [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

**Batching** (they batch prompts, but the general pattern is batching “requests”) [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

**GPU preprocessing** to avoid CPU bottlenecks and copies [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

**How to map this to streaming**

Replace “prompts per image” with “events per time window” or “frames per chunk”

If you have per frame preprocessing (resizes, colorspace, normalization), move it onto GPU

If you have cold starts (new stream session, new worker), prioritize AOT export

### 8) TorchCodec: video pipeline optimization hook

I could not retrieve the TorchCodec post content in this session, so I cannot accurately summarize the specific claims or APIs from that link. What you should have an agent evaluate immediately, given your stream recording context, is:

Does TorchCodec provide GPU accelerated decode or encode paths you can use?

Can it reduce Python overhead in frame ingest?

Does it integrate cleanly with your tensor pipeline to avoid extra copies?

If your profile shows decode, colorspace conversion, or CPU to GPU transfer is a bottleneck, TorchCodec or any similar “decode directly into tensors” approach can be high ROI.

## Integration blueprint for your scope stream recording work

I cannot currently inspect your repo or the notes/FA4 folder contents from here, so the steps below are designed to be directly executable by your coding agents on your branch.

### Step 0: Standardize the measurement harness

Require every performance PR to include:

a reproducible benchmark script

pinned versions and hardware notes

at least: throughput, p50 latency, p95 latency, and peak memory

breakdown: compile time vs steady state runtime

### Step 1: Decide what kind of bottleneck you have

Agents should classify each hotspot as one of:

A) CPU overhead inside the critical path
Examples: preprocessing, Python loops, sync points, data conversions
B) Compiler missed fusion
Many small kernels, lots of launch overhead, intermediate tensors
C) Attention or memory bound GPU kernels
High bandwidth use, low math utilization
D) Cold start or scale out problem
Startup time dominates user experience

### Step 2: Apply the right techniques in the right order

#### If A or B: compiler first

torch.compile the hot path

remove Python work from per frame loops

fuse preprocessing or move it onto GPU

reduce compilation overhead and reuse compiled artifacts (Flux Fast style emphasis) [PyTorch](https://pytorch.org/blog/presenting-flux-fast-making-flux-go-brrr-on-h100s/)

#### If D: deploy like SAM2

use torch.export + AOTInductor for ahead of time compiled artifacts

measure cold start and p90 under autoscaling style load (SAM2 methodology) [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

consider Helion if it materially simplifies multi optimization composition and low overhead inference deployment [PyTorch](https://pytorch.org/blog/helion/)

#### If C: attention and kernels

If attention pattern is standard: keep pushing FA4 style kernels and Inductor fusion

If attention is not standard or masks are complex: evaluate FlexAttention (compiled) vs your FA4 path [PyTorch+1](https://pytorch.org/blog/flexattention/)

For stubborn kernels: consider warp specialization patterns in custom Triton code [PyTorch](https://pytorch.org/blog/warp-specialization/)

If you need systematic kernel synthesis with verification: KernelAgent as the “hard mode” lever [GitHub](https://github.com/meta-pytorch/KernelAgent)

## A concrete “agent checklist” you can paste into your project

### Compilation and graph capture checklist

 Identify the smallest stable hot function for compile

 Bucket dynamic shapes to reduce recompiles

 Report compile time separately from runtime

 Verify numerics (unit tests plus output tolerance tests)

### Latency and cold start checklist

 Build torch.export path and AOTInductor artifact generation

 Measure cold start p50 and p90 end to end

 Validate reduced precision impact with task specific metrics (SAM2 style) [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

### Attention checklist

 Benchmark FA4 vs FlexAttention for your real mask patterns

 If you have streaming masks, prototype FlexAttention first (faster to iterate), then re implement in FA4 only if it is a clear win [PyTorch+1](https://pytorch.org/blog/flexattention/)

### Kernel synthesis checklist

 Confirm Inductor cannot fuse the subgraph you care about

 Isolate minimal subgraph, build microbench

 Run KernelAgent synthesis, keep runtime verification

 Accept only if speedup is durable across your production shape buckets [GitHub](https://github.com/meta-pytorch/KernelAgent)

## My recommended top priorities for you, based on the link set

If your feature is truly “stream recording” and the pipeline touches GPU work, the highest expected ROI order is:

**SAM2 style deployment optimizations** (torch.export + AOTI, reduced precision, GPU preprocessing, batching where possible) for latency and cold start. [PyTorch](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)

**torch.compile everywhere it makes sense**, using the workflow discipline from the diffusers guide and the overhead focus from Flux Fast. [PyTorch+1](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)

**FlexAttention as the flexible attention layer** for any non standard masking or streaming attention semantics, with a benchmark against FA4. [PyTorch+1](https://pytorch.org/blog/flexattention/)

**KernelAgent** only for the handful of hotspots where fusion and compilation do not get you there. [GitHub](https://github.com/meta-pytorch/KernelAgent)

**Warp specialization** only after you have stable, proven kernel bottlenecks that are clearly memory bound. [PyTorch](https://pytorch.org/blog/warp-specialization/)

## If you want me to integrate this directly with your branch

Paste either:

the contents of `notes/FA4` (or the most important note files), and

the profiler outputs you referenced (even screenshots or text summaries)

and I will map each of the techniques above to specific call sites and modules, and produce a prioritized “PR plan” with estimated difficulty and expected upside per change.

