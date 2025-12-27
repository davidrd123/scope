Below is a curated “cookbook” of the **Gemini integration patterns that already exist in this repo**, organized so you can lift them into another app. I’m sticking to **single‑turn, single‑prompt flows** (no chat/session continuation), and I’m separating patterns by **output kind**: **text**, **image‑edit (Flash Image / “Nano Banana”)**, and **image generation/edit (Gemini 3 Pro Image Preview)**.

---

## 1) Core building blocks you already have

### A. Client initialization (shared everywhere)

**Source:** `comfy_automation/analysis_providers/gemini_min.py:init_client`

Pattern:

* Reads `GEMINI_API_KEY` from env.
* Best-effort loads `.env` if `python-dotenv` is installed.
* Returns a `google.genai.Client`.

**Reuse as-is** if the new app can rely on env vars.
If you want the new app to be library‑friendly, consider adding an alternate entrypoint that accepts `api_key` explicitly and only falls back to env.

---

### B. Media preparation (single file → Gemini “contents” list)

**Source:** `comfy_automation/analysis_providers/gemini_min.py:prepare_content`

Pattern:

* Detect mimetype via `mimetypes.guess_type`.
* **Images** (`image/*`): load via Pillow → `[PIL.Image, user_prompt]`.
* **Videos** (`video/*`): upload to Files API → `[Part(file_data=...), user_prompt]` and returns uploaded file handle for cleanup.
* Errors if unsupported.

This gives you a stable “single prompt” format: `contents=[media_part, prompt_text]` (or `contents=[prompt_text, images...]` in Gemini 3 Pro Image helper; see below).

---

### C. Output parsing patterns (text vs images)

You have two parsing styles:

1. **Text** parsing:

* Prefer SDK convenience `resp.text` if present.
* Fallback to iterating `candidates[].content.parts[].text`.
* Skip thought parts (`part.thought`) defensively.

**Source:** `gemini_min.py:generate_text`, `generate_text_with_usage`

2. **Image** parsing:

* For Gemini 3 Pro Image Preview: prefer `part.as_image()` if available, else fallback to decoding `part.inline_data`.
* For Flash image-edit (`generate_image_edits`): decode `inline_data` only.

**Sources:**

* `comfy_automation/generation_providers/gemini_image.py:_decode_inline_image`, `generate_images`
* `comfy_automation/analysis_providers/gemini_min.py:generate_image_edits`

If you’re reusing patterns in a new app, the **best reusable micro‑pattern** is:

> “Try `part.as_image()` first, then `inline_data` decode fallback.”

That’s already implemented in `gemini_image.py` and is worth mirroring in `generate_image_edits` if you want to make Flash image parsing more robust.

---

## 2) Pattern: Single-turn **text** (image or video) → text output (+ usage)

### Where it exists

* Provider logic: `gemini_min.py:prepare_content`, `generate_text_with_usage`, `cleanup_uploaded`
* CLI usage: `tools/batch_video_analysis.py:_analyze_one` and `comfy_automation/cli_batch_video_analysis.py:_analyze_one`
* MCP server: `tools/mcp_video_analyzer_min.py:analyze_video`

### What to reuse

**The “RAII” cleanup shape**: upload may happen for videos; always cleanup in `finally`.

#### Minimal reusable function (template)

```py
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from comfy_automation.analysis_providers.gemini_min import (
    init_client,
    prepare_content,
    generate_text_with_usage,
    cleanup_uploaded,
)

def gemini_single_turn_text(
    media_path: Path,
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    fps: Optional[float] = None,
    max_output_tokens: int = 65536,
    temperature: float = 0.7,
    stop_sequences: Optional[list[str]] = None,
) -> Tuple[str, Optional[Dict[str, Optional[int]]]]:
    client = init_client()
    uploaded = None
    try:
        contents, uploaded = prepare_content(client, media_path, user_prompt, fps)
        text, usage = generate_text_with_usage(
            client,
            model=model,
            contents=contents,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        return text, usage
    finally:
        cleanup_uploaded(client, uploaded)
```

### Notes worth carrying into another app

* If you need deterministic-ish formatting (e.g., no timestamps), `tools/batch_video_analysis.py` uses `stop_sequences` when `--no-timestamps` is enabled (text mode only).
* The CLI logs usage to stdout (`[USAGE] …`) but keeps the return value clean.

---

## 3) Pattern: Single-turn **image-edit** (Flash Image / “Nano Banana”) → edited image(s)

### Where it exists

* Provider logic: `gemini_min.py:generate_image_edits`
* Captioner integration: `tools/batch_video_analysis.py:_edit_one_image`
* ADR design intent: `Notes/Architecture/adr/2025-11-18-adr-034-...`

### What the pattern is

* Input: image + instruction prompt
* Output: one or more images (PIL)
* Save images with predictable suffixes (`.clean.png`, `.clean_2.png`, …) in sidecar mode; or `*-gemini-clean.png` under analysis tree mode.

#### Minimal reusable function (template)

```py
from pathlib import Path
from typing import List, Optional
from PIL import Image

from comfy_automation.analysis_providers.gemini_min import (
    init_client,
    prepare_content,
    generate_image_edits,
)

def gemini_single_turn_flash_edit(
    image_path: Path,
    *,
    instruction: str,
    model: str,                 # e.g. "gemini-2.5-flash-image" (or your chosen flash-image id)
    system_prompt: str = "",
    temperature: float = 0.7,
) -> List[Image.Image]:
    client = init_client()
    # prepare_content(image) returns [PIL.Image, instruction]
    contents, _uploaded = prepare_content(client, image_path, instruction, fps=None)
    images = generate_image_edits(
        client,
        model=model,
        contents=contents,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    return images
```

### Important constraints your repo already codifies

* This mode should only run on `image/*` inputs (the captioner checks mimetype and errors otherwise).
* In `tools/batch_video_analysis.py`, Flash-image and Gemini 3 Pro Image are **explicitly rejected** in **Batch API file mode** (`--batch-mode file`) and only supported in direct calls. That’s aligned with ADR‑034 and ADR‑035.

### “Integration smell” to avoid in a new app

In `tools/batch_video_analysis.py`, routing is currently based on:

* `"flash-image" in args.model` → image-edit
* `"3-pro-image" in args.model` → gemini image generation

That’s convenient, but brittle if model naming changes. For a new app, prefer:

* an explicit `output_kind` selection, or
* a small registry table mapping `model_id -> capabilities` (similar to what the UI registry does).

---

## 4) Pattern: Single-turn **Gemini 3 Pro Image Preview** (generate/edit) → images (+ optional text, grounding)

### Where it exists

* Core helper module: `comfy_automation/generation_providers/gemini_image.py:generate_images`
* Used by:

  * `tools/image_generator_cli.py` (Gemini provider path)
  * `tools/image_batch_cli.py` (YAML batch provider=gemini)
  * `tools/batch_video_analysis.py:_generate_one_gemini3`
* ADR: `Notes/Architecture/adr/2025-11-19-adr-035-...`

### What the pattern is

* Input:

  * `user_prompt` text
  * optional `system_prompt`
  * zero or more images (base image + reference images)
  * optional `aspect_ratio`, `image_size` (1K/2K/4K), optional `google_search`
* Output:

  * `images`: list of PIL images
  * `text`: optional additional text
  * `grounding`: grounding metadata (if search tool enabled)
  * plus the config echo

#### Minimal reusable function (template)

```py
from pathlib import Path
from typing import List, Optional, Tuple, Any
from PIL import Image

from comfy_automation.analysis_providers.gemini_min import init_client
from comfy_automation.generation_providers.gemini_image import generate_images, MAX_REF_IMAGES

def gemini3_single_turn_images(
    *,
    prompt: str,
    model: str = "gemini-3-pro-image-preview",
    system_prompt: str = "",
    base_image: Optional[Path] = None,
    ref_images: Optional[List[Path]] = None,
    aspect_ratio: Optional[str] = None,   # "16:9", ...
    image_size: Optional[str] = None,     # "1K"|"2K"|"4K"
    temperature: float = 0.7,
    enable_search: bool = False,
) -> Tuple[List[Image.Image], Optional[str], Any]:
    refs = list(ref_images or [])
    total_imgs = (1 if base_image else 0) + len(refs)
    if total_imgs > MAX_REF_IMAGES:
        raise ValueError(f"Too many images ({total_imgs}); max {MAX_REF_IMAGES}.")

    client = init_client()
    result = generate_images(
        client=client,
        model=model,
        user_prompt=prompt,
        system_prompt=system_prompt,
        image_paths=[base_image] if base_image else [],
        ref_image_paths=refs,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        temperature=temperature,
        enable_search=enable_search,
    )
    return (result["images"], result.get("text"), result.get("grounding"))
```

### Key implementation details to preserve

* **Reference cap is enforced**: `MAX_REF_IMAGES = 14`.
* Config uses `response_modalities=["IMAGE","TEXT"]`.
* Optional `ImageConfig(aspect_ratio=…, image_size=…)` only if those are set.
* Optional `tools=[{"google_search": {}}]` only when enabled.

This is a *clean, portable* “single-turn” pattern.

---

## 5) Sidecar + naming patterns (portable across apps)

If you’re reusing these integrations in another app, one of the biggest wins is copying the **naming + sidecar** conventions so outputs remain traceable.

### A. Naming conventions

**Source:** `comfy_automation/naming_imagegen.py` usage in:

* `tools/image_generator_cli.py`
* `tools/image_batch_cli.py`

Pattern:

* `compute_base_dir(...)` creates a stable folder layout: user/project/session/model_slug
* `compute_base_name(...)` hashes prompt+params → stable name
* `dedupe_base_name(...)` avoids overwrites (`time|counter|none`)
* Optional “take numbers” are appended (`_t01`, `_t02`, …) in `tools/image_batch_cli.py`

This is especially important if your new app will re-run the same job multiple times.

### B. Sidecar schema (Gemini image jobs)

**Source:** `tools/image_generator_cli.py` (Gemini path) + `tools/image_batch_cli.py` (Gemini path)

A good portable schema (already used) looks like:

```json
{
  "status": "ok",
  "provider": "gemini",
  "model": {"id": "gemini-3-pro-image-preview", "version": null},
  "inputs": {
    "prompt": "...",
    "system": "...",
    "image_paths": ["..."],          // or base_image/ref_images
    "aspect_ratio": "16:9",
    "size": "2K",
    "google_search": false
  },
  "resolved_params": {...},
  "outputs": [{"path": "...", "url": null, "bytes": 12345}],
  "text_path": "...optional...",
  "grounding": "...optional...",
  "naming": {"base_dir": "...", "base_name": "...", "take_number": 3},
  "meta": {"job_id": "...", "job_title": "..."}
}
```

### C. “Always write an error sidecar”

Both `tools/image_batch_cli.py` and `tools/batch_video_analysis.py` will write a `status="error"` sidecar when resolution fails, so you don’t lose provenance.

That’s a very reusable pattern for another app, especially a UI: it simplifies “show me what happened”.

---

## 6) UI integration pattern (optional, but very reusable)

If the “another app” is a web UI, the `tools/gemini_ui` approach is worth copying.

### A. Registry-driven parameter forms

**Sources:**

* `ModelCatalog/unified_registry.yaml` (Gemini/OpenAI UI model definitions)
* `tools/gemini_ui/components.py:render_model_params`

Pattern:

* A YAML registry defines params with types (`enum`, `boolean`) + defaults.
* UI renders forms dynamically from that schema.
* This avoids hardcoding parameters in the UI.

### B. Delegate execution to a CLI subprocess

**Source:** `tools/gemini_ui/app.py` orchestration helpers:

* `_run_cli_plan(... --plan)` → parse JSON plan (even with logs mixed in)
* `_run_cli_execute_json(... --json-result)` → returns structured JSON to the UI

This is a strong “reuse pattern” when:

* you want the UI thin,
* you want CLI to remain the single source of truth for execution,
* you want local-only workflows without turning your UI server into the runtime.

### C. Provider override logic from job fields

**Source:** `tools/gemini_ui/state.py:create_temp_yaml`

Pattern:

* Decide provider from job content:

  * `catalog` ⇒ Replicate
  * model starts `gpt-image-*` ⇒ OpenAI
  * model contains `/` ⇒ Replicate raw id
  * model starts `gemini` ⇒ Gemini

This is a good portability pattern if your new app also supports multiple providers.

---

## 7) What to lift verbatim vs refactor into “portable” abstractions

### Lift verbatim (already clean, single-turn focused)

* `init_client`
* `prepare_content` (if you need image+video)
* `generate_text_with_usage` (and `generate_text`)
* `gemini_image.generate_images` (Gemini 3 Pro Image Preview helper)
* Sidecar writing strategy from `image_generator_cli.py` (Gemini path)

### Consider small refactors for portability

1. **Unify image extraction**

   * `gemini_image.generate_images` supports `part.as_image()` and inline fallback.
   * `gemini_min.generate_image_edits` currently only scans `inline_data`.
   * For a new app, consider a shared `extract_images_from_response(resp)` helper used by both.

2. **Replace model-name substring routing**

   * Current captioner routing: `"flash-image"` / `"3-pro-image"` substrings.
   * More robust portable approach: a capability registry like:

     ```py
     MODEL_CAPS = {
       "gemini-3-pro-image-preview": {"kind": "image-gen"},
       "gemini-2.5-flash-image": {"kind": "image-edit"},
       "gemini-2.5-pro": {"kind": "text"},
     }
     ```
   * Your UI already has the start of this idea in `unified_registry.yaml` (`type: [txt2img, edit]`, `max_images`).

3. **Make client injection easy**

   * Right now most code calls `init_client()` inside the function.
   * For reuse in another app (especially async/web), it’s often nicer to:

     * construct one client at startup
     * pass it down (or store in app context)

---

## 8) The three “single-turn recipes” in one place

If you only copy three things into the new app, copy these shapes:

1. **Text analysis** (image/video)
   `prepare_content` → `generate_text_with_usage` → cleanup

2. **Flash image-edit** (image → edited images)
   `prepare_content` → `generate_image_edits` → save `*.clean.png`

3. **Gemini 3 Pro Image Preview** (prompt + optional images → images + optional text)
   `gemini_image.generate_images` → enforce 14-image cap → save outputs + optional text + grounding

---

## 9) About the “Flash” model ambiguity (your note)

You called out that `unified_registry.yaml` doesn’t include Gemini Flash models, and that flash-image is currently keyed off `"flash-image"` substring matching.

For reuse in another app, the cleanest approach is:

* Treat **Flash (text)** models as **text-only** by always setting `response_mime_type="text/plain"` (your `generate_text*` does this).
* Treat **Flash Image** models as **image-edit** by expecting image parts and writing image outputs.
* Add Flash model ids to the same registry the UI already uses if you want them selectable in the UI (and add badges in `MODEL_BADGE_REGISTRY`).

You don’t need to solve that immediately if the new app only cares about the two stable single-turn paths: **text** and **Gemini 3 Pro Image Preview**.

---

If you want, I can also provide a **single consolidated “portable module”** (one file) that:

* exposes `run_text(...)`, `run_flash_edit(...)`, `run_gemini3_images(...)`
* normalizes outputs into one consistent `Result` dict (status/model/outputs/text/usage)
* includes the sidecar writer

…but everything above should already be enough to reuse the existing repo patterns without bringing over the CLI/UI machinery.

---

## 10) Action Items: Patterns to Copy from ComfyPromptByAPI

> Source: `/root/ComfyPromptByAPI` (cloned 2025-12-27)

The ComfyPromptByAPI repo demonstrates mature multi-provider patterns worth extracting. These action items represent concrete patterns to add or improve in this codebase.

### High Priority (Core Integration)

#### A. Provider Protocol & Contract
**Source:** `comfy_automation/generation_providers/interface.py`

Copy the `ImageGenerator` Protocol pattern:
- [ ] Define structural typing for provider interface (`generate()`, `edit()` methods)
- [ ] Standardize `ImageJobResult` TypedDict (status, error, outputs, metrics)
- [ ] Add error classification: `ok | error | rejected | timeout`

#### B. Image Generation YAML Spec
**Source:** `tools/image_batch_cli.py`

The YAML batch spec is more mature than what we have:
- [ ] Document complete schema (batch, defaults, jobs)
- [ ] Add parameter inheritance (job > batch defaults > global defaults)
- [ ] Support `provider` routing field (gemini, replicate, openai)

#### C. Handle System for Edit Chaining
**Source:** `comfy_automation/handles.py`

Enables iterative editing workflows:
- [ ] `@hash:abc123` — find output by content hash
- [ ] `@sidecar:/path/to.json` — explicit sidecar reference
- [ ] `@last:model:user:session` — query-based filter
- [ ] Resolution algorithm scans `index.ndjson` under configured roots

### Medium Priority (Robustness)

#### D. Sidecar Schema Alignment
**Source:** All providers write consistent sidecars

Our sidecars are ad-hoc; theirs are standardized:
```json
{
  "status": "ok",
  "provider": "gemini",
  "model": {"id": "...", "version": null},
  "inputs": {"prompt": "...", "system": null},
  "resolved_params": {...},
  "outputs": [{"path": "...", "url": null, "bytes": 12345}],
  "naming": {"base_dir": "...", "base_name": "...", "take_number": 1},
  "meta": {"job_id": "...", "job_title": "..."}
}
```
- [ ] Add `resolved_params` (what actually ran, after defaults applied)
- [ ] Add `naming` block for deterministic output organization
- [ ] Add `meta.job_id` for batch traceability

#### E. Index NDJSON for Query
**Source:** Output directories have `index.ndjson`

Append-only log enables handle resolution and history:
- [ ] Write one line per generation (sourced from sidecar)
- [ ] Use for `@hash:` and `@last:` resolution
- [ ] Enables reproducibility queries

#### F. Model Registry Design
**Source:** `ModelCatalog/unified_registry.yaml`

Cleaner than substring matching:
```yaml
models:
  gemini-3-pro-image-preview:
    provider: gemini
    type: [txt2img, edit]
    max_images: 14
    params:
      aspect_ratio: {type: enum, options: [...], default: "1:1"}
```
- [ ] Add capability flags (`txt2img`, `edit`, `multi_image_reference`)
- [ ] Add `param_map` for canonical → provider API names
- [ ] Add `unsupported` list for filtering invalid params

### Lower Priority (Nice to Have)

#### G. Session Kit Structure
**Source:** `WorkingSpace/Projects/.../Sessions/`

For project organization:
```
Sessions/MyProject/
├── session.yaml          # Batch metadata
├── groups/
│   └── shots/
│       ├── group.yaml    # Prompts + defaults
│       └── Images/       # Input images
└── out/
    └── batch.yaml        # Materialized from kit
```
- [ ] Consider adopting for daydream projects
- [ ] `group.yaml` has `prompt_body`, `neg_body`, `images_glob`

#### H. UI State Management (Three-Layer)
**Source:** `tools/gemini_ui/state.py`

If we build a UI:
1. In-memory caches (server lifetime)
2. `.state.json` (ephemeral, gitignored)
3. Source files (YAML, sidecars)

Key patterns:
- [ ] `StateManager` with conflict detection (yaml_hash, yaml_mtime)
- [ ] Delegate execution to CLI subprocess (don't reimplement naming)
- [ ] Take number reservation for concurrent requests

#### I. ADR References
**Source:** `Notes/Architecture/adr/`

Key decisions to reference:
- ADR-040: Gemini Image UI (FastHTML MVP)
- ADR-035: Gemini 3 Pro Image Preview integration
- ADR-018: Image Handles and Browser
- ADR-017: Image Generation (multi-provider)

### Patterns NOT in ComfyPromptByAPI (Future Work)

- Request/response tracing (correlation IDs)
- Rate limiting handling
- Persistent caching layer
- Multi-tenancy

---

## Summary: What to Lift vs Build

| Pattern | Source | Action |
|---------|--------|--------|
| `init_client`, `prepare_content` | gemini_min.py | Lift verbatim |
| `generate_images` (Gemini 3 Pro) | gemini_image.py | Lift verbatim |
| Sidecar writer | image_generator_cli.py | Adapt schema |
| Handle resolver | handles.py | Port if chaining needed |
| Model registry | unified_registry.yaml | Adopt format |
| YAML batch spec | image_batch_cli.py | Align schemas |
| UI state management | gemini_ui/state.py | Reference for future UI |
