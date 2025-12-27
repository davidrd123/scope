I compared `scope/notes/proposals/gemini-cookbook.md` against the actual implementations + ADR-034/035 you included. Overall it’s solid and maps correctly to the repo structure, but there are a few concrete mismatches and a couple “missing context” gotchas worth fixing **in the markdown**.

## What’s accurate (no change needed)

* `init_client` / `.env` best-effort / `GEMINI_API_KEY` requirement matches `comfy_automation/analysis_providers/gemini_min.py:init_client`.
* `prepare_content` behavior and the “single-turn contents list” conventions match:

  * Images: `[PIL.Image, user_prompt]`
  * Videos: `[gtypes.Part(file_data=...), user_prompt]` + uploaded handle for cleanup
* Text parsing description matches `generate_text` / `generate_text_with_usage` (prefers `resp.text`, then walks candidate parts; skips `thought` parts).
* Image parsing distinction is correct:

  * Gemini 3 Pro Image helper (`generation_providers/gemini_image.py`) does **`part.as_image()` first**, then inline fallback.
  * Flash image-edit helper (`analysis_providers/gemini_min.py:generate_image_edits`) decodes **inline_data only**.
* Batch API (`tools/batch_video_analysis.py --batch-mode file`) is correctly described as text-only, and correctly notes flash-image / 3-pro-image are rejected in file mode.
* UI registry-driven params and provider override rules match `ModelCatalog/unified_registry.yaml`, `tools/gemini_ui/components.py`, and `tools/gemini_ui/state.py`.

## Mismatches / missing context to fix

### 1) “Sidecar” terminology is overloaded (minor but confusing)

In `tools/batch_video_analysis.py`, “sidecar” means **a caption output file next to the media** (default `.txt`) or image outputs next to media in image modes — **not** the JSON metadata `*-sidecar.json` used by image generation tools.

The cookbook currently uses “sidecar” in both senses without explicitly calling out the difference.

### 2) Section 5B “Sidecar schema” conflates two different JSON sidecar schemas

You cite both:

* `tools/image_batch_cli.py` (batch runner) and
* `tools/image_generator_cli.py` (single-run generator)

…but then present a single “portable schema” example that doesn’t exactly match either:

* `image_generator_cli.py` **includes** `success: True` and `metrics: {}`, and uses `inputs.base_image` / `inputs.ref_images` / `inputs.size` (not `image_paths`), and **does not include** `naming` or `meta`.
* `image_batch_cli.py` **includes** `naming` + `meta`, and uses `inputs.image_paths`, but **error sidecars often omit `provider`** (provider is present in the “ok” sidecar, but missing in the resolution-failure + exception branches shown in the code you provided).

So the cookbook should split this into “two schemas” (or explicitly say “these vary by tool”).

### 3) Section 5C “Always write an error sidecar” is incorrect

* `tools/batch_video_analysis.py` **does not write JSON sidecars at all** (direct mode writes the primary artifact; batch file mode writes outputs plus manifest/index + `failed.txt`).
* `tools/image_generator_cli.py` (Gemini provider path) **returns early on exceptions** and does **not** write an error sidecar.
* `tools/image_batch_cli.py` does write error sidecars for per-item failures **during execution**, but there’s an exception: the **early** `--strict-take` conflict path returns `1` before execution and **does not write a sidecar** (it only prints/logs, optionally emitting `--json-result`).

### 4) Missing gotcha: `types.ImageConfig` SDK requirement

`generation_providers/gemini_image.py` will raise if the installed `google-genai` SDK doesn’t expose `types.ImageConfig` (it explicitly says “upgrade google-genai…”). This is worth one bullet in the cookbook under the Gemini 3 Pro Image section.

---

## Precise markdown-only edits (suggested patch)

Below is a **surgical diff** you can apply to `scope/notes/proposals/gemini-cookbook.md` to correct the above without changing code.

````diff
diff --git a/scope/notes/proposals/gemini-cookbook.md b/scope/notes/proposals/gemini-cookbook.md
--- a/scope/notes/proposals/gemini-cookbook.md
+++ b/scope/notes/proposals/gemini-cookbook.md
@@
 ### Notes worth carrying into another app
 
 * If you need deterministic-ish formatting (e.g., no timestamps), `tools/batch_video_analysis.py` uses `stop_sequences` when `--no-timestamps` is enabled (text mode only).
 * The CLI logs usage to stdout (`[USAGE] …`) but keeps the return value clean.
+* Note on terminology: in `tools/batch_video_analysis.py`, “sidecar” means the **caption/output file next to the media** (default `.txt`), not a JSON metadata sidecar (`*-sidecar.json`).
 
 ---
 
 ## 3) Pattern: Single-turn **image-edit** (Flash Image / “Nano Banana”) → edited image(s)
@@
 ### Key implementation details to preserve
 
 * **Reference cap is enforced**: `MAX_REF_IMAGES = 14`.
 * Config uses `response_modalities=["IMAGE","TEXT"]`.
 * Optional `ImageConfig(aspect_ratio=…, image_size=…)` only if those are set.
+  * Implementation gotcha: `generate_images()` raises if your installed `google-genai` SDK does not expose `types.ImageConfig` (it explicitly tells you to upgrade).
 * Optional `tools=[{"google_search": {}}]` only when enabled.
 
 This is a *clean, portable* “single-turn” pattern.
 
 ---
 
 ## 5) Sidecar + naming patterns (portable across apps)
@@
 ### B. Sidecar schema (Gemini image jobs)
 
-**Source:** `tools/image_generator_cli.py` (Gemini path) + `tools/image_batch_cli.py` (Gemini path)
-
-A good portable schema (already used) looks like:
+There are **two** similar-but-not-identical JSON “sidecar” schemas today, depending on which tool wrote it.
+
+#### 1) Batch runner: `tools/image_batch_cli.py` (provider=gemini/openai)
+
+This tool writes `*-sidecar.json` for each **executed** job item (success or error). The shape differs slightly between `status="ok"` and `status="error"`:
+
+* `status`: `"ok"` or `"error"` (always present)
+* `error`: present only when `status == "error"`
+* `provider`: present on many success sidecars (e.g. `"gemini"`), but is **not guaranteed** on all error sidecars in current code paths
+* `model`, `inputs`, `resolved_params`, `outputs`
+* `naming`: includes `base_dir`, `base_name`, and optional `take_number`
+* `meta`: includes `job_id` and `job_title`
+
+Example (success shape, Gemini):
 
 ```json
 {
   "status": "ok",
   "provider": "gemini",
   "model": {"id": "gemini-3-pro-image-preview", "version": null},
   "inputs": {
     "prompt": "...",
     "system": "...",
     "image_paths": ["..."]
   },
   "resolved_params": {...},
   "outputs": [{"path": "...", "url": null, "bytes": 12345}],
   "text_path": "...optional...",
   "grounding": "...optional...",
   "naming": {"base_dir": "...", "base_name": "...", "take_number": 3},
   "meta": {"job_id": "...", "job_title": "..."}
 }
````

+#### 2) Single-run generator: `tools/image_generator_cli.py --provider gemini`
+
+This tool writes a sidecar **only on success** (it returns early on errors). Its sidecar schema is slightly different:
+
+* Includes `success: true` and `metrics: {}`
+* Uses `inputs.base_image` / `inputs.ref_images` and `inputs.size` (rather than `inputs.image_paths`)
+* Does **not** currently include `naming` or `meta`
+
+Example:
+
+```json
+{

* "provider": "gemini",
* "success": true,
* "status": "ok",
* "model": {"id": "gemini-3-pro-image-preview", "version": null},
* "inputs": {
* "prompt": "...",
* "system": "...",
* "aspect_ratio": "16:9",
* "size": "2K",
* "google_search": false,
* "ref_images": ["..."],
* "base_image": "...or null..."
* },
* "resolved_params": {...},
* "outputs": [{"path": "...", "url": null, "bytes": 12345}],
* "text_path": "...optional...",
* "grounding": "...optional...",
* "metrics": {}
  +}
  +```
*

### C. “Always write an error sidecar”

-Both `tools/image_batch_cli.py` and `tools/batch_video_analysis.py` will write a `status="error"` sidecar when resolution fails, so you don’t lose provenance.
+What’s true today:
+
+* `tools/image_batch_cli.py` writes `status="error"` sidecars for per-job failures during execution (bad `from:`/handle resolution, provider exceptions, provider-returned errors, etc.).

* * Exception: the **early** `--strict-take` conflict path exits before execution and does **not** write a sidecar (it only logs / prints JSON result if `--json-result` is enabled).
    +* `tools/batch_video_analysis.py` does **not** write JSON sidecars:
* * Direct multi-file mode writes the primary artifact only (caption text file or edited image(s)) and logs failures.
* * Batch file mode writes outputs plus a manifest/index NDJSON and `failed.txt` (no per-item JSON sidecars).
    +* `tools/image_generator_cli.py` (Gemini provider path) currently returns early on exceptions and does **not** emit an error sidecar.

```

## Optional small doc-only improvement
If you want the cookbook to be extra “drop-in portable,” add one sentence somewhere near the “Sidecar + naming patterns” section stating explicitly:

- “Only the image-generation CLIs produce JSON metadata sidecars; the captioner (`batch_video_analysis.py`) does not.”

That single line prevents a lot of confusion when someone skims.

If you want, I can also rewrite just **Section 5** into a tighter “Schema Reference” block (still markdown-only) that’s easier to lift into another repo docs.
```
