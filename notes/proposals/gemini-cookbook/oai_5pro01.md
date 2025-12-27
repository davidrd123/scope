Got it – I've updated the task accordingly.


Here’s what I see after cross-checking **`notes/proposals/gemini-cookbook.md`** against the repo tree you shared **and** the `gemini_min.py` you pasted.

## Executive verdict

* **The cookbook is internally consistent with the `gemini_min.py` you pasted** (the text-first single-turn helper module with `init_client`, `prepare_content`, `generate_text_with_usage`, `generate_image_edits`, batch JSONL helpers, etc.).
* **But it is not accurate “against this repo” as currently structured**, because it repeatedly claims those patterns “already exist in this repo” and cites **paths (`comfy_automation/*`, `tools/*`, `ModelCatalog/*`) that are not present in the repo tree shown** (depth cap 3 would have shown `comfy_automation/analysis_providers/gemini_min.py` if it existed at repo root).
* The **actual in-repo Gemini integration** is centered on:

  * `src/scope/realtime/gemini_client.py` (GeminiCompiler / WorldChanger / PromptJiggler + `GEMINI_API_KEY`)
  * `src/scope/realtime/prompt_compiler.py` (wires GeminiCompiler into compilation via `create_compiler`, with `SCOPE_LLM_COMPILER` override)
* The cookbook also describes **image/video analysis/editing, UI registry, sidecar naming** that do **not** exist in this repo (at least not as codepaths in the tree you provided). Those capabilities are explicitly “NOT STARTED” in `notes/proposals/vlm-integration.md`.

So: **the cookbook reads like an imported external “comfy_automation” doc**, which is consistent with `notes/NOTES-INDEX.md` labeling it as “from comfy_automation,” but the cookbook itself doesn’t clearly say that—and that’s the main accuracy problem.

---

## 1) Missing or mismatched source references in the cookbook

These are referenced as “in-repo sources” inside `notes/proposals/gemini-cookbook.md`, but **do not exist in the repo tree you provided**:

### “comfy_automation” paths (missing in this repo)

* `comfy_automation/analysis_providers/gemini_min.py` (the cookbook’s main source)
* `comfy_automation/generation_providers/gemini_image.py`
* `comfy_automation/naming_imagegen.py`
* `comfy_automation/cli_batch_video_analysis.py`

### “tools” paths (missing in this repo)

* `tools/batch_video_analysis.py`
* `tools/mcp_video_analyzer_min.py`
* `tools/image_generator_cli.py`
* `tools/image_batch_cli.py`
* `tools/gemini_ui/*`

### “ModelCatalog” path (missing in this repo)

* `ModelCatalog/unified_registry.yaml`

### “Notes/Architecture/adr/*” paths (missing in this repo)

* The cookbook references ADR-034/ADR-035 locations that aren’t present as directories in the repo tree shown.

> Net: these citations are currently **broken** if you treat this repo as the ground truth.

---

## 2) What the repo actually contains (and what the cookbook should point to)

### ✅ Implemented in this repo: text-only Gemini integration for prompt compilation

**`src/scope/realtime/gemini_client.py`**

* `init_client()`

  * reads `GEMINI_API_KEY`
  * returns `genai.Client(api_key=api_key)` or `None` (logs warning) if not configured
  * **does not** do python-dotenv `.env` loading
* `GeminiCompiler`

  * implements `(system_prompt, user_message) -> str`
  * uses `response_mime_type="text/plain"`
  * has a simple `_rate_limit()` (10 calls/sec max)
* `GeminiWorldChanger`

  * asks Gemini for **JSON-only** output (`response_mime_type="application/json"`)
  * parses via `WorldState.model_validate_json(...)`
* `GeminiPromptJiggler`

  * generates prompt variations
  * gracefully falls back to original prompt if Gemini unavailable

**`src/scope/realtime/prompt_compiler.py`**

* `create_compiler(style, mode="auto")` selects:

  * TemplateCompiler (no Gemini)
  * or Gemini via `LLMCompiler(llm_callable=GeminiCompiler(), instruction_sheet=...)`
* env overrides:

  * `SCOPE_LLM_COMPILER=gemini|template|auto`
  * `GEMINI_API_KEY` required for Gemini mode
* optional instruction sheet loading:

  * `styles/<style.name>/instructions.md`

### 🔲 Not implemented in this repo: frame analysis / image editing / video tooling

This matches `notes/proposals/vlm-integration.md` which lists:

* frame analysis: “NOT STARTED”
* image editing: “NOT STARTED”

The cookbook currently presents those as already existing patterns “in this repo,” which is inaccurate.

---

## 3) The cookbook is accurate about `gemini_min.py`—but that file is out-of-tree here

Now that you pasted `gemini_min.py`, we can say more precisely:

### What the cookbook says about `gemini_min.py` that matches the code you pasted

* `init_client()`:

  * best-effort `.env` load via `python-dotenv` ✅
  * reads `GEMINI_API_KEY` ✅
  * raises if missing key / missing SDK ✅
* `prepare_content()`:

  * image: loads via PIL, returns `[img, user_prompt]` ✅
  * video: uploads to Files API, polls until ACTIVE, returns `[Part(file_data=...), user_prompt]` + uploaded file handle ✅
* `generate_text()` / `generate_text_with_usage()`:

  * uses `GenerateContentConfig(... response_mime_type="text/plain")` ✅
  * extracts `.text` when present, otherwise walks candidates/parts ✅
  * returns usage summary when available ✅
* `cleanup_uploaded()` deletes uploaded Files API objects ✅
* Batch helpers exist in this module (JSONL, polling, result parsing) ✅

  * The cookbook mentions batch/file mode patterns in places; those align with the existence of batch helpers, though the cookbook references additional wrapper scripts that aren’t in this repo.

### The real mismatch

The mismatch isn’t “cookbook vs gemini_min.py”; it’s **cookbook vs *this repo***.

So the cookbook needs an explicit provenance/scoping fix.

---

## 4) Concrete doc edits to make `notes/proposals/gemini-cookbook.md` accurate in *this* repo

You have two good options. Pick one depending on what you want that file to be.

### Option A: Keep it as an external reference, but label it correctly (minimal change)

Make these edits:

1. **Change the opening claim** (“patterns that already exist in this repo”) to something like:

> “This is a curated cookbook of Gemini integration patterns derived from the `comfy_automation` codebase (e.g., `gemini_min.py`). Most referenced paths do not exist in this repository. The in-repo Gemini integration lives in `src/scope/realtime/gemini_client.py` and is wired via `src/scope/realtime/prompt_compiler.py`.”

2. Add a **Status Legend** at the top:

* ✅ Implemented in Scope
* 🧩 External / from comfy_automation (not in this repo)
* 🔲 Planned (see `notes/proposals/vlm-integration.md`)

3. For every section header, tag it:

* 🧩 “Single-turn text (image/video) → text output” (external)
* 🧩 “Flash image-edit → edited images” (external)
* 🧩 “Gemini 3 Pro Image Preview → images” (external)
* ✅ Add a new section: “Prompt compilation + WorldState editing in Scope” (in-repo)

4. Add a short “Scope mapping” section (example below).

#### Suggested drop-in “Scope mapping” section (copy/paste)

Add near the top:

```md
## Scope repo mapping (what actually exists here)

✅ Implemented (in `src/scope/realtime/gemini_client.py`):
- init_client(): reads GEMINI_API_KEY and returns google.genai.Client or None
- GeminiCompiler: (system_prompt, user_message) -> prompt text
- GeminiWorldChanger: instruction -> updated WorldState JSON
- GeminiPromptJiggler: prompt -> subtle variation

✅ Wired into compilation (in `src/scope/realtime/prompt_compiler.py`):
- create_compiler(): selects Gemini when GEMINI_API_KEY is set (or SCOPE_LLM_COMPILER=gemini)

🔲 Planned (see `notes/proposals/vlm-integration.md`):
- GeminiFrameAnalyzer (frame -> description)
- GeminiFrameEditor (frame + instruction -> edited frame)
```

5. **Don’t pretend broken file paths are in-repo**
   When you cite `comfy_automation/...`, explicitly say “external reference; not currently in this repo tree”.

This aligns the cookbook with your NOTES-INDEX (“from comfy_automation”) and avoids misleading readers.

---

### Option B: Rewrite the cookbook to be a true “Scope repo cookbook” (bigger change, but cleanest)

If you want this file to be a reliable in-repo doc, restructure it around what exists:

* Section 1: “Gemini client in Scope”

  * `init_client()` behavior (note: returns `None` vs raising)
  * `_rate_limit()` behavior
  * `GeminiCompiler` call shape (`system_instruction`, `response_mime_type="text/plain"`)
* Section 2: “Prompt compilation integration”

  * how `LLMCompiler` builds system prompt with vocab + instruction sheet
  * how `create_compiler` selects template vs Gemini
  * env vars: `GEMINI_API_KEY`, `SCOPE_LLM_COMPILER`
* Section 3: “WorldState editing”

  * strict JSON-only contract
  * failure modes (invalid JSON → ValueError)
* Section 4: “Prompt jiggle”

  * fallback behavior if Gemini missing
* Section 5: “Future work”

  * summarize frame describe/edit from `notes/proposals/vlm-integration.md`
  * explicitly mark as not implemented

Then move the existing comfy_automation patterns to either:

* an Appendix, or
* a new doc name: `notes/reference/comfy_automation_gemini_cookbook.md`

---

## 5) Specific clarifications the cookbook should make (even if you keep it mostly as-is)

These will prevent confusion between `gemini_min.py` and Scope’s `gemini_client.py`:

### Clarify `init_client` behavior differences

* **Scope**: `init_client()` returns `None` when not configured (logs warning).
* **gemini_min**: `init_client()` raises RuntimeError when not configured; best-effort loads `.env`.

That matters because cookbook currently reads like “this repo behaves like gemini_min.”

### Clarify that Scope currently does not have media prep / Files API usage

* Scope’s `gemini_client.py` does not upload videos, does not have `prepare_content`, does not do batch JSONL, etc.
* Those are in the `gemini_min.py` you pasted (external).

### Clarify model IDs are repo-local defaults vs external examples

* Scope default: `"gemini-3-flash-preview"` (in code)
* The cookbook talks about “Gemini 3 Pro Image Preview” + “Flash Image” flows—those are not implemented in Scope today.

### Unify API key naming across notes

* Your `notes/reference/context-editing-code.md` snippet mentions `GOOGLE_API_KEY`; Scope uses `GEMINI_API_KEY`.
  Even though that’s a different doc, readers will collide here—worth normalizing in the cookbook (or adding a warning callout).

---

## 6) A minimal diff that would fix the biggest accuracy issue

If you want the smallest possible patch with maximum impact, edit just the intro + add a mapping section:

```diff
- Below is a curated “cookbook” of the **Gemini integration patterns that already exist in this repo**...
+ Below is a curated “cookbook” of Gemini integration patterns.
+ IMPORTANT: Much of this document is derived from an external codebase (“comfy_automation”),
+ and several referenced paths (comfy_automation/*, tools/*, ModelCatalog/*) do NOT exist in this repo.
+ The Gemini integration that actually exists in this repo is in:
+ - src/scope/realtime/gemini_client.py
+ - src/scope/realtime/prompt_compiler.py
+ For planned frame analysis/editing, see notes/proposals/vlm-integration.md.

+ ## Scope repo mapping (what exists here)
+ ✅ Prompt compilation: GeminiCompiler (text-only) + LLMCompiler wiring via create_compiler()
+ ✅ WorldState updates: GeminiWorldChanger (JSON-only)
+ ✅ Prompt variation: GeminiPromptJiggler
+ 🔲 Frame analysis + image editing: not implemented yet (see vlm-integration.md)
```

---

If you want, I can also propose a **fully rewritten** `notes/proposals/gemini-cookbook.md` outline that cleanly separates:

* ✅ “in Scope now”
* 🧩 “external comfy_automation reference”
* 🔲 “planned per vlm-integration.md”

…but the edits above are enough to make the current document **truthful and navigable** in this repo without losing the useful external patterns.
