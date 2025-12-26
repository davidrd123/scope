# VLM Integration Plan

> Status: Draft
> Date: 2025-12-26
> Related: `notes/proposals/gemini-cookbook.md`, `notes/plans/phase6-prompt-compilation.md`, `notes/realtime-roadmap.md`

## Summary

Consolidate all VLM (Vision-Language Model) integration into a coherent plan. Three capabilities, one integration:

| Capability | Input | Output | Status |
|------------|-------|--------|--------|
| **Prompt Compilation** | WorldState JSON | T2V prompt text | ✅ Implemented |
| **Frame Analysis** | Video frame | Description text | 🔲 Not started |
| **Image Editing** | Frame + instruction | Edited frame | 🔲 Not started |

All use Gemini models but with different configurations.

---

## 1. Prompt Compilation (Phase 6a) — IMPLEMENTED

**What:** WorldState + StyleManifest → LLM → compiled T2V prompt

**Status:** ✅ Working in `src/scope/realtime/gemini_client.py`

**Components:**
- `GeminiCompiler` - compiles WorldState to prompt via instruction sheet
- `GeminiWorldChanger` - natural language WorldState updates ("make Rooster angry")
- `GeminiPromptJiggler` - prompt variations without changing intent

**Model:** `gemini-3-flash-preview` (text-only, fast)

**Endpoints:**
- `POST /api/v1/realtime/world/change` - semantic WorldState update
- `POST /api/v1/prompt/jiggle` - prompt variation

**No additional work needed** - this is complete.

---

## 2. Frame Analysis (Phase 8) — NOT STARTED

**What:** Analyze generated frames, return description for agent evaluation loop.

**Use case:** Agent watches generation, decides if it matches intent, adjusts prompts.

```
Frame (RGB tensor) → Gemini Vision → "A clay rooster chasing a hen around a kitchen table"
```

**Model:** `gemini-2.0-flash` or `gemini-2.5-flash` (vision-capable)

### Proposed Implementation

#### Endpoint

```
GET /api/v1/realtime/frame/describe
POST /api/v1/realtime/frame/describe  (with custom prompt)
```

Response:
```json
{
  "description": "A clay rooster chasing a hen...",
  "chunk_index": 42,
  "confidence": 0.85
}
```

#### Core Function

```python
# src/scope/realtime/gemini_client.py

class GeminiFrameAnalyzer:
    """Analyze video frames via Gemini Vision."""

    SYSTEM_PROMPT = """You are a video frame analyzer for an animation system.

Describe what you see in the frame:
- Characters and their actions
- Environment and lighting
- Camera angle and composition
- Any text or UI elements visible

Be concise but specific. Output plain text description."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
    ):
        self.model = model
        self.temperature = temperature
        self._client = None

    def describe(
        self,
        frame: Image.Image | torch.Tensor,
        custom_prompt: str | None = None,
    ) -> str:
        """
        Describe what's in a frame.

        Args:
            frame: PIL Image or tensor (will be converted)
            custom_prompt: Optional custom analysis prompt

        Returns:
            Text description of frame contents
        """
        if self.client is None:
            raise RuntimeError("Gemini client not available")

        # Convert tensor to PIL if needed
        if isinstance(frame, torch.Tensor):
            frame = tensor_to_pil(frame)

        prompt = custom_prompt or "Describe this animation frame:"

        response = self.client.models.generate_content(
            model=self.model,
            contents=[frame, prompt],
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=256,
            ),
        )
        return response.text.strip()
```

#### CLI

```bash
video-cli describe-frame              # Describe current frame
video-cli describe-frame --prompt "Is Rooster visible?"
```

#### Agent Loop Pattern

```python
def direct_video(goal: str):
    run_cli(f'video-cli prompt "{goal}"')

    for i in range(max_iterations):
        run_cli("video-cli step")
        desc = run_cli("video-cli describe-frame")

        if "character not visible" in desc:
            run_cli('video-cli change "bring character to center"')
        elif "wrong emotion" in desc:
            run_cli('video-cli change "make character look happier"')
```

---

## 3. Image Editing (Context Editing) — NOT STARTED

**What:** Edit anchor frame via Gemini image model, propagate through KV cache.

**Use case:** Fix hallucinations, add/remove elements, correct identity drift.

```
Anchor Frame + "remove the extra arm" → Gemini Flash Image → Edited Frame → KV Cache
```

**Model:** `gemini-2.5-flash-image` (Flash Image / "Nano Banana")

### The Mechanism

From `recompute_kv_cache.py`:
```python
decoded_first_frame = state.decoded_frame_buffer[:, :1]
reencoded_latent = vae.encode_to_latent(decoded_first_frame)
```

Edit surface: modify `decoded_frame_buffer[:, :1]` → next recompute encodes the edit → KV cache reflects the change → future frames "remember" the edit.

### Proposed Implementation

#### Endpoint

```
POST /api/v1/realtime/frame/edit
{
  "instruction": "remove the extra arm on the left character"
}
```

Response:
```json
{
  "status": "applied",
  "edit_applied_at_chunk": 42,
  "instruction": "remove the extra arm..."
}
```

#### Core Function

```python
# src/scope/realtime/gemini_client.py

class GeminiFrameEditor:
    """Edit frames via Gemini Flash Image."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-image",
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        self._client = None

    def edit(
        self,
        frame: Image.Image,
        instruction: str,
    ) -> Image.Image:
        """
        Edit a frame with a natural language instruction.

        Args:
            frame: Input frame as PIL Image
            instruction: Edit instruction (e.g., "remove the extra arm")

        Returns:
            Edited frame as PIL Image
        """
        if self.client is None:
            raise RuntimeError("Gemini client not available")

        response = self.client.models.generate_content(
            model=self.model,
            contents=[frame, instruction],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
            ),
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data'):
                return decode_inline_image(part.inline_data)

        raise ValueError("No image in response")
```

#### Integration with Pipeline

```python
# In frame_processor.py or new module

async def apply_frame_edit(
    pipeline: KreaRealtimeVideoPipeline,
    instruction: str,
    editor: GeminiFrameEditor,
) -> bool:
    """Apply edit to anchor frame."""
    # Get current anchor frame
    decoded_buffer = pipeline.state.get('decoded_frame_buffer')
    anchor_tensor = decoded_buffer[:, :1]  # [B, 1, C, H, W]

    # Convert to PIL
    anchor_pil = tensor_to_pil(anchor_tensor[0, 0])

    # Edit via Gemini
    edited_pil = editor.edit(anchor_pil, instruction)

    # Convert back to tensor and inject
    edited_tensor = pil_to_tensor(edited_pil)
    decoded_buffer[:, :1] = edited_tensor.unsqueeze(0).unsqueeze(0)

    # Next recompute will encode the edited frame
    return True
```

#### CLI

```bash
video-cli edit "remove the extra arm"
video-cli edit "add rain to the scene"
video-cli edit --preview "change lighting to night"  # Preview without applying
```

---

## 4. Validation Spike (Context Editing)

Before building the full image editing integration, validate the mechanism works:

```python
# Simple color tint test - no Gemini needed
decoded_buffer = pipeline.state.get('decoded_frame_buffer')
anchor = decoded_buffer[:, :1].clone()
anchor[:, :, 2, :, :] = 1.0  # Max blue
anchor[:, :, 0, :, :] = 0.0  # Zero red
decoded_buffer[:, :1] = anchor
# Continue generating → should stay blue
```

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| Scene goes blue, stays blue | Edit propagates through KV cache | Build real edit integration |
| Flickers blue then reverts | Model "corrects" back | Try aligning prompt with edit |
| No visible change | Edit not reaching recompute | Check timing, buffer indices |
| Generation breaks | Edit too aggressive | Try subtler mutation |

---

## 5. Implementation Order

1. **Frame Analysis** (simpler, immediate value)
   - Add `GeminiFrameAnalyzer` to `gemini_client.py`
   - Add `/api/v1/realtime/frame/describe` endpoint
   - Add `video-cli describe-frame` command
   - Test with agent loop

2. **Validation Spike** (de-risk context editing)
   - Run color tint test
   - Document results
   - Decide if mechanism is viable

3. **Image Editing** (if validation passes)
   - Add `GeminiFrameEditor` to `gemini_client.py`
   - Add `/api/v1/realtime/frame/edit` endpoint
   - Add `video-cli edit` command
   - Integration tests

---

## 6. Dependencies

- `google-genai` package (already in deps for prompt compilation)
- `GEMINI_API_KEY` environment variable
- For image editing: Gemini Flash Image access (may need allowlist)

---

## 7. Related Files

| File | Role |
|------|------|
| `src/scope/realtime/gemini_client.py` | All Gemini integrations |
| `src/scope/server/app.py` | REST endpoints |
| `src/scope/cli/video_cli.py` | CLI commands |
| `src/scope/server/frame_processor.py` | Frame buffer access |
| `src/scope/core/pipelines/krea_realtime_video/blocks/recompute_kv_cache.py` | KV cache recompute (edit injection point) |

---

## 8. References

- `notes/proposals/gemini-cookbook.md` - Gemini integration patterns from comfy_automation
- `notes/plans/phase6-prompt-compilation.md` - Original Phase 6 spec
- `notes/realtime-roadmap.md` - Phase 8 VLM feedback loop
- `notes/capability-roadmap.md` - Context Editing section
