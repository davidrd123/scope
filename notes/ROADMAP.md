# Realtime Video Generation Roadmap

## Completed

### Phase 6a: Style Layer Scaffolding
- WorldState, StyleManifest, PromptCompiler abstractions
- TemplateCompiler, LLMCompiler, CachedCompiler
- ControlBus integration with FrameProcessor
- `/api/v1/realtime/world` + `/api/v1/realtime/style` endpoints + `video-cli world|style`
- See: `notes/plans/phase6-prompt-compilation.md`

### Phase 6b: Gemini Flash Integration
- GeminiCompiler, GeminiWorldChanger, GeminiPromptJiggler
- `/api/v1/realtime/world/change` - natural language WorldState updates
- `/api/v1/prompt/jiggle` - prompt variation generation
- `create_compiler()` factory with auto/gemini/template modes
- Requires `GEMINI_API_KEY` env var
- See: `notes/plans/phase6-prompt-compilation.md` (spec) and `src/scope/realtime/gemini_client.py` (implementation)

---

## Next Up (Pick One)

### 1. Multi-LoRA Hot-Switching
Load multiple LoRAs at startup, switch by setting scale (0.0/0.85).
- Env var: `SCOPE_PRELOAD_LORAS=rat,tmnt,yeti,hidari`
- See: `notes/issues/multi-lora-hot-switching.md`

### 2. TUI Director Console
Keyboard-driven control for live video direction.
- Textual-based TUI
- Style/beat/camera quick-keys
- See: `notes/plans/tui-director-console.md`

### 3. VLM Feedback Loop (Phase 8)
Use Gemini to analyze generated frames for automatic prompt refinement.

### 4. CLI Extensions (quick win)
```bash
video-cli change "make Rooster angry"   # POST /world/change
video-cli jiggle                        # POST /prompt/jiggle
```

---

## Key Files

| Component | Location |
|-----------|----------|
| Realtime control plane | `src/scope/realtime/` |
| Gemini integration | `src/scope/realtime/gemini_client.py` |
| Prompt compilation | `src/scope/realtime/prompt_compiler.py` |
| Style manifests | `styles/<name>/manifest.yaml` |
| Instruction sheets | `styles/<name>/instructions.md` |
| REST endpoints | `src/scope/server/app.py` |
| Frame processing | `src/scope/server/frame_processor.py` |

---

## Branch

All work is on `feature/stream-recording`
