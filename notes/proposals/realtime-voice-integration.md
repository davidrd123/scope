# Real-Time Voice Integration

> Status: Draft
> Date: 2025-12-26
> Research: `notes/research/2025-12-26/voice_integration/oai_5pro_01.md`

## Summary

Add real-time voice input for controlling video generation — either voice-to-prompt (STT) or full voice chat (audio-in/audio-out). Voice as a natural control surface for live performances and creative exploration.

## The Vision

```
Voice Input → [ Voice Engine ] → Commands/Prompts → Video Pipeline
   (mic)      (STT or Chat)        (text)          (generation)
                   │
                   ▼
             Audio Response (optional)
```

Speak naturally to control the generation: "Make it more dramatic", "Hard cut to a night scene", "What's happening in this frame?". The voice system can be simple STT-to-prompt or full conversational AI.

## Architecture Options

### Option A: Chained STT → Text (Simple)

```
Browser Mic → Web Speech API → Text → Prompt API
                   or
Browser Mic → OpenAI Whisper → Text → Prompt API
```

**Pros:**
- Simple integration — text goes to existing API
- Full transcript control
- Works offline (Web Speech API)
- Low cost (Whisper is cheap)

**Cons:**
- Higher latency (~1-2s for full utterance)
- No audio response
- No barge-in (can't interrupt)

### Option B: OpenAI Realtime API (Voice Chat)

```
Browser Mic → WebRTC → OpenAI Realtime → Audio + Text
                           │
                           ├── Tool calls → Control API
                           └── Text response → Prompt API
```

**Pros:**
- Low latency (~300ms)
- Barge-in support (interrupt mid-response)
- Audio response (can speak back)
- Tool calling for structured commands
- Server control channel for backend monitoring

**Cons:**
- More complex integration
- Vendor lock-in (OpenAI)
- Higher cost (realtime pricing)

### Option C: Google Gemini Live (Multimodal)

```
Browser → WebSocket → Gemini Live → Audio + Text
             │
             └── Can also send video frames (multimodal)
```

**Pros:**
- Multimodal — can send audio AND video together
- Native audio model (not chained)
- Google ecosystem integration
- Potentially lower cost than OpenAI

**Cons:**
- Preview status on Vertex AI
- WebSocket complexity
- Less mature than OpenAI for voice

## Use Cases

### 1. Voice Prompting

Speak prompts instead of typing:

```
User: "A cyberpunk cityscape at night, neon lights reflecting on wet streets"
System: [Updates prompt via API]
```

**Implementation:** STT → POST /api/v1/realtime/prompt

### 2. Voice Commands

Control the system with voice:

| Command | Action |
|---------|--------|
| "Hard cut" | POST /api/v1/realtime/hard-cut |
| "Next scene" | POST /api/v1/realtime/playlist/next |
| "Previous" | POST /api/v1/realtime/playlist/prev |
| "Save this" / "Record" | POST /api/v1/realtime/recording/start |
| "Stop recording" | POST /api/v1/realtime/recording/stop |
| "More energy" | POST /api/v1/realtime/world (update tension) |
| "Switch to RAT style" | POST /api/v1/realtime/style |

**Implementation:** STT → Command Parser → API call

### 3. Conversational Control

Natural language refinement:

```
User: "Make it more dramatic"
System: [Interprets as: increase contrast, add motion, adjust prompt modifiers]

User: "What am I looking at?"
System: [Uses VLM to describe current frame, speaks response]

User: "Keep the composition but change to daytime"
System: [Modifies prompt while preserving structure]
```

**Implementation:** Voice Chat API + LLM reasoning + Control API

### 4. Buddy Collaboration

Voice notes during live session:

```
User: "Note: this transition works well with the synth swell"
System: [Records timestamped note to session log]
```

**Implementation:** STT → Session Recorder metadata

## Implementation Phases

### Phase 1: Voice-to-Prompt (MVP)

**Goal:** Speak prompts, basic commands.

**Components:**

1. **Browser STT** — Web Speech API (free, offline-capable)
   ```javascript
   const recognition = new webkitSpeechRecognition();
   recognition.continuous = true;
   recognition.onresult = (event) => {
     const transcript = event.results[0][0].transcript;
     // Send to backend
   };
   ```

2. **Command Detection** — Simple keyword matching
   ```python
   COMMANDS = {
       "hard cut": lambda: api.hard_cut(),
       "next scene": lambda: api.playlist_next(),
       "next": lambda: api.playlist_next(),
       "previous": lambda: api.playlist_prev(),
       "record": lambda: api.recording_start(),
       "stop recording": lambda: api.recording_stop(),
   }

   def handle_transcript(text: str):
       text_lower = text.lower().strip()
       for trigger, action in COMMANDS.items():
           if text_lower.startswith(trigger):
               return action()
       # Not a command — treat as prompt
       return api.set_prompt(text)
   ```

3. **Frontend Integration**
   ```typescript
   // In StreamPage.tsx
   const [voiceEnabled, setVoiceEnabled] = useState(false);

   useVoiceInput({
     enabled: voiceEnabled,
     onTranscript: (text) => {
       // Send to /api/v1/realtime/voice/transcript
     },
     onCommand: (cmd) => {
       // Handle recognized command
     },
   });
   ```

**Deliverables:**
- [ ] `useVoiceInput` hook in frontend
- [ ] `POST /api/v1/realtime/voice/transcript` endpoint
- [ ] Command parser with configurable keywords
- [ ] Voice toggle in UI

### Phase 2: Enhanced Commands

**Goal:** Richer command vocabulary, parameter extraction.

**Examples:**

```
"Set energy to 80 percent" → world.tension = 0.8
"Go to scene 5" → playlist.goto(5)
"Fade to the forest prompt" → playlist.goto(name="forest", transition="soft")
```

**Components:**

1. **Intent Extraction** — Regex patterns or small LLM
   ```python
   patterns = [
       (r"set (\w+) to (\d+)", extract_parameter_command),
       (r"go to scene (\d+)", extract_goto_command),
       (r"fade to (.+)", extract_transition_command),
   ]
   ```

2. **Fuzzy Matching** — For prompt names, style names
   ```python
   from rapidfuzz import process

   def match_prompt_name(query: str, prompts: list[str]) -> str:
       match, score, _ = process.extractOne(query, prompts)
       return match if score > 70 else None
   ```

### Phase 3: Voice Chat

**Goal:** Full conversational control with audio response.

**OpenAI Realtime Integration:**

```python
# Server-side WebSocket handler
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def handle_voice_session(websocket):
    async with client.beta.realtime.connect(model="gpt-realtime") as conn:
        # Configure tools
        await conn.session.update(session={
            "tools": [
                {
                    "type": "function",
                    "name": "set_prompt",
                    "description": "Set the video generation prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"}
                        }
                    }
                },
                {
                    "type": "function",
                    "name": "hard_cut",
                    "description": "Trigger a hard cut to reset the scene"
                },
                # ... more tools
            ],
            "instructions": """
            You are controlling a real-time video generation system.
            The user can speak prompts or commands.
            Use the available tools to control the video.
            """
        })

        # Relay audio between browser and OpenAI
        async for event in conn:
            if event.type == "response.audio.delta":
                await websocket.send(event.delta)
            elif event.type == "response.function_call_arguments.done":
                # Execute tool call
                result = await execute_tool(event.name, event.arguments)
                await conn.response.create()
```

**With VLM for "What am I seeing?":**

```python
async def describe_current_frame():
    frame = await get_latest_frame()
    description = await vlm_client.describe(frame)
    return description

# Tool definition
{
    "name": "describe_frame",
    "description": "Describe what's currently being generated"
}
```

## Network Topology

### Option A: Browser-Direct (Phase 1-2)

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER                                                        │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  Web Speech API │────▶│  Voice Handler  │                   │
│  │  (local STT)    │     │  (JS)           │                   │
│  └─────────────────┘     └────────┬────────┘                   │
│                                   │                             │
└───────────────────────────────────┼─────────────────────────────┘
                                    │ HTTP
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│  SERVER                                                           │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────┐ │
│  │  Voice Endpoint │────▶│  Command Parser │────▶│  Control    │ │
│  │  (FastAPI)      │     │                 │     │  APIs       │ │
│  └─────────────────┘     └─────────────────┘     └─────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### Option B: Server-Mediated Voice Chat (Phase 3)

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER                                                        │
│  ┌─────────────────┐                                           │
│  │  Audio Capture  │◀────────────────────────────────────────┐ │
│  │  (MediaStream)  │                                         │ │
│  └────────┬────────┘                                         │ │
│           │ WebSocket                                        │ │
└───────────┼──────────────────────────────────────────────────┼─┘
            │                                                  │
            ▼                                                  │
┌───────────────────────────────────────────────────────────────────┐
│  SERVER                                                           │
│  ┌─────────────────┐     ┌─────────────────┐                     │
│  │  Voice Gateway  │────▶│  OpenAI Realtime│                     │
│  │  (WebSocket)    │◀────│  (WebRTC/WS)    │                     │
│  └────────┬────────┘     └─────────────────┘                     │
│           │                       │                               │
│           │              ┌────────▼────────┐                     │
│           │              │  Tool Executor  │                     │
│           │              │  (hard_cut, etc)│                     │
│           │              └─────────────────┘                     │
│           │                                                       │
│           └──────────────────────────────────────────────────────┤
│                          Audio Response                          │
└───────────────────────────────────────────────────────────────────┘
```

## API Design

### REST Endpoints

```
POST /api/v1/realtime/voice/transcript
  Body: { "text": "...", "is_final": true }
  Response: { "action": "prompt" | "command", "executed": true }

GET /api/v1/realtime/voice/status
  Response: { "enabled": true, "mode": "stt" | "chat", "listening": true }

POST /api/v1/realtime/voice/enable
POST /api/v1/realtime/voice/disable
```

### WebSocket (Phase 3)

```
ws://server/api/v1/realtime/voice/chat

Client → Server:
  { "type": "audio", "data": "<base64 PCM>" }
  { "type": "config", "language": "en" }

Server → Client:
  { "type": "audio", "data": "<base64 PCM>" }
  { "type": "transcript", "text": "...", "is_final": true }
  { "type": "action", "name": "hard_cut", "status": "executed" }
```

## Configuration

```python
# In schema.py

class VoiceConfig(BaseModel):
    enabled: bool = Field(default=False)
    mode: Literal["stt", "chat"] = Field(default="stt")
    language: str = Field(default="en-US")

    # Phase 1: STT
    stt_provider: Literal["browser", "whisper"] = Field(default="browser")

    # Phase 3: Voice Chat
    chat_provider: Literal["openai", "gemini"] = Field(default="openai")
    chat_model: str = Field(default="gpt-realtime")
```

**Environment variables:**

```bash
SCOPE_VOICE_ENABLED=1
SCOPE_VOICE_MODE=stt  # or "chat"
SCOPE_VOICE_PROVIDER=browser  # or "whisper", "openai", "gemini"

# For Phase 3
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

## Dependencies

**Phase 1 (MVP):**
- None — Web Speech API is built into browsers

**Phase 2:**
- `rapidfuzz` — For fuzzy matching prompt/style names

**Phase 3:**
- `openai>=1.0` — For Realtime API
- `google-cloud-aiplatform` — For Gemini Live (optional)
- WebSocket support in frontend

## Performance Considerations

| Phase | Latency | Cost |
|-------|---------|------|
| Web Speech API | ~500ms | Free |
| OpenAI Whisper | ~1-2s | $0.006/min |
| OpenAI Realtime | ~300ms | $0.06/min audio |
| Gemini Live | ~300ms | TBD |

**For live performance:** Phase 1 (Web Speech) is surprisingly usable. Phase 3 (Realtime) is better for conversational flow.

## Open Questions

- [ ] **Which vendor for Phase 3?** OpenAI Realtime vs Gemini Live vs both?
- [ ] **Audio output needed?** Or is text response sufficient?
- [ ] **Push-to-talk vs always listening?** Push-to-talk better for noisy environments
- [ ] **Command confirmation?** "Did you say hard cut?" before executing destructive commands
- [ ] **Multilingual?** Web Speech API supports many languages; Whisper supports 100+

## Files to Create

| File | Purpose |
|------|---------|
| `frontend/src/hooks/useVoiceInput.ts` | Browser STT hook |
| `frontend/src/components/VoiceControls.tsx` | UI for voice toggle, status |
| `src/scope/server/voice/handler.py` | Transcript processing, command parsing |
| `src/scope/server/voice/commands.py` | Command registry and execution |
| `src/scope/server/voice/chat.py` | Phase 3: Voice chat gateway |
| `src/scope/server/schema.py` | VoiceConfig addition |

## References

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [OpenAI Realtime WebRTC](https://platform.openai.com/docs/guides/realtime-webrtc)
- [OpenAI Voice Agents Guide](https://platform.openai.com/docs/guides/voice-agents)
- [Gemini Live API (Vertex AI)](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api)
- [Web Speech API (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- [OpenAI Whisper](https://platform.openai.com/docs/guides/speech-to-text)
