What both OpenAI and Google offer today generally falls into two buckets:

1. **Real-time transcription (audio → text)**

   * You stream mic audio and get partial / final text back quickly (captions, command-and-control, meeting notes).

2. **Real-time “voice chat” (audio → audio, optionally with text/transcripts)**

   * A stateful, low-latency session where the model *hears you* continuously and *speaks back*, often with **barge‑in** (user can interrupt) and tool/function calling.

Below is what each vendor has “on the table” for those two buckets, and what you’d typically use each for.

---

## OpenAI: real-time voice input options

### 1) Realtime API (native speech-to-speech “voice chat”)

**What it is:** A low-latency, stateful API for **multimodal** interaction: audio/text/image in, audio/text out. It’s positioned as the main way to build **voice agents** and also supports realtime transcription. ([OpenAI Platform][1])

**How you connect (important for app architecture):**

* **WebRTC**: recommended for browser/mobile client connections for more consistent realtime performance. ([OpenAI Platform][2])
* **WebSocket**: recommended for server-to-server realtime. ([OpenAI Platform][3])
* **SIP**: supported for VoIP/telephony-style connections. ([OpenAI Platform][1])

**Model(s):**

* `gpt-realtime` is described as OpenAI’s **first general-availability realtime model**, supporting realtime audio + text over WebRTC/WebSocket/SIP. ([OpenAI Platform][4])

**What “streaming audio in” looks like:**

* Over **WebRTC**, media handling is mostly “native” (audio tracks).
* Over **WebSockets**, you push audio into an input buffer using JSON events that carry **base64-encoded audio**. ([OpenAI Platform][5])

**If you need server-side control while the client is connected:**

* OpenAI supports a **sideband**/server control channel so your backend can monitor the same session, update instructions, and handle tool calls while the user is connected via WebRTC or SIP. ([OpenAI Platform][6])

**When you’d pick this:**

* You want the closest thing to “voice chat” UX: fast turn-taking, interruptions, streaming audio out, and richer conversational feel. OpenAI explicitly frames this as the “speech-to-speech” architecture vs a chained pipeline. ([OpenAI Platform][7])

---

### 2) Realtime transcription mode (audio → text, realtime)

If you *don’t* want the model to talk back—just to caption/transcribe—you can create a Realtime **transcription session** (still over WebRTC or WebSocket). In this mode, the model typically doesn’t generate responses. ([OpenAI Platform][8])

**When you’d pick this:**

* Live captions/subtitles
* Streaming transcripts for search/indexing
* Voice-to-text input for an app UI (without TTS)

---

### 3) Audio API (speech-to-text and text-to-speech, “non-Realtime” building blocks)

OpenAI also provides “classic” endpoints you can chain:

* **Speech-to-text** (`audio/transcriptions`) with models like `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1`, etc. ([OpenAI Platform][9])
* **Text-to-speech** (`audio/speech`) with models like `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`. ([OpenAI Platform][10])

OpenAI’s docs explicitly describe two architectures:

* **Speech-to-speech (Realtime API)** = lowest latency / most natural
* **Chained** (STT → text LLM → TTS) = more predictable + easier transcript control ([OpenAI Platform][7])

**When you’d pick this:**

* You want tighter control and an always-available text transcript
* You already have a text-agent and just want to “add voice” reliably

---

## Google: real-time voice input options

Google basically offers **(A)** “Gemini Live” for voice chat, and **(B)** Cloud Speech-to-Text for pure transcription.

### 1) Gemini Live API on Vertex AI (voice chat: audio/video/text in ↔ audio/text out)

**What it is:** A **low-latency**, bidirectional, stateful Live API for **voice and video** interaction with Gemini. It supports barge‑in and tool use. ([Google Cloud Documentation][11])

**Transport & formats (good to know up front):**

* Vertex AI docs describe it as a **stateful WebSocket (WSS) connection**. ([Google Cloud Documentation][11])
* The overview page lists input/output audio specs like **raw 16-bit PCM** (16kHz input; 24kHz output) in its technical specs section. ([Google Cloud Documentation][11])

**Models / availability:**

* Vertex AI lists `gemini-live-2.5-flash-native-audio` as **generally available** and “recommended” for low-latency voice agents. ([Google Cloud Documentation][11])
* Google’s Cloud blog (Dec 2025) also states the **general availability** of Gemini Live API on Vertex AI powered by Gemini 2.5 Flash Native Audio, and frames it as moving from a STT→LLM→TTS pipeline to a single native-audio model. ([Google Cloud][12])

**Client apps (web/mobile) note:**

* Vertex AI reference docs emphasize Live API is designed for **server-to-server**, and suggest partner integrations for web/mobile scenarios. ([Google Cloud Documentation][13])
* The Vertex AI overview also lists partners (Daily, LiveKit, Twilio, Voximplant) that integrate over **WebRTC** to simplify realtime media handling. ([Google Cloud Documentation][11])

**When you’d pick this:**

* You want Google’s end-to-end “voice chat” style UX (audio in/out, interruptions, multimodal).
* You’re already on Google Cloud / Vertex AI and want managed infra + enterprise controls.

---

### 2) Gemini “Live API” via Google AI for Developers (Gemini API) + ephemeral tokens

Google also documents a “Live API” in the Gemini Developer ecosystem (AI Studio / Gemini API). It supports streaming audio input/output with raw PCM expectations (little-endian 16‑bit PCM; output 24kHz; input is natively 16kHz but can be resampled). ([Google AI for Developers][14])

For connecting directly from the client, Google provides **ephemeral tokens**:

* They’re **short-lived** auth tokens intended for client-to-server **WebSocket** access to Gemini Live, to reduce the risk of shipping a long-lived API key in a browser/mobile app. ([Google AI for Developers][15])
* The doc notes defaults like ~**1 minute** to start a new Live session and ~**30 minutes** for messaging on that connection (unless configured). ([Google AI for Developers][15])

**When you’d pick this:**

* You want a direct-to-client Live connection pattern (or a hybrid), and you’re building around Gemini API / AI Studio workflows.

---

### 3) Firebase AI Logic (client SDK path to Gemini Live)

If you’re building client apps (especially mobile), Google offers Firebase AI Logic docs for getting started with Gemini Live:

* It explicitly describes Live API as **bidirectional realtime voice/video** streaming (audio responses), and says it creates a **WebSocket connection** for the session. ([Firebase][16])
* The Firebase AI Logic integration is called out as **Preview** (may change, no SLA). ([Firebase][16])
* Android docs reinforce that you can call Live API from Android via Firebase AI Logic without a backend, also in developer preview. ([Android Developers][17])

**When you’d pick this:**

* You want a “no backend” or lighter-backend path for mobile prototypes (and can accept preview constraints).

---

### 4) Cloud Speech-to-Text (streaming transcription only)

If you just need **real-time speech recognition** (audio → text), Google Cloud Speech-to-Text supports streaming recognition:

* You stream audio (e.g., mic input) and get recognition results **in real time** as audio is processed. ([Google Cloud Documentation][18])
* The docs note streaming recognition is available **through gRPC only**. ([Google Cloud Documentation][18])

**When you’d pick this:**

* You need very strong standalone ASR (captions, commands, call transcription), and you’ll feed the text into another system (Gemini text model, your own NLU, etc.).

---

## Quick “which should I use?” guide

### If you want “voice chat” UX (audio in ↔ audio out, barge‑in)

* **OpenAI:** Realtime API (WebRTC for client, WebSocket for server; SIP available). ([OpenAI Platform][1])
* **Google:** Gemini Live API (Vertex AI Live via WSS; partners for WebRTC; Firebase AI Logic option). ([Google Cloud Documentation][11])

### If you only need realtime voice input as text (captions / dictation / commands)

* **OpenAI:** Realtime transcription mode (WebRTC/WebSocket) or Audio transcriptions endpoint. ([OpenAI Platform][8])
* **Google:** Cloud Speech-to-Text streaming (gRPC). ([Google Cloud Documentation][18])

### If you want maximum control + a guaranteed transcript (classic pipeline)

* **OpenAI’s recommended “chained architecture”**: STT → text LLM → TTS. ([OpenAI Platform][7])
* Google’s equivalent is typically Cloud Speech-to-Text streaming → Gemini text model → TTS (not detailed above, but that’s the standard decomposition).

---

## Two practical integration questions that decide a lot

If you tell me these two things, I can recommend a “best-fit” path and a reference architecture (client/server split, protocols, buffering strategy):

1. Is your app **web**, **mobile**, **desktop**, or **telephony**?
2. Do you want the assistant to **speak back in real time**, or is **text output** enough?

[1]: https://platform.openai.com/docs/guides/realtime "Realtime API | OpenAI API"
[2]: https://platform.openai.com/docs/guides/realtime-webrtc "Realtime API with WebRTC | OpenAI API"
[3]: https://platform.openai.com/docs/guides/realtime-websocket "Realtime API with WebSocket | OpenAI API"
[4]: https://platform.openai.com/docs/models/gpt-realtime "gpt-realtime Model | OpenAI API"
[5]: https://platform.openai.com/docs/guides/realtime-conversations "Realtime conversations | OpenAI API"
[6]: https://platform.openai.com/docs/guides/realtime-server-controls "Webhooks and server-side controls | OpenAI API"
[7]: https://platform.openai.com/docs/guides/voice-agents "Voice agents | OpenAI API"
[8]: https://platform.openai.com/docs/guides/realtime-transcription "Realtime transcription | OpenAI API"
[9]: https://platform.openai.com/docs/guides/speech-to-text?utm_source=chatgpt.com "Speech to text | OpenAI API"
[10]: https://platform.openai.com/docs/guides/audio "Audio and speech | OpenAI API"
[11]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api "Gemini Live API overview  |  Generative AI on Vertex AI  |  Google Cloud Documentation"
[12]: https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai "How to use Gemini Live API Native Audio in Vertex AI | Google Cloud Blog"
[13]: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-live "Gemini Live API reference  |  Generative AI on Vertex AI  |  Google Cloud Documentation"
[14]: https://ai.google.dev/gemini-api/docs/live-guide "Live API capabilities guide  |  Gemini API  |  Google AI for Developers"
[15]: https://ai.google.dev/gemini-api/docs/ephemeral-tokens "Ephemeral tokens  |  Gemini API  |  Google AI for Developers"
[16]: https://firebase.google.com/docs/ai-logic/live-api "Get started with the Gemini Live API using Firebase AI Logic  |  Firebase AI Logic"
[17]: https://developer.android.com/ai/gemini/live "Gemini Live API  |  AI  |  Android Developers"
[18]: https://docs.cloud.google.com/speech-to-text/docs/streaming-recognize "Transcribe audio from streaming input  |  Cloud Speech-to-Text  |  Google Cloud Documentation"
