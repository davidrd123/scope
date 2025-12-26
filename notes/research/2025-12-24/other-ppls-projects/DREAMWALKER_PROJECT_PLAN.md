# Open-World Narrative AI Video Exploration

## Project Overview

An interactive storytelling experience where a player with a camera walks through physical space while an AI Storyteller narrates and responds to their choices. The player's camera feed is transformed in real-time using Scope's AI video processing, creating a "reimagined" view of reality. Secondary participants (NPCs, Dungeon Masters) can influence the narrative and see the same transformed stream.

### Goals

- Real-time AI video transformation of player's camera feed
- Interactive narrative driven by Storyteller LLM
- Dynamic prompt and reference image control based on story events

### Stretch Goals
- Multiple participant roles: Player Character, NPC, Dungeon Master
- Unified transformed video stream visible to all participants
- **NPCs**: Secondary users choose character avatars, receive story-relevant dialogue prompts via headphones
- **6DoF Tracking**: Prompt and reference image changes based on physical location
- **Additional Hardware**: VR headset, Rayban Meta glasses, Android/iOS support
- **Real-time Preprocessing**: Depth and pose extraction from camera feed
- **Basic Quest System**: Fetch quests encouraging return to previous locations
- **Twitch Integration**: Audience votes on story events via chat

### Non-Goals

- 3D video projection (output is flat 2D video)
- Robust game mechanics (focus is exploration and narrative)
- Multi-user scaling (prototype targets single play session)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              WINDOWS HOST PC                                     │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │ Player Camera   │    │ Preprocessing   │    │ Scope                       │ │
│  │ (Android/PC)    │───►│ Service (GPU 2) │───►│ (GPU 1)                     │ │
│  │                 │    │ Depth + Pose    │    │ VACE + R2V                  │ │
│  └─────────────────┘    └─────────────────┘    │                             │ │
│                                                │ Inputs:                     │ │
│                                                │ • Preprocessed video        │ │
│  ┌─────────────────┐                          │ • Text prompt               │ │
│  │ Storyteller LLM │─────────────────────────►│ • Reference image           │ │
│  │ (Claude API)    │                          │                             │ │
│  │ Vision-capable  │                          └──────────────┬──────────────┘ │
│  └────────┬────────┘                                         │ Spout          │
│           │                                                  ▼                │
│           │              ┌────────────────────────────────────────────────┐   │
│           │              │ Unity Host (Compositor + Game Logic)           │   │
│           │              │                                                │   │
│           │ TTS Audio    │ • Receives Spout video                         │   │
│           └─────────────►│ • Mixes TTS + Music + SFX                      │   │
│                          │ • Manages game state                           │   │
│                          │ • Routes to participants via WebRTC            │   │
│                          └──────────────┬─────────────────────────────────┘   │
└─────────────────────────────────────────┼─────────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
        ▼                                 ▼                                 ▼
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ Player          │              │ NPC             │              │ DM              │
│ (Local/Android) │              │ (Android)       │              │ (Android)       │
│                 │              │                 │              │                 │
│ • Transformed   │              │ • Video + Audio │              │ • Video + Audio │
│   video         │              │ • Whisper       │              │ • Meta-controls │
│ • Narrator      │              │   prompts       │              │ • Story         │
│   audio         │              │ • "In view"     │              │   direction     │
│ • Choice UI     │              │   button        │              │                 │
└─────────────────┘              └─────────────────┘              └─────────────────┘

                    ┌─────────────────────────────────────┐
                    │           RUNPOD (Remote)           │
                    │                                     │
                    │  ComfyUI Image Generation           │
                    │  • Quest item references            │
                    │  • NPC avatar images                │
                    │  • Composite reference images       │
                    │  • Scene style references           │
                    │                                     │
                    │  Triggered by Storyteller LLM       │
                    └─────────────────────────────────────┘
```

---

## Component Specifications

### 1. Unity Application (Host)

Multi-mode application running on Windows PC as authoritative host.

#### User Modes

| Mode | Input | Output | Controls |
|------|-------|--------|----------|
| **Player Character** | Device camera, choice buttons | Transformed video, narrator audio | Story choices |
| **NPC** | Avatar selection, "in view" button | Transformed video, narrator audio, whisper prompts | Dialogue delivery |
| **Dungeon Master** | Meta-choice buttons | Transformed video, narrator audio | Story direction |
| **Twitch DM** | Chat vote results | Stream output | Audience-driven choices |

#### Responsibilities

- Aggregate inputs from all participants
- Communicate with Storyteller LLM
- Route TTS audio (narrator to all, whispers to NPC only)
- Manage game state and quest progression
- Send prompts and reference images to Scope
- Relay video stream to remote participants via WebRTC

### 2. Preprocessing Service

Dedicated Python service running on GPU 2, converting raw RGB camera feed into VACE-compatible control signals.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 PREPROCESSING SERVICE (GPU 2)                    │
│                                                                 │
│  Input: RGB frames via shared memory / socket from Unity        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Frame Router                          │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│           ┌────────────────┴────────────────┐                  │
│           ▼                                 ▼                  │
│  ┌─────────────────────┐        ┌─────────────────────────┐   │
│  │  Depth Anything V2  │        │  MediaPipe Pose         │   │
│  │  (GPU inference)    │        │  (CPU, fast enough)     │   │
│  └──────────┬──────────┘        └────────────┬────────────┘   │
│             │                                │                  │
│             └────────────┬───────────────────┘                  │
│                          ▼                                      │
│             ┌─────────────────────────┐                        │
│             │  Composite Generator    │                        │
│             │  R=Depth, G=Pose, B=0   │                        │
│             └────────────┬────────────┘                        │
│                          │                                      │
│  Output: Control frames via Spout sender to Scope              │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
                    Scope (GPU 1)
```

#### Implementation

```python
# preprocessing_service.py

import torch
import cv2
import mediapipe as mp
from depth_anything_v2.dpt import DepthAnythingV2
import SpoutGL

class PreprocessingService:
    def __init__(self, device="cuda:1"):
        self.device = device

        # Depth model on GPU 2
        self.depth_model = DepthAnythingV2(
            encoder='vits',  # Small model for speed
            features=64,
            out_channels=[48, 96, 192, 384]
        ).to(device)
        self.depth_model.load_state_dict(torch.load('depth_anything_v2_vits.pth'))
        self.depth_model.eval()

        # MediaPipe on CPU (fast enough, keeps GPU free)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5
        )

        # Spout output to Scope
        self.spout_sender = SpoutGL.SpoutSender()
        self.spout_sender.setSenderName("PreprocessedControl")

    @torch.no_grad()
    def process_frame(self, rgb_frame):
        # Depth estimation
        depth = self.depth_model.infer_image(rgb_frame)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint8 = (depth_normalized * 255).astype('uint8')

        # Pose estimation
        pose_result = self.pose.process(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
        pose_overlay = self._render_pose(rgb_frame.shape, pose_result)

        # Composite: R=Depth, G=Pose, B=0
        composite = np.zeros((*rgb_frame.shape[:2], 3), dtype='uint8')
        composite[:, :, 0] = depth_uint8
        composite[:, :, 1] = pose_overlay

        return composite

    def _render_pose(self, shape, pose_result):
        overlay = np.zeros(shape[:2], dtype='uint8')
        if pose_result.pose_landmarks:
            # Draw skeleton on overlay
            mp.solutions.drawing_utils.draw_landmarks(
                overlay, pose_result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS
            )
        return overlay

    def run(self, input_spout_name="UnityCamera"):
        spout_receiver = SpoutGL.SpoutReceiver()

        while True:
            frame = spout_receiver.receiveTexture(input_spout_name)
            if frame is not None:
                processed = self.process_frame(frame)
                self.spout_sender.sendTexture(processed)
```

### 3. Scope Configuration

Local Windows installation on GPU 1 with Spout I/O enabled.

#### Pipeline Selection

| Pipeline | VRAM | Quality | Speed | VACE Support |
|----------|------|---------|-------|--------------|
| `longlive` | ~18GB | High | ~6fps | Full |
| `reward-forcing` | ~18GB | High | ~6fps | Full |
| `streamdiffusionv2` | ~12GB | Medium | ~15fps | Limited |

**Recommended**: `longlive` or `reward-forcing` for quality, `streamdiffusionv2` if frame rate is priority.

#### VACE Mode Configuration

When VACE is enabled, Scope automatically routes incoming video to `vace_input_frames` (conditioning signal). The preprocessing pipeline output (depth + pose) becomes the structural guide while reference images provide style.

#### Spout Configuration

- **Input**: `PreprocessedControl` (from Preprocessing Service)
- **Output**: `ScopeSyphonSpoutOut` (to Unity Host)
- Enable VACE in Settings panel
- Enable Spout Receiver, set name to `PreprocessedControl`
- Enable Spout Sender

#### DataChannel Control Messages

```javascript
// Update text prompt
{ prompts: [{ text: "forest scene, mystical", weight: 1.0 }] }

// Smooth prompt transition
{
  transition: {
    target_prompts: [{ text: "dragon appears", weight: 1.0 }],
    num_steps: 8,
    temporal_interpolation_method: "slerp"
  }
}

// Set reference image (uploaded via REST API first)
{ vace_ref_images: ["/assets/composite_ref.png"], vace_context_scale: 1.0 }

// Pipeline control
{ denoising_step_list: [700, 500], noise_scale: 0.7 }
```

### 4. Storyteller LLM Agent

Vision-capable Claude model for narrative generation with scene understanding.

#### Input Context

```json
{
  "scene_history": ["Player entered forest", "Found mysterious artifact"],
  "current_location": "Ancient ruins, north section",
  "active_quest": { "name": "Find the Crystal", "progress": 2, "total": 5 },
  "npc_states": [
    { "id": "wizard", "present": true, "in_view": true, "disposition": "friendly" }
  ],
  "player_last_choice": "Examine the artifact",
  "dm_directive": "Introduce danger",
  "current_frame": "<base64 encoded frame or vision analysis>"
}
```

#### Structured Output

```json
{
  "scope_prompt": "ancient ruins, glowing crystal, mystical energy, fantasy",
  "image_gen": {
    "trigger": true,
    "prompt": "glowing blue crystal on stone pedestal, fantasy art, [npc_avatar:wizard], [quest_item:crystal]",
    "type": "composite"
  },
  "narration": "As your fingers brush the crystal's surface, a pulse of energy ripples through the chamber...",
  "npc_whisper": "Look surprised, then warn the player about the crystal's power",
  "player_choices": [
    "Take the crystal",
    "Leave it alone",
    "Ask the wizard about it"
  ],
  "dm_choices": [
    "Crystal is cursed",
    "Crystal grants vision",
    "Guardian awakens"
  ],
  "prefetch_images": [
    { "prompt": "stone guardian awakening, fantasy", "probability": 0.4 },
    { "prompt": "magical vision effect, ethereal", "probability": 0.3 }
  ]
}
```

### 5. ComfyUI Image Generation (Runpod)

Remote ComfyUI instance generating composite reference images for VACE.

#### Composite Reference Strategy

Since VACE accepts only one reference image, ComfyUI composites multiple elements into a single image:

```
┌─────────────────────────────────────────────────────────────────┐
│              COMPOSITE REFERENCE IMAGE                           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ NPC Avatar  │  │ Quest Item  │  │ Scene Style │             │
│  │ (if in view)│  │ (if active) │  │ (always)    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│                 ┌─────────────────┐                            │
│                 │ ComfyUI Node    │                            │
│                 │ Grid Composite  │                            │
│                 │ or Blend        │                            │
│                 └────────┬────────┘                            │
│                          │                                      │
│                          ▼                                      │
│                 Single Reference Image                          │
│                 (uploaded to Scope)                             │
└─────────────────────────────────────────────────────────────────┘
```

#### Workflow

1. Storyteller triggers `image_gen` with composite requirements
2. ComfyUI on Runpod generates/retrieves component images
3. Components composited into grid or blended image
4. Result uploaded to Scope via `POST /api/v1/assets`
5. DataChannel message updates `vace_ref_images`

#### Pre-generation Strategy

- **Prefetch queue**: Generate likely next images from `prefetch_images`
- **Asset cache**: Pre-authored images for common elements (stored on Runpod)
- **NPC avatars**: Generated once when NPC selects appearance, cached for session

### 6. Audio System

#### Channels

| Channel | Source | Recipients | Purpose |
|---------|--------|------------|---------|
| Narrator | ElevenLabs TTS | All participants | Story narration |
| NPC Whisper | ElevenLabs TTS | NPC only (headphones) | Dialogue prompts |
| Background Music | Pre-authored | All participants | Atmosphere |
| SFX | Triggered events | All participants | Feedback |

#### Unity Audio Routing

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unity AudioMixer                              │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Narrator    │  │ NPC Whisper │  │ Music/SFX   │             │
│  │ (TTS)       │  │ (TTS)       │  │             │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────────────────────────────────────────┐           │
│  │              Master Output                       │           │
│  │  (to all participants via WebRTC AudioTrack)    │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │              NPC Private Channel                 │           │
│  │  (whisper only, separate WebRTC AudioTrack)     │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 7. Video Distribution

#### Local Distribution (Windows)

**Spout** handles all local video routing with zero-copy GPU texture sharing:

```
Scope Spout Sender ("ScopeSyphonSpoutOut")
        │
        ├──► Unity Host (game logic + compositing)
        └──► OBS (optional backup / Twitch source)
```

#### Remote Distribution (Android NPC/DM)

**Unity-to-Unity WebRTC** for participants on Android devices (same network):

```csharp
// PC Unity Host - sends to remote participants
public class RemoteVideoRelay : MonoBehaviour
{
    private SpoutReceiver _spoutReceiver;
    private VideoStreamTrack _videoTrack;
    private AudioStreamTrack _audioTrack;
    private Dictionary<string, RTCPeerConnection> _connections;

    void Start()
    {
        _spoutReceiver = GetComponent<SpoutReceiver>();
        _videoTrack = new VideoStreamTrack(_spoutReceiver.targetTexture);
        _audioTrack = new AudioStreamTrack(GetComponent<AudioListener>());
    }

    public async Task ConnectRemoteParticipant(string odparticipantId, RTCSessionDescription offer)
    {
        var pc = new RTCPeerConnection();
        pc.AddTrack(_videoTrack);
        pc.AddTrack(_audioTrack);

        await pc.SetRemoteDescription(offer);
        var answer = await pc.CreateAnswer();
        await pc.SetLocalDescription(answer);

        _connections[participantId] = pc;
        NetworkManager.SendWebRTCAnswer(participantId, answer);
    }
}
```

```csharp
// Android Unity - receives video + audio
public class RemoteVideoReceiver : MonoBehaviour
{
    private RTCPeerConnection _peerConnection;
    private RawImage _videoDisplay;
    private AudioSource _audioOutput;

    public async Task ConnectToHost()
    {
        _peerConnection = new RTCPeerConnection();

        _peerConnection.OnTrack = (RTCTrackEvent e) =>
        {
            if (e.Track is VideoStreamTrack video)
                video.OnVideoReceived += tex => _videoDisplay.texture = tex;
            else if (e.Track is AudioStreamTrack audio)
            {
                _audioOutput.SetTrack(audio);
                _audioOutput.Play();
            }
        };

        var offer = await _peerConnection.CreateOffer();
        await _peerConnection.SetLocalDescription(offer);
        NetworkManager.SendWebRTCOffer(offer);
    }
}
```

#### Twitch Streaming

Unity Compositor outputs combined video + audio to Twitch via OBS:

```
Unity Host
    │
    └──► Spout Out ──► OBS ──► RTMP ──► Twitch
              │
              └── Audio via Virtual Audio Cable from Unity
```

---

## Data Flow: Complete Processing Chain

### Frame Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: CAPTURE                                                            │
│                                                                             │
│ Player Camera (Android/PC)                                                  │
│     │                                                                       │
│     │ 512x512 @ 30fps                                                       │
│     │                                                                       │
│     ▼                                                                       │
│ WebRTC to PC (if Android) OR direct capture (if PC)                        │
└─────┼───────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING (GPU 2)                                              │
│                                                                             │
│ RGB Frame                                                                   │
│     │                                                                       │
│     ├──► Depth Anything V2 ──► Depth Map                                   │
│     │                              │                                        │
│     └──► MediaPipe Pose (CPU) ──► Skeleton Overlay                         │
│                                    │                                        │
│                                    ▼                                        │
│                          Composite Control Frame                            │
│                          (R=Depth, G=Pose, B=0)                            │
│                                    │                                        │
│                                    │ Spout: "PreprocessedControl"           │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: AI VIDEO TRANSFORMATION (GPU 1)                                    │
│                                                                             │
│ Scope (VACE Mode)                                                           │
│     │                                                                       │
│     ├── Spout Receiver: "PreprocessedControl" → vace_input_frames          │
│     ├── vace_ref_images: Composite from ComfyUI (Runpod)                   │
│     ├── prompts: Text guidance from Storyteller                            │
│     │                                                                       │
│     ▼                                                                       │
│ Diffusion Processing (longlive / reward-forcing)                           │
│     │                                                                       │
│     ▼                                                                       │
│ Transformed Video Output                                                    │
│     │                                                                       │
│     │ Spout: "ScopeSyphonSpoutOut"                                         │
└─────┼───────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: COMPOSITING & DISTRIBUTION                                         │
│                                                                             │
│ Unity Host (Spout Receiver)                                                 │
│     │                                                                       │
│     ├── Video: Transformed frames from Scope                               │
│     ├── Audio: Mixed TTS + Music + SFX                                     │
│     ├── UI: Overlaid choice buttons, status                                │
│     │                                                                       │
│     ▼                                                                       │
│ Distribution:                                                               │
│     ├──► Local display (Player on PC)                                      │
│     ├──► WebRTC to Android (NPC, DM) - same network                        │
│     └──► OBS → RTMP to Twitch                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Storyteller Decision Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EVENT TRIGGERS                                                              │
│                                                                             │
│ • Player selects choice                                                     │
│ • DM selects meta-choice                                                    │
│ • NPC delivers dialogue / presses "in view" button                         │
│ • Timer expires (ambient narration)                                         │
│ • Location change detected (6DoF stretch goal)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STORYTELLER LLM (Claude API - Vision Capable)                               │
│                                                                             │
│ Context:                                                                    │
│ • System prompt (narrator personality, world rules)                         │
│ • Scene history (last N events)                                            │
│ • Current state (location, quest, NPCs, NPC in-view status)                │
│ • Trigger event details                                                     │
│ • Current frame (periodic vision analysis)                                  │
│                                                                             │
│ Structured output generated                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT ROUTING                                                              │
│                                                                             │
│ scope_prompt ──────────────► Scope DataChannel                             │
│                              { prompts: [...], transition: {...} }          │
│                                                                             │
│ image_gen ─────────────────► ComfyUI (Runpod)                              │
│                              Generate composite → Upload → Scope            │
│                              { vace_ref_images: [...] }                     │
│                                                                             │
│ narration ─────────────────► ElevenLabs TTS → Unity (all participants)     │
│                                                                             │
│ npc_whisper ───────────────► ElevenLabs TTS → Unity (NPC only)             │
│                                                                             │
│ player_choices ────────────► Unity UI (Player device)                      │
│                                                                             │
│ dm_choices ────────────────► Unity UI (DM device)                          │
│                                                                             │
│ prefetch_images ───────────► ComfyUI background queue                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Reference

### Scope Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/pipeline/load` | POST | Load VACE-enabled pipeline |
| `/api/v1/pipeline/status` | GET | Check pipeline ready state |
| `/api/v1/webrtc/offer` | POST | Establish WebRTC connection (player camera input) |
| `/api/v1/webrtc/ice-servers` | GET | Get TURN/STUN configuration |
| `/api/v1/assets` | POST | Upload reference image |
| `/api/v1/assets` | GET | List available assets |

### Scope DataChannel Protocol

```javascript
// Text prompt update
{ prompts: [{ text: "mystical forest, glowing mushrooms", weight: 1.0 }] }

// Smooth prompt transition (over N frames)
{
  transition: {
    target_prompts: [{ text: "dragon emerges from shadows", weight: 1.0 }],
    num_steps: 8,
    temporal_interpolation_method: "slerp"
  }
}

// Reference image update (after uploading via REST)
{ vace_ref_images: ["/assets/composite_ref.png"], vace_context_scale: 1.0 }

// Pipeline parameters
{ denoising_step_list: [700, 500], noise_scale: 0.7 }

// Pause/resume processing
{ paused: true }
```

### Storyteller LLM Schema

```typescript
interface StorytellerInput {
  scene_history: string[];
  current_location: string;
  active_quest: { name: string; progress: number; total: number } | null;
  npc_states: { id: string; present: boolean; in_view: boolean; disposition: string }[];
  player_last_choice: string | null;
  dm_directive: string | null;
  current_frame: string; // Base64 or vision analysis result
}

interface StorytellerOutput {
  scope_prompt: string;
  image_gen: {
    trigger: boolean;
    prompt: string;
    type: "scene_element" | "npc_avatar" | "quest_item" | "composite";
  } | null;
  narration: string;
  npc_whisper: string | null;
  player_choices: string[];
  dm_choices: string[];
  prefetch_images: { prompt: string; probability: number }[];
}
```

---

## Hardware Configuration

### Windows Host PC

| Component | Specification | Assignment |
|-----------|---------------|------------|
| GPU 1 | RTX 3090/4090 (24GB) | Scope |
| GPU 2 | RTX 3080+ (12GB+) | Preprocessing |
| CPU | 12-core+ | Unity Host, MediaPipe |
| RAM | 64GB | All services |
| Storage | NVMe SSD | Models, assets |

### Android Devices (NPC/DM/Player)

| Requirement | Specification |
|-------------|---------------|
| Android | 10+ |
| RAM | 4GB+ |
| Network | Same WiFi as host PC |

### Runpod Instance (ComfyUI)

| Requirement | Specification |
|-------------|---------------|
| GPU | RTX 4090 or A100 |
| VRAM | 24GB+ |
| Template | ComfyUI |

---

## Dependencies

### Software Stack

| Component | Technology | Location |
|-----------|------------|----------|
| AI Video | Scope | Local (GPU 1) |
| Preprocessing | Python + Depth Anything V2 + MediaPipe | Local (GPU 2) |
| LLM | Claude API (vision-capable) | Cloud |
| TTS | ElevenLabs API | Cloud |
| Image Gen | ComfyUI | Runpod |
| Game Engine | Unity 2022+ | Local + Android |
| Multiplayer | Mirror or Unity Netcode | Unity |
| Video Streaming | Unity WebRTC | Unity |

### Unity Packages

```
com.unity.webrtc
KlakSpout (Spout receiver/sender)
Mirror or com.unity.netcode.gameobjects
```

### Python Packages (Preprocessing Service)

```
torch
depth-anything-v2
mediapipe
opencv-python
numpy
SpoutGL
```

---

## Implementation Phases

### Phase 1: Core Pipeline
- [ ] Scope installation and VACE configuration on GPU 1
- [ ] Preprocessing service on GPU 2 (Depth Anything + MediaPipe)
- [ ] Spout connection: Preprocessing → Scope → Unity
- [ ] Unity Host displaying transformed video
- [ ] Manual prompt control via Unity debug UI

### Phase 2: Storyteller Integration
- [ ] Claude API integration with vision and structured outputs
- [ ] ElevenLabs TTS integration
- [ ] Player choice UI and response handling
- [ ] Basic narration loop with scene history

### Phase 3: Multi-Participant
- [ ] Unity-to-Unity WebRTC for Android participants
- [ ] NPC mode with whisper audio channel
- [ ] NPC "in view" button triggering avatar inclusion
- [ ] DM mode with meta-choice controls
- [ ] Audio channel separation (master vs NPC-private)

### Phase 4: Image Generation
- [ ] ComfyUI setup on Runpod
- [ ] Composite reference image workflow
- [ ] REST upload pipeline to Scope
- [ ] Prefetch queue for predictive generation
- [ ] NPC avatar generation and caching

### Phase 5: Polish & Stretch Goals
- [ ] Twitch integration (OBS + RTMP + vote overlay)
- [ ] Quest system with location-based triggers
- [ ] 6DoF tracking integration
- [ ] Additional device support (VR, Meta glasses)
