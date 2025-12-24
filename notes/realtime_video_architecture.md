# Real-Time Video Generation System Architecture

## Executive Summary

A real-time video generation system built on the KREA/Scope pipeline that separates **content** (what happens) from **style** (how it looks), enabling rapid prompt iteration, narrative branching, and multi-style rendering from a single world state.

Core insight: The pipeline produces 3-frame chunks, each requiring ~5 forward passes. Build the interaction system on top of a stable generator API, then optimize the generator as a drop-in replacement.

---

## Two-Axis Architecture

The system has two orthogonal organizing principles:

**Axis A: Semantic Layers** (what the system means)
- Define the transformation from intent to pixels
- Vertical stack: Presentation → World Logic → Style Logic → Render

**Axis B: Operational Planes** (how it runs in time)
- Define how the system executes and responds
- Horizontal cut: Control, Generation, Media, Timeline, Analysis

```
                    OPERATIONAL PLANES (horizontal)
                    
         Control    Generation    Media    Timeline    Analysis
            │           │           │          │           │
    ┌───────┼───────────┼───────────┼──────────┼───────────┼───────┐
    │       │           │           │          │           │       │
P   │       ▼           │           ▼          │           ▼       │
r   │   [Events]        │       [WebRTC]       │       [VLM]       │
e   │   [DevConsole]    │       [Record]       │       [Metrics]   │
s   │       │           │           │          │           │       │
e   ├───────┼───────────┼───────────┼──────────┼───────────┼───────┤
n   │       │           │           │          │           │       │
t   │       ▼           │           │          ▼           │       │
    │  [WorldState      │           │     [Snapshots]      │       │
W   │   updates]        │           │     [BranchGraph]    │       │
o   │       │           │           │          │           │       │
r   ├───────┼───────────┼───────────┼──────────┼───────────┼───────┤
l   │       │           │           │          │           │       │
d   │       ▼           │           │          │           │       │
    │  [PromptCompiler] │           │          │           │       │
S   │  [StyleManifest]  │           │          │           │       │
t   │       │           │           │          │           │       │
y   ├───────┼───────────┼───────────┼──────────┼───────────┼───────┤
l   │       │           │           │          │           │       │
e   │       ▼           ▼           │          │           │       │
    │  [ControlState]──►[Pipeline]──┼──────────┼───────────┘       │
R   │                   [Driver]    │          │                   │
e   │                      │        │          │                   │
n   │                      ▼        ▼          ▼                   │
d   │                  [Frames]──►[FrameBus]──►[ChunkStore]        │
e   │                                                              │
r   └──────────────────────────────────────────────────────────────┘

SEMANTIC LAYERS (vertical)
```

**Key insight**: Semantic layers define "what" at each level of abstraction. Operational planes cut across all layers to handle "when" and "how." Events flow down through semantic layers; frames flow right through operational planes.

---

## System Overview (Semantic Layers Detail)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PRESENTATION LAYER                                                       │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│   │   Curated    │  │   Freeform   │  │ Dev Console  │                  │
│   │   Choices    │  │    Input     │  │ (State Edit) │                  │
│   │  (2-4 opts)  │  │  (open box)  │  │(direct access)│                  │
│   └──────────────┘  └──────────────┘  └──────────────┘                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ WORLD LOGIC LAYER (Domain-Agnostic "Truth")                             │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ NARRATIVE LOGIC                                                  │   │
│   │ Genre conventions, beat patterns, arc templates                  │   │
│   │ "Slapstick: setup → escalation → payoff → reset"                │   │
│   │ "Noir: tension builds, betrayal, moral ambiguity"               │   │
│   └─────────────────────────────┬───────────────────────────────────┘   │
│                                 │ shapes                                 │
│                                 ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ CHARACTER INTERNAL STATE                                         │   │
│   │ Emotions, motivations, knowledge, relationships                  │   │
│   │ "Rooster: frustrated, doesn't know Terry hid the key"           │   │
│   │ Updates: events + interpretation (slower, stickier)             │   │
│   └─────────────────────────────┬───────────────────────────────────┘   │
│                                 │ drives                                 │
│                                 ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ EXTERNAL WORLD STATE                                             │   │
│   │ Props, locations, physical positions, visibility                 │   │
│   │ "Banana peel: on floor, near door, visible to camera"           │   │
│   │ Updates: actions (fast, concrete)                                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STYLE LOGIC LAYER (Per-World-Pack Prompt Compiler)                      │
│                                                                          │
│   Translates intent → prompt for target LoRA                            │
│   "menacing walk" → material vocab + motion vocab                       │
│                                                                          │
│   Example translations for same WorldState:                             │
│   ┌─────────────────┬─────────────────┬─────────────────┐              │
│   │ Rudolph 1964    │ TMNT Mutant     │ Rooster & Terry │              │
│   │                 │ Mayhem          │                 │              │
│   ├─────────────────┼─────────────────┼─────────────────┤              │
│   │ "stop-motion    │ "spray paint    │ "2D animation   │              │
│   │ puppet walk     │ style, aggres-  │ smear frames,   │              │
│   │ cycle, felt     │ sive stride,    │ exaggerated     │              │
│   │ texture"        │ graffiti tags"  │ silhouette"     │              │
│   └─────────────────┴─────────────────┴─────────────────┘              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ RENDER LAYER                                                             │
│   LoRA + Scope/Krea Pipeline                                            │
│   (Visual Output)                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Primitives

### 1. WorldState

Domain-agnostic truth layer. No LoRA knowledge. Speaks in abstract terms.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class Beat(Enum):
    SETUP = "setup"
    ESCALATION = "escalation"
    PAYOFF = "payoff"
    RESET = "reset"
    TENSION_BUILD = "tension_build"
    BETRAYAL = "betrayal"

class ArcTemplate(Enum):
    SLAPSTICK = "slapstick"
    NOIR = "noir"
    CHASE = "chase"
    DISCOVERY = "discovery"
    CONFRONTATION = "confrontation"

@dataclass
class CharacterState:
    emotion: str  # "frustrated", "suspicious", "elated"
    motivation: str  # "find the key", "escape", "confront Terry"
    knowledge: dict[str, bool] = field(default_factory=dict)
    # e.g. {"terry_hid_key": False, "door_is_locked": True}
    relationships: dict[str, str] = field(default_factory=dict)
    # e.g. {"terry": "distrustful", "narrator": "unaware"}

@dataclass
class Prop:
    name: str
    location: str  # "floor_near_door", "table_center", "character_hand"
    visible_to_camera: bool = True
    state: str = "default"  # "peeled", "broken", "glowing"

@dataclass
class WorldState:
    """Domain-agnostic truth layer. No LoRA knowledge."""
    
    # Narrative Logic (slowest to change)
    current_beat: Beat = Beat.SETUP
    arc_template: ArcTemplate = ArcTemplate.SLAPSTICK
    tension_level: float = 0.0  # 0.0 to 1.0
    
    # Character Internal State (medium update rate)
    characters: dict[str, CharacterState] = field(default_factory=dict)
    
    # External World State (fastest updates)
    props: list[Prop] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    active_location: str = ""
    
    # Current moment (abstract descriptions)
    current_action: str = ""  # "character enters menacingly"
    camera_intent: str = ""   # "low angle, tracking"
    lighting_intent: str = "" # "dramatic shadows", "flat bright"
    
    # Time tracking
    chunk_index: int = 0
    
    def to_dict(self) -> dict:
        return {
            "narrative": {
                "beat": self.current_beat.value,
                "arc": self.arc_template.value,
                "tension": self.tension_level,
            },
            "characters": {
                name: {
                    "emotion": c.emotion,
                    "motivation": c.motivation,
                    "knowledge": c.knowledge,
                    "relationships": c.relationships,
                }
                for name, c in self.characters.items()
            },
            "world": {
                "location": self.active_location,
                "props": [
                    {"name": p.name, "location": p.location, 
                     "visible": p.visible_to_camera, "state": p.state}
                    for p in self.props
                ],
            },
            "current": {
                "action": self.current_action,
                "camera": self.camera_intent,
                "lighting": self.lighting_intent,
            },
            "chunk_index": self.chunk_index,
        }
```

### 2. StyleManifest

Per-LoRA translation rules. Lives in `/styles/{lora_name}/manifest.yaml`.

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class StyleManifest:
    """Per-LoRA translation rules."""
    
    name: str
    lora_path: str
    lora_default_scale: float = 0.8
    
    # Vocabulary mappings: abstract term → LoRA-specific tokens
    material_vocab: dict[str, str] = field(default_factory=dict)
    motion_vocab: dict[str, str] = field(default_factory=dict)
    camera_vocab: dict[str, str] = field(default_factory=dict)
    lighting_vocab: dict[str, str] = field(default_factory=dict)
    emotion_vocab: dict[str, str] = field(default_factory=dict)
    
    # Beat-specific modifiers
    beat_modifiers: dict[str, str] = field(default_factory=dict)
    
    # Negative prompts that work for this LoRA
    default_negative: str = ""
    
    # Token budget guidance
    max_prompt_tokens: int = 75
    priority_order: list[str] = field(
        default_factory=lambda: ["action", "material", "camera", "mood"]
    )
    
    # Trigger words (some LoRAs require specific activation tokens)
    trigger_words: list[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "StyleManifest":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


# Example manifest for Rudolph 1964 style
RUDOLPH_1964_MANIFEST = StyleManifest(
    name="rudolph_1964",
    lora_path="/loras/rudolph_1964_v2.safetensors",
    lora_default_scale=0.85,
    
    material_vocab={
        "skin": "felt texture, visible stitching, matte surface",
        "metal": "painted tin, slightly reflective, vintage toy",
        "wood": "carved balsa wood, visible grain, handcrafted",
        "snow": "cotton batting snow, sparkle dust, miniature drifts",
        "default": "stop-motion puppet, Rankin-Bass style",
    },
    
    motion_vocab={
        "walk": "deliberate puppet walk cycle, 12fps, slight wobble",
        "run": "held cels, blur frames, exaggerated stride",
        "enter_menacing": "slow deliberate steps, looming presence, shadow first",
        "fall": "replacement animation tumble, squash on impact",
        "idle": "subtle breathing motion, micro-adjustments",
    },
    
    camera_vocab={
        "low_angle": "miniature set perspective, forced depth, looking up",
        "tracking": "smooth dolly move, tabletop track",
        "close_up": "macro lens softness, shallow depth of field",
        "wide": "full set visible, diorama framing",
    },
    
    lighting_vocab={
        "dramatic": "strong key light, deep shadows, rim lighting",
        "flat": "soft even lighting, minimal shadows, cheerful",
        "night": "blue fill, warm practicals, moonlight key",
    },
    
    emotion_vocab={
        "frustrated": "furrowed brow, clenched pose, agitated micro-movements",
        "happy": "wide eyes, bouncy movement, warm lighting",
        "suspicious": "narrowed eyes, guarded posture, side glance",
    },
    
    beat_modifiers={
        "setup": "establishing shot, calm pacing",
        "escalation": "quicker cuts, tighter framing",
        "payoff": "dramatic pause, reaction shot",
        "reset": "wide shot, normalized lighting",
    },
    
    default_negative="realistic, photographic, 3D render, modern CGI, smooth plastic",
    
    trigger_words=["rudolph1964", "rankinbass"],
    
    max_prompt_tokens=77,
    priority_order=["trigger", "action", "material", "camera", "mood"],
)
```

### 3. PromptCompiler

Translates WorldState → ControlState using active StyleManifest.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class CompiledPrompt:
    positive: list[dict]  # [{"text": "...", "weight": 1.0}]
    negative: str
    lora_stack: list[dict]  # [{"path": "...", "scale": 0.8}]
    
class PromptCompiler:
    """Translates WorldState → pipeline-ready prompts using StyleManifest."""
    
    def __init__(self, manifest: StyleManifest):
        self.style = manifest
    
    def compile(self, world: WorldState) -> CompiledPrompt:
        """Produce pipeline-ready prompts from abstract world state."""
        
        parts = []
        
        # 1. Trigger words first (if required by LoRA)
        if self.style.trigger_words:
            parts.append(", ".join(self.style.trigger_words))
        
        # 2. Current action (highest priority content)
        action_phrase = self._translate_action(world.current_action)
        if action_phrase:
            parts.append(action_phrase)
        
        # 3. Material/style context
        material_phrase = self._get_default_material()
        parts.append(material_phrase)
        
        # 4. Camera
        camera_phrase = self._translate_camera(world.camera_intent)
        if camera_phrase:
            parts.append(camera_phrase)
        
        # 5. Lighting
        lighting_phrase = self._translate_lighting(world.lighting_intent)
        if lighting_phrase:
            parts.append(lighting_phrase)
        
        # 6. Beat modifier
        beat_mod = self.style.beat_modifiers.get(world.current_beat.value, "")
        if beat_mod:
            parts.append(beat_mod)
        
        # 7. Character emotion (if characters present)
        for char_name, char_state in world.characters.items():
            emotion_phrase = self._translate_emotion(char_state.emotion)
            if emotion_phrase:
                parts.append(f"{char_name} {emotion_phrase}")
                break  # Only first character for now
        
        # Assemble and truncate
        prompt_text = ", ".join(filter(None, parts))
        prompt_text = self._truncate_to_tokens(prompt_text)
        
        return CompiledPrompt(
            positive=[{"text": prompt_text, "weight": 1.0}],
            negative=self.style.default_negative,
            lora_stack=[{
                "path": self.style.lora_path,
                "scale": self.style.lora_default_scale,
            }],
        )
    
    def _translate_action(self, action: str) -> str:
        """Look up action in motion vocab, fall back to literal."""
        # Normalize action string for lookup
        key = action.lower().replace(" ", "_")
        return self.style.motion_vocab.get(key, action)
    
    def _translate_camera(self, camera: str) -> str:
        key = camera.lower().replace(" ", "_")
        return self.style.camera_vocab.get(key, camera)
    
    def _translate_lighting(self, lighting: str) -> str:
        key = lighting.lower().replace(" ", "_")
        return self.style.lighting_vocab.get(key, lighting)
    
    def _translate_emotion(self, emotion: str) -> str:
        key = emotion.lower().replace(" ", "_")
        return self.style.emotion_vocab.get(key, emotion)
    
    def _get_default_material(self) -> str:
        return self.style.material_vocab.get("default", "")
    
    def _truncate_to_tokens(self, text: str) -> str:
        """Rough truncation to token budget (assumes ~0.75 tokens per word)."""
        words = text.split()
        max_words = int(self.style.max_prompt_tokens * 0.75)
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text
    
    def preview(self, world: WorldState) -> dict:
        """Return compilation breakdown for debugging."""
        compiled = self.compile(world)
        return {
            "input_world": world.to_dict(),
            "style_manifest": self.style.name,
            "output_prompt": compiled.positive[0]["text"],
            "output_negative": compiled.negative,
            "lora": compiled.lora_stack,
        }
```

### 4. ControlState

The immediate control surface for the generator. Updated by PromptCompiler or directly via Dev Console.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class GenerationMode(Enum):
    T2V = "text_to_video"
    V2V = "video_to_video"

@dataclass
class ControlState:
    """Immediate control surface for the generator."""
    
    # Prompts (output of PromptCompiler, or direct override)
    prompts: list[dict] = field(default_factory=list)
    # [{"text": "...", "weight": 1.0}]
    
    negative_prompt: str = ""
    
    # LoRA configuration
    lora_stack: list[dict] = field(default_factory=list)
    # [{"path": "...", "scale": 0.8}]
    
    # Generation parameters
    mode: GenerationMode = GenerationMode.T2V
    num_frames_per_block: int = 3
    denoising_step_list: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    
    # Determinism
    base_seed: int = 42
    branch_seed_offset: int = 0  # For deterministic branching
    
    # KV cache behavior (0.3 is KREA default - higher = more stable, less responsive)
    kv_cache_attention_bias: float = 0.3
    
    # Transition state (for prompt ramping)
    transition_chunks_remaining: int = 0
    transition_from_prompts: Optional[list[dict]] = None
    
    # Pipeline state tracking
    current_start_frame: int = 0
    
    def effective_seed(self) -> int:
        return self.base_seed + self.branch_seed_offset
    
    def to_pipeline_kwargs(self) -> dict:
        """Convert to kwargs for pipeline call."""
        return {
            "prompts": self.prompts,
            "negative_prompt": self.negative_prompt,
            "num_frame_per_block": self.num_frames_per_block,
            "denoising_step_list": self.denoising_step_list,
            "base_seed": self.effective_seed(),
            # Include LoRA and cache behavior
            "lora_stack": self.lora_stack,
            "kv_cache_attention_bias": self.kv_cache_attention_bias,
        }
```

### 5. ControlBus (Event Queue)

A timestamped queue of typed events applied at chunk boundaries. This makes control flow explicit, debuggable, and replayable.

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from enum import Enum
from collections import deque
import time

class EventType(Enum):
    # Prompt and style
    SET_PROMPT = "set_prompt"
    SET_WORLD_STATE = "set_world_state"
    SET_STYLE_MANIFEST = "set_style_manifest"
    SET_LORA_STACK = "set_lora_stack"
    
    # Generation parameters
    SET_DENOISE_STEPS = "set_denoise_steps"
    SET_SEED = "set_seed"
    
    # Lifecycle
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    STOP = "stop"
    
    # Branching
    SNAPSHOT_REQUEST = "snapshot_request"
    FORK_REQUEST = "fork_request"
    ROLLOUT_REQUEST = "rollout_request"
    SELECT_BRANCH = "select_branch"
    RESTORE_SNAPSHOT = "restore_snapshot"

class ApplyMode(Enum):
    NEXT_BOUNDARY = "next_boundary"      # Apply at start of next chunk
    IMMEDIATE_IF_PAUSED = "immediate"    # Apply now if paused, else next boundary

@dataclass
class ControlEvent:
    """A single control event with timing and application semantics."""
    
    type: EventType
    payload: dict = field(default_factory=dict)
    apply_mode: ApplyMode = ApplyMode.NEXT_BOUNDARY
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(time.time_ns()))
    
    # For debugging/replay
    source: str = "api"  # "api", "vlm", "timeline", "dev_console"

@dataclass
class ControlBus:
    """
    Timestamped event queue with chunk-boundary semantics.
    
    Events are queued immediately but applied at chunk boundaries,
    ensuring the generator always sees consistent state.
    """
    
    pending: deque[ControlEvent] = field(default_factory=deque)
    history: list[ControlEvent] = field(default_factory=list)
    max_history: int = 1000
    
    def enqueue(
        self,
        event_type: EventType,
        payload: dict = None,
        apply_mode: ApplyMode = ApplyMode.NEXT_BOUNDARY,
        source: str = "api",
    ) -> ControlEvent:
        """Add an event to the queue."""
        event = ControlEvent(
            type=event_type,
            payload=payload or {},
            apply_mode=apply_mode,
            source=source,
        )
        self.pending.append(event)
        return event
    
    def drain_pending(self, is_paused: bool = False) -> list[ControlEvent]:
        """
        Get all events that should be applied now.
        Called at chunk boundaries (or immediately if checking for pause-mode events).
        """
        to_apply = []
        remaining = deque()
        
        for event in self.pending:
            should_apply = (
                event.apply_mode == ApplyMode.NEXT_BOUNDARY or
                (event.apply_mode == ApplyMode.IMMEDIATE_IF_PAUSED and is_paused)
            )
            
            if should_apply:
                to_apply.append(event)
                self._add_to_history(event)
            else:
                remaining.append(event)
        
        self.pending = remaining
        return to_apply
    
    def _add_to_history(self, event: ControlEvent):
        """Store event in history for debugging/replay."""
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(
        self,
        since_timestamp: float = 0,
        event_types: list[EventType] = None,
    ) -> list[ControlEvent]:
        """Query event history for debugging or replay."""
        filtered = [e for e in self.history if e.timestamp >= since_timestamp]
        if event_types:
            filtered = [e for e in filtered if e.type in event_types]
        return filtered
    
    def clear_pending(self):
        """Clear all pending events (e.g., on stop)."""
        self.pending.clear()


# Convenience functions for common event patterns
def prompt_event(
    prompts: list[dict],
    ramp_chunks: int = 0,
    source: str = "api",
) -> ControlEvent:
    """Create a prompt update event."""
    return ControlEvent(
        type=EventType.SET_PROMPT,
        payload={"prompts": prompts, "ramp_chunks": ramp_chunks},
        source=source,
    )

def world_state_event(updates: dict, source: str = "api") -> ControlEvent:
    """Create a world state update event."""
    return ControlEvent(
        type=EventType.SET_WORLD_STATE,
        payload=updates,
        source=source,
    )

def pause_event(source: str = "api") -> ControlEvent:
    """Create a pause event (applies immediately if possible)."""
    return ControlEvent(
        type=EventType.PAUSE,
        apply_mode=ApplyMode.IMMEDIATE_IF_PAUSED,
        source=source,
    )

def fork_event(
    from_snapshot_id: str,
    name: str = "",
    source: str = "api",
) -> ControlEvent:
    """Create a fork request event."""
    return ControlEvent(
        type=EventType.FORK_REQUEST,
        payload={"from_snapshot_id": from_snapshot_id, "name": name},
        source=source,
    )
```

### 6. GeneratorDriver

The tick loop that owns the pipeline and applies control events.

```python
import asyncio
from dataclasses import dataclass
from typing import Callable, Optional
from enum import Enum

class DriverState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"

@dataclass
class GenerationResult:
    frames: any  # Tensor or numpy array
    chunk_index: int
    control_state_snapshot: dict
    world_state_snapshot: dict
    timing_ms: float

class GeneratorDriver:
    """Tick loop that owns the pipeline and applies control events."""
    
    def __init__(
        self,
        pipeline,  # KreaRealtimeVideoPipeline instance
        on_chunk: Callable[[GenerationResult], None],
        on_state_change: Callable[[DriverState], None],
    ):
        self.pipeline = pipeline
        self.on_chunk = on_chunk
        self.on_state_change = on_state_change
        
        self.state = DriverState.STOPPED
        self.control_state = ControlState()
        self.world_state = WorldState()
        self.compiler: Optional[PromptCompiler] = None
        
        self._pending_control_updates: list[dict] = []
        self._pending_world_updates: list[dict] = []
        self._run_task: Optional[asyncio.Task] = None  # Guard against multiple loops
        
    def set_compiler(self, compiler: PromptCompiler):
        """Set the active style compiler."""
        self.compiler = compiler
        # Recompile immediately
        if self.compiler:
            compiled = self.compiler.compile(self.world_state)
            self.control_state.prompts = compiled.positive
            self.control_state.negative_prompt = compiled.negative
            self.control_state.lora_stack = compiled.lora_stack
    
    def update_world(self, updates: dict):
        """Queue world state updates (applied at next chunk boundary)."""
        self._pending_world_updates.append(updates)
    
    def update_control(self, updates: dict):
        """Queue direct control state updates (bypass compiler)."""
        self._pending_control_updates.append(updates)
    
    def _apply_pending_updates(self):
        """Apply queued updates at chunk boundary."""
        # Track if world changed BEFORE clearing
        world_changed = bool(self._pending_world_updates)
        
        # Apply world updates
        for update in self._pending_world_updates:
            for key, value in update.items():
                if hasattr(self.world_state, key):
                    setattr(self.world_state, key, value)
        self._pending_world_updates.clear()
        
        # Recompile if we have a compiler and world changed
        if self.compiler and world_changed:
            compiled = self.compiler.compile(self.world_state)
            self.control_state.prompts = compiled.positive
            self.control_state.negative_prompt = compiled.negative
        
        # Apply direct control overrides
        for update in self._pending_control_updates:
            for key, value in update.items():
                if hasattr(self.control_state, key):
                    setattr(self.control_state, key, value)
        self._pending_control_updates.clear()
    
    async def run(self):
        """Main generation loop."""
        self.state = DriverState.RUNNING
        self.on_state_change(self.state)
        
        while self.state == DriverState.RUNNING:
            await self._generate_chunk()
            await asyncio.sleep(0)  # Yield to event loop
    
    async def step(self) -> GenerationResult:
        """Generate exactly one chunk (for Dev Console)."""
        self.state = DriverState.STEPPING
        self.on_state_change(self.state)
        
        result = await self._generate_chunk()
        
        self.state = DriverState.PAUSED
        self.on_state_change(self.state)
        
        return result
    
    async def _generate_chunk(self) -> GenerationResult:
        """Generate one chunk of frames."""
        import time
        
        # Apply pending updates at chunk boundary
        self._apply_pending_updates()
        
        start_time = time.perf_counter()
        
        # Call pipeline
        output = self.pipeline(**self.control_state.to_pipeline_kwargs())
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Update frame counter
        self.control_state.current_start_frame += self.control_state.num_frames_per_block
        self.world_state.chunk_index += 1
        
        result = GenerationResult(
            frames=output,
            chunk_index=self.world_state.chunk_index,
            control_state_snapshot=self.control_state.__dict__.copy(),
            world_state_snapshot=self.world_state.to_dict(),
            timing_ms=elapsed_ms,
        )
        
        self.on_chunk(result)
        return result
    
    def pause(self):
        self.state = DriverState.PAUSED
        self.on_state_change(self.state)
    
    def resume(self):
        """Resume generation. Guards against spawning multiple loops."""
        if self.state != DriverState.PAUSED:
            return  # Can only resume from paused
        if self._run_task and not self._run_task.done():
            return  # Already have an active loop
        self._run_task = asyncio.create_task(self.run())
    
    def stop(self):
        """Stop generation and cancel any running task."""
        self.state = DriverState.STOPPED
        self.on_state_change(self.state)
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        self._run_task = None
    
    def snapshot(self) -> dict:
        """
        Create a restorable snapshot of current state.
        
        Includes generator continuity buffers needed for seamless continuation.
        Without these, restore produces a hard cut, not seamless continuation.
        """
        return {
            # Narrative/control layer
            "world_state": self.world_state.to_dict(),
            "control_state": {
                "prompts": self.control_state.prompts,
                "negative_prompt": self.control_state.negative_prompt,
                "lora_stack": self.control_state.lora_stack,
                "base_seed": self.control_state.base_seed,
                "branch_seed_offset": self.control_state.branch_seed_offset,
                "current_start_frame": self.control_state.current_start_frame,
                "denoising_step_list": self.control_state.denoising_step_list,
                "mode": self.control_state.mode.value,
            },
            "chunk_index": self.world_state.chunk_index,
            "style_manifest": self.compiler.style.name if self.compiler else None,
            
            # Generator continuity layer (REQUIRED for seamless continuation)
            # These are the buffers KREA's recompute block needs to rebuild KV cache
            "generator_continuity": self._capture_continuity_state(),
        }
    
    def _capture_continuity_state(self) -> dict:
        """
        Capture pipeline-internal state needed for seamless continuation.
        
        The KREA pipeline maintains continuity via:
        - first_context_frame: anchor for temporal consistency
        - context_frame_buffer: recent frames for KV cache recompute
        - decoded_frame_buffer: for re-encoding if needed
        
        Without these, restore triggers a hard cut (cache rebuilt from scratch).
        """
        # Access pipeline's internal buffers
        # NOTE: Actual attribute names depend on KREA pipeline implementation
        continuity = {
            "current_start_frame": self.control_state.current_start_frame,
        }
        
        # Attempt to capture pipeline buffers if available
        if hasattr(self.pipeline, 'first_context_frame'):
            continuity["first_context_frame"] = self.pipeline.first_context_frame
        if hasattr(self.pipeline, 'context_frame_buffer'):
            continuity["context_frame_buffer"] = self.pipeline.context_frame_buffer
        if hasattr(self.pipeline, 'decoded_frame_buffer'):
            continuity["decoded_frame_buffer"] = self.pipeline.decoded_frame_buffer
        
        return continuity
    
    def restore(self, snapshot: dict):
        """
        Restore from a snapshot.
        
        If generator_continuity is present and valid, this produces seamless
        continuation. Otherwise, it's a hard cut (acceptable for branching).
        """
        # Restore world state
        world_data = snapshot["world_state"]
        self.world_state.current_beat = Beat(world_data["narrative"]["beat"])
        self.world_state.arc_template = ArcTemplate(world_data["narrative"]["arc"])
        self.world_state.tension_level = world_data["narrative"]["tension"]
        self.world_state.active_location = world_data["world"]["location"]
        self.world_state.current_action = world_data["current"]["action"]
        self.world_state.camera_intent = world_data["current"]["camera"]
        self.world_state.chunk_index = snapshot["chunk_index"]
        
        # Restore control state
        ctrl_data = snapshot["control_state"]
        self.control_state.prompts = ctrl_data["prompts"]
        self.control_state.negative_prompt = ctrl_data["negative_prompt"]
        self.control_state.lora_stack = ctrl_data["lora_stack"]
        self.control_state.base_seed = ctrl_data["base_seed"]
        self.control_state.branch_seed_offset = ctrl_data["branch_seed_offset"]
        self.control_state.current_start_frame = ctrl_data["current_start_frame"]
        
        # Restore generator continuity (if available)
        if "generator_continuity" in snapshot:
            self._restore_continuity_state(snapshot["generator_continuity"])
    
    def _restore_continuity_state(self, continuity: dict):
        """
        Restore pipeline-internal buffers for seamless continuation.
        
        If buffers are missing or incompatible, the pipeline's recompute
        block will rebuild from scratch (hard cut, but still works).
        """
        if hasattr(self.pipeline, 'first_context_frame') and "first_context_frame" in continuity:
            self.pipeline.first_context_frame = continuity["first_context_frame"]
        if hasattr(self.pipeline, 'context_frame_buffer') and "context_frame_buffer" in continuity:
            self.pipeline.context_frame_buffer = continuity["context_frame_buffer"]
        if hasattr(self.pipeline, 'decoded_frame_buffer') and "decoded_frame_buffer" in continuity:
            self.pipeline.decoded_frame_buffer = continuity["decoded_frame_buffer"]
```

### 7. FrameBus

Pub-sub hub with ring buffer for frame distribution.

```python
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Any
import threading
import time

@dataclass
class FramePacket:
    frames: Any  # Tensor or numpy (contains frames_per_chunk frames, default 3)
    chunk_index: int
    timestamp: float
    metadata: dict

class FrameBus:
    """Pub-sub hub with ring buffer for frame distribution."""
    
    def __init__(
        self, 
        buffer_seconds: float = 30.0, 
        fps: float = 24.0,
        frames_per_chunk: int = 3,
    ):
        # Buffer sized in CHUNKS, not frames
        # 30 seconds at 24fps with 3-frame chunks = 30 * 24 / 3 = 240 chunks
        chunks_per_second = fps / frames_per_chunk
        self.buffer_size = int(buffer_seconds * chunks_per_second)
        self.buffer: deque[FramePacket] = deque(maxlen=self.buffer_size)
        self.subscribers: dict[str, Callable[[FramePacket], None]] = {}
        self._lock = threading.Lock()
        
        # Store config for later reference
        self.fps = fps
        self.frames_per_chunk = frames_per_chunk
    
    def publish(self, frames: Any, chunk_index: int, metadata: dict = None):
        """Publish new frames to all subscribers."""
        packet = FramePacket(
            frames=frames,
            chunk_index=chunk_index,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        with self._lock:
            self.buffer.append(packet)
        
        # Notify subscribers (non-blocking)
        for name, callback in self.subscribers.items():
            try:
                callback(packet)
            except Exception as e:
                print(f"Subscriber {name} error: {e}")
    
    def subscribe(self, name: str, callback: Callable[[FramePacket], None]):
        """Register a subscriber."""
        self.subscribers[name] = callback
    
    def unsubscribe(self, name: str):
        """Remove a subscriber."""
        self.subscribers.pop(name, None)
    
    def get_range(self, start_chunk: int, end_chunk: int) -> list[FramePacket]:
        """Get frames in a chunk range (for scrubbing)."""
        with self._lock:
            return [
                p for p in self.buffer 
                if start_chunk <= p.chunk_index <= end_chunk
            ]
    
    def get_last_n_seconds(self, seconds: float) -> list[FramePacket]:
        """Get frames from the last N seconds."""
        cutoff = time.time() - seconds
        with self._lock:
            return [p for p in self.buffer if p.timestamp >= cutoff]
    
    def get_latest(self) -> FramePacket | None:
        """Get most recent frame packet."""
        with self._lock:
            return self.buffer[-1] if self.buffer else None
    
    def buffer_duration_seconds(self) -> float:
        """Return actual duration of buffered content."""
        with self._lock:
            if len(self.buffer) < 2:
                return 0.0
            return self.buffer[-1].timestamp - self.buffer[0].timestamp
```

### 8. BranchGraph

DAG of checkpoints for pause/fork/resume functionality.

```python
from dataclasses import dataclass, field
from typing import Optional
import uuid
from datetime import datetime

@dataclass
class BranchNode:
    id: str
    parent_id: Optional[str]
    name: str
    snapshot: dict  # From GeneratorDriver.snapshot()
    created_at: datetime
    chunk_index: int
    preview_path: Optional[str] = None  # Path to preview video
    children: list[str] = field(default_factory=list)
    is_active: bool = False

class BranchGraph:
    """DAG of checkpoints for branching functionality."""
    
    def __init__(self):
        self.nodes: dict[str, BranchNode] = {}
        self.active_branch_id: Optional[str] = None
        self.root_id: Optional[str] = None
    
    def create_root(self, snapshot: dict, name: str = "root") -> BranchNode:
        """Create the initial root node."""
        node = BranchNode(
            id=str(uuid.uuid4()),
            parent_id=None,
            name=name,
            snapshot=snapshot,
            created_at=datetime.now(),
            chunk_index=snapshot.get("chunk_index", 0),
            is_active=True,
        )
        self.nodes[node.id] = node
        self.root_id = node.id
        self.active_branch_id = node.id
        return node
    
    def fork(
        self, 
        from_node_id: str, 
        snapshot: dict, 
        name: str = ""
    ) -> BranchNode:
        """Create a new branch from an existing node."""
        parent = self.nodes.get(from_node_id)
        if not parent:
            raise ValueError(f"Parent node {from_node_id} not found")
        
        if not name:
            name = f"branch_{len(parent.children) + 1}"
        
        node = BranchNode(
            id=str(uuid.uuid4()),
            parent_id=from_node_id,
            name=name,
            snapshot=snapshot,
            created_at=datetime.now(),
            chunk_index=snapshot.get("chunk_index", 0),
        )
        
        self.nodes[node.id] = node
        parent.children.append(node.id)
        
        return node
    
    def set_active(self, node_id: str):
        """Set a branch as the active continuation."""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Deactivate all
        for node in self.nodes.values():
            node.is_active = False
        
        # Activate selected
        self.nodes[node_id].is_active = True
        self.active_branch_id = node_id
    
    def get_active_snapshot(self) -> dict | None:
        """Get the snapshot of the active branch."""
        if self.active_branch_id and self.active_branch_id in self.nodes:
            return self.nodes[self.active_branch_id].snapshot
        return None
    
    def get_lineage(self, node_id: str) -> list[BranchNode]:
        """Get all ancestors of a node (root to node)."""
        lineage = []
        current_id = node_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            lineage.append(node)
            current_id = node.parent_id
        
        return list(reversed(lineage))
    
    def get_children(self, node_id: str) -> list[BranchNode]:
        """Get all direct children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]
    
    def to_dict(self) -> dict:
        """Serialize the entire graph."""
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "parent_id": n.parent_id,
                    "name": n.name,
                    "snapshot": n.snapshot,
                    "created_at": n.created_at.isoformat(),
                    "chunk_index": n.chunk_index,
                    "preview_path": n.preview_path,
                    "children": n.children,
                    "is_active": n.is_active,
                }
                for nid, n in self.nodes.items()
            },
            "active_branch_id": self.active_branch_id,
            "root_id": self.root_id,
        }
```

---

## API Surface

### Session Lifecycle

```
POST   /v1/sessions                    → {session_id}
POST   /v1/sessions/{id}/start
POST   /v1/sessions/{id}/pause
POST   /v1/sessions/{id}/resume
POST   /v1/sessions/{id}/step          # Generate exactly 1 chunk
DELETE /v1/sessions/{id}
```

### World State (Presentation Layer → World Logic Layer)

```
GET    /v1/sessions/{id}/world         → WorldState
PUT    /v1/sessions/{id}/world         # Partial update
       {
         "current_action": "character enters menacingly",
         "camera_intent": "low angle",
         "tension_level": 0.7
       }

PUT    /v1/sessions/{id}/world/character/{name}
       {
         "emotion": "frustrated",
         "motivation": "find the key"
       }

POST   /v1/sessions/{id}/world/props
       {"name": "banana_peel", "location": "floor_near_door"}

DELETE /v1/sessions/{id}/world/props/{name}
```

### Style (Style Logic Layer)

```
GET    /v1/styles                      → [{name, path}]
GET    /v1/styles/{name}               → StyleManifest
PUT    /v1/sessions/{id}/style         # Switch active style
       {"name": "rudolph_1964"}

POST   /v1/styles/{name}/compile       # Preview compilation without generating
       {world_state}                   → {compiled_prompt, lora_stack}
```

### Direct Control (Dev Console Bypass)

```
GET    /v1/sessions/{id}/control       → ControlState
PUT    /v1/sessions/{id}/control       # Direct override (bypass compiler)
       {
         "prompts": [{"text": "...", "weight": 1.0}],
         "denoising_step_list": [1000, 500],
         "lora_stack": [{"path": "...", "scale": 0.9}]
       }

GET    /v1/sessions/{id}/last_prompt   # See what was actually sent to pipeline
```

### Prompt Control with Transitions

```
PUT    /v1/sessions/{id}/prompt
       {
         "prompts": [{"text": "Laura running in neon alley", "weight": 1.0}],
         "transition": {"type": "ramp", "chunks": 4}
       }
```

### Snapshots and Branching

```
POST   /v1/sessions/{id}/snapshots     → {snapshot_id}
GET    /v1/sessions/{id}/snapshots     → [{snapshot_id, chunk_index, preview_url}]
GET    /v1/sessions/{id}/snapshots/{snapshot_id}

POST   /v1/sessions/{id}/fork
       {
         "from_snapshot_id": "snap_123",
         "name": "cooler lighting take"
       }

POST   /v1/sessions/{id}/rollout
       {
         "from_snapshot_id": "snap_123",
         "count": 4,
         "horizon_chunks": 6,
         "mutations": [
           {"type": "prompt_variation", "strength": 0.3},
           {"type": "seed_variation"},
           {"type": "world_variation", "field": "tension_level", "delta": 0.2}
         ]
       }

POST   /v1/sessions/{id}/select_branch
       {"branch_id": "branch_abc"}

GET    /v1/sessions/{id}/branches      → BranchGraph
```

### Timeline and Playback

```
GET    /v1/sessions/{id}/timeline      → [{chunk_index, timestamp, preview_url}]
GET    /v1/sessions/{id}/frames/{start}/{end}  → video segment or frame list
```

### Streaming

```
WebSocket /v1/sessions/{id}/stream     # Real-time frame stream
WebSocket /v1/sessions/{id}/events     # Control events and state changes
```

---

## Sequence Diagrams

### Steady-State Generation with Prompt Update

```
sequenceDiagram
  autonumber
  participant UI as Presentation Layer
  participant API as Control API
  participant World as WorldState
  participant Compiler as PromptCompiler
  participant Control as ControlState
  participant Driver as GeneratorDriver
  participant Pipe as Pipeline
  participant Bus as FrameBus
  participant RTC as WebRTC

  UI->>API: PUT /world {action: "character enters menacingly"}
  API->>World: update current_action
  
  Note over Driver: Next chunk boundary
  
  Driver->>Driver: apply pending updates
  Driver->>Compiler: compile(WorldState)
  Compiler->>Compiler: translate via StyleManifest
  Compiler->>Control: prompts=[{text: "stop-motion puppet..."}]
  
  Driver->>Pipe: __call__(prompts, lora_stack, ...)
  
  Note over Pipe: TextConditioningBlock
  Note over Pipe: EmbeddingBlendingBlock
  Note over Pipe: SetTimestepsBlock
  Note over Pipe: SetupCachesBlock
  
  alt current_start_frame > 0
    Note over Pipe: RecomputeKVCacheBlock
  end
  
  loop for each t in denoising_step_list
    Note over Pipe: DenoiseBlock
  end
  
  Note over Pipe: DecodeBlock
  Note over Pipe: PrepareNextBlock
  
  Pipe-->>Driver: frames (3)
  Driver->>Bus: publish(frames, metadata)
  Bus->>RTC: push to viewers
```

### Dev Console: Step Mode Iteration

```
sequenceDiagram
  autonumber
  participant Dev as Dev Console
  participant API as Control API
  participant Driver as GeneratorDriver
  participant Compiler as PromptCompiler
  
  Dev->>API: PUT /control {prompts: [...]} (direct override)
  API->>Driver: update_control()
  
  Dev->>API: POST /step
  API->>Driver: step()
  
  Driver->>Driver: apply pending (uses direct prompts, skips compiler)
  Driver->>Driver: generate one chunk
  Driver-->>API: GenerationResult
  API-->>Dev: {frames, timing_ms, control_state_snapshot}
  
  Dev->>Dev: inspect result, adjust prompts
  
  Dev->>API: PUT /control {prompts: [...]} (refined)
  Dev->>API: POST /step
  Note over Dev: iterate until satisfied
  
  Dev->>API: POST /styles/rudolph_1964/compile
  Note over Dev: codify working prompts into StyleManifest
```

### Pause → Fork → Rollout → Select

```
sequenceDiagram
  autonumber
  participant UI as UI
  participant API as Control API
  participant Driver as GeneratorDriver
  participant Graph as BranchGraph
  participant Bus as FrameBus

  UI->>API: POST /pause
  API->>Driver: pause()

  UI->>API: POST /snapshots
  API->>Driver: snapshot()
  Driver->>Graph: create node from snapshot
  Graph-->>API: {snapshot_id}

  UI->>Bus: get_last_n_seconds(10)
  Bus-->>UI: frames for review

  UI->>API: POST /rollout {count: 4, horizon: 6, mutations: [...]}
  
  par Branch 1
    API->>Driver: restore(snapshot)
    API->>Driver: apply mutation (seed +1)
    loop 6 chunks
      Driver->>Driver: generate chunk
      Driver->>Graph: store preview
    end
  and Branch 2
    API->>Driver: restore(snapshot)
    API->>Driver: apply mutation (tension +0.2)
    loop 6 chunks
      Driver->>Driver: generate chunk
    end
  and Branch 3, 4...
  end

  Graph-->>UI: preview URLs for all branches
  UI->>UI: review previews

  UI->>API: POST /select_branch {branch_id: "branch_2"}
  API->>Graph: set_active("branch_2")
  API->>Driver: restore(branch_2.snapshot)

  UI->>API: POST /resume
  API->>Driver: resume()
```

---

## File Structure

```
/project_root/
│
├── /styles/                          # Style manifests and LoRAs
│   ├── rudolph_1964/
│   │   ├── manifest.yaml
│   │   ├── lora/
│   │   │   └── rudolph_1964_v2.safetensors
│   │   ├── examples/
│   │   │   ├── menacing_walk.mp4
│   │   │   └── setup_beat.mp4
│   │   └── PROMPTING_GUIDE.md        # Human-readable notes
│   │
│   ├── tmnt_mutant_mayhem/
│   │   ├── manifest.yaml
│   │   ├── lora/
│   │   └── examples/
│   │
│   └── rooster_terry/
│       ├── manifest.yaml
│       ├── lora/
│       └── examples/
│
├── /src/
│   ├── /core/
│   │   ├── world_state.py            # WorldState, CharacterState, Prop
│   │   ├── control_state.py          # ControlState
│   │   ├── style_manifest.py         # StyleManifest
│   │   └── prompt_compiler.py        # PromptCompiler
│   │
│   ├── /engine/
│   │   ├── generator_driver.py       # GeneratorDriver
│   │   ├── control_bus.py            # ControlBus, ControlEvent
│   │   ├── frame_bus.py              # FrameBus
│   │   └── branch_graph.py           # BranchGraph
│   │
│   ├── /api/
│   │   ├── main.py                   # FastAPI app
│   │   ├── routes/
│   │   │   ├── sessions.py
│   │   │   ├── world.py
│   │   │   ├── styles.py
│   │   │   ├── control.py
│   │   │   ├── snapshots.py
│   │   │   └── timeline.py
│   │   └── websockets/
│   │       ├── stream.py
│   │       └── events.py
│   │
│   └── /skills/                      # Claude skill integration
│       ├── prompt_iteration.py
│       └── style_authoring.py
│
├── /tests/
│   ├── test_prompt_compiler.py
│   ├── test_world_state.py
│   └── test_branch_graph.py
│
└── /docs/
    ├── ARCHITECTURE.md               # This document
    └── API.md                        # OpenAPI spec
```

---

## Skeptical Checks (Before You Overbuild)

Two important questions to keep asking yourself:

### 1. Do you really want "narrative logic" early?

The narrative layer (beat templates, arc state machines, character knowledge graphs) is seductive because it feels like "the real creative tool." But it can easily become a full project unto itself.

**The 80/20 version**: Make "beats" just a small vocabulary (setup, escalation, payoff, reset) plus a few manual state transitions. Let the human drive. You get most of the value without building a narrative engine.

**Build it properly only when**: You have a working instrument and find yourself manually doing the same narrative transitions repeatedly.

### 2. Do you really need true KV cache serialization for branching?

Given that your pipeline's recompute block already rebuilds cache from stored context frames, "true state branching" (serializing the full KV cache) is a **performance optimization**, not a prerequisite.

**What you need to branch**: 
- `current_start_frame`
- `context_frame_buffer`
- `decoded_frame_buffer`
- ControlState snapshot
- Seed stream state

**What you don't need**: The full `kv_cache` and `crossattn_cache` tensors.

The recompute block will rebuild the cache when you restore. It's slower than a true cache restore, but it works today with no additional engineering.

**Build true cache serialization only when**: Branching is working and the recompute time is actually your bottleneck (measure first).

### 3. Are you blocked on GPU performance?

If you're stuck at 8.8 fps and tempted to dive into kernel optimization, remember:

- Each chunk is ~5 forward passes (1 recompute + 4 denoise)
- Optimizing one pass may not move the wall clock
- The interaction system doesn't need fast generation to be useful

**The productive move**: Get step mode working first. Then you can iterate prompts at *any* fps. Performance becomes a parallel track, not a blocker.

---

## Build Order (16 Days)

### Minimum Viable Day 2 (Get Unstuck Today)

If you implement only these three things, you have a working instrument:

```
POST /sessions           → Create session
PUT  /sessions/{id}/prompt  → Update prompt
POST /sessions/{id}/step    → Generate 1 chunk
```

Wire `step` to "call the pipeline once and publish 3 frames." That proves the control architecture is real and makes every later feature an extension of the same loop.

### Full Build Order

| Days | Deliverable | Files Created | Demo Capability |
|------|-------------|---------------|-----------------|
| **2** | **Minimum Viable Instrument** | | |
| | Session + step endpoint | `main.py`, `sessions.py` | |
| | Prompt update endpoint | `control.py` | "Change prompt, step, see 3 frames" |
| **3-4** | **Dev Console Foundation** | | |
| | ControlState + ControlBus | `control_state.py`, `control_bus.py` | |
| | Step mode with state inspection | | "See exactly what was sent to pipeline" |
| | Basic FrameBus (latest frame only) | `frame_bus.py` | |
| **5-7** | **Style Layer** | | |
| | StyleManifest schema | `style_manifest.py` | |
| | PromptCompiler | `prompt_compiler.py` | |
| | Style API routes | `styles.py` | "Same WorldState, different LoRA output" |
| **8-10** | **World Layer** | | |
| | WorldState schema | `world_state.py` | |
| | World API routes | `world.py` | |
| | Integration (World→Style→Control) | | "Change tension, see style-appropriate prompt" |
| **11-13** | **Branching** | | |
| | FrameBus ring buffer (full) | | |
| | BranchGraph | `branch_graph.py` | |
| | Snapshots + Fork API | `snapshots.py` | "Pause, fork, rollout 4 options, pick one" |
| **14-16** | **Polish + VLM Hooks** | | |
| | Timeline API | `timeline.py` | |
| | WebSocket streaming | `stream.py`, `events.py` | |
| | VLM controller hooks | | "Automated suggestions in pause mode" |

---

## Performance Knobs (When You Need FPS Now)

These bypass deep optimization work and give immediate speed gains:

| Knob | Default | Fast Setting | Tradeoff |
|------|---------|--------------|----------|
| `denoising_step_list` | `[1000,750,500,250]` (4 steps) | `[1000,500]` (2 steps) | ~2x faster, lower quality |
| `num_frames_per_block` | 3 | 1 | More overhead, finer control |
| Resolution | Full | 50% | 4x fewer tokens |
| KV recompute | Every block | Every N blocks | More drift, faster |

---

## Profiling Strategy

Only profile after the control loop works:

1. **`PROFILE_PIPELINE_BLOCKS=1`** - Find which block dominates (RecomputeKV, Denoise, Decode)
2. **`PROFILE_ATTENTION=1`** - If attention is bottleneck, see FFN vs cross-attn
3. **`SCOPE_KV_BIAS_BACKEND`** - Check when switching GPUs (V200 vs V300 may select different backends)

---

## Key Design Decisions

1. **Branch only at chunk boundaries** (every 3 frames) - aligns with KV cache machinery
2. **WorldState has no LoRA knowledge** - content is domain-agnostic
3. **StyleManifest is the codified prompt knowledge** - your experiments become reusable
4. **Dev Console bypasses compiler** - direct ControlState access for iteration
5. **Snapshots store context buffers, not KV cache** - recompute block rebuilds cache

---

## Integration Points for Claude Skills

Your existing prompt engineering skills become StyleManifest authoring tools:

```
┌─────────────────────────────────────────────────────────┐
│ PROMPT ITERATION SKILL                                  │
│                                                         │
│ Input:  LoRA + test WorldState                         │
│ Output: Working prompt tokens                           │
│                                                         │
│ Method: Dev Console step mode                          │
│         - Generate variations                           │
│         - A/B test with step()                         │
│         - Identify effective tokens                     │
└────────────────────────┬────────────────────────────────┘
                         │ codify
                         ▼
┌─────────────────────────────────────────────────────────┐
│ STYLE AUTHORING SKILL                                   │
│                                                         │
│ Input:  Working prompts from iteration                 │
│ Output: StyleManifest YAML                             │
│                                                         │
│ Method: Extract patterns                               │
│         - Map abstract terms to effective tokens        │
│         - Define vocab dictionaries                     │
│         - Set priority order and token budget          │
└─────────────────────────────────────────────────────────┘
```

---

## Appendix: Example Style Manifest (YAML)

```yaml
# /styles/rudolph_1964/manifest.yaml

name: rudolph_1964
lora_path: /styles/rudolph_1964/lora/rudolph_1964_v2.safetensors
lora_default_scale: 0.85

trigger_words:
  - rudolph1964
  - rankinbass

material_vocab:
  skin: "felt texture, visible stitching, matte surface"
  metal: "painted tin, slightly reflective, vintage toy"
  wood: "carved balsa wood, visible grain, handcrafted"
  snow: "cotton batting snow, sparkle dust, miniature drifts"
  default: "stop-motion puppet, Rankin-Bass style"

motion_vocab:
  walk: "deliberate puppet walk cycle, 12fps, slight wobble"
  run: "held cels, blur frames, exaggerated stride"
  enter_menacing: "slow deliberate steps, looming presence, shadow first"
  fall: "replacement animation tumble, squash on impact"
  idle: "subtle breathing motion, micro-adjustments"

camera_vocab:
  low_angle: "miniature set perspective, forced depth, looking up"
  tracking: "smooth dolly move, tabletop track"
  close_up: "macro lens softness, shallow depth of field"
  wide: "full set visible, diorama framing"

lighting_vocab:
  dramatic: "strong key light, deep shadows, rim lighting"
  flat: "soft even lighting, minimal shadows, cheerful"
  night: "blue fill, warm practicals, moonlight key"

emotion_vocab:
  frustrated: "furrowed brow, clenched pose, agitated micro-movements"
  happy: "wide eyes, bouncy movement, warm lighting"
  suspicious: "narrowed eyes, guarded posture, side glance"

beat_modifiers:
  setup: "establishing shot, calm pacing"
  escalation: "quicker cuts, tighter framing"
  payoff: "dramatic pause, reaction shot"
  reset: "wide shot, normalized lighting"

default_negative: "realistic, photographic, 3D render, modern CGI, smooth plastic"

max_prompt_tokens: 77
priority_order:
  - trigger
  - action
  - material
  - camera
  - mood
```

---

## Known Limitations & Upgrade Paths

This document intentionally simplifies several things to keep the Day 2-16 build tractable. Each limitation has a clear upgrade path when you need it.

### Snapshot Tiers

**Current (Day 2-10)**: Hard-cut snapshots
- Stores WorldState, ControlState, chunk_index
- Attempts to capture pipeline continuity buffers if available
- If buffers are missing or incompatible, restore produces a visible cut (acceptable for branching)

**Upgrade 1**: Continuity snapshots
- Validate that captured buffers actually restore correctly
- Add buffer compatibility checks (resolution, format)
- Test that KV cache recompute produces seamless continuation

**Upgrade 2**: True KV cache serialization
- Serialize full `kv_cache` and `crossattn_cache` tensors
- Skip recompute on restore (faster branching)
- Memory heavy: ~2-4GB per snapshot depending on model size
- Only implement if recompute time is actually your bottleneck (measure first)

### Seed Determinism

**Current**: Simple offset scheme
- `effective_seed = base_seed + branch_seed_offset`
- Changing seed changes everything including frames before fork point
- Good enough for "try 4 different directions" creative branching

**Upgrade**: Seed stream scheme
- Stable seed stream for frames up to fork point
- Different seed stream starts only after fork boundary
- Enables "prefix-identical, then diverge" reproduction
- Requires tracking per-step seed state in snapshot

### Token Budget

**Current**: Heuristic (0.75 tokens per word)
- Simple, fast, usually works
- Will silently truncate on weird LoRA trigger words or long compounds
- Fine for Day 2-7

**Upgrade**: Real tokenizer integration
- Use the same tokenizer the pipeline uses for text encoding
- Exact token counting, no silent truncation
- Implement when you notice prompts behaving unexpectedly

### Character Handling in PromptCompiler

**Current**: First character only
- `break` after first character in emotion loop
- Fine for single-character scenes

**Upgrade**: Visibility-aware multi-character
- Track which characters are visible to camera
- Sort by narrative importance
- Include top K in prompt, others implied
- Implement when you have multi-character scenes that look wrong

### Streaming Output

**Current**: Unspecified (whatever works)
- Doc mentions both WebRTC and WebSocket without deciding
- For Day 2, any preview that shows frames is fine

**Upgrade**: Dual output paths
```python
class OutputPaths:
    preview_stream: WebSocketStream   # Debug, higher latency OK
    primary_stream: WebRTCStream      # Production, latency critical
```
- FrameBus fans out to both
- Implement when latency matters for your use case

### Pipeline Kwargs Coverage

**Current**: Partial
- `to_pipeline_kwargs()` includes core params + lora_stack + kv_cache_attention_bias
- May not cover all KREA pipeline options

**Upgrade**: Full pipeline config
- Audit KREA pipeline for all accepted kwargs
- Add missing params to ControlState
- Add validation that kwargs match pipeline expectations

### ControlBus Event Replay

**Current**: History stored but not replayed
- Events stored in `ControlBus.history` for debugging
- No replay mechanism implemented

**Upgrade**: Deterministic replay
- Given initial state + event history, reproduce exact session
- Useful for debugging, demos, regression testing
- Requires seed stream scheme (above) to be meaningful

---

## Document History

This architecture emerged from synthesizing two independent analysis passes that converged on the same design:

1. **Implementation-focused pass**: Complete Python implementations, data structures, sequence diagrams
2. **Architecture-focused pass**: Two-axis framing, explicit event semantics, skeptical checks
3. **Technical review pass**: Surgical fixes to example code, Known Limitations section

The convergence validates the design. Key additions from synthesis:
- Two-axis diagram (semantic layers × operational planes)
- Explicit ControlBus with typed events for debugging/replay
- Skeptical checks section to prevent overbuilding
- Aggressive Day 2 scope (3 endpoints to get unstuck immediately)

**Surgical fixes applied (v1.1)**:
- Fixed world update recompile bug (was checking after clear, always false)
- Fixed resume() concurrency (guards against spawning multiple loops)
- Fixed to_pipeline_kwargs() (now includes lora_stack, kv_cache_attention_bias)
- Fixed kv_cache_attention_bias default (0.3, matching KREA)
- Fixed FrameBus buffer sizing (chunks, not frames)
- Added generator continuity buffers to snapshot/restore
- Added Known Limitations & Upgrade Paths section

*Last updated: Day 2 of 16-day sprint (v1.1 with surgical fixes).*
