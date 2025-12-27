# Tidal Cycles meets AI video: the live-coded audiovisual frontier

**Real-time AI video synchronized to live-coded music represents the next evolutionary leap in algorithmic performance.** The ecosystem surrounding Tidal Cycles already includes robust video integration patterns—from Hydra and Punctual to sophisticated OSC routing architectures—that can directly inform a Tidal-to-AI-video pipeline. The community has developed battle-tested synchronization approaches, hardware control paradigms, and nascent AI experiments that provide both technical foundations and creative inspiration for your proposal.

This report maps the landscape of Tidal Cycles visual integrations, algorithmic AV synchronization techniques, narrative-driven music generation research, and emerging 2024-2025 patterns including diffusion-model experiments. Key takeaways: **OSC-based pattern routing** is the dominant integration architecture, **semantic control parameters** (energy, tension, density) offer promising abstraction layers, and the Epidemic Sound MCP server demonstrates the first production-grade Model Context Protocol integration for music—a pattern directly applicable to AI video control.

---

## Hydra and Strudel lead the visual integration ecosystem

The most mature Tidal-video integration is **Hydra**, Olivia Jack's browser-based video synth using WebGL shaders with syntax inspired by analog modular synthesizers. Diego Dorado's `tidal-hydra-tutorial` documents four distinct integration methods: forwarding SuperDirt events through SuperCollider, sending OSC directly from Tidal, custom OSC messages with user-defined parameters, and routing RMS audio levels from orbits for audio-reactive visuals.

The technical pattern involves defining a custom OSC target in Tidal's `BootTidal.hs`:

```haskell
let target = Target {
    oName = "visualiser",
    oAddress = "localhost",
    oPort = 5050,
    oLatency = 0.2,
    oSchedule = Live
}
```

Hydra receives these via JavaScript OSC listeners, mapping pattern values to shader parameters. A critical limitation: this only works with desktop editors (Atom/Pulsar)—browser-based Hydra cannot receive OSC directly.

**Strudel**, the official JavaScript port of TidalCycles by Felix Roos and Alex McLean, offers the most seamless visual integration via its `@strudel/hydra` package. Running entirely in-browser at strudel.cc, it enables patterns like:

```javascript
await initHydra()
let pattern = "3 4 5 [6 7]*2".slow(2)
note(pattern).play()
osc(H(pattern)).out()  // H() converts pattern to hydra input
```

Built-in visualizations include pianoroll, punchcard, spiral, oscilloscope, and spectrum displays. The **Estuary** platform extends this further, supporting multiple visual languages in a single collaborative environment: MiniTidal (Tidal patterns), Punctual (David Ogborn's audio-visual language), CineCer0 (video/typography), and Hydra.

---

## OSC patterns form the universal integration backbone

Tidal's OSC configuration system provides a flexible foundation for any video integration. The scheduling system offers three modes: **Live** (messages sent at correct time minus latency), **Pre BundleStamp** (bundled with OSC timestamps), and **Pre MessageStamp** (timestamp embedded in message). For AI video systems requiring precise timing, Pre BundleStamp likely offers the best balance of accuracy and compatibility.

Every Tidal event automatically includes `cps` (cycles per second), `cycle` (event start position), and `delta` (event duration)—providing the timing metadata an AI video system would need for synchronization. Custom parameters can be routed through pattern functions:

```haskell
-- Multi-format OSC for complex visual control
OSC "/{asccolour}/speed" $ ArgList [("ascspeed", Nothing)]
OSC "/{asccolour}/mode" $ ArgList [("ascmode", Nothing)]
```

Controller input flows back via port 6010 using `/ctrl` messages with typed accessors: `cF` for floats, `cI` for integers, `cS` for strings, and notably `cP` for entire patterns in mini notation. This bidirectional capability enables an AI video system to send feedback parameters back to Tidal, creating genuine dialog between audio and visual generation.

---

## Four architectural patterns for audio-visual synchronization

The community has converged on four primary AV sync architectures, each with distinct tradeoffs for AI video integration:

**Pattern 1: Shared Clock (Tight Sync)** uses Ableton Link or ESPGrid for tempo synchronization. Tidal 1.9+ includes native Link support; earlier versions use the **Carabiner** bridge. Link maintains shared tempo and phase across network-connected devices with sub-millisecond accuracy. Configuration requires setting `cQuantum` (beats for phase sync) and `cBeatsPerCycle` (BPM/CPS conversion factor). For an AI video system, this ensures frame generation aligns precisely to musical phrases.

**Pattern 2: OSC Event-Driven** routes pattern events as discrete triggers. Each note, sample, or parameter change fires an OSC message that the video system processes independently. This works well for systems that generate video frames on-demand but requires careful latency compensation.

**Pattern 3: Audio-Reactive (Analysis-Driven)** processes the audio stream through FFT analysis, extracting features like onset detection, spectral centroid, and energy bands. Libraries like **Meyda** (JavaScript/Web Audio), **Essentia** (C++ with JS bindings), and **Aubio** (C with Python bindings) provide real-time feature extraction. This approach offers organic visual response but loses pattern intentionality.

**Pattern 4: Hybrid** combines pattern-based triggers with audio reactivity—the approach most algorave performers use. Pattern events control major visual transitions while audio analysis modulates continuous parameters like color saturation or motion intensity.

For AI video generation, **a hybrid of Pattern 2 and semantic control** appears most promising: OSC events carrying high-level descriptors (scene mood, energy level, visual style hints) rather than low-level audio features, allowing the AI sufficient context to generate meaningful imagery.

---

## Semantic control offers the crucial abstraction layer

Research on crossmodal correspondence reveals consistent perceptual mappings between audio and visual parameters. The **arousal-valence model** from music psychology provides a framework: arousal (energy/intensity) and valence (positive/negative) can be predicted from audio features and mapped to visual parameters.

| Audio Feature | Visual Parameter |
|--------------|------------------|
| High arousal | Saturated colors, red tones, faster motion |
| Low arousal | Desaturated colors, blue tones, slower motion |
| High valence | Bright colors, yellow tones, expanding forms |
| Low valence | Dark colors, contracting forms |
| Spectral flux | Object morphing/transitions |
| Pitch | Vertical position, color lightness |

Essentia's pre-trained models can predict arousal (R²=0.88) and valence (R²=0.74) from audio features, providing a bridge between raw musical events and semantic descriptors an AI video system could interpret. The **Tidal-MerzA** project (2024) demonstrates this approach, combining affective modeling with autonomous code generation using reinforcement learning agents aligned to emotional state.

---

## Narrative music generation inverts the proposal's direction

Understanding how narrative drives music illuminates the inverse challenge. Google's **MusicLM** generates music from rich text captions, supporting sequential descriptions that create melodic "stories" (e.g., "time to meditate" → "time to wake up" → "time to run"). The **Generative Theory of Tonal Music (GTTM)** and **Tonal Tension Model (TTM)** by Lerdahl and colleagues provide theoretical foundations: music expresses meaning through hierarchical structures, with tension arcs operating at multiple timescales from individual phrases to entire compositions.

Game audio middleware demonstrates mature "world state to music" systems. **Wwise** uses Real-Time Parameter Control (RTPCs) mapping game variables (player health, proximity to enemies) to audio parameters. **FMOD** offers similar capabilities with a DAW-like interface. Architectural patterns include:

- **Vertical layering**: Multiple simultaneous layers (drums, bass, melody) that fade in/out based on state
- **Horizontal resequencing**: Discrete musical sections triggered by events
- **Procedural generation**: Markov chains, genetic algorithms, or neural networks creating variation

For a Tidal-to-AI-video system, these patterns suggest the video generator should receive not just immediate events but **accumulated context**: current tension level, recent pattern density, stylistic consistency hints, and narrative arc position.

---

## AI agents enter the live coding conversation

**Cibo** (Jeremy Stewart, ICLC 2019) represents the first sequence-to-sequence neural network generating TidalCycles code, operating purely on code without audio processing. **Tidal-MerzA** (2024) advances this with reinforcement learning, using ALCAA (Affective Live Coding Autonomous Agent) plus Tidal Fuzz for mini-notation generation aligned to target emotional states.

The **Epidemic Sound MCP Server** (September 2025) marks the first major music industry Model Context Protocol implementation. It provides AI agents with context-aware music discovery: "mood: calm" or "scene: dark forest at dawn" returns curated tracks. The integration pattern—natural language context → structured query → creative asset retrieval—directly maps to AI video generation from musical context.

Human-AI collaboration research accelerates. The **Revival** project (K-Phi-A Collective) blends percussionist, electronic performer, and AI agents (MASOM, SpireMuse) trained on deceased composers' works, orchestrated via Chataigne software over OSC. Google Labs collaborations with Jacob Collier, Lupe Fiasco, and Dan Deacon explore MusicFX DJ for generative continuous music flow alongside human performance.

No native Copilot-style real-time completion exists yet for Tidal—LLM integration currently requires copy-paste workflows. However, community experiments with ChatGPT generating Sonic Pi code produce functional generative patterns, suggesting the path toward AI-assisted pattern generation during live performance.

---

## Hardware control surfaces enable physical semantics

**Monome Grid** integration with Tidal uses SuperCollider as a negotiator via the **Grrr toolkit**, mapping button presses to code block evaluation through VIM/TMUX. **Norns** (Monome's sound computer) can run TidalCycles via SSH terminal, though memory constraints require lazy sample loading and swap files during Haskell compilation.

For Ableton workflow integration, **tidal4live** provides Max4Live devices controlling software instruments and audio parameters, while **HackYourDaw** enables direct Ableton control from Tidal without SuperCollider via the Live Object Model.

**Open Stage Control** offers the most accessible custom control surface path: define widgets with `/ctrl` preArgs, forward OSC through SuperCollider, and use Tidal's control functions (`cF`, `cS`, `cI`, `cP`) to receive values. TouchOSC provides cross-platform support with GPU-powered editing and Lua scripting.

For physical "semantic" parameters—intensity sliders, tension knobs—DIY Arduino MIDI controllers using the MIDIUSB library or Teensy boards offer budget options ($30-100). The **MIDIbox** project provides modular open-source hardware supporting up to 64 encoders. These physical controls could modulate the semantic parameters sent to an AI video system: intensity affecting visual dynamism, tension influencing color palette and composition complexity.

---

## The 2024-2025 landscape shows browser-first and AI-hybrid trends

**Browser-based tools dominate recent development.** Strudel 1.0 released January 2024, Hydra continues active development, and Flok provides P2P collaborative editing supporting up to 8 simultaneous code slots across TidalCycles, Hydra, SuperCollider, Mercury, and Strudel. The zero-installation paradigm lowers barriers to entry dramatically.

**ChuGL** (Stanford CCRMA, NIME 2024) extends ChucK with unified audiovisual programming: Graphics Generators (GGen) alongside audio unit generators (UGens) with sample-synchronous scheduling. **RayTone** offers node-based audiovisual sequencing combining ChucK DSP with GLSL shaders. These represent a philosophical shift toward equal treatment of audio and graphics in a single language.

**Diffusion models meet live audio** through tools like **Neural Frames**, which extracts stems (kick, snare, vocals) for granular audio-reactive parameter mapping to Stable Diffusion generation at 25 FPS with 4K output. Turbo Mode (June 2024) achieved 400% faster rendering. The "stable-diffusion-dance" model from Pollinations synchronizes generated images to input audio in real-time. An ACM IMX 2025 paper describes integrating Music Information Retrieval + LLMs + Image Generation using adversarial diffusion distillation for speed.

**ICLC 2024** (Shanghai—first in Asia) centered on LLMs and generative AI in live coding, exploring agency in human-machine conversation. ICLC 2025 heads to Barcelona. The community actively debates AI's role: collaborator, tool, or threat to the improvisational ethos.

---

## Key artists and researchers to follow

**Alex McLean (Yaxu)** created TidalCycles and co-founded Algorave and TOPLAP; he directs the AlgoMech festival and performs with Slub and CCAI. **Kindohm (Mike Hodnick)** composes exclusively with TidalCycles using conditional patterning logic, known for 170 BPM "off-kilter dancefloor interruptions" on Conditional Records. **Atsushi Tadokoro** pioneers TidalCycles + TouchDesigner audiovisual performance in Japan and researches live coding of laser beams. **Renick Bell** authored Conductive (Haskell live coding library) and researches algorithmic composition at Tama Art University.

Research centers include **Stanford CCRMA** (ChuGL, RayTone, ChAI), **Goldsmiths University of London** (Computational Arts program, creative AI), **Queen Mary Centre for Digital Music** (human-machine agencies research), and **McMaster University** (Estuary platform). **Olivia Jack** (Hydra creator) and **David Ogborn** (Punctual, Estuary) drive visual live coding tool development.

---

## Architectural implications for Tidal + AI video

Synthesizing these patterns, a Tidal-to-AI-video system could adopt this architecture:

1. **Event capture layer**: OSC messages from Tidal carrying pattern events, cycle position, tempo, and custom semantic parameters
2. **Context accumulator**: Maintains rolling state of musical tension, density, stylistic consistency, and narrative arc position
3. **Prompt generator**: Translates accumulated context into structured prompts for the AI video model, using arousal-valence mappings and scene descriptors
4. **Latency compensator**: Accounts for AI generation time by using Tidal's `delta` values and scheduling modes to predict ahead
5. **Feedback channel**: Returns generated visual characteristics to Tidal via `/ctrl` messages, enabling bidirectional influence

Novel approaches not yet explored include: using Tidal's pattern algebra to directly control diffusion model conditioning vectors; training a model specifically on Tidal pattern semantics; implementing MCP as the integration protocol between Tidal and AI video services; and creating "visual patterns" that mirror Tidal's combinatorial pattern language for video parameter spaces.

---

## Essential repositories and resources

| Resource | Purpose | URL |
|----------|---------|-----|
| **TidalCycles** | Main Haskell pattern library | github.com/tidalcycles/tidal |
| **Strudel** | Browser-based JavaScript port | codeberg.org/uzu/strudel |
| **Hydra** | Browser visual synth | github.com/hydra-synth/hydra |
| **tidal-hydra-tutorial** | OSC integration guide | github.com/diegodorado/tidal-hydra-tutorial |
| **Flok** | Collaborative editor | codeberg.org/munshkr/flok |
| **Estuary** | Multi-language platform | estuary.mcmaster.ca |
| **Carabiner** | Ableton Link bridge | github.com/Deep-Symmetry/carabiner |
| **awesome-livecoding** | Comprehensive resource list | github.com/toplap/awesome-livecoding |
| **Neural Frames** | Audio-reactive diffusion | neuralframes.com |
| **Live Coding Manual** | Open-access MIT Press book | livecodingbook.toplap.org |

The landscape reveals a community actively experimenting with AI integration while maintaining commitment to improvisation, transparency, and collaborative performance. Your proposal sits at the convergence of mature integration patterns (OSC, Link, semantic mapping) and emerging capabilities (diffusion models, LLM assistants, MCP protocols)—a genuinely novel synthesis with strong foundations to build upon.
