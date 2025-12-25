# Akira Captioning Style and Prompting Guide (v4)

## Purpose

- Provide a compact, repeatable prompting style that mirrors the captioning language used for training while remaining flexible for novel and out‑of‑distribution (OOD) scenes.
- Help humans and agents compose new prompts that stay "on model" for visual grammar, materials, lighting, motion, and camera language specific to the 1988 cel animation look.

## Core Voice and Structure

- Optional opener: "1988 Cel Animation — …" anchors style and era.
- Single paragraph per shot; declarative, visual-first phrasing.
- Order: composition → subject → background → palette/materials → camera → motion → effects → timing.
- Prefer precise nouns and fixed phrases; avoid modern/CG terms unless explicitly contrasting.

## Visual Grammar (Materials and Surfaces)

- FLAT-COLORED CELS; BOLD BLACK INK OUTLINES
- DENSELY DETAILED PAINTED BACKGROUND (DDPB)
- OPAQUE PAINTED SMOKE PLUMES; painted dust/debris; hand‑drawn sparks
- GRADIENT AIRBRUSHING (glows/halos); TRANSLUCENT ENERGY AURA
- ANAMORPHIC-STYLE LENS FLARE; SOFT BOKEH
- LIGHT TRAILS; MOTION BLUR LINES

## Lighting Language

- DEEP BLACKS (shadow mass); BRILLIANT HIGHLIGHTS (specular hits)
- Overexposure/whiteout/impact frame; silhouette against glow
- Colored neons (pink, teal, magenta), warm firelight cores (yellow/orange)

## Camera and Framing

- Static wide/medium/close; low/high angle; top‑down; fisheye
- TRACKING SHOT; push‑in/pull‑out; simulated zoom; iris wipe; whip pan
- POV shots; over‑the‑shoulder; dolly parallax (MULTI-LAYER PARALLAX)

## Motion and Timing

- "animated on ones/twos"; subtle vibration; slow push‑in; fast pull‑out
- Impact frame (white); strobing muzzle flashes; drifting smoke; debris float

## Canon Tokens (consistent identifiers)

- KANEDAS RED MOTORCYCLE; RED RIDING SUIT; THE PILL CAPSULE SYMBOL
- TETSUOS MUTATING ARM / BODY HORROR; RED CAPE; CYBERNETIC PROSTHETICS
- LASER CANNON (ground); LASER CANNON SATELLITE (SOL)

## Palette and Atmosphere

- Nocturnal NEO‑TOKYO: teal/blue shadows, neon pinks, magentas, cyans
- Fire/explosion cores: yellow→orange with dark red/black plumes
- Industrial greens, ochres, dusty violets for rubble/smoke/water grime

## Negative Space (avoid unless specified)

- Modern anime style, cute/moe/chibi, CGI/rendered/3D look, sketch/comic tropes
- Over‑smoothing/airbrushed skin, plastic look, compression artifacts, overlays

## Prompt Scaffolds

1. **Baseline shot**
   - 1988 Cel Animation — [composition + angle + lens], [subject + pose/expression], against a [DDPB of setting]. Materials: [flat‑colored cels with BOLD BLACK INK OUTLINES], lighting with [DEEP BLACKS and BRILLIANT HIGHLIGHTS]. Camera: [TRACKING SHOT/push‑in/pull‑out/fisheye]. Motion: [animated on ones/twos], [MOTION BLUR LINES/LIGHT TRAILS]. FX: [OPAQUE PAINTED SMOKE PLUMES/ANAMORPHIC-STYLE LENS FLARE].

2. **Action FX**
   - 1988 Cel Animation — [impact action], painted [yellow/orange] core with [OPAQUE PAINTED SMOKE PLUMES], debris [shards/sparks]. A [white IMPACT FRAME] punches the cut before returning to [DDPB].

3. **Water/Reflections**
   - 1988 Cel Animation — Low‑angle close on a wheel plowing a puddle; hand‑drawn white spray; quick pull‑up into a tracking shot with MULTI‑LAYER PARALLAX; turquoise water reflects hub assembly on a separate cel.

4. **SOL / Laser**
   - 1988 Cel Animation — A razor‑thin green‑white beam from the LASER CANNON SATELLITE (SOL) slices through stylized cloud plumes; the strike resolves in a white impact frame, then expanding RED AND ORANGE PAINTED EXPLOSION.

## Composition Building Blocks (phrase bank)

- **Materials:** flat‑colored cels; bold outlines; painted background; airbrushed glow; opaque smoke plumes; translucent aura; hand‑drawn sparks; motion blur lines; light trails
- **Camera:** static wide; low‑angle; top‑down; fisheye; POV; tracking shot; dolly; push‑in; pull‑out; simulated zoom; iris wipe; whip pan
- **Lighting:** deep blacks; brilliant highlights; overexposed bloom; silhouettes; colored neon spill; warm core/cool rim
- **Motion:** animated on ones/twos; slow push‑in; rapid pull‑out; subtle vibration; parallax layers

## OOD Prompting Strategies (staying in‑style)

- Preserve grammar, materials, and timing terms while swapping setting/props.
- Anchor with DDPB + cel/outline; then introduce novel subject matter (e.g., maritime cranes, elevated gardens, storm‑lit rooftops) described in the same painted/inked terms.
- Use one or two signature FX max (e.g., lens flare + smoke), not a laundry list.
- Keep color narrative coherent (neon night vs firelight vs overcast day).

## Mini Examples (mix-and-match)

- 1988 Cel Animation — A low‑angle TRACKING SHOT along a rain‑slick causeway as KANEDAS RED MOTORCYCLE leans into a slide, hand‑drawn white spray arcing from the wheels. The DENSELY DETAILED PAINTED BACKGROUND recedes in MULTI‑LAYER PARALLAX under DEEP BLACKS with neon magenta spill, animated on twos.

- 1988 Cel Animation — Top‑down static wide of a rooftop plaza; a glowing fountain ripples with airbrushed light while thin MOTION BLUR LINES trace two distant couriers crossing, their flat‑colored cels edged in BOLD BLACK INK OUTLINES.

- 1988 Cel Animation — An iris wipe reveals a fisheye view of an industrial stadium shell as a green‑white SOL beam cuts the overcast, snapping to a white IMPACT FRAME before RED AND ORANGE PAINTED EXPLOSION plumes fill the frame.

## Choice Example Captions (verbatim style)

### Alley Crash with Spotlights (099)

1988 Cel Animation — From a high, canted angle overlooking a DENSELY DETAILED PAINTED BACKGROUND of a dark NEO-TOKYO alley, a single bright spotlight sweeps across the grimy building facades, rendered in deep blacks and muted blues. A second spotlight joins it, illuminating two small motorcycles as they race through the chasm below, their brilliant headlights and red taillights creating faint LIGHT TRAILS. The scene cuts to a low-angle view from within a trash-choked side street, where a clown gang biker on a custom chopper plows into frame. The biker wears a purple helmet with a painted clown face and a light blue mask. His bike's headlights cast a bright glow as he crashes through piles of debris with fluid motion. A glass bottle shatters, sending a splash of hand-drawn purple liquid arcing through the air while wooden crates splinter into sharp fragments. He bursts out of the alley, tumbling violently with his motorcycle in a shower of debris before coming to rest in a heap in front of a detailed storefront with a red, dragon-painted security gate.

### SOL Bridge Strike (1769)

1988 Cel Animation — From a high-angle perspective, dozens of civilians stand scattered across a wide concrete bridge, observing an off-screen event, set against a DENSELY DETAILED PAINTED BACKGROUND of a massive construction site. A brilliant green-and-white horizontal beam from a LASER CANNON SATELLITE streaks across the scene from the left, striking the bridge with a bright white IMPACT FRAME. Instantly, a massive RED AND ORANGE PAINTED EXPLOSION erupts, its force violently throwing the civilian figures into the air. Their bodies tumble like ragdolls as OPAQUE PAINTED SMOKE PLUMES with BOLD BLACK INK OUTLINES billow outwards and upwards, filling the screen with dark red and fiery yellow animated clouds. Sharp shards of debris are violently ejected from the blast's epicenter as the explosion continues to expand with immense force.

### Akira Slide Reflection (2341)

1988 Cel Animation — From a high-angle perspective, the front wheel of KANEDAS RED MOTORCYCLE executes an AKIRA SLIDE, scraping sideways from right to left across a cracked, DENSELY DETAILED PAINTED BACKGROUND of dark asphalt. The wheel, rendered with BOLD BLACK INK OUTLINES and flat red and green colored cels, casts a hard shadow as the camera performs a smooth tracking shot. The rear wheel slides into view, and small, hand-drawn white sparks with yellow centers arc from the friction point of the front wheel. As the slide continues, the rear wheel passes over a puddle of turquoise water, its intricate red hub assembly creating a perfect, clear reflection on the water's surface, painted on a separate cel layer to create depth.

For more examples, see the batch YAMLs in `WorkingSpace/davidrd/Contexts/Akira/Prompting/` (Motion, CameraFX, ActionFX, SOL, Water).

## Operational Tips (for agents)

- Use this guide as a token bank: sample a few phrases from each category rather than copying entire sentences.
- Keep shots atomic: one location, one primary action, one camera move.
- Log requested vs applied values at boundaries (resolution/frames/LoRA/negatives) and use applied values thereafter.

## Appendix: Suggested Negative Prompt (copyable)

```text
色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 杂乱的背景, 背景人很多, 倒着走, split screen, text, signature, watermark, logo, timestamp, cartoon, modern anime style, moe, chibi, illustration, 3d render, CGI, rendered, painting, sketch, comic book style, plastic skin, overly smooth features, airbrushed, doll-like, mannequin, untextured surfaces, artifacting, compression artifacts, glitch, digital noise, banding, pixelation, frame, border, collage, oversaturated, pastel colors, soft colors, flat lighting, low contrast, low detail background, empty background, out of focus detail, extreme color cast, monochrome unless specified, generic AI look, uncanny valley, poorly fused elements, inconsistent lighting unless specified
```
