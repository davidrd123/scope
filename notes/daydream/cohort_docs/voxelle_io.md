Title: Frost Bytes — Voxelle io

URL Source: https://app.daydream.live/creators/voxelle_io/frost-bytes

Markdown Content:
Daydream AI Video Program - Scope Track Project
-----------------------------------------------

"Frost Bytes" is an immersive Unreal Engine 5 virtual production where X4NTHA performs a live-coded Tidal Cycles electronic music set, orchestrating a soundscape that is visually mirrored by a real-time, generative AI video stream of an abstract Aurora Borealis. This project explores the interplay between rigid algorithmic code and fluid, dreamlike AI imagery, creating a cozy atmosphere that turns technical limitations (like video artifacts) into an immersive artistic statement. The pipeline intelligently synchronizes OSC audio data and prompt text generated in real time by Tidal Cycles from code with a remote WebRTC AI video stream to drive reactive lighting and skybox textures, resulting in a cohesive, latency-managed audiovisual broadcast.

![Image 1: Conceptual Mock-up](https://api.daydream.live/v1/assets/resolve?key=assets%2F1766616676610-d9aa8e29.png)

Conceptual Mock-up

Technical Architecture
----------------------

*   Core Engine: Unreal Engine 5.7.
*   Audio/Control: Tidal Cycles → SuperCollider → OSC (LAN) → UE5 OSC Plugin.
*   Code Display: Secondary Machine IDE (Neovim) → HDMI Capture Card → UE5 Media Player Framework.
*   AI Visuals: UE5 OSC/prompt processing → RunPod (Daydream Scope API) → WebRTC Stream → UE5 C++ YUV decode/HLSL node mapping logic.

Current Status
--------------

I am working toward an MVP implementation where UE5 ingests OSC data to actively drive a localhost Scope instance, resulting in correctly rendered video within the engine.
