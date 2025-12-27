Title: Scope V2V Integration for Unity — Hupey

URL Source: https://app.daydream.live/creators/hupey/scope-v2v-integration-for-unity

Warning: This page contains shadow DOM that are currently hidden, consider enabling shadow DOM processing.

Markdown Content:
Part of the Scope Workshop 25

My Idea
-------

The implementation of a Scope (LongLive) Unity integration that dynamically injects real-time actions and environmental data into the output context.

I previously started working on a Daydream API integration into Unity. However, since the underlying models are image models, the output is creative yet very inconsistent. Changing the backend to Scope and its more consistent video models could improve that. From there, I would then try to incorporate more interactivity and events in Unity to drive prompt generation, etc.

Your browser does not support the video tag.

My original Daydream API integration in Unity

I/O
---

*   Input: Camera stream from Unity scene + programmatically triggered prompts
*   Output: Video stream back into Unity
*   Potential add-on: Additional LLM integration

First Steps
-----------

Making LongLive work in Unity with the video-to-video model via Runscope and WebRTC streaming.
