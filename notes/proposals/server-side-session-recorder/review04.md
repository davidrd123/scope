Below is a concrete, repo-ready patch that wires the **server-side Session Recording MVP** end-to-end as specified:

* **FrameProcessor** integrates `SessionRecorder` and records:

  * ControlBus **prompt + transition** events
  * **hard cuts** via `reset_cache=True` → recorded as `initCache: true` (only when actually executed on a generated chunk)
  * **soft cuts** via `_rcp_soft_transition` → recorded as `softCut` with `restoreBias/restoreWasSet`
* **PipelineManager** gets a **non-mutating** `peek_status_info()` (so recorder start doesn’t clear errors)
* **FastAPI** adds:

  * `POST /api/v1/realtime/session-recording/start`
  * `POST /api/v1/realtime/session-recording/stop`
  * `GET  /api/v1/realtime/session-recording/status`
* Saves **v1.1 timeline JSON** to:

  * `~/.daydream-scope/recordings/session_YYYY-MM-DD_HHMMSS.timeline.json`
* Status endpoint exposes `last_timeline_path` (DoD #4)

---

## 1) `src/scope/server/pipeline_manager.py` — add `peek_status_info()`

```diff
diff --git a/src/scope/server/pipeline_manager.py b/src/scope/server/pipeline_manager.py
index 1111111..2222222 100644
--- a/src/scope/server/pipeline_manager.py
+++ b/src/scope/server/pipeline_manager.py
@@ class PipelineManager:
     def get_status_info(self) -> dict[str, Any]:
         """Get detailed status information (thread-safe).

         Note: If status is ERROR, the error message is returned once and then cleared
         to prevent persistence across page reloads.
         """
         with self._lock:
@@
             return {
                 "status": current_status.value,
                 "pipeline_id": pipeline_id,
                 "load_params": load_params,
                 "loaded_lora_adapters": loaded_lora_adapters,
                 "error": error_message,
             }

+    def peek_status_info(self) -> dict[str, Any]:
+        """Non-mutating status read (does NOT clear errors or reset state).
+
+        Used by SessionRecorder start gating; recorder shouldn't consume/clear pipeline errors.
+        """
+        with self._lock:
+            loaded_lora_adapters = None
+            if self._pipeline is not None and hasattr(self._pipeline, "loaded_lora_adapters"):
+                loaded_lora_adapters = getattr(self._pipeline, "loaded_lora_adapters", None)
+
+            load_params = self._load_params
+            if isinstance(load_params, dict):
+                load_params = load_params.copy()
+
+            return {
+                "status": self._status.value,  # e.g. "loaded"
+                "pipeline_id": self._pipeline_id,
+                "load_params": load_params,
+                "loaded_lora_adapters": loaded_lora_adapters,
+                "error": self._error_message,
+            }
```

---

## 2) `src/scope/server/frame_processor.py` — wire `SessionRecorder` + record events/cuts

### a) Add imports + init fields + helper for baseline prompt extraction

```diff
diff --git a/src/scope/server/frame_processor.py b/src/scope/server/frame_processor.py
index 3333333..4444444 100644
--- a/src/scope/server/frame_processor.py
+++ b/src/scope/server/frame_processor.py
@@
 import copy
 import logging
 import queue
 import threading
 import time
 import uuid
 from collections import deque
 from dataclasses import dataclass, field
+from datetime import datetime
 from pathlib import Path
 from typing import Any

 import torch
@@
 from scope.realtime.control_bus import ControlBus, EventType

 from .pipeline_manager import PipelineManager, PipelineNotAvailableException
+from .session_recorder import SessionRecorder

 logger = logging.getLogger(__name__)
@@ class FrameProcessor:
     def __init__(
         self,
         pipeline_manager: PipelineManager,
@@
         # Soft transition state (temporary KV cache bias adjustment)
         self._soft_transition_active: bool = False
         self._soft_transition_chunks_remaining: int = 0
         self._soft_transition_temp_bias: float | None = None
         self._soft_transition_original_bias: float | None = None
         self._soft_transition_original_bias_was_set: bool = False
+        # Session recorder needs a one-shot “record this soft cut” latch per trigger.
+        self._soft_transition_record_pending: bool = False

+        # Session recorder (server-side timeline export)
+        self.session_recorder = SessionRecorder()
+        self._last_recording_path: Path | None = None

@@
         # Input mode is signaled by the frontend at stream start.
         # This determines whether we wait for video frames or generate immediately.
         self._video_mode = (initial_parameters or {}).get("input_mode") == "video"
+
+    def _get_current_effective_prompt(self) -> tuple[str | None, float]:
+        """Best-effort extraction of the pipeline-facing prompt for baseline/edge detection.
+
+        Precedence (per proposal):
+        1) transition.target_prompts[0]
+        2) parameters["prompts"][0]
+        3) pipeline.state["prompts"][0] (fallback)
+        4) style layer compiled prompt (handles multiple shapes)
+        """
+        # 1) transition targets
+        transition = self.parameters.get("transition")
+        if isinstance(transition, dict):
+            targets = transition.get("target_prompts")
+            if isinstance(targets, list) and targets:
+                first = targets[0]
+                if isinstance(first, dict):
+                    return first.get("text"), float(first.get("weight", 1.0))
+
+        # 2) current parameters prompts
+        prompts = self.parameters.get("prompts")
+        if isinstance(prompts, list) and prompts:
+            first = prompts[0]
+            if isinstance(first, dict):
+                return first.get("text"), float(first.get("weight", 1.0))
+            if hasattr(first, "text"):
+                return getattr(first, "text", None), float(getattr(first, "weight", 1.0))
+
+        # 3) pipeline.state prompts fallback
+        try:
+            pipeline = self.pipeline_manager.get_pipeline()
+        except Exception:
+            pipeline = None
+        if pipeline is not None and hasattr(pipeline, "state"):
+            state = getattr(pipeline, "state", None)
+            state_prompts = None
+            if hasattr(state, "get"):
+                state_prompts = state.get("prompts")
+            if isinstance(state_prompts, list) and state_prompts:
+                first = state_prompts[0]
+                if isinstance(first, dict):
+                    return first.get("text"), float(first.get("weight", 1.0))
+
+        # 4) compiled prompt fallback (two shapes in repo)
+        compiled = getattr(self, "_compiled_prompt", None)
+        if compiled is not None:
+            # prompt_compiler.CompiledPrompt: .prompts is list[PromptEntry]
+            cps = getattr(compiled, "prompts", None)
+            if isinstance(cps, list) and cps:
+                first = cps[0]
+                if hasattr(first, "text"):
+                    return getattr(first, "text", None), float(getattr(first, "weight", 1.0))
+                if isinstance(first, dict):
+                    return first.get("text"), float(first.get("weight", 1.0))
+
+            # control_state.CompiledPrompt: .positive is list[dict]
+            pos = getattr(compiled, "positive", None)
+            if isinstance(pos, list) and pos:
+                first = pos[0]
+                if isinstance(first, dict):
+                    return first.get("text"), float(first.get("weight", 1.0))
+
+            # string convenience
+            prompt_str = getattr(compiled, "prompt", None)
+            if isinstance(prompt_str, str) and prompt_str.strip():
+                return prompt_str, 1.0
+
+        return None, 1.0
```

### b) In `process_chunk()`: handle start/stop reserved keys + soft-cut latch + record after pipeline call

Patch is shown against your provided `process_chunk` excerpt; the key idea is:

* **Start/stop** are consumed from `merged_updates` (not forwarded)
* **Soft cut** sets `_soft_transition_record_pending = True` when triggered; cleared when canceled/completed or recorded
* **Recording** happens only for chunks where we actually run the pipeline (after the pause early-return), and uses:

  * **last SET_PROMPT applied in this chunk** OR
  * **fallback “prompt edge” detection** (critical for “prompt changed while paused” fidelity)
  * plus hard/soft cut metadata bundled into the same recorded segment when applicable

```diff
diff --git a/src/scope/server/frame_processor.py b/src/scope/server/frame_processor.py
index 4444444..5555555 100644
--- a/src/scope/server/frame_processor.py
+++ b/src/scope/server/frame_processor.py
@@ def process_chunk(self):
         # ========================================================================
         # RESERVED KEYS: Handle snapshot/restore commands (not forwarded to pipeline)
         # ========================================================================
@@
         # Step: generate exactly one chunk even while paused.
@@
         if "_rcp_step" in merged_updates:
@@
             self._pending_steps += step_count

+        # ========================================================================
+        # RESERVED KEYS: Session recording start/stop (thread-safe via process_chunk)
+        # ========================================================================
+        if "_rcp_session_recording_start" in merged_updates:
+            merged_updates.pop("_rcp_session_recording_start", None)
+            try:
+                status = (
+                    self.pipeline_manager.peek_status_info()
+                    if hasattr(self.pipeline_manager, "peek_status_info")
+                    else self.pipeline_manager.get_status_info()
+                )
+            except Exception as e:
+                logger.warning(f"Session recording start: failed to read pipeline status: {e}")
+                status = {}
+
+            if status.get("status") != "loaded":
+                logger.warning(
+                    "Session recording start ignored: pipeline not loaded (status=%s)",
+                    status.get("status"),
+                )
+            else:
+                pipeline_id = status.get("pipeline_id")
+                lp = status.get("load_params") or {}
+                runtime_params: dict[str, Any] = dict(lp) if isinstance(lp, dict) else {"load_params": lp}
+
+                # Include current runtime params (used by export_timeline/settings)
+                if "kv_cache_attention_bias" in self.parameters:
+                    runtime_params["kv_cache_attention_bias"] = self.parameters.get("kv_cache_attention_bias")
+                if "denoising_step_list" in self.parameters:
+                    runtime_params["denoising_step_list"] = self.parameters.get("denoising_step_list")
+                if "seed" not in runtime_params:
+                    if "seed" in self.parameters:
+                        runtime_params["seed"] = self.parameters.get("seed")
+                    elif "base_seed" in self.parameters:
+                        runtime_params["seed"] = self.parameters.get("base_seed")
+
+                baseline_prompt, baseline_weight = self._get_current_effective_prompt()
+                try:
+                    self.session_recorder.start(
+                        chunk_index=self.chunk_index,
+                        pipeline_id=pipeline_id,
+                        load_params=runtime_params,
+                        baseline_prompt=baseline_prompt,
+                        baseline_weight=baseline_weight,
+                    )
+                    self._last_recording_path = None
+                    # If a soft transition is already active, record remaining window at t=0.
+                    self._soft_transition_record_pending = bool(self._soft_transition_active)
+                    logger.info("Session recording started at chunk=%d", self.chunk_index)
+                except Exception as e:
+                    logger.error(f"Session recording start failed: {e}")
+
+        if "_rcp_session_recording_stop" in merged_updates:
+            merged_updates.pop("_rcp_session_recording_stop", None)
+            try:
+                recording = self.session_recorder.stop(chunk_index=self.chunk_index)
+            except Exception as e:
+                logger.error(f"Session recording stop failed: {e}")
+                recording = None
+
+            if recording is not None:
+                ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
+                path = Path.home() / ".daydream-scope" / "recordings" / f"session_{ts}.timeline.json"
+                try:
+                    saved = self.session_recorder.save(recording, path)
+                    self._last_recording_path = saved
+                    logger.info("Session recording saved: %s", saved)
+                except Exception as e:
+                    logger.error(f"Failed to save session recording timeline: {e}")

         # Soft transition: temporarily lower KV cache bias for N chunks
         if "_rcp_soft_transition" in merged_updates:
             soft_data = merged_updates.pop("_rcp_soft_transition")
             if isinstance(soft_data, dict):
@@
                 # Apply temporary bias immediately
                 self.parameters["kv_cache_attention_bias"] = temp_bias
+                # Record soft cut ONCE on the next generated chunk (may be delayed by pause/video wait).
+                self._soft_transition_record_pending = True
                 logger.info(
                     f"Soft transition: bias -> {temp_bias} for {num_chunks} chunks "
                     f"(will restore to "
                     f"{self._soft_transition_original_bias if self._soft_transition_original_bias_was_set else '<unset>'})"
                 )

@@
         if self._soft_transition_active and "kv_cache_attention_bias" in merged_updates:
             logger.info(
                 "Soft transition canceled: explicit kv_cache_attention_bias update received"
             )
             self._soft_transition_active = False
             self._soft_transition_chunks_remaining = 0
             self._soft_transition_temp_bias = None
             self._soft_transition_original_bias = None
             self._soft_transition_original_bias_was_set = False
+            self._soft_transition_record_pending = False

@@
         # Check if paused after applying events (step overrides pause)
         if self.paused and not step_requested:
             # Sleep briefly to avoid busy waiting
             self.shutdown_event.wait(SLEEP_TIME)
             return

+        # Recorder prompt-edge detection (captures prompt changes applied while paused/video-waiting)
+        prev_recorded_prompt = self.session_recorder._last_prompt if self.session_recorder.is_recording else None  # type: ignore[attr-defined]
+        fallback_prompt: str | None = None
+        fallback_weight: float = 1.0
+        if self.session_recorder.is_recording:
+            cur_prompt, cur_weight = self._get_current_effective_prompt()
+            if cur_prompt is not None and cur_prompt != prev_recorded_prompt:
+                fallback_prompt = cur_prompt
+                fallback_weight = float(cur_weight)

         # Get the current pipeline using sync wrapper
         pipeline = self.pipeline_manager.get_pipeline()

         # prepare() will handle any required preparation based on parameters internally
         reset_cache = self.parameters.pop("reset_cache", None)
+        hard_cut_executed = bool(reset_cache)

@@
         chunk_error: Exception | None = None
         try:
@@
             output = pipeline(**call_params)
@@
             # Update FPS calculation based on processing time and frame count
             self._calculate_pipeline_fps(start_time, num_frames)
         except Exception as e:
             chunk_error = e
             if self._is_recoverable(e):
                 # Handle recoverable errors with full stack trace and continue processing
                 logger.error(f"Error processing chunk: {e}", exc_info=True)
             else:
                 raise e

+        # ====================================================================
+        # SessionRecorder: record prompt/transition + hard/soft cuts for this chunk
+        # Record AFTER pipeline call decision (so "paused" churn doesn't create phantom segments).
+        # ====================================================================
+        if self.session_recorder.is_recording:
+            wall_time = time.monotonic()
+
+            # Soft cut metadata should be recorded ONCE per trigger, at the first generated chunk.
+            soft_cut_bias = None
+            soft_cut_chunks = None
+            soft_restore_bias = None
+            soft_restore_was_set = False
+            if self._soft_transition_record_pending and self._soft_transition_active:
+                soft_cut_bias = self._soft_transition_temp_bias
+                soft_cut_chunks = self._soft_transition_chunks_remaining
+                soft_restore_bias = self._soft_transition_original_bias
+                soft_restore_was_set = bool(self._soft_transition_original_bias_was_set)
+                self._soft_transition_record_pending = False
+
+            recorded_prompt_event = False
+
+            # Prefer the last SET_PROMPT applied this boundary (authoritative for this chunk).
+            last_prompt_event = None
+            for ev in reversed(events):
+                if ev.type == EventType.SET_PROMPT:
+                    last_prompt_event = ev
+                    break
+
+            if last_prompt_event is not None:
+                payload = last_prompt_event.payload or {}
+                prompt_text = None
+                prompt_weight = 1.0
+
+                p = payload.get("prompts")
+                if isinstance(p, list) and p:
+                    first = p[0]
+                    if isinstance(first, dict):
+                        prompt_text = first.get("text")
+                        prompt_weight = float(first.get("weight", 1.0))
+
+                if prompt_text is None:
+                    tr = payload.get("transition")
+                    if isinstance(tr, dict):
+                        tps = tr.get("target_prompts")
+                        if isinstance(tps, list) and tps:
+                            first = tps[0]
+                            if isinstance(first, dict):
+                                prompt_text = first.get("text")
+                                prompt_weight = float(first.get("weight", 1.0))
+
+                transition_steps = None
+                transition_method = None
+                tr = payload.get("transition")
+                if isinstance(tr, dict):
+                    transition_steps = tr.get("num_steps")
+                    transition_method = tr.get("temporal_interpolation_method")
+
+                if prompt_text is not None:
+                    self.session_recorder.record_event(
+                        chunk_index=self.chunk_index,
+                        wall_time=wall_time,
+                        prompt=prompt_text,
+                        prompt_weight=prompt_weight,
+                        transition_steps=transition_steps,
+                        transition_method=transition_method,
+                        hard_cut=hard_cut_executed,
+                        soft_cut_bias=soft_cut_bias,
+                        soft_cut_chunks=soft_cut_chunks,
+                        soft_cut_restore_bias=soft_restore_bias,
+                        soft_cut_restore_was_set=soft_restore_was_set,
+                    )
+                    recorded_prompt_event = True
+                    hard_cut_executed = False
+                    soft_cut_bias = None
+
+            # Fallback: prompt changed since last recorded chunk (e.g. edits while paused)
+            if not recorded_prompt_event and fallback_prompt is not None:
+                self.session_recorder.record_event(
+                    chunk_index=self.chunk_index,
+                    wall_time=wall_time,
+                    prompt=fallback_prompt,
+                    prompt_weight=fallback_weight,
+                    hard_cut=hard_cut_executed,
+                    soft_cut_bias=soft_cut_bias,
+                    soft_cut_chunks=soft_cut_chunks,
+                    soft_cut_restore_bias=soft_restore_bias,
+                    soft_cut_restore_was_set=soft_restore_was_set,
+                )
+                recorded_prompt_event = True
+                hard_cut_executed = False
+                soft_cut_bias = None
+
+            # Cut-only event (no prompt change): recorder carries forward last prompt
+            if (not recorded_prompt_event) and (hard_cut_executed or soft_cut_bias is not None):
+                self.session_recorder.record_event(
+                    chunk_index=self.chunk_index,
+                    wall_time=wall_time,
+                    prompt=None,
+                    hard_cut=hard_cut_executed,
+                    soft_cut_bias=soft_cut_bias,
+                    soft_cut_chunks=soft_cut_chunks,
+                    soft_cut_restore_bias=soft_restore_bias,
+                    soft_cut_restore_was_set=soft_restore_was_set,
+                )

         self.is_prepared = True

         # Soft transition countdown and auto-restore at chunk boundary
         if self._soft_transition_active:
             self._soft_transition_chunks_remaining -= 1
             if self._soft_transition_chunks_remaining <= 0:
@@
                 self._soft_transition_active = False
                 self._soft_transition_chunks_remaining = 0
                 self._soft_transition_temp_bias = None
                 self._soft_transition_original_bias = None
                 self._soft_transition_original_bias_was_set = False
+                self._soft_transition_record_pending = False
```

That’s the minimum “wiring + semantics” needed to satisfy the MVP contract you wrote (including the crucial **paused-churn edge case**, via prompt-edge fallback).

---

## 3) `src/scope/server/app.py` — add REST endpoints

Add these models + endpoints in the existing **Realtime Control API** section (near the other `/api/v1/realtime/...` endpoints):

```diff
diff --git a/src/scope/server/app.py b/src/scope/server/app.py
index 6666666..7777777 100644
--- a/src/scope/server/app.py
+++ b/src/scope/server/app.py
@@
 class RealtimeControlResponse(BaseModel):
     """Response for control operations."""
     status: str
     chunk_index: int | None = None

+class SessionRecordingControlResponse(BaseModel):
+    """Response for session recording control operations."""
+    status: str
+
+class SessionRecordingStatusResponse(BaseModel):
+    """Thread-safe snapshot of recording status + last saved timeline path."""
+    is_recording: bool
+    duration_seconds: float = 0.0
+    start_chunk: int | None = None
+    events_count: int = 0
+    last_timeline_path: str | None = None

@@
 @app.get("/api/v1/realtime/frame/latest")
 async def get_latest_frame(
@@
         raise HTTPException(500, str(e)) from e

+@app.post(
+    "/api/v1/realtime/session-recording/start",
+    response_model=SessionRecordingControlResponse,
+)
+async def start_session_recording(
+    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
+):
+    """Start server-side session recording (timeline export)."""
+    try:
+        session = get_active_session(webrtc_manager)
+        if not apply_control_message(session, {"_rcp_session_recording_start": True}):
+            raise HTTPException(503, "Failed to apply session recording start")
+        return SessionRecordingControlResponse(status="recording_start_requested")
+    except ValueError as e:
+        raise HTTPException(400, str(e)) from e
+    except Exception as e:
+        logger.error(f"Error starting session recording: {e}")
+        raise HTTPException(500, str(e)) from e
+
+
+@app.post(
+    "/api/v1/realtime/session-recording/stop",
+    response_model=SessionRecordingControlResponse,
+)
+async def stop_session_recording(
+    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
+):
+    """Stop server-side session recording.
+
+    Non-blocking: actual save happens inside FrameProcessor.process_chunk().
+    Poll /status to observe last_timeline_path (DoD #4).
+    """
+    try:
+        session = get_active_session(webrtc_manager)
+        if not apply_control_message(session, {"_rcp_session_recording_stop": True}):
+            raise HTTPException(503, "Failed to apply session recording stop")
+        return SessionRecordingControlResponse(status="stop_requested")
+    except ValueError as e:
+        raise HTTPException(400, str(e)) from e
+    except Exception as e:
+        logger.error(f"Error stopping session recording: {e}")
+        raise HTTPException(500, str(e)) from e
+
+
+@app.get(
+    "/api/v1/realtime/session-recording/status",
+    response_model=SessionRecordingStatusResponse,
+)
+async def get_session_recording_status(
+    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
+):
+    """Get session recording status (thread-safe snapshot)."""
+    try:
+        session = get_active_session(webrtc_manager)
+        vt = session.video_track
+        if vt is None:
+            raise HTTPException(400, "No video track")
+
+        vt.initialize_output_processing()
+        fp = vt.frame_processor
+        if fp is None:
+            raise HTTPException(400, "FrameProcessor not ready")
+
+        snapshot = (
+            fp.session_recorder.get_status_snapshot()
+            if hasattr(fp, "session_recorder")
+            else {"is_recording": False}
+        )
+
+        last_path = None
+        if getattr(fp, "_last_recording_path", None) is not None:
+            last_path = str(fp._last_recording_path)
+
+        return SessionRecordingStatusResponse(
+            is_recording=bool(snapshot.get("is_recording", False)),
+            duration_seconds=float(snapshot.get("duration_seconds", 0.0) or 0.0),
+            start_chunk=snapshot.get("start_chunk"),
+            events_count=int(snapshot.get("events_count", 0) or 0),
+            last_timeline_path=last_path,
+        )
+    except ValueError as e:
+        raise HTTPException(400, str(e)) from e
+    except HTTPException:
+        raise
+    except Exception as e:
+        logger.error(f"Error getting session recording status: {e}")
+        raise HTTPException(500, str(e)) from e
```

---

## Validation steps

### A) Unit tests

```bash
pytest -q
```

### B) Manual API flow (thread-safe, end-to-end)

1. Start recording:

```bash
curl -X POST http://localhost:8000/api/v1/realtime/session-recording/start
```

2. Do some actions (prompt changes / hard-cut / soft-cut)

3. Stop recording:

```bash
curl -X POST http://localhost:8000/api/v1/realtime/session-recording/stop
```

4. Poll status until you see `last_timeline_path`:

```bash
curl http://localhost:8000/api/v1/realtime/session-recording/status
```

### C) Offline replay using chunk timebase (your requested validation)

```bash
uv run python -m scope.cli.render_timeline \
  ~/.daydream-scope/recordings/session_YYYY-MM-DD_HHMMSS.timeline.json \
  out.mp4 \
  --timebase chunk
```

(You can also add `--preset quality` etc as desired.)

---

## Notes on semantics matching your DoD

* **Hard cut** is recorded only when `reset_cache=True` is popped and the pipeline call is happening (so no “warmup init_cache” false positives).
* **Soft cut** records `restoreBias` + `restoreWasSet` directly from the FrameProcessor’s own soft transition bookkeeping, and records it **once per trigger** at the **first generated chunk** where it actually affects replay.
* **Paused churn** is handled: prompt changes made while paused don’t create phantom segments; instead, the **first generated chunk** after unpausing will record the prompt edge via `_get_current_effective_prompt()`.

If you want, I can also add a small pytest that exercises the new REST endpoints with a FakeFrameProcessor and asserts the reserved keys get enqueued (mirrors the patterns in `tests/test_realtime_rest_api.py`).
