import copy
import logging
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

try:
    from aiortc.mediastreams import VideoFrame
except ImportError:  # pragma: no cover
    VideoFrame = Any  # type: ignore[misc,assignment]

from scope.realtime.control_bus import ControlBus, EventType

from .pipeline_manager import PipelineManager, PipelineNotAvailableException

logger = logging.getLogger(__name__)


# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
DEFAULT_FPS = 30.0  # Default FPS
SLEEP_TIME = 0.01

# Input FPS measurement constants
INPUT_FPS_SAMPLE_SIZE = 30  # Number of frame intervals to track
INPUT_FPS_MIN_SAMPLES = 5  # Minimum samples needed before using input FPS

# Snapshot constants
MAX_SNAPSHOTS = 10  # Maximum number of snapshots to keep (LRU eviction)

# Continuity keys from pipeline.state that define generation continuity
CONTINUITY_KEYS = [
    "current_start_frame",
    "first_context_frame",
    "context_frame_buffer",
    "decoded_frame_buffer",
    "context_frame_buffer_max_size",
    "decoded_frame_buffer_max_size",
]


@dataclass
class Snapshot:
    """Server-side snapshot of generation state at a chunk boundary.

    Snapshots are stored in-memory and contain cloned GPU tensors.
    Clients receive only snapshot_id + metadata, not the actual tensor data.
    """

    snapshot_id: str
    chunk_index: int
    created_at: float

    # Continuity state (cloned tensors from pipeline.state)
    current_start_frame: int = 0
    first_context_frame: torch.Tensor | None = None
    context_frame_buffer: torch.Tensor | None = None
    decoded_frame_buffer: torch.Tensor | None = None
    context_frame_buffer_max_size: int = 0
    decoded_frame_buffer_max_size: int = 0

    # Control state (deep copy of parameters)
    parameters: dict[str, Any] = field(default_factory=dict)
    paused: bool = False
    video_mode: bool = False

    # Compatibility metadata (for future validation)
    pipeline_id: str | None = None
    resolution: tuple[int, int] | None = None


class _SpoutFrame:
    """Lightweight wrapper for Spout frames to match VideoFrame interface."""

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data


class FrameProcessor:
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        max_output_queue_size: int = 8,
        max_parameter_queue_size: int = 8,
        max_buffer_size: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        self.pipeline_manager = pipeline_manager

        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.frame_buffer_lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=max_output_queue_size)
        self.output_queue_lock = threading.Lock()  # Protects queue resize and flush

        # Non-destructive latest frame buffer for REST /api/frame/latest
        self.latest_frame_cpu: torch.Tensor | None = None
        self.latest_frame_lock = threading.Lock()

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=max_parameter_queue_size)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # Callback to notify when frame processor stops
        self.notification_callback = notification_callback

        # FPS tracking variables
        self.processing_time_per_frame = deque(
            maxlen=2
        )  # Keep last 2 processing_time/num_frames values for averaging
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.min_fps = MIN_FPS
        self.max_fps = MAX_FPS
        self.current_pipeline_fps = DEFAULT_FPS
        self.fps_lock = threading.Lock()  # Lock for thread-safe FPS updates

        # Input FPS tracking variables
        self.input_frame_times = deque(maxlen=INPUT_FPS_SAMPLE_SIZE)
        self.current_input_fps = DEFAULT_FPS
        self.last_input_fps_update = time.time()
        self.input_fps_lock = threading.Lock()

        self.paused = False

        # Control bus for deterministic event ordering at chunk boundaries
        self.control_bus = ControlBus()
        self.chunk_index = 0

        # Step mode: allow generating N chunks even while paused.
        # Stored on the worker thread for deterministic semantics.
        self._pending_steps = 0

        # Snapshot store (server-side, in-memory)
        # Keys are snapshot_id, values are Snapshot objects with cloned tensors
        self.snapshots: dict[str, Snapshot] = {}
        self.snapshot_order: list[str] = []  # For LRU eviction (oldest first)
        self.snapshot_response_callback: callable | None = None

        # Spout integration
        self.spout_sender = None
        self.spout_sender_enabled = False
        self.spout_sender_name = "ScopeSyphonSpoutOut"
        self._frame_spout_count = 0
        self.spout_sender_queue = queue.Queue(
            maxsize=30
        )  # Queue for async Spout sending
        self.spout_sender_thread = None

        # Spout input
        self.spout_receiver = None
        self.spout_receiver_enabled = False
        self.spout_receiver_name = ""
        self.spout_receiver_thread = None

        # Input mode is signaled by the frontend at stream start.
        # This determines whether we wait for video frames or generate immediately.
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

    def start(self):
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        # Process any Spout settings from initial parameters
        if "spout_sender" in self.parameters:
            spout_config = self.parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        if "spout_receiver" in self.parameters:
            spout_config = self.parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("FrameProcessor started")

    def stop(self, error_message: str = None):
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            # Don't join if we're calling stop() from within the worker thread
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        self.flush_output_queue()

        with self.frame_buffer_lock:
            self.frame_buffer.clear()

        # Clean up Spout sender
        self.spout_sender_enabled = False
        if self.spout_sender_thread and self.spout_sender_thread.is_alive():
            # Signal thread to stop by putting None in queue
            try:
                self.spout_sender_queue.put_nowait(None)
            except queue.Full:
                pass
            self.spout_sender_thread.join(timeout=2.0)
        if self.spout_sender is not None:
            try:
                self.spout_sender.release()
            except Exception as e:
                logger.error(f"Error releasing Spout sender: {e}")
            self.spout_sender = None

        # Clean up Spout receiver
        self.spout_receiver_enabled = False
        if self.spout_receiver is not None:
            try:
                self.spout_receiver.release()
            except Exception as e:
                logger.error(f"Error releasing Spout receiver: {e}")
            self.spout_receiver = None

        # Clear input frame times
        with self.input_fps_lock:
            self.input_frame_times.clear()

        logger.info("FrameProcessor stopped")

        # Notify callback that frame processor has stopped
        if self.notification_callback:
            try:
                message = {"type": "stream_stopped"}
                if error_message:
                    message["error_message"] = error_message
                self.notification_callback(message)
            except Exception as e:
                logger.error(f"Error in frame processor stop callback: {e}")

    def put(self, frame: VideoFrame) -> bool:
        if not self.running:
            return False

        # Track input frame timestamp for FPS measurement
        self.track_input_frame()

        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)
            return True

    def flush_output_queue(self) -> int:
        """Flush all frames from output queue.

        Thread-safe: uses output_queue_lock to prevent race with queue resize.

        Returns:
            Number of frames flushed
        """
        count = 0
        with self.output_queue_lock:
            while True:
                try:
                    self.output_queue.get_nowait()
                    count += 1
                except queue.Empty:
                    break
        return count

    def get_latest_frame(self) -> torch.Tensor | None:
        """Get the most recent frame without consuming from output queue.

        Returns a clone of the latest frame, or None if no frames produced yet.
        Thread-safe: uses latest_frame_lock.
        """
        with self.latest_frame_lock:
            if self.latest_frame_cpu is not None:
                return self.latest_frame_cpu.clone()
            return None

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        try:
            frame = self.output_queue.get_nowait()
            # Enqueue frame for async Spout sending (non-blocking)
            if self.spout_sender_enabled and self.spout_sender is not None:
                try:
                    # Frame is (H, W, C) uint8 [0, 255]
                    frame_np = frame.numpy()
                    self.spout_sender_queue.put_nowait(frame_np)
                except queue.Full:
                    # Queue full, drop frame (non-blocking)
                    logger.debug("Spout output queue full, dropping frame")
                except Exception as e:
                    logger.error(f"Error enqueueing Spout frame: {e}")

            return frame
        except queue.Empty:
            return None

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS"""
        with self.fps_lock:
            return self.current_pipeline_fps

    def get_output_fps(self) -> float:
        """Get the output FPS that frames should be sent at.

        Returns the minimum of input FPS and pipeline FPS to ensure:
        1. We don't send frames faster than they were captured (maintains temporal accuracy)
        2. We don't try to output faster than the pipeline can produce (prevents frame starvation)
        """
        input_fps = self._get_input_fps()
        pipeline_fps = self.get_current_pipeline_fps()

        if input_fps is None:
            return pipeline_fps

        # Use minimum to respect both input rate and pipeline capacity
        return min(input_fps, pipeline_fps)

    def _get_input_fps(self) -> float | None:
        """Get the current measured input FPS.

        Returns the measured input FPS if enough samples are available,
        otherwise returns None to indicate fallback should be used.
        """
        with self.input_fps_lock:
            if len(self.input_frame_times) < INPUT_FPS_MIN_SAMPLES:
                return None
            return self.current_input_fps

    def _calculate_input_fps(self):
        """Calculate and update input FPS from recent frame timestamps.

        Uses the same time-based update logic as pipeline FPS for consistency.
        Only updates if enough time has passed since the last update.
        """
        # Update FPS if enough time has passed
        current_time = time.time()
        if current_time - self.last_input_fps_update >= self.fps_update_interval:
            with self.input_fps_lock:
                if len(self.input_frame_times) >= INPUT_FPS_MIN_SAMPLES:
                    # Calculate FPS from frame intervals
                    times = list(self.input_frame_times)
                    if len(times) >= 2:
                        # Time span from first to last frame
                        time_span = times[-1] - times[0]
                        if time_span > 0:
                            # FPS = (number of intervals) / time_span
                            num_intervals = len(times) - 1
                            estimated_fps = num_intervals / time_span

                            # Clamp to reasonable bounds (same as pipeline FPS)
                            estimated_fps = max(
                                self.min_fps, min(self.max_fps, estimated_fps)
                            )
                            self.current_input_fps = estimated_fps

            self.last_input_fps_update = current_time

    def track_input_frame(self):
        """Track timestamp of an incoming frame for FPS measurement"""
        with self.input_fps_lock:
            self.input_frame_times.append(time.time())

        # Update input FPS calculation using same logic as pipeline FPS
        self._calculate_input_fps()

    def _calculate_pipeline_fps(self, start_time: float, num_frames: int):
        """Calculate FPS based on processing time and number of frames created"""
        processing_time = time.time() - start_time
        if processing_time <= 0 or num_frames <= 0:
            return

        # Store processing time per frame for averaging
        time_per_frame = processing_time / num_frames
        self.processing_time_per_frame.append(time_per_frame)

        # Update FPS if enough time has passed
        current_time = time.time()
        if current_time - self.last_fps_update >= self.fps_update_interval:
            if len(self.processing_time_per_frame) >= 1:
                # Calculate average processing time per frame
                avg_time_per_frame = sum(self.processing_time_per_frame) / len(
                    self.processing_time_per_frame
                )

                # Calculate FPS: 1 / average_time_per_frame
                # This gives us the actual frames per second output
                with self.fps_lock:
                    current_fps = self.current_pipeline_fps
                estimated_fps = (
                    1.0 / avg_time_per_frame if avg_time_per_frame > 0 else current_fps
                )

                # Clamp to reasonable bounds
                estimated_fps = max(self.min_fps, min(self.max_fps, estimated_fps))
                with self.fps_lock:
                    self.current_pipeline_fps = estimated_fps

            self.last_fps_update = current_time

    def _get_pipeline_dimensions(self) -> tuple[int, int]:
        """Get current pipeline dimensions from pipeline manager."""
        try:
            status_info = self.pipeline_manager.get_status_info()
            load_params = status_info.get("load_params") or {}
            width = load_params.get("width", 512)
            height = load_params.get("height", 512)
            return width, height
        except Exception as e:
            logger.warning(f"Could not get pipeline dimensions: {e}")
            return 512, 512

    def update_parameters(self, parameters: dict[str, Any]) -> bool:
        """Update parameters that will be used in the next pipeline call.

        Returns:
            True if the update was queued successfully, False otherwise.
        """
        # Handle Spout output settings
        if "spout_sender" in parameters:
            spout_config = parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        # Handle Spout input settings
        if "spout_receiver" in parameters:
            spout_config = parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

        # Put new parameters in queue with mailbox semantics:
        # If queue is full, drop oldest (not newest) to ensure latest control commands apply
        try:
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            # Drop oldest to make room for newest (mailbox semantics)
            try:
                self.parameters_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.parameters_queue.put_nowait(parameters)
            except queue.Full:
                logger.warning("Parameter queue still full after dropping oldest")
                return False
        return True

    def _update_spout_sender(self, config: dict):
        """Update Spout output configuration."""
        logger.info(f"Spout output config received: {config}")

        enabled = config.get("enabled", False)
        sender_name = config.get("name", "ScopeSyphonSpoutOut")

        # Get dimensions from active pipeline
        width, height = self._get_pipeline_dimensions()

        logger.info(
            f"Spout output: enabled={enabled}, name={sender_name}, size={width}x{height}"
        )

        # Lazy import SpoutSender
        try:
            from scope.server.spout import SpoutSender
        except ImportError:
            if enabled:
                logger.warning("Spout module not available on this platform")
            return

        if enabled and not self.spout_sender_enabled:
            # Enable Spout output
            try:
                self.spout_sender = SpoutSender(sender_name, width, height)
                if self.spout_sender.create():
                    self.spout_sender_enabled = True
                    self.spout_sender_name = sender_name
                    # Start background thread for async sending
                    if (
                        self.spout_sender_thread is None
                        or not self.spout_sender_thread.is_alive()
                    ):
                        self.spout_sender_thread = threading.Thread(
                            target=self._spout_sender_loop, daemon=True
                        )
                        self.spout_sender_thread.start()
                    logger.info(f"Spout output enabled: '{sender_name}'")
                else:
                    logger.error("Failed to create Spout sender")
                    self.spout_sender = None
            except Exception as e:
                logger.error(f"Error creating Spout sender: {e}")
                self.spout_sender = None

        elif not enabled and self.spout_sender_enabled:
            # Disable Spout output
            if self.spout_sender is not None:
                self.spout_sender.release()
                self.spout_sender = None
            self.spout_sender_enabled = False
            logger.info("Spout output disabled")

        elif enabled and (
            sender_name != self.spout_sender_name
            or (
                self.spout_sender
                and (
                    self.spout_sender.width != width
                    or self.spout_sender.height != height
                )
            )
        ):
            # Name or dimensions changed, recreate sender
            if self.spout_sender is not None:
                self.spout_sender.release()
            try:
                self.spout_sender = SpoutSender(sender_name, width, height)
                if self.spout_sender.create():
                    self.spout_sender_name = sender_name
                    # Ensure output thread is running
                    if (
                        self.spout_sender_thread is None
                        or not self.spout_sender_thread.is_alive()
                    ):
                        self.spout_sender_thread = threading.Thread(
                            target=self._spout_sender_loop, daemon=True
                        )
                        self.spout_sender_thread.start()
                    logger.info(
                        f"Spout output updated: '{sender_name}' ({width}x{height})"
                    )
                else:
                    logger.error("Failed to recreate Spout sender")
                    self.spout_sender = None
                    self.spout_sender_enabled = False
            except Exception as e:
                logger.error(f"Error recreating Spout sender: {e}")
                self.spout_sender = None
                self.spout_sender_enabled = False

    def _update_spout_receiver(self, config: dict):
        """Update Spout input configuration."""
        enabled = config.get("enabled", False)
        sender_name = config.get("name", "")

        # Lazy import SpoutReceiver
        try:
            from scope.server.spout import SpoutReceiver
        except ImportError:
            if enabled:
                logger.warning("Spout module not available on this platform")
            return

        if enabled and not self.spout_receiver_enabled:
            # Enable Spout input
            try:
                self.spout_receiver = SpoutReceiver(sender_name, 512, 512)
                if self.spout_receiver.create():
                    self.spout_receiver_enabled = True
                    self.spout_receiver_name = sender_name
                    # Start receiving thread
                    self.spout_receiver_thread = threading.Thread(
                        target=self._spout_receiver_loop, daemon=True
                    )
                    self.spout_receiver_thread.start()
                    logger.info(f"Spout input enabled: '{sender_name or 'any'}'")
                else:
                    logger.error("Failed to create Spout receiver")
                    self.spout_receiver = None
            except Exception as e:
                logger.error(f"Error creating Spout receiver: {e}")
                self.spout_receiver = None

        elif not enabled and self.spout_receiver_enabled:
            # Disable Spout input
            self.spout_receiver_enabled = False
            if self.spout_receiver is not None:
                self.spout_receiver.release()
                self.spout_receiver = None
            logger.info("Spout input disabled")

        elif enabled and sender_name != self.spout_receiver_name:
            # Name changed, recreate receiver
            self.spout_receiver_enabled = False
            if self.spout_receiver is not None:
                self.spout_receiver.release()
            try:
                self.spout_receiver = SpoutReceiver(sender_name, 512, 512)
                if self.spout_receiver.create():
                    self.spout_receiver_enabled = True
                    self.spout_receiver_name = sender_name
                    # Restart receiving thread if not running
                    if (
                        self.spout_receiver_thread is None
                        or not self.spout_receiver_thread.is_alive()
                    ):
                        self.spout_receiver_thread = threading.Thread(
                            target=self._spout_receiver_loop, daemon=True
                        )
                        self.spout_receiver_thread.start()
                    logger.info(f"Spout input changed to: '{sender_name or 'any'}'")
                else:
                    logger.error("Failed to recreate Spout receiver")
                    self.spout_receiver = None
            except Exception as e:
                logger.error(f"Error recreating Spout receiver: {e}")
                self.spout_receiver = None

    def _spout_sender_loop(self):
        """Background thread that sends frames to Spout asynchronously."""
        logger.info("Spout output thread started")
        frame_count = 0

        while (
            self.running and self.spout_sender_enabled and self.spout_sender is not None
        ):
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_np = self.spout_sender_queue.get(timeout=0.1)
                    # None is a sentinel value to stop the thread
                    if frame_np is None:
                        break
                except queue.Empty:
                    continue

                # Send frame to Spout
                success = self.spout_sender.send(frame_np)
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(
                        f"Spout sent frame {frame_count}, "
                        f"shape={frame_np.shape}, success={success}"
                    )
                self._frame_spout_count = frame_count

            except Exception as e:
                logger.error(f"Error in Spout output loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout output thread stopped after {frame_count} frames")

    def _spout_receiver_loop(self):
        """Background thread that receives frames from Spout and adds to buffer."""
        logger.info("Spout input thread started")

        # Initial target frame rate
        target_fps = self.get_current_pipeline_fps()
        frame_interval = 1.0 / target_fps
        last_frame_time = 0.0
        frame_count = 0

        while (
            self.running
            and self.spout_receiver_enabled
            and self.spout_receiver is not None
        ):
            try:
                # Update target FPS dynamically from pipeline performance
                current_pipeline_fps = self.get_current_pipeline_fps()
                if current_pipeline_fps > 0:
                    target_fps = current_pipeline_fps
                    frame_interval = 1.0 / target_fps

                current_time = time.time()

                # Frame rate limiting - don't receive faster than target FPS
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    time.sleep(frame_interval - time_since_last)
                    continue

                # Receive directly as RGB (avoids extra copy from RGBA slice)
                rgb_frame = self.spout_receiver.receive(as_rgb=True)
                if rgb_frame is not None:
                    last_frame_time = time.time()
                    spout_frame = _SpoutFrame(rgb_frame)

                    with self.frame_buffer_lock:
                        self.frame_buffer.append(spout_frame)

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f"Spout input received {frame_count} frames")
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"Error in Spout input loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout input thread stopped after {frame_count} frames")

    def worker_loop(self):
        logger.info("Worker thread started")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.process_chunk()

            except PipelineNotAvailableException as e:
                logger.debug(f"Pipeline temporarily unavailable: {e}")
                # Flush frame buffer to prevent buildup
                with self.frame_buffer_lock:
                    if self.frame_buffer:
                        logger.debug(
                            f"Flushing {len(self.frame_buffer)} frames due to pipeline unavailability"
                        )
                        self.frame_buffer.clear()
                continue
            except Exception as e:
                if self._is_recoverable(e):
                    logger.error(f"Error in worker loop: {e}")
                    continue
                else:
                    logger.error(
                        f"Non-recoverable error in worker loop: {e}, stopping frame processor"
                    )
                    self.stop(error_message=str(e))
                    break
        logger.info("Worker thread stopped")

    def process_chunk(self):
        start_time = time.time()

        # Legacy safety: ensure we don't persist "paused" inside self.parameters.
        # Pause state is tracked separately in self.paused and updated via events.
        paused = self.parameters.pop("paused", None)
        if paused is not None and paused != self.paused:
            self.paused = paused

        # ========================================================================
        # INGEST: Drain ALL pending queue entries (mailbox semantics)
        # ========================================================================
        # Intentional behavior change from "drain 1" to "drain all":
        # - Old: at most 1 update per chunk (10 rapid updates → 10 chunks to apply)
        # - New: all pending updates per chunk (commit at boundary)
        merged_updates: dict = {}
        while True:
            try:
                update = self.parameters_queue.get_nowait()
                # Last-write-wins merge
                merged_updates = {**merged_updates, **update}
            except queue.Empty:
                break

        # ========================================================================
        # RESERVED KEYS: Handle snapshot/restore commands (not forwarded to pipeline)
        # ========================================================================
        # These reserved keys route through parameters_queue for thread safety,
        # but are consumed here and never forwarded to the pipeline or events.
        if "_rcp_snapshot_request" in merged_updates:
            merged_updates.pop("_rcp_snapshot_request")
            try:
                snapshot = self._create_snapshot()
                # Send response via callback if registered
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {
                            "type": "snapshot_response",
                            "snapshot_id": snapshot.snapshot_id,
                            "chunk_index": snapshot.chunk_index,
                            "current_start_frame": snapshot.current_start_frame,
                        }
                    )
            except Exception as e:
                logger.error(f"Error creating snapshot: {e}")
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {"type": "snapshot_response", "error": str(e)}
                    )

        if "_rcp_restore_snapshot" in merged_updates:
            restore_data = merged_updates.pop("_rcp_restore_snapshot")
            snapshot_id = restore_data.get("snapshot_id") if restore_data else None
            if snapshot_id:
                success = self._restore_snapshot(snapshot_id)
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {
                            "type": "restore_response",
                            "snapshot_id": snapshot_id,
                            "success": success,
                        }
                    )
            else:
                logger.warning("restore_snapshot called without snapshot_id")
                if self.snapshot_response_callback:
                    self.snapshot_response_callback(
                        {
                            "type": "restore_response",
                            "error": "snapshot_id required",
                            "success": False,
                        }
                    )

        # Step: generate exactly one chunk even while paused.
        # Keep a small backlog so step isn't dropped when input frames aren't ready.
        if "_rcp_step" in merged_updates:
            step_val = merged_updates.pop("_rcp_step")
            step_count = 1
            if isinstance(step_val, int) and not isinstance(step_val, bool):
                step_count = max(1, step_val)
            self._pending_steps += step_count

        step_requested = self._pending_steps > 0

        # ========================================================================
        # TRANSLATE: Convert dict updates to typed events for ordering
        # ========================================================================
        if merged_updates:
            # Handle pause/resume via events
            if "paused" in merged_updates:
                paused_val = merged_updates.pop("paused")
                if paused_val:
                    self.control_bus.enqueue(EventType.PAUSE)
                else:
                    self.control_bus.enqueue(EventType.RESUME)

            # Handle prompts/transition via events
            if "prompts" in merged_updates or "transition" in merged_updates:
                payload = {}
                if "prompts" in merged_updates:
                    payload["prompts"] = merged_updates.pop("prompts")
                if "transition" in merged_updates:
                    payload["transition"] = merged_updates.pop("transition")
                self.control_bus.enqueue(EventType.SET_PROMPT, payload=payload)

            # Handle lora_scales via events
            if "lora_scales" in merged_updates:
                self.control_bus.enqueue(
                    EventType.SET_LORA_SCALES,
                    payload={"lora_scales": merged_updates.pop("lora_scales")},
                )

            # Handle base_seed via events
            if "base_seed" in merged_updates:
                self.control_bus.enqueue(
                    EventType.SET_SEED,
                    payload={"base_seed": merged_updates.pop("base_seed")},
                )

            # Handle denoising_step_list via events
            if "denoising_step_list" in merged_updates:
                self.control_bus.enqueue(
                    EventType.SET_DENOISE_STEPS,
                    payload={
                        "denoising_step_list": merged_updates.pop("denoising_step_list")
                    },
                )

            # Update video mode if input_mode parameter changes
            if "input_mode" in merged_updates:
                self._video_mode = merged_updates.get("input_mode") == "video"

            # Remaining keys merge directly into self.parameters (no event needed)
            if merged_updates:
                self.parameters = {**self.parameters, **merged_updates}

        # ========================================================================
        # ORDER + APPLY: Apply events in deterministic order
        # ========================================================================
        events = self.control_bus.drain_pending(
            is_paused=self.paused, chunk_index=self.chunk_index
        )

        for event in events:
            if event.type == EventType.PAUSE:
                self.paused = True
            elif event.type == EventType.RESUME:
                self.paused = False
            elif event.type == EventType.SET_PROMPT:
                # Clear stale transition when new prompts arrive without transition
                if (
                    "prompts" in event.payload
                    and "transition" not in event.payload
                    and "transition" in self.parameters
                ):
                    self.parameters.pop("transition", None)
                # Apply prompt/transition to parameters
                if "prompts" in event.payload:
                    self.parameters["prompts"] = event.payload["prompts"]
                if "transition" in event.payload:
                    self.parameters["transition"] = event.payload["transition"]
            elif event.type == EventType.SET_LORA_SCALES:
                self.parameters["lora_scales"] = event.payload["lora_scales"]
            elif event.type == EventType.SET_SEED:
                self.parameters["base_seed"] = event.payload["base_seed"]
            elif event.type == EventType.SET_DENOISE_STEPS:
                self.parameters["denoising_step_list"] = event.payload[
                    "denoising_step_list"
                ]

        # Check if paused after applying events (step overrides pause)
        if self.paused and not step_requested:
            # Sleep briefly to avoid busy waiting
            self.shutdown_event.wait(SLEEP_TIME)
            return

        # Get the current pipeline using sync wrapper
        pipeline = self.pipeline_manager.get_pipeline()

        # prepare() will handle any required preparation based on parameters internally
        reset_cache = self.parameters.pop("reset_cache", None)

        # Pop lora_scales to prevent re-processing on every frame
        lora_scales = self.parameters.pop("lora_scales", None)

        # Clear output buffer queue when reset_cache is requested to prevent old frames
        if reset_cache:
            logger.info("Clearing output buffer queue due to reset_cache request")
            self.flush_output_queue()

        requirements = None
        if hasattr(pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            if self._video_mode:
                # Signal to prepare() that video input is expected.
                # This allows resolve_input_mode() to detect video mode correctly.
                prepare_params["video"] = True  # Placeholder, actual data passed later
            requirements = pipeline.prepare(
                **prepare_params,
            )

        video_input = None
        if requirements is not None:
            current_chunk_size = requirements.input_size
            with self.frame_buffer_lock:
                if not self.frame_buffer or len(self.frame_buffer) < current_chunk_size:
                    # Sleep briefly to avoid busy waiting
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                video_input = self.prepare_chunk(current_chunk_size)
        chunk_error: Exception | None = None
        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            # Pass lora_scales only when present (one-time update)
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            # Route video input based on VACE status
            # We do not support combining normal V2V (denoising from noisy video latents) and VACE V2V editing
            if video_input is not None:
                vace_enabled = getattr(pipeline, "vace_enabled", False)
                if vace_enabled:
                    # VACE V2V editing mode: route to vace_input_frames
                    call_params["vace_input_frames"] = video_input
                else:
                    # Normal V2V mode: route to video
                    call_params["video"] = video_input

            output = pipeline(**call_params)

            # Clear vace_ref_images from parameters after use to prevent sending them on subsequent chunks
            # vace_ref_images should only be sent when explicitly provided in parameter updates
            if (
                "vace_ref_images" in call_params
                and "vace_ref_images" in self.parameters
            ):
                self.parameters.pop("vace_ref_images", None)

            # Clear transition when complete (blocks signal completion via _transition_active)
            # Contract: Modular pipelines manage prompts internally; frame_processor manages lifecycle
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = False
                if hasattr(pipeline, "state"):
                    transition_active = pipeline.state.get("_transition_active", False)

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    target_prompts = None
                    if isinstance(transition, dict):
                        target_prompts = transition.get("target_prompts")
                    elif transition is not None and hasattr(
                        transition, "target_prompts"
                    ):
                        target_prompts = getattr(transition, "target_prompts", None)

                    if target_prompts is not None:
                        self.parameters["prompts"] = target_prompts
                    self.parameters.pop("transition", None)

            processing_time = time.time() - start_time
            num_frames = output.shape[0]
            logger.debug(
                f"Processed pipeline in {processing_time:.4f}s, {num_frames} frames"
            )

            # Normalize to [0, 255] and convert to uint8
            output = (
                (output * 255.0)
                .clamp(0, 255)
                .to(dtype=torch.uint8)
                .contiguous()
                .detach()
                .cpu()
            )

            # Store latest frame for non-destructive REST reads
            with self.latest_frame_lock:
                self.latest_frame_cpu = output[-1].clone()

            # Resize output queue to meet target max size
            # Lock protects against race with flush_output_queue()
            with self.output_queue_lock:
                target_output_queue_max_size = num_frames * OUTPUT_QUEUE_MAX_SIZE_FACTOR
                if self.output_queue.maxsize < target_output_queue_max_size:
                    logger.info(
                        f"Increasing output queue size to {target_output_queue_max_size}, current size {self.output_queue.maxsize}, num_frames {num_frames}"
                    )

                    # Transfer frames from old queue to new queue
                    old_queue = self.output_queue
                    self.output_queue = queue.Queue(maxsize=target_output_queue_max_size)
                    while not old_queue.empty():
                        try:
                            frame = old_queue.get_nowait()
                            self.output_queue.put_nowait(frame)
                        except queue.Empty:
                            break

            for frame in output:
                try:
                    self.output_queue.put_nowait(frame)
                except queue.Full:
                    logger.warning("Output queue full, dropping processed frame")
                    # Update FPS calculation based on processing time and frame count
                    self._calculate_pipeline_fps(start_time, num_frames)
                    continue

            # Update FPS calculation based on processing time and frame count
            self._calculate_pipeline_fps(start_time, num_frames)
        except Exception as e:
            chunk_error = e
            if self._is_recoverable(e):
                # Handle recoverable errors with full stack trace and continue processing
                logger.error(f"Error processing chunk: {e}", exc_info=True)
            else:
                raise e

        self.is_prepared = True
        self.chunk_index += 1

        # Send step response after completing a step-driven chunk generation.
        if self._pending_steps > 0:
            self._pending_steps = max(0, self._pending_steps - 1)

        if step_requested and self.snapshot_response_callback:
            self.snapshot_response_callback(
                {
                    "type": "step_response",
                    "chunk_index": self.chunk_index,
                    "success": chunk_error is None,
                    "error": str(chunk_error) if chunk_error is not None else None,
                }
            )

    def prepare_chunk(self, chunk_size: int) -> list[torch.Tensor]:
        """
        Sample frames uniformly from the buffer, convert them to tensors, and remove processed frames.

        This function implements uniform sampling across the entire buffer to ensure
        temporal coverage of input frames. It samples frames at evenly distributed
        indices and removes all frames up to the last sampled frame to prevent
        buffer buildup.

        Note:
            This function must be called with self.frame_buffer_lock held to ensure
            thread safety. The caller is responsible for acquiring the lock.

        Example:
            With buffer_len=8 and chunk_size=4:
            - step = 8/4 = 2.0
            - indices = [0, 2, 4, 6] (uniformly distributed)
            - Returns frames at positions 0, 2, 4, 6
            - Removes frames 0-6 from buffer (7 frames total)

        Returns:
            List of tensor frames, each (1, H, W, C) for downstream preprocess_chunk
        """
        # Calculate uniform sampling step
        step = len(self.frame_buffer) / chunk_size
        # Generate indices for uniform sampling
        indices = [round(i * step) for i in range(chunk_size)]
        # Extract VideoFrames at sampled indices
        video_frames = [self.frame_buffer[i] for i in indices]

        # Drop all frames up to and including the last sampled frame
        last_idx = indices[-1]
        for _ in range(last_idx + 1):
            self.frame_buffer.popleft()

        # Convert VideoFrames to tensors
        tensor_frames = []
        for video_frame in video_frames:
            # Convert VideoFrame into (1, H, W, C) tensor on cpu
            # The T=1 dimension is expected by preprocess_chunk which rearranges T H W C -> T C H W
            tensor = (
                torch.from_numpy(video_frame.to_ndarray(format="rgb24"))
                .float()
                .unsqueeze(0)
            )
            tensor_frames.append(tensor)

        return tensor_frames

    def _create_snapshot(self) -> Snapshot:
        """Create a snapshot of current generation state.

        Captures:
        - Continuity state from pipeline.state (cloned tensors)
        - Control state (deep copy of parameters)
        - Metadata (chunk_index, timestamp, resolution)

        Returns:
            Snapshot object with unique ID
        """
        snapshot_id = str(uuid.uuid4())
        pipeline = self.pipeline_manager.get_pipeline()

        # Capture continuity state from pipeline.state
        current_start_frame = 0
        first_context_frame = None
        context_frame_buffer = None
        decoded_frame_buffer = None
        context_frame_buffer_max_size = 0
        decoded_frame_buffer_max_size = 0

        if hasattr(pipeline, "state"):
            state = pipeline.state
            current_start_frame = state.get("current_start_frame", 0)
            context_frame_buffer_max_size = state.get(
                "context_frame_buffer_max_size", 0
            )
            decoded_frame_buffer_max_size = state.get(
                "decoded_frame_buffer_max_size", 0
            )

            # Clone tensors to avoid mutation
            fcf = state.get("first_context_frame")
            if fcf is not None and isinstance(fcf, torch.Tensor):
                first_context_frame = fcf.detach().clone()

            cfb = state.get("context_frame_buffer")
            if cfb is not None and isinstance(cfb, torch.Tensor):
                context_frame_buffer = cfb.detach().clone()

            dfb = state.get("decoded_frame_buffer")
            if dfb is not None and isinstance(dfb, torch.Tensor):
                decoded_frame_buffer = dfb.detach().clone()

        # Get resolution from pipeline manager
        resolution = self._get_pipeline_dimensions()

        # Get pipeline_id if available
        pipeline_id = None
        try:
            status_info = self.pipeline_manager.get_status_info()
            pipeline_id = status_info.get("pipeline_id")
        except Exception:
            pass

        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            chunk_index=self.chunk_index,
            created_at=time.time(),
            current_start_frame=current_start_frame,
            first_context_frame=first_context_frame,
            context_frame_buffer=context_frame_buffer,
            decoded_frame_buffer=decoded_frame_buffer,
            context_frame_buffer_max_size=context_frame_buffer_max_size,
            decoded_frame_buffer_max_size=decoded_frame_buffer_max_size,
            parameters=copy.deepcopy(self.parameters),
            paused=self.paused,
            video_mode=self._video_mode,
            pipeline_id=pipeline_id,
            resolution=resolution,
        )

        # Store snapshot with LRU eviction
        self.snapshots[snapshot_id] = snapshot
        self.snapshot_order.append(snapshot_id)

        # Evict oldest if over limit
        while len(self.snapshots) > MAX_SNAPSHOTS:
            oldest_id = self.snapshot_order.pop(0)
            old_snapshot = self.snapshots.pop(oldest_id, None)
            if old_snapshot:
                # Release tensor memory explicitly
                old_snapshot.first_context_frame = None
                old_snapshot.context_frame_buffer = None
                old_snapshot.decoded_frame_buffer = None
                logger.debug(f"Evicted snapshot {oldest_id} (LRU)")

        logger.info(
            f"Created snapshot {snapshot_id} at chunk {self.chunk_index}, "
            f"total snapshots: {len(self.snapshots)}"
        )

        return snapshot

    def _restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore generation state from a snapshot.

        Restores:
        - Continuity state to pipeline.state
        - Control state to self.parameters
        - Clears output_queue to prevent stale frames
        - Sets is_prepared=True to avoid accidental cache reset

        Args:
            snapshot_id: ID of snapshot to restore

        Returns:
            True if restore succeeded, False if snapshot not found
        """
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            logger.warning(f"Snapshot {snapshot_id} not found")
            return False

        # LRU: move restored snapshot to end of order (most recently used)
        if snapshot_id in self.snapshot_order:
            self.snapshot_order.remove(snapshot_id)
            self.snapshot_order.append(snapshot_id)

        pipeline = self.pipeline_manager.get_pipeline()

        # Restore continuity state to pipeline.state
        if hasattr(pipeline, "state"):
            state = pipeline.state
            state.set("current_start_frame", snapshot.current_start_frame)
            state.set(
                "context_frame_buffer_max_size", snapshot.context_frame_buffer_max_size
            )
            state.set(
                "decoded_frame_buffer_max_size", snapshot.decoded_frame_buffer_max_size
            )

            # Restore tensors back to pipeline.state (or clear when None).
            state.set(
                "first_context_frame",
                snapshot.first_context_frame.detach().clone()
                if snapshot.first_context_frame is not None
                else None,
            )
            state.set(
                "context_frame_buffer",
                snapshot.context_frame_buffer.detach().clone()
                if snapshot.context_frame_buffer is not None
                else None,
            )
            state.set(
                "decoded_frame_buffer",
                snapshot.decoded_frame_buffer.detach().clone()
                if snapshot.decoded_frame_buffer is not None
                else None,
            )

        # Restore control state
        self.parameters = copy.deepcopy(snapshot.parameters)
        self.paused = snapshot.paused
        self._video_mode = snapshot.video_mode
        self.chunk_index = snapshot.chunk_index

        # Clear output_queue to prevent stale pre-restore frames
        self.flush_output_queue()

        # Clear frame_buffer in V2V mode to prevent stale input frames
        if self._video_mode:
            with self.frame_buffer_lock:
                self.frame_buffer.clear()

        # Set is_prepared=True to avoid accidental cache reset on next chunk
        self.is_prepared = True

        logger.info(
            f"Restored snapshot {snapshot_id} to chunk {snapshot.chunk_index}"
        )

        return True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """
        Check if an error is recoverable (i.e., processing can continue).
        Non-recoverable errors will cause the stream to stop.
        """
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        # Add more non-recoverable error types here as needed
        return True
