import argparse
import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import version
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .download_models import download_models
from .download_progress_manager import download_progress_manager
from .logs_config import (
    cleanup_old_logs,
    ensure_logs_dir,
    get_current_log_file,
    get_logs_dir,
    get_most_recent_log_file,
)
from .models_config import (
    ensure_models_dir,
    get_assets_dir,
    get_models_dir,
    models_are_downloaded,
)
from .pipeline_manager import PipelineManager
from .schema import (
    AssetFileInfo,
    AssetsResponse,
    HardwareInfoResponse,
    HealthResponse,
    IceCandidateRequest,
    IceServerConfig,
    IceServersResponse,
    PipelineLoadRequest,
    PipelineSchemasResponse,
    PipelineStatusResponse,
    WebRTCOfferRequest,
    WebRTCOfferResponse,
)
from .webrtc import WebRTCManager, apply_control_message, get_active_session


class STUNErrorFilter(logging.Filter):
    """Filter to suppress STUN/TURN connection errors that are not critical."""

    def filter(self, record):
        # Suppress STUN  exeception that occurrs always during the stream restart
        if "Task exception was never retrieved" in record.getMessage():
            return False
        return True


# Ensure logs directory exists and clean up old logs
logs_dir = ensure_logs_dir()
cleanup_old_logs(max_age_days=1)  # Delete logs older than 1 day
log_file = get_current_log_file()

# Configure logging - set root to WARNING to keep non-app libraries quiet by default
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler handles INFO
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, RotatingFileHandler
    ):
        handler.setLevel(logging.INFO)

# Add rotating file handler
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=5,  # Keep 5 backup files
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Add the filter to suppress STUN/TURN errors
stun_filter = STUNErrorFilter()
logging.getLogger("asyncio").addFilter(stun_filter)

# Set INFO level for your app modules
logging.getLogger("scope.server").setLevel(logging.INFO)
logging.getLogger("scope.core").setLevel(logging.INFO)

# Set INFO level for uvicorn
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

# Enable verbose logging for other libraries when needed
if os.getenv("VERBOSE_LOGGING"):
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("aiortc").setLevel(logging.INFO)

# Select pipeline depending on the "PIPELINE" environment variable
PIPELINE = os.getenv("PIPELINE", None)

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Git commit hash if available, otherwise a fallback message.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
            cwd=Path(__file__).parent,  # Run in the project directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown (not a git repository)"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown (git error)"
    except FileNotFoundError:
        return "unknown (git not installed)"
    except Exception:
        return "unknown"


def print_version_info():
    """Print version information and exit."""
    try:
        pkg_version = version("daydream-scope")
    except Exception:
        pkg_version = "unknown"

    git_hash = get_git_commit_hash()

    print(f"daydream-scope: {pkg_version}")
    print(f"git commit: {git_hash}")


def configure_static_files():
    """Configure static file serving for production."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount(
            "/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets"
        )
        logger.info(f"Serving static assets from {frontend_dist / 'assets'}")
    else:
        logger.info("Frontend dist directory not found - running in development mode")


# Global WebRTC manager instance
webrtc_manager = None
# Global pipeline manager instance
pipeline_manager = None


async def prewarm_pipeline(pipeline_id: str):
    """Background task to pre-warm the pipeline without blocking startup."""
    try:
        await asyncio.wait_for(
            pipeline_manager.load_pipeline(pipeline_id),
            timeout=300,  # 5 minute timeout for pipeline loading
        )
    except Exception as e:
        logger.error(f"Error pre-warming pipeline {pipeline_id} in background: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Startup
    global webrtc_manager, pipeline_manager

    # Check CUDA availability and warn if not available
    if not torch.cuda.is_available():
        warning_msg = (
            "CUDA is not available on this system. "
            "Some pipelines may not work without a CUDA-compatible GPU. "
            "The application will start, but pipeline functionality may be limited."
        )
        logger.warning(warning_msg)

    # Log logs directory
    logs_dir = get_logs_dir()
    logger.info(f"Logs directory: {logs_dir}")

    # Ensure models directory and subdirectories exist
    models_dir = ensure_models_dir()
    logger.info(f"Models directory: {models_dir}")

    # Ensure assets directory exists for VACE reference images and other media (at same level as models)
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Assets directory: {assets_dir}")

    # Initialize pipeline manager (but don't load pipeline yet)
    pipeline_manager = PipelineManager()
    logger.info("Pipeline manager initialized")

    # Pre-warm the default pipeline
    if PIPELINE is not None:
        asyncio.create_task(prewarm_pipeline(PIPELINE))

    webrtc_manager = WebRTCManager()
    logger.info("WebRTC manager initialized")

    yield

    # Shutdown
    if webrtc_manager:
        logger.info("Shutting down WebRTC manager...")
        await webrtc_manager.stop()
        logger.info("WebRTC manager shutdown complete")

    if pipeline_manager:
        logger.info("Shutting down pipeline manager...")
        pipeline_manager.unload_pipeline()
        logger.info("Pipeline manager shutdown complete")


def get_webrtc_manager() -> WebRTCManager:
    """Dependency to get WebRTC manager instance."""
    return webrtc_manager


def get_pipeline_manager() -> PipelineManager:
    """Dependency to get pipeline manager instance."""
    return pipeline_manager


app = FastAPI(
    lifespan=lifespan,
    title="Scope",
    description="A tool for running and customizing real-time, interactive generative AI pipelines and models",
    version=version("daydream-scope"),
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat())


@app.get("/")
async def root():
    """Serve the frontend at the root URL."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        return {"message": "Scope API - Frontend not built"}

    # Serve the frontend index.html with no-cache headers
    # This ensures clients like Electron alway fetch the latest HTML (which references hashed assets)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return {"message": "Scope API - Frontend index.html not found"}


@app.post("/api/v1/pipeline/load")
async def load_pipeline(
    request: PipelineLoadRequest,
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Load a pipeline."""
    try:
        # Convert pydantic model to dict for pipeline manager
        load_params_dict = None
        if request.load_params:
            load_params_dict = request.load_params.model_dump()

        # Start loading in background without blocking
        asyncio.create_task(
            pipeline_manager.load_pipeline(request.pipeline_id, load_params_dict)
        )
        return {"message": "Pipeline loading initiated successfully"}
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Get current pipeline status."""
    try:
        status_info = await pipeline_manager.get_status_info_async()
        return PipelineStatusResponse(**status_info)
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/pipelines/schemas", response_model=PipelineSchemasResponse)
async def get_pipeline_schemas():
    """Get configuration schemas and defaults for all available pipelines.

    Returns the output of each pipeline's get_schema_with_metadata() method,
    which includes:
    - Pipeline metadata (id, name, description, version)
    - supported_modes: List of supported input modes ("text", "video")
    - default_mode: Default input mode for this pipeline
    - mode_defaults: Mode-specific default overrides (if any)
    - config_schema: Full JSON schema with defaults

    The frontend should use this as the source of truth for parameter defaults.
    """
    from scope.core.pipelines.schema import PIPELINE_CONFIGS

    pipelines: dict = {}

    for pipeline_id, config_class in PIPELINE_CONFIGS.items():
        # get_schema_with_metadata() now includes supported_modes, default_mode,
        # and mode_defaults directly from the config class
        schema_data = config_class.get_schema_with_metadata()
        pipelines[pipeline_id] = schema_data

    return PipelineSchemasResponse(pipelines=pipelines)


@app.get("/api/v1/webrtc/ice-servers", response_model=IceServersResponse)
async def get_ice_servers(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Return ICE server configuration for frontend WebRTC connection."""
    ice_servers = []

    for server in webrtc_manager.rtc_config.iceServers:
        ice_servers.append(
            IceServerConfig(
                urls=server.urls,
                username=server.username if hasattr(server, "username") else None,
                credential=server.credential if hasattr(server, "credential") else None,
            )
        )

    return IceServersResponse(iceServers=ice_servers)


@app.post("/api/v1/webrtc/offer", response_model=WebRTCOfferResponse)
async def handle_webrtc_offer(
    request: WebRTCOfferRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
    pipeline_manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Handle WebRTC offer and return answer."""
    try:
        # Ensure pipeline is loaded before proceeding
        status_info = await pipeline_manager.get_status_info_async()
        if status_info["status"] != "loaded":
            raise HTTPException(
                status_code=400,
                detail="Pipeline not loaded. Please load pipeline first.",
            )

        return await webrtc_manager.handle_offer(request, pipeline_manager)

    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.patch(
    "/api/v1/webrtc/offer/{session_id}", status_code=204, response_class=Response
)
async def add_ice_candidate(
    session_id: str,
    candidate_request: IceCandidateRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Add ICE candidate(s) to an existing WebRTC session (Trickle ICE).

    This endpoint follows the Trickle ICE pattern, allowing clients to send
    ICE candidates as they are discovered.
    """
    # TODO: Validate that the Content-Type is 'application/trickle-ice-sdpfrag'
    # At the moment FastAPI defaults to validating that it is 'application/json'
    try:
        for candidate_init in candidate_request.candidates:
            await webrtc_manager.add_ice_candidate(
                session_id=session_id,
                candidate=candidate_init.candidate,
                sdp_mid=candidate_init.sdpMid,
                sdp_mline_index=candidate_init.sdpMLineIndex,
            )

            logger.debug(
                f"Added {len(candidate_request.candidates)} ICE candidates to session {session_id}"
            )

        # Return 204 No Content on success
        return Response(status_code=204)

    except ValueError as e:
        # Session not found or invalid candidate
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error adding ICE candidate to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ModelStatusResponse(BaseModel):
    downloaded: bool


class DownloadModelsRequest(BaseModel):
    pipeline_id: str


class LoRAFileInfo(BaseModel):
    """Metadata for an available LoRA file on disk."""

    name: str
    path: str
    size_mb: float
    folder: str | None = None


class LoRAFilesResponse(BaseModel):
    """Response containing all discoverable LoRA files."""

    lora_files: list[LoRAFileInfo]


@app.get("/api/v1/lora/list", response_model=LoRAFilesResponse)
async def list_lora_files():
    """List available LoRA files in the models/lora directory and its subdirectories."""

    def process_lora_file(file_path: Path, lora_dir: Path) -> LoRAFileInfo:
        """Extract LoRA file metadata."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        relative_path = file_path.relative_to(lora_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )
        return LoRAFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
        )

    try:
        lora_dir = get_models_dir() / "lora"
        lora_files: list[LoRAFileInfo] = []

        if lora_dir.exists() and lora_dir.is_dir():
            for pattern in ("*.safetensors", "*.bin", "*.pt"):
                for file_path in lora_dir.rglob(pattern):
                    if file_path.is_file():
                        lora_files.append(process_lora_file(file_path, lora_dir))

        lora_files.sort(key=lambda x: (x.folder or "", x.name))
        return LoRAFilesResponse(lora_files=lora_files)

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"list_lora_files: Error listing LoRA files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/assets", response_model=AssetsResponse)
async def list_assets(
    type: str | None = Query(None, description="Filter by asset type (image, video)"),
):
    """List available asset files in the assets directory and its subdirectories."""

    def process_asset_file(
        file_path: Path, assets_dir: Path, asset_type: str
    ) -> AssetFileInfo:
        """Extract asset file metadata."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        created_at = file_path.stat().st_ctime
        relative_path = file_path.relative_to(assets_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )
        return AssetFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
            type=asset_type,
            created_at=created_at,
        )

    try:
        assets_dir = get_assets_dir()
        asset_files: list[AssetFileInfo] = []

        if assets_dir.exists() and assets_dir.is_dir():
            # Define patterns based on type filter
            if type == "image" or type is None:
                image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
                for pattern in image_patterns:
                    for file_path in assets_dir.rglob(pattern):
                        if file_path.is_file():
                            asset_files.append(
                                process_asset_file(file_path, assets_dir, "image")
                            )

            if type == "video" or type is None:
                video_patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
                for pattern in video_patterns:
                    for file_path in assets_dir.rglob(pattern):
                        if file_path.is_file():
                            asset_files.append(
                                process_asset_file(file_path, assets_dir, "video")
                            )

        # Sort by created_at (most recent first), then by folder and name
        asset_files.sort(key=lambda x: (-x.created_at, x.folder or "", x.name))
        return AssetsResponse(assets=asset_files)

    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"list_assets: Error listing asset files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/assets", response_model=AssetFileInfo)
async def upload_asset(request: Request, filename: str = Query(...)):
    """Upload an asset file (image or video) to the assets directory."""
    try:
        # Validate file type - support both images and videos
        allowed_image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        allowed_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        allowed_extensions = allowed_image_extensions | allowed_video_extensions

        file_extension = Path(filename).suffix.lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}",
            )

        # Determine asset type
        if file_extension in allowed_image_extensions:
            asset_type = "image"
        else:
            asset_type = "video"

        # Ensure assets directory exists
        assets_dir = get_assets_dir()
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Read file content from request body
        content = await request.body()

        # Validate file size (50MB limit)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {max_size / (1024 * 1024):.0f}MB",
            )

        # Save file to assets directory
        file_path = assets_dir / filename
        file_path.write_bytes(content)

        # Return file info matching AssetFileInfo structure
        size_mb = len(content) / (1024 * 1024)
        created_at = file_path.stat().st_ctime
        relative_path = file_path.relative_to(assets_dir)
        folder = (
            str(relative_path.parent) if relative_path.parent != Path(".") else None
        )

        logger.info(f"upload_asset: Uploaded {asset_type} file: {file_path}")
        return AssetFileInfo(
            name=file_path.stem,
            path=str(file_path),
            size_mb=round(size_mb, 2),
            folder=folder,
            type=asset_type,
            created_at=created_at,
        )

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"upload_asset: Error uploading asset file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/assets/{asset_path:path}")
async def serve_asset(asset_path: str):
    """Serve an asset file (for thumbnails/previews)."""
    try:
        assets_dir = get_assets_dir()
        file_path = assets_dir / asset_path

        # Security check: ensure the path is within assets directory
        try:
            file_path = file_path.resolve()
            assets_dir_resolved = assets_dir.resolve()
            if not str(file_path).startswith(str(assets_dir_resolved)):
                raise HTTPException(status_code=403, detail="Access denied")
        except Exception:
            raise HTTPException(status_code=403, detail="Invalid path") from None

        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Asset not found")

        # Determine media type based on extension
        file_extension = file_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
        }
        media_type = media_types.get(file_extension, "application/octet-stream")

        return FileResponse(file_path, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.error(f"serve_asset: Error serving asset file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/models/status")
async def get_model_status(pipeline_id: str):
    """Check if models for a pipeline are downloaded and get download progress."""
    try:
        progress = download_progress_manager.get_progress(pipeline_id)

        # If download is in progress, always report as not downloaded
        if progress and progress.get("is_downloading"):
            return {"downloaded": False, "progress": progress}

        # Check if files actually exist
        downloaded = models_are_downloaded(pipeline_id)

        # Clean up progress if download is complete
        if downloaded and progress:
            download_progress_manager.clear_progress(pipeline_id)
            progress = None

        return {"downloaded": downloaded, "progress": progress}
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/models/download")
async def download_pipeline_models(request: DownloadModelsRequest):
    """Download models for a specific pipeline."""
    try:
        if not request.pipeline_id:
            raise HTTPException(status_code=400, detail="pipeline_id is required")

        pipeline_id = request.pipeline_id

        # Check if download already in progress
        existing_progress = download_progress_manager.get_progress(pipeline_id)
        if existing_progress and existing_progress.get("is_downloading"):
            raise HTTPException(
                status_code=409,
                detail=f"Download already in progress for {pipeline_id}",
            )

        # Download in a background thread to avoid blocking
        import threading

        def download_in_background():
            """Run download in background thread."""
            try:
                download_models(pipeline_id)
                download_progress_manager.mark_complete(pipeline_id)
            except Exception as e:
                logger.error(f"Error downloading models for {pipeline_id}: {e}")
                download_progress_manager.clear_progress(pipeline_id)

        thread = threading.Thread(target=download_in_background)
        thread.daemon = True
        thread.start()

        return {"message": f"Model download started for {pipeline_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model download: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def is_spout_available() -> bool:
    """Check if Spout is available (native Windows only, not WSL)."""
    # Spout requires native Windows - it won't work in WSL/Linux
    return sys.platform == "win32"


@app.get("/api/v1/hardware/info", response_model=HardwareInfoResponse)
async def get_hardware_info():
    """Get hardware information including available VRAM and Spout availability."""
    try:
        vram_gb = None

        if torch.cuda.is_available():
            # Get total VRAM from the first GPU (in bytes), convert to GB
            _, total_mem = torch.cuda.mem_get_info(0)
            vram_gb = total_mem / (1024**3)

        return HardwareInfoResponse(
            vram_gb=vram_gb,
            spout_available=is_spout_available(),
        )
    except Exception as e:
        logger.error(f"Error getting hardware info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Realtime Control API - REST endpoints for CLI/agent control
# =============================================================================


class RealtimeStateResponse(BaseModel):
    """Current state of the realtime video generation."""

    paused: bool
    chunk_index: int
    prompt: str | None = None
    session_id: str

    # Style layer state
    world_state: dict | None = None
    active_style: str | None = None
    compiled_prompt: str | None = None


class RealtimeControlResponse(BaseModel):
    """Response for control operations."""

    status: str
    chunk_index: int | None = None


class PromptRequest(BaseModel):
    """Request to set prompt."""

    prompt: str


class HardCutRequest(BaseModel):
    """Request to perform a hard cut (cache reset) with optional new prompt."""

    prompt: str | None = None  # Optional prompt to apply after cache reset


class WorldStateRequest(BaseModel):
    """Request to set world state (full replace)."""

    world_state: dict


class SetStyleRequest(BaseModel):
    """Request to set active style."""

    name: str


class StyleInfo(BaseModel):
    """Summary info about a style."""

    name: str
    description: str
    lora_path: str | None = None
    trigger_words: list[str] = []


class StyleListResponse(BaseModel):
    """List of available styles."""

    styles: list[StyleInfo]
    active_style: str | None = None


class WorldChangeRequest(BaseModel):
    """Request to change world state via natural language instruction."""

    instruction: str


class WorldChangeResponse(BaseModel):
    """Response from world state change operation."""

    status: str
    world_state: dict | None = None
    compiled_prompt: str | None = None
    chunk_index: int | None = None


class PromptJiggleRequest(BaseModel):
    """Request to generate a prompt variation."""

    prompt: str | None = None  # None = use current compiled prompt
    intensity: float = 0.3  # 0-1, how different the variation should be


class PromptJiggleResponse(BaseModel):
    """Response from prompt jiggle operation."""

    status: str
    original_prompt: str | None = None
    jiggled_prompt: str | None = None


# =============================================================================
# Prompt Playlist Schemas
# =============================================================================


class PlaylistLoadRequest(BaseModel):
    """Request to load a prompt playlist from a file."""

    file_path: str
    old_trigger: str | None = None  # Trigger phrase to replace
    new_trigger: str | None = None  # Replacement trigger phrase


class PlaylistGotoRequest(BaseModel):
    """Request to go to a specific playlist index."""

    index: int


class PlaylistResponse(BaseModel):
    """Response with current playlist state."""

    status: str
    source_file: str | None = None
    current_index: int = 0
    total: int = 0
    current_prompt: str | None = None
    has_next: bool = False
    has_prev: bool = False
    trigger_swap: list[str] | None = None
    prompt_applied: bool = False


class PlaylistPreviewResponse(BaseModel):
    """Response with playlist preview window."""

    prompts: list[dict]
    current_index: int
    total: int


# Global playlist instance (per-server, not per-session for simplicity)
_prompt_playlist: "PromptPlaylist | None" = None


@app.get("/api/v1/realtime/state", response_model=RealtimeStateResponse)
async def get_realtime_state(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get current realtime generation state."""
    try:
        session = get_active_session(webrtc_manager)
        vt = session.video_track
        if vt is None:
            raise HTTPException(400, "No video track")

        vt.initialize_output_processing()
        fp = vt.frame_processor
        if fp is None:
            raise HTTPException(400, "FrameProcessor not ready")

        # Extract prompt from parameters
        prompts = fp.parameters.get("prompts", [])
        prompt_text = None
        if prompts and len(prompts) > 0:
            first_prompt = prompts[0]
            if isinstance(first_prompt, dict):
                prompt_text = first_prompt.get("text")
            elif hasattr(first_prompt, "text"):
                prompt_text = first_prompt.text

        return RealtimeStateResponse(
            paused=fp.paused,
            chunk_index=fp.chunk_index,
            prompt=prompt_text,
            session_id=session.id,
            # Style layer state
            world_state=fp.world_state.model_dump() if fp.world_state else None,
            active_style=fp.style_manifest.name if fp.style_manifest else None,
            compiled_prompt=(
                fp._compiled_prompt.prompt if fp._compiled_prompt else None
            ),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f"Error getting realtime state: {e}")
        raise HTTPException(500, str(e)) from e


@app.post("/api/v1/realtime/pause", response_model=RealtimeControlResponse)
async def pause_realtime(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Pause realtime generation."""
    try:
        session = get_active_session(webrtc_manager)
        if not apply_control_message(session, {"paused": True}):
            raise HTTPException(503, "Failed to apply pause control message")
        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status="paused",
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f"Error pausing realtime: {e}")
        raise HTTPException(500, str(e)) from e


@app.post("/api/v1/realtime/run", response_model=RealtimeControlResponse)
async def run_realtime(
    chunks: int | None = None,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Resume realtime generation, optionally for N chunks."""
    try:
        session = get_active_session(webrtc_manager)
        if chunks is not None and chunks < 0:
            raise HTTPException(400, "chunks must be >= 0")
        if chunks is not None and chunks > 0:
            # Generate N chunks while staying paused
            if not apply_control_message(session, {"paused": True, "_rcp_step": chunks}):
                raise HTTPException(503, "Failed to apply run control message")
            status = f"stepping_{chunks}"
        else:
            # Resume continuous generation
            if not apply_control_message(session, {"paused": False}):
                raise HTTPException(503, "Failed to apply run control message")
            status = "running"

        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status=status,
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running realtime: {e}")
        raise HTTPException(500, str(e)) from e


@app.post("/api/v1/realtime/step", response_model=RealtimeControlResponse)
async def step_realtime(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Generate one chunk while paused."""
    try:
        session = get_active_session(webrtc_manager)
        # Ensure paused + trigger step
        if not apply_control_message(session, {"paused": True, "_rcp_step": 1}):
            raise HTTPException(503, "Failed to apply step control message")
        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status="step_queued",
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f"Error stepping realtime: {e}")
        raise HTTPException(500, str(e)) from e


@app.post("/api/v1/realtime/hard-cut", response_model=RealtimeControlResponse)
async def hard_cut(
    request: HardCutRequest | None = None,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Perform a hard cut (reset KV cache) with optional new prompt.

    A hard cut resets the generation context, allowing a clean scene transition
    instead of morphing from the current frame. Use this for:
    - Scene changes in sequences/playlists
    - Breaking out of error accumulation
    - Starting fresh with a new prompt
    """
    try:
        session = get_active_session(webrtc_manager)

        # Build control message with reset_cache
        msg: dict = {"reset_cache": True}

        # Optionally include new prompt
        if request and request.prompt:
            msg["prompts"] = [{"text": request.prompt, "weight": 1.0}]

        if not apply_control_message(session, msg):
            raise HTTPException(503, "Failed to apply hard cut")

        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status="hard_cut_applied",
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f"Error performing hard cut: {e}")
        raise HTTPException(500, str(e)) from e


@app.put("/api/v1/realtime/prompt", response_model=RealtimeControlResponse)
async def set_realtime_prompt(
    request: PromptRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Set the generation prompt."""
    try:
        session = get_active_session(webrtc_manager)
        if not apply_control_message(
            session, {"prompts": [{"text": request.prompt, "weight": 1.0}]}
        ):
            raise HTTPException(503, "Failed to apply prompt control message")
        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status="prompt_set",
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error(f"Error setting prompt: {e}")
        raise HTTPException(500, str(e)) from e


@app.get("/api/v1/realtime/frame/latest")
async def get_latest_frame(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Get the latest generated frame as PNG image."""
    try:
        session = get_active_session(webrtc_manager)
        vt = session.video_track
        if vt is None:
            raise HTTPException(400, "No video track")

        vt.initialize_output_processing()
        fp = vt.frame_processor
        if fp is None:
            raise HTTPException(400, "FrameProcessor not ready")

        frame = fp.get_latest_frame()
        if frame is None:
            raise HTTPException(404, "No frames generated yet")

        # Encode to PNG without consuming from output_queue.
        # Frame is (H, W, C) uint8 on CPU.
        from torchvision.io import encode_png

        frame_chw = frame.permute(2, 0, 1).contiguous()
        png_bytes = encode_png(frame_chw).numpy().tobytes()

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=frame.png"},
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest frame: {e}")
        raise HTTPException(500, str(e)) from e


# =============================================================================
# Style Layer API - WorldState, Style, and Prompt Compilation
# =============================================================================


@app.put("/api/v1/realtime/world", response_model=RealtimeControlResponse)
async def set_world_state(
    request: WorldStateRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Replace WorldState (full replace, not patch)."""
    try:
        session = get_active_session(webrtc_manager)
        if not apply_control_message(session, {"_rcp_world_state": request.world_state}):
            raise HTTPException(503, "Failed to apply world state")
        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status="world_state_updated",
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting world state: {e}")
        raise HTTPException(500, str(e)) from e


@app.put("/api/v1/realtime/style", response_model=RealtimeControlResponse)
async def set_active_style(
    request: SetStyleRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Set active style by name."""
    try:
        session = get_active_session(webrtc_manager)
        if not apply_control_message(session, {"_rcp_set_style": request.name}):
            raise HTTPException(503, "Failed to apply style change")
        fp = session.video_track.frame_processor
        return RealtimeControlResponse(
            status="style_set",
            chunk_index=fp.chunk_index if fp else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting style: {e}")
        raise HTTPException(500, str(e)) from e


@app.get("/api/v1/realtime/style/list", response_model=StyleListResponse)
async def list_styles(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """List available styles from the registry."""
    try:
        session = get_active_session(webrtc_manager)
        vt = session.video_track
        if vt is None:
            raise HTTPException(400, "No video track")

        vt.initialize_output_processing()
        fp = vt.frame_processor
        if fp is None:
            raise HTTPException(400, "FrameProcessor not ready")

        styles = []
        for style_name in fp.style_registry.list_styles():
            manifest = fp.style_registry.get(style_name)
            if manifest:
                styles.append(
                    StyleInfo(
                        name=manifest.name,
                        description=manifest.description,
                        lora_path=manifest.lora_path,
                        trigger_words=manifest.trigger_words,
                    )
                )

        return StyleListResponse(
            styles=styles,
            active_style=fp.style_manifest.name if fp.style_manifest else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing styles: {e}")
        raise HTTPException(500, str(e)) from e


@app.post("/api/v1/realtime/world/change", response_model=WorldChangeResponse)
async def change_world_state(
    request: WorldChangeRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Change WorldState using natural language instruction.

    Uses Gemini to interpret the instruction and apply changes to WorldState.
    Requires GEMINI_API_KEY environment variable.

    Example: {"instruction": "make Rooster angry and have him storm out"}
    """
    try:
        from scope.realtime.gemini_client import GeminiWorldChanger, is_gemini_available

        if not is_gemini_available():
            raise HTTPException(
                503,
                "Gemini not available - set GEMINI_API_KEY environment variable",
            )

        session = get_active_session(webrtc_manager)
        vt = session.video_track
        if vt is None:
            raise HTTPException(400, "No video track")

        vt.initialize_output_processing()
        fp = vt.frame_processor
        if fp is None:
            raise HTTPException(400, "FrameProcessor not ready")

        # Get current WorldState
        current_world = fp.world_state

        # Apply change via Gemini (sync call, should be fast for Flash)
        changer = GeminiWorldChanger()
        new_world = changer.change(current_world, request.instruction)

        # Apply to frame processor via control message
        if not apply_control_message(session, {"_rcp_world_state": new_world.model_dump()}):
            raise HTTPException(503, "Failed to apply world state change")

        # Get compiled prompt from frame processor (will be updated on next chunk)
        compiled_prompt = None
        if fp._compiled_prompt:
            compiled_prompt = fp._compiled_prompt.prompt

        return WorldChangeResponse(
            status="changed",
            world_state=new_world.model_dump(),
            compiled_prompt=compiled_prompt,
            chunk_index=fp.chunk_index,
        )

    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing world state: {e}")
        raise HTTPException(500, str(e)) from e


@app.post("/api/v1/prompt/jiggle", response_model=PromptJiggleResponse)
async def jiggle_prompt(
    request: PromptJiggleRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Generate a variation of the current or provided prompt.

    Uses Gemini to create a semantically similar but textually different prompt.
    Useful for adding variety without changing scene intent.

    Requires GEMINI_API_KEY environment variable.
    """
    try:
        from scope.realtime.gemini_client import GeminiPromptJiggler, is_gemini_available

        session = get_active_session(webrtc_manager)
        vt = session.video_track
        if vt is None:
            raise HTTPException(400, "No video track")

        vt.initialize_output_processing()
        fp = vt.frame_processor
        if fp is None:
            raise HTTPException(400, "FrameProcessor not ready")

        # Get prompt to jiggle
        original = request.prompt
        if original is None:
            # Use current compiled prompt
            if fp._compiled_prompt:
                original = fp._compiled_prompt.prompt
            else:
                raise HTTPException(400, "No prompt available to jiggle")

        # If Gemini not available, return original
        if not is_gemini_available():
            return PromptJiggleResponse(
                status="unchanged",
                original_prompt=original,
                jiggled_prompt=original,
            )

        # Jiggle via Gemini
        jiggler = GeminiPromptJiggler()
        jiggled = jiggler.jiggle(original, intensity=request.intensity)

        return PromptJiggleResponse(
            status="jiggled",
            original_prompt=original,
            jiggled_prompt=jiggled,
        )

    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error jiggling prompt: {e}")
        raise HTTPException(500, str(e)) from e


# =============================================================================
# Prompt Playlist API
# =============================================================================


def _apply_playlist_prompt(
    webrtc_manager: WebRTCManager, prompt: str
) -> bool:
    """Apply the current playlist prompt to the session."""
    try:
        session = get_active_session(webrtc_manager)
        return apply_control_message(
            session, {"prompts": [{"text": prompt, "weight": 1.0}]}
        )
    except Exception:
        return False


@app.post("/api/v1/realtime/playlist/load", response_model=PlaylistResponse)
async def load_playlist(
    request: PlaylistLoadRequest,
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Load a prompt playlist from a caption file.

    Optionally swap trigger phrases (e.g., "1988 Cel Animation" -> "Rankin/Bass Animagic Stop-Motion").
    """
    global _prompt_playlist
    try:
        from scope.realtime.prompt_playlist import PromptPlaylist

        trigger_swap = None
        if request.old_trigger and request.new_trigger:
            trigger_swap = (request.old_trigger, request.new_trigger)

        _prompt_playlist = PromptPlaylist.from_file(
            request.file_path,
            trigger_swap=trigger_swap,
        )

        # Apply first prompt
        applied = _apply_playlist_prompt(webrtc_manager, _prompt_playlist.current)

        return PlaylistResponse(
            status="loaded",
            source_file=_prompt_playlist.source_file,
            current_index=_prompt_playlist.current_index,
            total=_prompt_playlist.total,
            current_prompt=_prompt_playlist.current,
            has_next=_prompt_playlist.has_next,
            has_prev=_prompt_playlist.has_prev,
            trigger_swap=list(trigger_swap) if trigger_swap else None,
            prompt_applied=applied,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        logger.error(f"Error loading playlist: {e}")
        raise HTTPException(500, str(e)) from e


@app.get("/api/v1/realtime/playlist", response_model=PlaylistResponse)
async def get_playlist():
    """Get current playlist state."""
    global _prompt_playlist
    if _prompt_playlist is None:
        return PlaylistResponse(status="no_playlist")

    return PlaylistResponse(
        status="ok",
        source_file=_prompt_playlist.source_file,
        current_index=_prompt_playlist.current_index,
        total=_prompt_playlist.total,
        current_prompt=_prompt_playlist.current,
        has_next=_prompt_playlist.has_next,
        has_prev=_prompt_playlist.has_prev,
        trigger_swap=list(_prompt_playlist.trigger_swap) if _prompt_playlist.trigger_swap else None,
        prompt_applied=False,
    )


@app.get("/api/v1/realtime/playlist/preview", response_model=PlaylistPreviewResponse)
async def preview_playlist(context: int = Query(default=3, ge=1, le=10)):
    """Get a preview window around current playlist position."""
    global _prompt_playlist
    if _prompt_playlist is None:
        return PlaylistPreviewResponse(prompts=[], current_index=0, total=0)

    preview = _prompt_playlist.preview(context=context)
    return PlaylistPreviewResponse(**preview)


@app.post("/api/v1/realtime/playlist/next", response_model=PlaylistResponse)
async def playlist_next(
    apply: bool = Query(default=True, description="Apply prompt after navigating"),
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Move to the next prompt in the playlist, optionally applying it."""
    global _prompt_playlist
    if _prompt_playlist is None:
        raise HTTPException(400, "No playlist loaded")

    prompt = _prompt_playlist.next()
    applied = _apply_playlist_prompt(webrtc_manager, prompt) if apply else False

    return PlaylistResponse(
        status="next",
        source_file=_prompt_playlist.source_file,
        current_index=_prompt_playlist.current_index,
        total=_prompt_playlist.total,
        current_prompt=prompt,
        has_next=_prompt_playlist.has_next,
        has_prev=_prompt_playlist.has_prev,
        trigger_swap=list(_prompt_playlist.trigger_swap) if _prompt_playlist.trigger_swap else None,
        prompt_applied=applied,
    )


@app.post("/api/v1/realtime/playlist/prev", response_model=PlaylistResponse)
async def playlist_prev(
    apply: bool = Query(default=True, description="Apply prompt after navigating"),
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Move to the previous prompt in the playlist, optionally applying it."""
    global _prompt_playlist
    if _prompt_playlist is None:
        raise HTTPException(400, "No playlist loaded")

    prompt = _prompt_playlist.prev()
    applied = _apply_playlist_prompt(webrtc_manager, prompt) if apply else False

    return PlaylistResponse(
        status="prev",
        source_file=_prompt_playlist.source_file,
        current_index=_prompt_playlist.current_index,
        total=_prompt_playlist.total,
        current_prompt=prompt,
        has_next=_prompt_playlist.has_next,
        has_prev=_prompt_playlist.has_prev,
        trigger_swap=list(_prompt_playlist.trigger_swap) if _prompt_playlist.trigger_swap else None,
        prompt_applied=applied,
    )


@app.post("/api/v1/realtime/playlist/goto", response_model=PlaylistResponse)
async def playlist_goto(
    request: PlaylistGotoRequest,
    apply: bool = Query(default=True, description="Apply prompt after navigating"),
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Go to a specific index in the playlist, optionally applying it."""
    global _prompt_playlist
    if _prompt_playlist is None:
        raise HTTPException(400, "No playlist loaded")

    prompt = _prompt_playlist.goto(request.index)
    applied = _apply_playlist_prompt(webrtc_manager, prompt) if apply else False

    return PlaylistResponse(
        status="goto",
        source_file=_prompt_playlist.source_file,
        current_index=_prompt_playlist.current_index,
        total=_prompt_playlist.total,
        current_prompt=prompt,
        has_next=_prompt_playlist.has_next,
        has_prev=_prompt_playlist.has_prev,
        trigger_swap=list(_prompt_playlist.trigger_swap) if _prompt_playlist.trigger_swap else None,
        prompt_applied=applied,
    )


@app.post("/api/v1/realtime/playlist/apply", response_model=PlaylistResponse)
async def playlist_apply(
    webrtc_manager: WebRTCManager = Depends(get_webrtc_manager),
):
    """Re-apply the current playlist prompt without changing position."""
    global _prompt_playlist
    if _prompt_playlist is None:
        raise HTTPException(400, "No playlist loaded")

    applied = _apply_playlist_prompt(webrtc_manager, _prompt_playlist.current)

    return PlaylistResponse(
        status="applied",
        source_file=_prompt_playlist.source_file,
        current_index=_prompt_playlist.current_index,
        total=_prompt_playlist.total,
        current_prompt=_prompt_playlist.current,
        has_next=_prompt_playlist.has_next,
        has_prev=_prompt_playlist.has_prev,
        trigger_swap=list(_prompt_playlist.trigger_swap) if _prompt_playlist.trigger_swap else None,
        prompt_applied=applied,
    )


@app.get("/api/v1/logs/current")
async def get_current_logs():
    """Get the most recent application log file for bug reporting."""
    try:
        log_file_path = get_most_recent_log_file()

        if log_file_path is None or not log_file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Log file not found. The application may not have logged anything yet.",
            )

        # Read the entire file into memory to avoid Content-Length issues
        # with actively written log files.
        # Use errors='replace' to handle non-UTF-8 bytes gracefully (e.g., Windows-1252
        # characters from subprocess output or exception messages on Windows).
        log_content = log_file_path.read_text(encoding="utf-8", errors="replace")

        # Return as a text response with proper headers for download
        return Response(
            content=log_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="{log_file_path.name.replace(".log", ".txt")}"'
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving log file: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/{path:path}")
async def serve_frontend(request: Request, path: str):
    """Serve the frontend for all non-API routes (fallback for client-side routing)."""
    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"

    # Only serve SPA if frontend dist exists (production mode)
    if not frontend_dist.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")

    # Check if requesting a specific file that exists
    file_path = frontend_dist / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    # Fallback to index.html for SPA routing
    # This ensures clients like Electron alway fetch the latest HTML (which references hashed assets)
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    raise HTTPException(status_code=404, detail="Frontend index.html not found")


def open_browser_when_ready(host: str, port: int, server):
    """Open browser when server is ready, with fallback to URL logging."""
    # Wait for server to be ready
    while not getattr(server, "started", False):
        time.sleep(0.1)

    # Determine the URL to open
    url = (
        f"http://localhost:{port}"
        if host in ["0.0.0.0", "127.0.0.1"]
        else f"http://{host}:{port}"
    )

    try:
        success = webbrowser.open(url)
        if success:
            logger.info(f"🌐 Opened browser at {url}")
    except Exception:
        success = False

    if not success:
        logger.info(f"🌐 UI is available at: {url}")


def main():
    """Main entry point for the daydream-scope command."""
    parser = argparse.ArgumentParser(
        description="A tool for running and customizing real-time, interactive generative AI pipelines and models"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "-N",
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser window after the server starts",
    )

    args = parser.parse_args()

    # Handle version flag
    if args.version:
        print_version_info()
        sys.exit(0)

    # Configure static file serving
    configure_static_files()

    # Check if we're in production mode (frontend dist exists)
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    is_production = frontend_dist.exists()

    if is_production:
        # Create server instance for production mode
        config = uvicorn.Config(
            "scope.server.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_config=None,  # Use our logging config, don't override it
        )
        server = uvicorn.Server(config)

        # Start browser opening thread (unless disabled)
        if not args.no_browser:
            browser_thread = threading.Thread(
                target=open_browser_when_ready,
                args=(args.host, args.port, server),
                daemon=True,
            )
            browser_thread.start()
        else:
            logger.info("main: Skipping browser auto-launch due to --no-browser")

        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # Clean shutdown on Ctrl+C
    else:
        # Development mode - just run normally
        uvicorn.run(
            "scope.server.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_config=None,  # Use our logging config, don't override it
        )


if __name__ == "__main__":
    main()
