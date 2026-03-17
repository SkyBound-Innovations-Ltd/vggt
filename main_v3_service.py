"""
VGGT-P Service: Full pipeline from video to geolocalized object tracks with state estimation.

Pipeline:
1. Accept video + flight log + tracking JSON via API
2. Run VGGT depth estimation
3. Run state estimation with Kalman Filter
4. Return geolocalized tracks JSON

FastAPI service with single synchronous endpoint.
"""

import torch
import cv2
import os
import tempfile
import shutil
import json
import numpy as np
import httpx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

# Import VGGT components
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Import processing functions from main.py and main_v2_state_est.py
# We'll inline the necessary functions here to avoid circular imports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VGGT-P Service", version="1.0")

# Create router with /vggt_p prefix
router = APIRouter(prefix="/vggt_p")


# Request models
class VGGTParameters(BaseModel):
    """VGGT processing parameters."""
    fps: float = Field(default=5.0, description="Frame extraction rate")
    batch_size: int = Field(default=5, description="VGGT inference batch size")
    kf_sigma_a: float = Field(default=0.5, description="KF process noise std (acceleration)")
    kf_sigma_meas_h: float = Field(default=5.0, description="KF measurement noise std (horizontal)")
    kf_sigma_meas_v: float = Field(default=2.0, description="KF measurement noise std (vertical)")
    yaw_offset: float = Field(default=0.0, description="Yaw calibration offset in degrees")
    magnetic_declination: float = Field(default=0.0, description="Magnetic declination in degrees")
    add_drone_yaw: bool = Field(default=False, description="Add drone yaw to gimbal yaw")
    use_osd_yaw: bool = Field(default=False, description="Use OSD.yaw instead of GIMBAL.yaw")
    timezone: str = Field(default="UTC", description="Timezone of flight log local timestamps (e.g., 'UTC', 'Europe/London')")
    hdbscan_min_cluster_size: int = Field(default=10, description="HDBSCAN min_cluster_size: minimum unique person tracks to form a crowd", ge=2)
    hdbscan_min_samples: int = Field(default=3, description="HDBSCAN min_samples: higher = more conservative clustering", ge=1)
    hdbscan_coherence_weight: float = Field(default=10.0, description="Weight on velocity features for crowd clustering", ge=0.0)
    hdbscan_max_speed_mps: float = Field(default=2.0, description="Max walking speed (m/s) for velocity normalisation", gt=0.0)
    hdbscan_cluster_selection_epsilon: float = Field(default=2.0, description="HDBSCAN epsilon: prevents over-fragmenting nearby sub-clusters (metres)", ge=0.0)
    hdbscan_max_match_dist: float = Field(default=20.0, description="Max cost for Hungarian crowd-cluster matching across frames (metres)", gt=0.0)
    hdbscan_ema_alpha: float = Field(default=0.4, description="EMA smoothing factor for cluster centroid/momentum (1.0 = no smoothing)", ge=0.0, le=1.0)
    hdbscan_memory_frames: int = Field(default=15, description="Frames to remember absent clusters (e.g. 15 = 1.5s at 10fps)", ge=1)
    tracking_fps: Optional[float] = Field(default=None, description="FPS of tracking JSON (if different from extraction fps, frame IDs are remapped)")


class VGGTRequest(BaseModel):
    """VGGT service request payload."""
    job_id: int = Field(..., description="Job ID from backend")
    input_file_url: str = Field(..., description="Video file URL")
    flight_log_url: str = Field(..., description="Flight log CSV URL")
    tracking_json_url: str = Field(..., description="MOT tracking JSON URL")
    file_id: str = Field(..., description="File identifier")
    parameters: VGGTParameters = Field(default_factory=VGGTParameters)

# Global model singleton (loaded once at startup)
MODEL = None
DEVICE = None
DTYPE = None

# Fixed resolution for 720p
SOURCE_RES = (1280, 720)


@app.on_event("startup")
async def load_model():
    """Load VGGT model once at startup."""
    global MODEL, DEVICE, DTYPE

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    logger.info(f"Loading VGGT model on {DEVICE} with dtype {DTYPE}")
    MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
    MODEL.eval()
    logger.info("Model loaded successfully")


async def download_file(url: str, destination: Path) -> None:
    """Download file from URL to destination path."""
    logger.info(f"Downloading {url} to {destination}")
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(destination, "wb") as f:
            f.write(response.content)
    logger.info(f"Downloaded {destination.name} ({destination.stat().st_size} bytes)")


@router.get("/")
async def root():
    """Root endpoint - Service information and available endpoints."""
    return {
        "service": "VGGT-P Service",
        "version": "1.0",
        "description": "Full pipeline from video to geolocalized object tracks with state estimation",
        "status": "online",
        "model_loaded": MODEL is not None,
        "endpoints": {
            "GET /vggt_p/": "Service information (this page)",
            "GET /vggt_p/health": "Health check endpoint",
            "POST /vggt_p/process": "Process video via URLs (video, flight log, tracking JSON)",
            "POST /vggt_p/process-upload": "Process video via file upload"
        },
        "documentation": "See /docs/VGGT_P_API.md for detailed API documentation"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "gpu_available": torch.cuda.is_available()
    }


@router.post("/process")
async def process_video(request: VGGTRequest):
    """
    Process video through full VGGT + state estimation pipeline.

    Returns geolocalized object tracks with state estimation (position, velocity, heading).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Processing job {request.job_id} (file_id: {request.file_id})")

    # Create temporary working directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            # Step 1: Download input files from URLs
            logger.info("Downloading input files")
            video_path = tmp_path / "video.mp4"
            flight_log_path = tmp_path / "flight_log.csv"
            tracking_json_path = tmp_path / "tracking.json"

            await download_file(request.input_file_url, video_path)
            await download_file(request.flight_log_url, flight_log_path)
            await download_file(request.tracking_json_url, tracking_json_path)

            # Step 2: Run VGGT depth estimation
            logger.info("Running VGGT depth estimation")
            depth_dir = tmp_path / "depth_metric"
            intrinsics_path = tmp_path / "cameras" / "intrinsics.npy"

            run_vggt_inference(
                video_path=str(video_path),
                flight_log_path=str(flight_log_path),
                output_dir=str(tmp_path),
                fps=request.parameters.fps,
                batch_size=request.parameters.batch_size
            )

            # Step 3: Run state estimation
            logger.info("Running state estimation with Kalman Filter")
            output_json = run_state_estimation(
                tracking_json_path=str(tracking_json_path),
                flight_log_path=str(flight_log_path),
                depth_dir=str(depth_dir),
                intrinsics_path=str(intrinsics_path),
                source_res=SOURCE_RES,
                fps=request.parameters.fps,
                kf_sigma_a=request.parameters.kf_sigma_a,
                kf_sigma_meas_h=request.parameters.kf_sigma_meas_h,
                kf_sigma_meas_v=request.parameters.kf_sigma_meas_v,
                yaw_offset=request.parameters.yaw_offset,
                magnetic_declination=request.parameters.magnetic_declination,
                add_drone_yaw=request.parameters.add_drone_yaw,
                use_osd_yaw=request.parameters.use_osd_yaw,
                timezone=request.parameters.timezone,
                hdbscan_min_cluster_size=request.parameters.hdbscan_min_cluster_size,
                hdbscan_min_samples=request.parameters.hdbscan_min_samples,
                hdbscan_coherence_weight=request.parameters.hdbscan_coherence_weight,
                hdbscan_max_speed_mps=request.parameters.hdbscan_max_speed_mps,
                hdbscan_cluster_selection_epsilon=request.parameters.hdbscan_cluster_selection_epsilon,
                hdbscan_max_match_dist=request.parameters.hdbscan_max_match_dist,
                hdbscan_ema_alpha=request.parameters.hdbscan_ema_alpha,
                hdbscan_memory_frames=request.parameters.hdbscan_memory_frames,
                tracking_fps=request.parameters.tracking_fps,
            )

            # Return JSON response
            logger.info(f"Job {request.job_id} completed successfully")
            return JSONResponse(content=output_json)

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/process-upload")
async def process_video_upload(
    video: UploadFile = File(..., description="MP4 video file"),
    flight_log: UploadFile = File(..., description="DJI flight log CSV"),
    tracking_json: UploadFile = File(..., description="MOT tracking JSON"),
    fps: float = Form(5.0),
    batch_size: int = Form(5),
    # Kalman Filter parameters
    kf_sigma_a: float = Form(0.5),
    kf_sigma_meas_h: float = Form(5.0),
    kf_sigma_meas_v: float = Form(2.0),
    # Yaw parameters
    yaw_offset: float = Form(0.0),
    magnetic_declination: float = Form(0.0),
    add_drone_yaw: bool = Form(False),
    use_osd_yaw: bool = Form(False),
    timezone: str = Form("UTC"),
    # Crowd clustering (HDBSCAN) parameters
    hdbscan_min_cluster_size: int = Form(10),
    hdbscan_min_samples: int = Form(3),
    hdbscan_coherence_weight: float = Form(10.0),
    hdbscan_max_speed_mps: float = Form(2.0),
    hdbscan_cluster_selection_epsilon: float = Form(2.0),
    hdbscan_max_match_dist: float = Form(20.0),
    hdbscan_ema_alpha: float = Form(0.4),
    hdbscan_memory_frames: int = Form(15),
    tracking_fps: Optional[float] = Form(None),
):
    """
    Process video through full VGGT + state estimation pipeline (file upload version).

    Use this endpoint when sending files directly instead of URLs.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Processing uploaded files: video={video.filename}, flight_log={flight_log.filename}, tracking={tracking_json.filename}")

    # Create temporary working directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            # Step 1: Save uploaded files
            logger.info("Saving uploaded files")
            video_path = tmp_path / "video.mp4"
            flight_log_path = tmp_path / "flight_log.csv"
            tracking_json_path = tmp_path / "tracking.json"

            with open(video_path, "wb") as f:
                f.write(await video.read())
            with open(flight_log_path, "wb") as f:
                f.write(await flight_log.read())
            with open(tracking_json_path, "wb") as f:
                f.write(await tracking_json.read())

            # Step 2: Run VGGT depth estimation
            logger.info("Running VGGT depth estimation")
            depth_dir = tmp_path / "depth_metric"
            intrinsics_path = tmp_path / "cameras" / "intrinsics.npy"

            run_vggt_inference(
                video_path=str(video_path),
                flight_log_path=str(flight_log_path),
                output_dir=str(tmp_path),
                fps=fps,
                batch_size=batch_size
            )

            # Step 3: Run state estimation
            logger.info("Running state estimation with Kalman Filter")
            output_json = run_state_estimation(
                tracking_json_path=str(tracking_json_path),
                flight_log_path=str(flight_log_path),
                depth_dir=str(depth_dir),
                intrinsics_path=str(intrinsics_path),
                source_res=SOURCE_RES,
                fps=fps,
                kf_sigma_a=kf_sigma_a,
                kf_sigma_meas_h=kf_sigma_meas_h,
                kf_sigma_meas_v=kf_sigma_meas_v,
                yaw_offset=yaw_offset,
                magnetic_declination=magnetic_declination,
                add_drone_yaw=add_drone_yaw,
                use_osd_yaw=use_osd_yaw,
                timezone=timezone,
                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                hdbscan_min_samples=hdbscan_min_samples,
                hdbscan_coherence_weight=hdbscan_coherence_weight,
                hdbscan_max_speed_mps=hdbscan_max_speed_mps,
                hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                hdbscan_max_match_dist=hdbscan_max_match_dist,
                hdbscan_ema_alpha=hdbscan_ema_alpha,
                hdbscan_memory_frames=hdbscan_memory_frames,
                tracking_fps=tracking_fps,
            )

            # Return JSON response
            logger.info("File upload processing completed successfully")
            return JSONResponse(content=output_json)

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def run_vggt_inference(
    video_path: str,
    flight_log_path: str,
    output_dir: str,
    fps: float,
    batch_size: int
):
    """Run VGGT depth estimation and save outputs."""
    import pandas as pd
    from main import (
        extract_frames_from_video,
        save_frames_and_load,
        parse_dji_flight_record,
        scale_depth_sequence_to_metric
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract frames
    logger.info(f"Extracting frames at {fps} FPS")
    frames = extract_frames_from_video(video_path, fps=fps)
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Preprocess frames
    tmp_frames_dir = output_path / "frames"
    images = save_frames_and_load(frames, str(tmp_frames_dir))
    images = images.to(DEVICE)
    logger.info(f"Loaded {images.shape[0]} frames")

    # Run inference in batches
    logger.info(f"Running inference (batch size: {batch_size})")
    all_depths = []
    all_pose_encs = []

    num_frames = images.shape[0]
    num_batches = (num_frames + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_frames)
        batch_images = images[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=DTYPE):
                batch_predictions = MODEL(batch_images)

        if "depth" in batch_predictions:
            all_depths.append(batch_predictions["depth"].cpu())
        if "pose_enc" in batch_predictions:
            all_pose_encs.append(batch_predictions["pose_enc"].cpu())

        del batch_predictions
        torch.cuda.empty_cache()

    # Combine results
    predictions = {}
    if all_depths:
        predictions["depth"] = torch.cat(all_depths, dim=1).squeeze(0)
    if all_pose_encs:
        pose_enc = torch.cat(all_pose_encs, dim=1)
        predictions["pose_enc"] = pose_enc.squeeze(0)

        # Decode camera intrinsics
        depth_shape = all_depths[0].shape
        image_size_hw = (depth_shape[2], depth_shape[3])
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=image_size_hw, pose_encoding_type="absT_quaR_FoV"
        )
        predictions["intrinsic"] = intrinsics.squeeze(0)

    # Apply metric scaling with flight record
    logger.info("Applying metric depth scaling")
    video_duration = len(frames) / fps
    telemetry = parse_dji_flight_record(
        flight_log_path,
        video_duration_sec=video_duration,
        target_fps=fps
    )

    if telemetry is not None:
        depth_maps_np = [d.cpu().numpy().squeeze() for d in predictions["depth"]]
        metric_depth_maps = scale_depth_sequence_to_metric(
            canonical_depths=depth_maps_np,
            altitudes=telemetry['altitudes'],
            pitch_angles=telemetry['pitch_below_horizontal']
        )
        predictions["depth_metric"] = metric_depth_maps
    else:
        raise ValueError("Invalid gimbal data in flight record")

    # Save outputs
    metric_depth_dir = output_path / "depth_metric"
    metric_depth_dir.mkdir(exist_ok=True)

    for i, depth in enumerate(predictions["depth_metric"]):
        np.save(metric_depth_dir / f"depth_metric_{i:04d}.npy", depth)
    logger.info(f"Saved {len(predictions['depth_metric'])} depth maps")

    # Save intrinsics
    cameras_dir = output_path / "cameras"
    cameras_dir.mkdir(exist_ok=True)
    intrinsics_np = predictions["intrinsic"].cpu().numpy()
    np.save(cameras_dir / "intrinsics.npy", intrinsics_np)
    logger.info(f"Saved intrinsics: {intrinsics_np.shape}")


def run_state_estimation(
    tracking_json_path: str,
    flight_log_path: str,
    depth_dir: str,
    intrinsics_path: str,
    source_res: Tuple[int, int],
    fps: float,
    kf_sigma_a: float,
    kf_sigma_meas_h: float,
    kf_sigma_meas_v: float,
    yaw_offset: float,
    magnetic_declination: float,
    add_drone_yaw: bool,
    use_osd_yaw: bool,
    timezone: str = "UTC",
    hdbscan_min_cluster_size: int = 10,
    hdbscan_min_samples: int = 3,
    hdbscan_coherence_weight: float = 10.0,
    hdbscan_max_speed_mps: float = 2.0,
    hdbscan_cluster_selection_epsilon: float = 2.0,
    hdbscan_max_match_dist: float = 20.0,
    hdbscan_ema_alpha: float = 0.4,
    hdbscan_memory_frames: int = 15,
    tracking_fps: Optional[float] = None,
) -> Dict:
    """Run state estimation and return JSON result."""
    import pandas as pd
    from main_v2_state_est import (
        load_flight_record,
        process_tracks
    )

    # Load tracking JSON
    with open(tracking_json_path, 'r') as f:
        tracking_data = json.load(f)

    tracks = tracking_data.get('tracks', [])
    if not tracks:
        raise ValueError("No tracks found in tracking JSON")

    # Resample tracking frame IDs if tracking was done at a different FPS
    if tracking_fps is not None and tracking_fps != fps:
        ratio = tracking_fps / fps
        logger.info(f"Resampling tracking frame IDs: {tracking_fps} fps -> {fps} fps (ratio {ratio:.2f})")
        for t in tracks:
            t['frame_id'] = round(t['frame_id'] / ratio)
        # Deduplicate: multiple 30fps frames map to the same 10fps frame
        # Keep all detections (they'll share the same depth map)
        unique_frames = len(set(t['frame_id'] for t in tracks))
        logger.info(f"  Remapped to {unique_frames} unique frames")

    # Load flight record
    flight_data, home_position = load_flight_record(flight_log_path, timezone=timezone)

    # Load intrinsics
    intrinsics = np.load(intrinsics_path)

    # Get depth resolution from first depth map
    depth_files = sorted(Path(depth_dir).glob("depth_metric_*.npy"))
    if not depth_files:
        raise ValueError("No depth maps found")

    first_depth = np.load(depth_files[0])
    depth_res = first_depth.shape  # (H, W)

    logger.info(f"Processing {len(tracks)} tracks")
    logger.info(f"Source resolution: {source_res}, Depth resolution: {depth_res}")

    # Process tracks with state estimation
    processed_tracks = process_tracks(
        tracks=tracks,
        flight_data=flight_data,
        home_position=home_position,
        depth_dir=depth_dir,
        intrinsics=intrinsics,
        source_res=source_res,
        depth_res=depth_res,
        fps=fps,
        velocity_window=5,  # Not used with KF
        use_kf=True,
        kf_sigma_a=kf_sigma_a,
        kf_sigma_meas_h=kf_sigma_meas_h,
        kf_sigma_meas_v=kf_sigma_meas_v,
        yaw_offset_deg=yaw_offset,
        magnetic_declination_deg=magnetic_declination,
        add_drone_yaw=add_drone_yaw,
        use_osd_yaw=use_osd_yaw,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_coherence_weight=hdbscan_coherence_weight,
        hdbscan_max_speed_mps=hdbscan_max_speed_mps,
        hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
        hdbscan_max_match_dist=hdbscan_max_match_dist,
        hdbscan_ema_alpha=hdbscan_ema_alpha,
        hdbscan_memory_frames=hdbscan_memory_frames,
    )

    logger.info(f"Processed {len(processed_tracks)} track entries")

    # Prepare output JSON
    output_json = {
        "metadata": {
            "source_resolution": {"width": source_res[0], "height": source_res[1]},
            "depth_resolution": {"height": depth_res[0], "width": depth_res[1]},
            "fps": fps,
            "total_frames": len(flight_data),
            "home_position": {
                "latitude": home_position[0],
                "longitude": home_position[1],
                "altitude": home_position[2]
            },
            "kalman_filter": {
                "enabled": True,
                "sigma_a": kf_sigma_a,
                "sigma_meas_h": kf_sigma_meas_h,
                "sigma_meas_v": kf_sigma_meas_v
            },
            "crowd_clustering": {
                "method": "HDBSCAN_per_frame_hungarian",
                "min_cluster_size": hdbscan_min_cluster_size,
                "min_samples": hdbscan_min_samples,
                "coherence_weight": hdbscan_coherence_weight,
                "max_speed_mps": hdbscan_max_speed_mps,
                "cluster_selection_epsilon": hdbscan_cluster_selection_epsilon,
                "max_match_dist": hdbscan_max_match_dist,
                "ema_alpha": hdbscan_ema_alpha,
                "memory_frames": hdbscan_memory_frames,
                "note": "Per-frame HDBSCAN with Hungarian centroid matching, EMA smoothing, and multi-frame memory for stable crowd IDs"
            }
        },
        "tracks": processed_tracks
    }

    return output_json


# Include router with /vggt_p prefix (must be after all routes are defined)
app.include_router(router)


def create_detection_video(
    video_path: str,
    tracking_json_path: str,
    output_path: str,
    fps: float = 10.0,
):
    """Overlay detection bboxes on original video frames and save as MP4.

    Draws class-colored bounding boxes with track_id labels on each frame.
    """
    from main import extract_frames_from_video

    CLASS_COLORS_BGR = {
        'person':     (0, 255, 0),
        'car':        (0, 0, 255),
        'vehicle':    (255, 0, 0),
        'cycle':      (255, 255, 0),
        'bus':        (255, 0, 255),
        'track_leaf': (0, 215, 255),
    }
    FALLBACK_COLOR = (170, 170, 170)

    # Load tracking JSON
    with open(tracking_json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        tracks = data.get('tracks', [])
    else:
        tracks = data

    # Group detections by frame_id (skip entries without bbox)
    by_frame = {}
    for t in tracks:
        fid = t.get('frame_id')
        bbox = t.get('bbox')
        if fid is None or bbox is None:
            continue
        by_frame.setdefault(fid, []).append(t)

    # Extract frames
    frames = extract_frames_from_video(video_path, fps=fps)
    if len(frames) == 0:
        logger.warning("No frames extracted, skipping detection video")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx, frame in enumerate(frames):
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        dets = by_frame.get(frame_idx, [])
        for det in dets:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cls = det.get('class_name', 'unknown')
            tid = det.get('track_id', '')
            conf = det.get('confidence')
            color = CLASS_COLORS_BGR.get(cls, FALLBACK_COLOR)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f'{cls} #{tid}'
            if conf is not None:
                label += f' {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(canvas, f'Frame: {frame_idx}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    logger.info(f"Detection video: {len(frames)} frames -> {output_path}")


def run_offline(args):
    """Run the full pipeline offline (no FastAPI server)."""
    global MODEL, DEVICE, DTYPE

    # --- Load model --------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    logger.info(f"Loading VGGT model on {DEVICE} with dtype {DTYPE}")
    MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)
    MODEL.eval()
    logger.info("Model loaded successfully")

    # --- Prepare working directory ----------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: VGGT depth inference -------------------------------------
    logger.info("Step 1: Running VGGT depth estimation")
    depth_dir = output_dir / "depth_metric"
    intrinsics_path = output_dir / "cameras" / "intrinsics.npy"

    run_vggt_inference(
        video_path=args.video,
        flight_log_path=args.flight_log,
        output_dir=str(output_dir),
        fps=args.fps,
        batch_size=args.batch_size,
    )

    # --- Step 2: State estimation -----------------------------------------
    logger.info("Step 2: Running state estimation with Kalman Filter")
    output_json = run_state_estimation(
        tracking_json_path=args.tracking_json,
        flight_log_path=args.flight_log,
        depth_dir=str(depth_dir),
        intrinsics_path=str(intrinsics_path),
        source_res=tuple(args.source_res),
        fps=args.fps,
        kf_sigma_a=args.kf_sigma_a,
        kf_sigma_meas_h=args.kf_sigma_meas_h,
        kf_sigma_meas_v=args.kf_sigma_meas_v,
        yaw_offset=args.yaw_offset,
        magnetic_declination=args.magnetic_declination,
        add_drone_yaw=args.add_drone_yaw,
        use_osd_yaw=args.use_osd_yaw,
        timezone=args.timezone,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_coherence_weight=args.hdbscan_coherence_weight,
        hdbscan_max_speed_mps=args.hdbscan_max_speed_mps,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        hdbscan_max_match_dist=args.hdbscan_max_match_dist,
        hdbscan_ema_alpha=args.hdbscan_ema_alpha,
        hdbscan_memory_frames=args.hdbscan_memory_frames,
        tracking_fps=args.tracking_fps,
    )

    # --- Save output ------------------------------------------------------
    output_path = output_dir / args.output
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    logger.info(f"Saved output to {output_path}")
    logger.info(f"Tracks: {len(output_json['tracks'])}")
    print(f"\nDone! Output saved to: {output_path}")

    # --- Step 3: Detection video ------------------------------------------
    logger.info("Step 3: Generating detection video")
    det_video_path = output_dir / "detections.mp4"
    create_detection_video(
        video_path=args.video,
        tracking_json_path=args.tracking_json,
        output_path=str(det_video_path),
        fps=args.fps,
    )
    logger.info(f"Detection video saved to {det_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VGGT-P: full pipeline (offline or service)")
    sub = parser.add_subparsers(dest="mode")

    # --- 'serve' sub-command (default FastAPI) ----------------------------
    srv = sub.add_parser("serve", help="Run as FastAPI service")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8003)

    # --- 'offline' sub-command --------------------------------------------
    off = sub.add_parser("offline", help="Run full pipeline on local files")
    off.add_argument("--video", "-v", required=True, help="Path to input MP4 video")
    off.add_argument("--flight-log", "-f", required=True, help="Path to DJI flight log CSV")
    off.add_argument("--tracking-json", "-t", required=True, help="Path to DET-TRACK output JSON")
    off.add_argument("--output-dir", "-d", default="outputs", help="Output directory (default: outputs)")
    off.add_argument("--output", "-o", default="state_estimation.json", help="Output JSON filename (default: state_estimation.json)")
    off.add_argument("--source-res", type=int, nargs=2, default=[1920, 1080], metavar=("W", "H"), help="Source video resolution (default: 1920 1080)")
    off.add_argument("--fps", type=float, default=10.0, help="Frame extraction rate (default: 10.0)")
    off.add_argument("--batch-size", type=int, default=5, help="VGGT inference batch size (default: 5)")
    # Kalman Filter
    off.add_argument("--kf-sigma-a", type=float, default=0.5, help="KF process noise std (default: 0.5)")
    off.add_argument("--kf-sigma-meas-h", type=float, default=5.0, help="KF horizontal measurement noise std (default: 5.0)")
    off.add_argument("--kf-sigma-meas-v", type=float, default=2.0, help="KF vertical measurement noise std (default: 2.0)")
    # Yaw
    off.add_argument("--yaw-offset", type=float, default=0.0, help="Yaw calibration offset in degrees")
    off.add_argument("--magnetic-declination", type=float, default=0.0, help="Magnetic declination in degrees")
    off.add_argument("--add-drone-yaw", action="store_true", help="Add drone heading to gimbal yaw")
    off.add_argument("--use-osd-yaw", action="store_true", help="Use OSD.yaw instead of GIMBAL.yaw")
    off.add_argument("--timezone", default="UTC", help="Flight log timezone (default: UTC)")
    # HDBSCAN
    off.add_argument("--hdbscan-min-cluster-size", type=int, default=10, help="HDBSCAN min cluster size (default: 10)")
    off.add_argument("--hdbscan-min-samples", type=int, default=3, help="HDBSCAN min samples (default: 3)")
    off.add_argument("--hdbscan-coherence-weight", type=float, default=5.0, help="Velocity feature weight (default: 5.0)")
    off.add_argument("--hdbscan-max-speed-mps", type=float, default=2.0, help="Max walking speed m/s (default: 2.0)")
    off.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=2.0, help="HDBSCAN epsilon to prevent over-fragmentation (metres, default: 2.0)")
    off.add_argument("--hdbscan-max-match-dist", type=float, default=20.0, help="Max cost for Hungarian crowd-cluster matching (metres, default: 20.0)")
    off.add_argument("--hdbscan-ema-alpha", type=float, default=0.4, help="EMA smoothing for cluster centroid/momentum (0-1, default: 0.4)")
    off.add_argument("--hdbscan-memory-frames", type=int, default=15, help="Frames to remember absent clusters (default: 15 = 1.5s at 10fps)")
    off.add_argument("--tracking-fps", type=float, default=None, help="FPS of tracking JSON (remaps frame IDs if different from --fps)")

    args = parser.parse_args()

    if args.mode == "offline":
        run_offline(args)
    else:
        import uvicorn
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8003)
        uvicorn.run(app, host=host, port=port)
