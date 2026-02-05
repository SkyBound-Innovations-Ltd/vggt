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
                use_osd_yaw=request.parameters.use_osd_yaw
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
    use_osd_yaw: bool = Form(False)
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
                use_osd_yaw=use_osd_yaw
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
    use_osd_yaw: bool
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

    # Load flight record
    flight_data, home_position = load_flight_record(flight_log_path)

    # Load intrinsics
    intrinsics = np.load(intrinsics_path)

    # Get depth resolution from first depth map
    depth_files = sorted(Path(depth_dir).glob("depth_metric_*.npy"))
    if not depth_files:
        raise ValueError("No depth maps found")

    first_depth = np.load(depth_files[0])
    depth_res = first_depth.shape  # (H, W)

    # Get number of depth frames
    num_depth_frames = len(depth_files)

    # Resample flight data to match depth frame count
    # flight_data has all telemetry (e.g., 450 rows), but we only have 225 depth frames
    if len(flight_data) != num_depth_frames:
        logger.info(f"Resampling flight data: {len(flight_data)} entries -> {num_depth_frames} frames")
        indices = np.linspace(0, len(flight_data) - 1, num_depth_frames, dtype=int)
        flight_data = flight_data.iloc[indices].reset_index(drop=True)
        logger.info(f"Resampled flight data: {len(flight_data)} entries")

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
        use_osd_yaw=use_osd_yaw
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
            }
        },
        "tracks": processed_tracks
    }

    return output_json


# Include router with /vggt_p prefix (must be after all routes are defined)
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
