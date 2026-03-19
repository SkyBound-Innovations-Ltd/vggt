"""
VGGT-P XP Service: Full pipeline from video to geolocalized object tracks.

Uses the v5 3-phase pipeline (segment -> infer -> globalise) with:
- Subprocess-based inference for GPU memory safety
- Mini-batch inference to avoid OOM from VGGT's global attention
- Overlap alignment for long-video depth consistency

Pipeline:
1. Accept video + flight log + tracking JSON via API
2. Run VGGT 3-phase depth estimation (segment, infer, globalise)
3. Run state estimation with Kalman Filter
4. Return geolocalized tracks JSON

FastAPI service with single synchronous endpoint.
"""

import torch
import os
import tempfile
import json
import numpy as np
import httpx
from pathlib import Path
from typing import Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from main_v5_global_scale import (
    phase_segment,
    phase_infer,
    phase_globalise,
    _output_dir_for_video,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VGGT-P XP Service", version="3.1")

# Create router with /vggt_p prefix
router = APIRouter(prefix="/vggt_p")

# Fixed resolution for 720p
SOURCE_RES = (1280, 720)


# Request models
class VGGTParameters(BaseModel):
    """VGGT processing parameters."""
    fps: float = Field(default=5.0, description="Frame extraction rate")
    batch_size: int = Field(default=20, description="Frames per VGGT inference batch")
    segment_size: int = Field(default=600, description="Frames per segment")
    overlap_frames: int = Field(default=10, description="Overlap between segments")
    kf_sigma_a: float = Field(default=0.5, description="KF process noise std (acceleration)")
    kf_sigma_meas_h: float = Field(default=5.0, description="KF measurement noise std (horizontal)")
    kf_sigma_meas_v: float = Field(default=2.0, description="KF measurement noise std (vertical)")
    yaw_offset: float = Field(default=0.0, description="Yaw calibration offset in degrees")
    magnetic_declination: float = Field(default=0.0, description="Magnetic declination in degrees")
    add_drone_yaw: bool = Field(default=False, description="Add drone yaw to gimbal yaw")
    use_gimbal_yaw: bool = Field(default=False, description="Use GIMBAL.yaw instead of OSD.yaw for yaw angle. Default: OSD.yaw (drone heading)")
    gimbal_pitch_override: float = Field(default=None, description="Manual gimbal pitch in DJI degrees (e.g. -90 for nadir)")
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
        "pipeline": "v5-subprocess",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_available": torch.cuda.is_available()
    }


@router.post("/process")
async def process_video(request: VGGTRequest):
    """
    Process video through full VGGT v5 + state estimation pipeline.

    Returns geolocalized object tracks with state estimation (position, velocity, heading).
    """
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

            # Step 2: Run VGGT v5 3-phase pipeline
            logger.info("Running VGGT v5 pipeline (segment -> infer -> globalise)")
            resolved_output = run_vggt_pipeline(
                video_path=str(video_path),
                flight_log_path=str(flight_log_path),
                output_dir=str(tmp_path),
                fps=request.parameters.fps,
                batch_size=request.parameters.batch_size,
                segment_size=request.parameters.segment_size,
                overlap_frames=request.parameters.overlap_frames,
                use_gimbal_yaw=request.parameters.use_gimbal_yaw,
                gimbal_pitch_override=request.parameters.gimbal_pitch_override,
            )

            # Step 3: Run state estimation
            logger.info("Running state estimation with Kalman Filter")
            depth_dir = os.path.join(resolved_output, "depth_metric")
            intrinsics_path = os.path.join(resolved_output, "cameras", "intrinsics.npy")

            output_json = run_state_estimation(
                tracking_json_path=str(tracking_json_path),
                flight_log_path=str(flight_log_path),
                depth_dir=depth_dir,
                intrinsics_path=intrinsics_path,
                source_res=SOURCE_RES,
                fps=request.parameters.fps,
                kf_sigma_a=request.parameters.kf_sigma_a,
                kf_sigma_meas_h=request.parameters.kf_sigma_meas_h,
                kf_sigma_meas_v=request.parameters.kf_sigma_meas_v,
                yaw_offset=request.parameters.yaw_offset,
                magnetic_declination=request.parameters.magnetic_declination,
                add_drone_yaw=request.parameters.add_drone_yaw,
                use_gimbal_yaw=request.parameters.use_gimbal_yaw,
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
    batch_size: int = Form(20),
    segment_size: int = Form(600),
    overlap_frames: int = Form(10),
    # Kalman Filter parameters
    kf_sigma_a: float = Form(0.5),
    kf_sigma_meas_h: float = Form(5.0),
    kf_sigma_meas_v: float = Form(2.0),
    # Yaw parameters
    yaw_offset: float = Form(0.0),
    magnetic_declination: float = Form(0.0),
    add_drone_yaw: bool = Form(False),
    use_gimbal_yaw: bool = Form(False),
    gimbal_pitch_override: float = Form(None),
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
    Process video through full VGGT v5 + state estimation pipeline (file upload version).

    Use this endpoint when sending files directly instead of URLs.
    """
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

            # Step 2: Run VGGT v5 3-phase pipeline
            logger.info("Running VGGT v5 pipeline (segment -> infer -> globalise)")
            resolved_output = run_vggt_pipeline(
                video_path=str(video_path),
                flight_log_path=str(flight_log_path),
                output_dir=str(tmp_path),
                fps=fps,
                batch_size=batch_size,
                segment_size=segment_size,
                overlap_frames=overlap_frames,
                use_gimbal_yaw=use_gimbal_yaw,
                gimbal_pitch_override=gimbal_pitch_override,
            )

            # Step 3: Run state estimation
            logger.info("Running state estimation with Kalman Filter")
            depth_dir = os.path.join(resolved_output, "depth_metric")
            intrinsics_path = os.path.join(resolved_output, "cameras", "intrinsics.npy")

            output_json = run_state_estimation(
                tracking_json_path=str(tracking_json_path),
                flight_log_path=str(flight_log_path),
                depth_dir=depth_dir,
                intrinsics_path=intrinsics_path,
                source_res=SOURCE_RES,
                fps=fps,
                kf_sigma_a=kf_sigma_a,
                kf_sigma_meas_h=kf_sigma_meas_h,
                kf_sigma_meas_v=kf_sigma_meas_v,
                yaw_offset=yaw_offset,
                magnetic_declination=magnetic_declination,
                add_drone_yaw=add_drone_yaw,
                use_gimbal_yaw=use_gimbal_yaw,
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


def run_vggt_pipeline(
    video_path: str,
    flight_log_path: str,
    output_dir: str,
    fps: float,
    batch_size: int,
    segment_size: int,
    overlap_frames: int,
    use_gimbal_yaw: bool = False,
    gimbal_pitch_override: float = None,
) -> str:
    """Run VGGT v5 3-phase pipeline and return resolved output directory.

    Phase 1 (segment): Extract frames into overlapping segment directories.
    Phase 2 (infer): Subprocess-based mini-batch inference per segment.
    Phase 3 (globalise): Overlap alignment, metric scaling, save outputs.

    Returns:
        Resolved output directory path (e.g. output_dir/video/).
    """
    # Phase 1: Segment
    logger.info(f"Phase 1: Segmenting video (fps={fps}, segment_size={segment_size}, overlap={overlap_frames})")
    phase_segment(
        video_path=video_path,
        output_dir=output_dir,
        fps=fps,
        segment_size=segment_size,
        overlap_frames=overlap_frames,
    )

    # Resolve output dir (phase_segment appends video stem)
    resolved_output = _output_dir_for_video(output_dir, video_path)

    # Phase 2: Infer (subprocess mode with mini-batching)
    logger.info(f"Phase 2: Inference (subprocess, batch_size={batch_size})")
    phase_infer(
        output_dir=resolved_output,
        batch_size=batch_size,
    )

    # Phase 3: Globalise (with cameras and metric scaling)
    logger.info("Phase 3: Globalise (overlap alignment + metric scaling)")
    phase_globalise(
        output_dir=resolved_output,
        flight_record=flight_log_path,
        fps=fps,
        save_cameras_flag=True,
        save_video=False,
        no_video=True,
        use_gimbal_yaw=use_gimbal_yaw,
        gimbal_pitch_override=gimbal_pitch_override,
    )

    return resolved_output


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
    use_gimbal_yaw: bool,
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
        raise ValueError("No metric depth maps found. Check flight record gimbal data.")

    first_depth = np.load(depth_files[0])
    depth_res = first_depth.shape  # (H, W)

    # Get number of depth frames
    num_depth_frames = len(depth_files)

    # Resample flight data to match depth frame count
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
        use_gimbal_yaw=use_gimbal_yaw,
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
            "pipeline": "v5-subprocess",
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


# Include router with /vggt_p prefix
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
