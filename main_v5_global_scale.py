import torch
import cv2
import os
import argparse
import numpy as np
import time
import json
import glob
import datetime
import tempfile
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# Metadata helpers (contract between phases)
# ---------------------------------------------------------------------------

def _meta_path(output_dir):
    return os.path.join(output_dir, "segments", "segments_meta.json")


def _load_meta(output_dir):
    path = _meta_path(output_dir)
    with open(path, "r") as f:
        return json.load(f)


def _save_meta(output_dir, meta):
    """Atomic write: write to a tmp file, then rename."""
    path = _meta_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, path)
    except:
        os.unlink(tmp)
        raise


def parse_dji_flight_record(csv_path: str, video_duration_sec: float, target_fps: float,
                            use_gimbal_yaw: bool = False,
                            gimbal_pitch_override: float = None) -> dict:
    """
    Parse DJI flight record CSV and extract telemetry synchronized with video frames.

    Extracts: altitude (AGL), GIMBAL.pitch, GIMBAL.roll, yaw, and resamples
    to match the target frame count.

    Yaw source: OSD.yaw (drone heading) by default; --use-gimbal-yaw switches to GIMBAL.yaw.

    Args:
        csv_path: Path to DJI flight record CSV file
        video_duration_sec: Duration of the input video in seconds
        target_fps: Target frames per second for extraction
        use_gimbal_yaw: If True, use GIMBAL.yaw instead of OSD.yaw
        gimbal_pitch_override: Manual gimbal pitch (DJI degrees) when GIMBAL.pitch is missing

    Returns:
        dict with keys, or None if gimbal pitch data is invalid:
            - 'altitudes': altitude values (meters) per frame
            - 'pitch_gimbal': GIMBAL.pitch angles (degrees, DJI convention) per frame
            - 'roll_gimbal': GIMBAL.roll angles (degrees) per frame
            - 'pitch_below_horizontal': pitch below horizontal (0=horizon, 90=nadir) per frame
            - 'yaw': yaw angles (degrees) per frame
            - 'timestamps': timestamps per frame
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Find altitude column
    altitude_col = None
    for col in ['OSD.height [m]', 'OSD.vpsHeight [m]']:
        if col in df.columns:
            altitude_col = col
            break
    if altitude_col is None:
        raise ValueError(f"Could not find altitude column in {csv_path}")
    print(f"  Using altitude column: {altitude_col}")

    # Yaw: OSD.yaw by default, GIMBAL.yaw with flag
    if use_gimbal_yaw:
        yaw_col = 'GIMBAL.yaw'
        print(f"  Using yaw column: GIMBAL.yaw (gimbal heading)")
    else:
        yaw_col = 'OSD.yaw'
        print(f"  Using yaw column: OSD.yaw (drone heading)")

    # Required columns
    required_cols = {
        'altitude': altitude_col,
        'pitch_gimbal': 'GIMBAL.pitch',
        'roll_gimbal': 'GIMBAL.roll',
        'yaw': yaw_col,
        'is_video': 'CAMERA.isVideo',
        'timestamp': 'CUSTOM.updateTime [local]'
    }

    for name, col in required_cols.items():
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available: {list(df.columns)[:20]}...")

    # Filter for video recording segments
    df['is_video_bool'] = df[required_cols['is_video']].apply(
        lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else False
    )
    video_df = df[df['is_video_bool']].copy()
    if len(video_df) == 0:
        raise ValueError("No rows found where CAMERA.isVideo is TRUE")

    print(f"  Flight record: {len(df)} total rows, {len(video_df)} during video recording")

    # Extract telemetry
    altitudes_raw = video_df[required_cols['altitude']].values
    pitch_gimbal_raw = video_df[required_cols['pitch_gimbal']].values
    roll_gimbal_raw = video_df[required_cols['roll_gimbal']].values
    yaw_raw = video_df[required_cols['yaw']].values
    timestamps_raw = video_df[required_cols['timestamp']].values

    # Check if gimbal pitch data is valid (not all zeros)
    if np.all(pitch_gimbal_raw == 0):
        if gimbal_pitch_override is not None:
            print(f"  WARNING: GIMBAL.pitch is all zeros — using override: {gimbal_pitch_override}°")
            pitch_gimbal_raw[:] = gimbal_pitch_override
        else:
            print("  WARNING: GIMBAL.pitch is all zeros (data not logged or corrupted)")
            print("  Metric depth scaling will be skipped.")
            print("  Hint: use --gimbal-pitch to supply a manual gimbal angle (e.g. --gimbal-pitch -90 for nadir)")
            return None

    # Handle NaN values
    altitudes_raw = np.nan_to_num(altitudes_raw, nan=np.nanmean(altitudes_raw))
    pitch_gimbal_raw = np.nan_to_num(pitch_gimbal_raw, nan=0.0)
    roll_gimbal_raw = np.nan_to_num(roll_gimbal_raw, nan=0.0)
    yaw_raw = np.nan_to_num(yaw_raw, nan=0.0)

    # Calculate target frame count
    target_frame_count = int(video_duration_sec * target_fps)
    print(f"  Video duration: {video_duration_sec:.1f}s, Target frames: {target_frame_count}")
    print(f"  Flight record video segment: {len(video_df)} samples")

    # Resample telemetry to match target frame count
    if len(altitudes_raw) != target_frame_count:
        old_indices = np.linspace(0, len(altitudes_raw) - 1, len(altitudes_raw))
        new_indices = np.linspace(0, len(altitudes_raw) - 1, target_frame_count)

        altitudes = np.interp(new_indices, old_indices, altitudes_raw)
        pitch_gimbal = np.interp(new_indices, old_indices, pitch_gimbal_raw)
        roll_gimbal = np.interp(new_indices, old_indices, roll_gimbal_raw)
        yaw = np.interp(new_indices, old_indices, yaw_raw)

        timestamp_indices = np.linspace(0, len(timestamps_raw) - 1, target_frame_count).astype(int)
        timestamps = [timestamps_raw[i] for i in timestamp_indices]
        print(f"  Resampled {len(altitudes_raw)} → {target_frame_count} samples")
    else:
        altitudes = altitudes_raw
        pitch_gimbal = pitch_gimbal_raw
        roll_gimbal = roll_gimbal_raw
        yaw = yaw_raw
        timestamps = list(timestamps_raw)

    # Convert gimbal pitch to "pitch below horizontal"
    # DJI convention: 0° = horizontal, negative = pointing down
    # Our convention: 0° = horizon, 90° = nadir (straight down)
    pitch_below_horizontal = -pitch_gimbal
    # Clamp to [0, 90] but do NOT force a minimum — let compute_global_scale_factor
    # skip near-horizon frames itself rather than feeding them fake angles.
    pitch_below_horizontal = np.clip(pitch_below_horizontal, 0, 90)

    # Summary
    print(f"  Altitude range: [{altitudes.min():.1f}, {altitudes.max():.1f}] m (mean: {altitudes.mean():.1f} m)")
    print(f"  GIMBAL.pitch range: [{pitch_gimbal.min():.1f}, {pitch_gimbal.max():.1f}]°")
    print(f"  GIMBAL.roll range: [{roll_gimbal.min():.1f}, {roll_gimbal.max():.1f}]°")
    print(f"  Pitch below horizontal: [{pitch_below_horizontal.min():.1f}, {pitch_below_horizontal.max():.1f}]°")
    print(f"  Yaw range: [{yaw.min():.1f}, {yaw.max():.1f}]°")

    return {
        'altitudes': list(altitudes),
        'pitch_gimbal': list(pitch_gimbal),
        'roll_gimbal': list(roll_gimbal),
        'pitch_below_horizontal': list(pitch_below_horizontal),
        'yaw': list(yaw),
        'timestamps': timestamps
    }


def compute_global_scale_factor(canonical_depths: list,
                                 altitudes: list,
                                 pitch_angles: list,
                                 num_reference_frames: int = 10,
                                 min_pitch_from_horizon: float = 5.0) -> float:
    """
    Compute a global scale factor from the first N frames.

    This approach is more robust than per-frame scaling when there are elevated
    objects at the frame center. The assumption is that the first frames in the
    video segment are looking at flat ground without obstructions.

    Mathematical derivation:
        For each reference frame i:
            slant_range_i = altitude_i / sin(pitch_i)
            scale_factor_i = slant_range_i / d_center_canonical_i

        Global scale factor = median(scale_factor_1, ..., scale_factor_N)

    Args:
        canonical_depths: List of canonical depth maps [N × [H, W]]
        altitudes: List of altitude values in meters [N]
        pitch_angles: List of pitch angles below horizontal in degrees [N]
        num_reference_frames: Number of frames to use for computing global scale (default: 10)
        min_pitch_from_horizon: Minimum valid pitch angle (default: 5°)

    Returns:
        global_scale_factor: Single scale factor to apply to all frames

    Raises:
        ValueError: If any reference frame has invalid center depth
    """
    print(f"\n  Computing global scale factor (need {num_reference_frames} valid frames, "
          f"min pitch {min_pitch_from_horizon}°)...")

    scale_factors = []
    skipped = 0

    for i in range(len(canonical_depths)):
        if len(scale_factors) >= num_reference_frames:
            break

        depth = canonical_depths[i]
        altitude = altitudes[i]
        pitch = pitch_angles[i]

        # Skip frames with insufficient pitch (camera too close to horizontal)
        if pitch < min_pitch_from_horizon:
            skipped += 1
            continue

        if not 0 < pitch <= 90:
            skipped += 1
            continue

        if altitude <= 0:
            skipped += 1
            continue

        # Calculate real slant range
        pitch_rad = np.deg2rad(pitch)
        sin_pitch = np.sin(pitch_rad)
        real_slant_range = altitude / sin_pitch

        # Get canonical depth at principal point (image center)
        h, w = depth.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Use a small window around center for robustness against noise
        window_size = max(1, min(h, w) // 20)  # 5% of smaller dimension
        y_start = max(0, center_y - window_size)
        y_end = min(h, center_y + window_size + 1)
        x_start = max(0, center_x - window_size)
        x_end = min(w, center_x + window_size + 1)

        center_region = depth[y_start:y_end, x_start:x_end]
        d_center = np.median(center_region)

        # Skip frames with invalid center depth
        if d_center <= 0 or not np.isfinite(d_center):
            skipped += 1
            continue

        # Calculate scale factor for this frame
        scale_factor = real_slant_range / d_center
        scale_factors.append(scale_factor)

        print(f"    Frame {i}: alt={altitude:.1f}m, pitch={pitch:.1f}°, "
              f"slant={real_slant_range:.1f}m, d_center={d_center:.4f}, scale={scale_factor:.4f}")

    if skipped > 0:
        print(f"    (skipped {skipped} frames with insufficient pitch/altitude/depth)")

    if len(scale_factors) == 0:
        raise ValueError(
            "No valid reference frames found for global scale computation. "
            "All frames had pitch < {min_pitch_from_horizon}°, altitude <= 0, or invalid depth."
        )

    # Compute global scale factor as median of individual scale factors
    global_scale = float(np.median(scale_factors))

    scale_factors_arr = np.array(scale_factors)
    print(f"\n  Scale factor statistics (n={len(scale_factors)}):")
    print(f"    Min:    {scale_factors_arr.min():.4f}")
    print(f"    Max:    {scale_factors_arr.max():.4f}")
    print(f"    Mean:   {scale_factors_arr.mean():.4f}")
    print(f"    Median: {global_scale:.4f} (SELECTED)")
    print(f"    Std:    {scale_factors_arr.std():.4f}")

    return global_scale


def scale_depth_sequence_with_global_factor(canonical_depths: list,
                                             global_scale_factor: float,
                                             altitudes: list = None,
                                             pitch_angles: list = None,
                                             min_pitch_from_horizon: float = 5.0) -> tuple:
    """
    Convert a sequence of canonical depth maps to metric depth using per-frame
    scale factors derived from flight telemetry. Frames with invalid telemetry
    (low pitch or altitude) fall back to the global scale factor.

    Args:
        canonical_depths: List of 2D numpy arrays [N × [H, W]]
        global_scale_factor: Fallback scale factor for frames without valid telemetry
        altitudes: List of altitude values in metres [N] (or None for global-only)
        pitch_angles: List of pitch angles below horizontal in degrees [N] (or None)
        min_pitch_from_horizon: Minimum valid pitch angle for per-frame scaling

    Returns:
        (metric_depths, per_frame_scales): list of metric depth maps,
            list of per-frame scale factors used
    """
    per_frame = altitudes is not None and pitch_angles is not None
    mode = "per-frame (with global fallback)" if per_frame else "global constant"
    print(f"\n  Scaling {len(canonical_depths)} frames → metric depth ({mode})...")

    metric_depths = []
    per_frame_scales = []

    for i, depth in enumerate(canonical_depths):
        sf = global_scale_factor  # default fallback

        if per_frame and i < len(altitudes):
            alt = altitudes[i]
            pitch = pitch_angles[i]

            if alt > 0 and pitch >= min_pitch_from_horizon:
                pitch_rad = np.deg2rad(pitch)
                real_slant_range = alt / np.sin(pitch_rad)

                # Center depth of canonical map
                h, w = depth.shape[:2]
                cy, cx = h // 2, w // 2
                ws = max(1, min(h, w) // 20)
                center_region = depth[max(0, cy - ws):min(h, cy + ws + 1),
                                      max(0, cx - ws):min(w, cx + ws + 1)]
                d_center = float(np.median(center_region))

                if d_center > 1e-6 and np.isfinite(d_center):
                    sf = real_slant_range / d_center

        per_frame_scales.append(sf)
        metric_depths.append(depth * sf)

    # Report statistics
    sf_arr = np.array(per_frame_scales)
    all_metric = np.array(metric_depths)
    print(f"  Scale factor range: [{sf_arr.min():.4f}, {sf_arr.max():.4f}]  "
          f"(global fallback={global_scale_factor:.4f})")
    print(f"  Metric depth range: [{all_metric.min():.2f}, {all_metric.max():.2f}] m")

    return metric_depths, per_frame_scales


def get_video_info_ffprobe(video_path):
    """Get accurate frame count and duration using ffprobe.

    Returns (total_frames, duration, fps) or None if ffprobe is unavailable.
    """
    import subprocess
    import shutil

    if not shutil.which('ffprobe'):
        return None

    try:
        # Get accurate frame count by decoding
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_read_frames,r_frame_rate,duration',
             '-of', 'csv=p=0:s=,', video_path],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(',')
        if len(parts) < 2:
            return None

        # Parse r_frame_rate (e.g. "30000/1001")
        fps_str = parts[0]
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        # Duration may or may not be present in stream metadata
        duration = float(parts[1]) if len(parts) >= 2 and parts[1] != 'N/A' else None

        # nb_read_frames is the last field
        total_frames = int(parts[-1])

        # If stream duration wasn't available, compute from frames/fps
        if duration is None or duration <= 0:
            duration = total_frames / fps if fps > 0 else 0

        return total_frames, duration, fps

    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def extract_frames_from_video(video_path, fps=1):
    """Extract frames from video using ffmpeg for accurate frame timing.

    Uses ffmpeg's fps filter which handles variable frame rate videos correctly
    and avoids sync issues with OpenCV's CAP_PROP_FRAME_COUNT metadata.
    """
    import subprocess
    import shutil
    import tempfile

    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg for accurate frame extraction.")

    # Get video duration using ffprobe
    ffprobe_info = get_video_info_ffprobe(video_path)
    if ffprobe_info is not None:
        total_frames, video_duration, video_fps = ffprobe_info
        print(f"  Video info: {video_duration:.1f}s, {video_fps:.3f} fps, {total_frames} frames")
    else:
        # Fallback: get duration from ffmpeg
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, text=True, check=True, timeout=30
            )
            video_duration = float(result.stdout.strip())
            print(f"  Video duration: {video_duration:.1f}s (ffprobe)")
        except:
            print("  WARNING: Could not determine video duration. Proceeding anyway...")
            video_duration = None

    # Calculate expected frame count
    if video_duration:
        expected_frames = int(video_duration * fps)
        print(f"Target: {fps} fps → expecting ~{expected_frames} frames")
    else:
        print(f"Target: {fps} fps")

    # Create temporary directory for extracted frames
    tmp_dir = tempfile.mkdtemp(prefix="vggt_frames_")

    try:
        # Use ffmpeg to extract frames at specified fps
        # -vf fps=X ensures accurate temporal resampling (not metadata-based)
        output_pattern = os.path.join(tmp_dir, "frame_%06d.png")

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',  # Temporal filter (accurate!)
            '-q:v', '2',          # High quality
            '-start_number', '0',
            output_pattern
        ]

        print(f"  Extracting frames with ffmpeg (fps={fps})...")
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"  FFmpeg stderr: {result.stderr[-500:]}")  # Last 500 chars
            raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

        # Load extracted frames in order
        frame_files = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.png")))

        if len(frame_files) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path)
            frames.append(np.array(img))  # Already RGB from PNG

        actual_duration = len(frames) / fps
        print(f"Extracted {len(frames)} frames")
        print(f"Output video will be: {actual_duration:.1f}s at {fps} fps")

        return frames

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def extract_frames_from_video_streaming(video_path, fps=1):
    """Extract frames from video using ffmpeg for accurate frame timing.

    This is a compatibility wrapper that yields frames one at a time, but
    internally uses ffmpeg batch extraction to avoid OpenCV sync issues.

    Yields (extracted_idx, frame_rgb) tuples.
    """
    import subprocess
    import shutil
    import tempfile

    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg for accurate frame extraction.")

    # Get video duration using ffprobe
    ffprobe_info = get_video_info_ffprobe(video_path)
    if ffprobe_info is not None:
        total_frames, video_duration, video_fps = ffprobe_info
        print(f"  Video info: {video_duration:.1f}s, {video_fps:.3f} fps, {total_frames} frames")
    else:
        # Fallback: get duration from ffmpeg
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                capture_output=True, text=True, check=True, timeout=30
            )
            video_duration = float(result.stdout.strip())
            print(f"  Video duration: {video_duration:.1f}s (ffprobe)")
        except:
            print("  WARNING: Could not determine video duration. Proceeding anyway...")
            video_duration = None

    # Calculate expected frame count
    if video_duration:
        expected_frames = int(video_duration * fps)
        print(f"Target: {fps} fps → expecting ~{expected_frames} frames")
    else:
        print(f"Target: {fps} fps")

    # Create temporary directory for extracted frames
    tmp_dir = tempfile.mkdtemp(prefix="vggt_frames_stream_")

    try:
        # Use ffmpeg to extract frames at specified fps
        # -vf fps=X ensures accurate temporal resampling (not metadata-based)
        output_pattern = os.path.join(tmp_dir, "frame_%06d.png")

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',  # Temporal filter (accurate!)
            '-q:v', '2',          # High quality
            '-start_number', '0',
            output_pattern
        ]

        print(f"  Extracting frames with ffmpeg (fps={fps})...")
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"  FFmpeg stderr: {result.stderr[-500:]}")  # Last 500 chars
            raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")

        # Load and yield extracted frames in order
        frame_files = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.png")))

        if len(frame_files) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        for extracted_idx, frame_path in enumerate(frame_files):
            img = Image.open(frame_path)
            frame_rgb = np.array(img)  # Already RGB from PNG
            yield (extracted_idx, frame_rgb)

        print(f"Streamed {len(frame_files)} frames")

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


def save_depth_maps(depth_maps, output_dir):
    """Save depth maps as both .npy and visualization .png files."""
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    for i, depth in enumerate(depth_maps):
        depth_np = depth.cpu().numpy().squeeze()  # [H, W]

        # Save raw depth as numpy
        np.save(os.path.join(depth_dir, f"depth_{i:04d}.npy"), depth_np)

        # Save normalized visualization
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        depth_img = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(os.path.join(depth_dir, f"depth_{i:04d}.png"))

    print(f"Saved {len(depth_maps)} depth maps to {depth_dir}/")


def save_point_maps(point_maps, output_dir):
    """Save world point maps as .npy files."""
    points_dir = os.path.join(output_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i, points in enumerate(point_maps):
        points_np = points.cpu().numpy()  # [H, W, 3]
        np.save(os.path.join(points_dir, f"points_{i:04d}.npy"), points_np)

    print(f"Saved {len(point_maps)} point maps to {points_dir}/")


def save_cameras(predictions, output_dir):
    """Save camera intrinsics and extrinsics."""
    camera_dir = os.path.join(output_dir, "cameras")
    os.makedirs(camera_dir, exist_ok=True)

    if "extrinsic" in predictions:
        extrinsics = predictions["extrinsic"].cpu().numpy()
        np.save(os.path.join(camera_dir, "extrinsics.npy"), extrinsics)
        print(f"Saved extrinsics: {extrinsics.shape}")

    if "intrinsic" in predictions:
        intrinsics = predictions["intrinsic"].cpu().numpy()
        np.save(os.path.join(camera_dir, "intrinsics.npy"), intrinsics)
        print(f"Saved intrinsics: {intrinsics.shape}")


def create_depth_colormap(depth_np, colormap='turbo', vmin=None, vmax=None):
    """Convert depth map to colored visualization using matplotlib colormap.

    Args:
        depth_np: 2D numpy array of depth values
        colormap: Matplotlib colormap name
        vmin: Optional minimum value for normalization (global min)
        vmax: Optional maximum value for normalization (global max)

    Returns:
        depth_colored: Colorized depth image [H, W, 3]
        depth_min: Actual min value used for normalization
        depth_max: Actual max value used for normalization
    """
    # Use provided global range or fall back to per-frame range
    depth_min = vmin if vmin is not None else depth_np.min()
    depth_max = vmax if vmax is not None else depth_np.max()

    # Normalize depth to [0, 1] using the specified range
    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    # Clip to [0, 1] in case values fall outside the global range
    depth_normalized = np.clip(depth_normalized, 0, 1)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)

    return depth_colored, depth_min, depth_max


def create_colorbar(height, width=120, depth_min=0, depth_max=1, colormap='turbo'):
    """Create a vertical colorbar image with depth range labels."""
    # Create figure with colorbar and labels - wider figure for readable text
    fig_height = max(6, height / 60)
    fig = plt.figure(figsize=(2.5, fig_height))
    fig.patch.set_facecolor('black')

    # Create axes for colorbar with more space for labels
    ax = fig.add_axes([0.1, 0.08, 0.25, 0.84])  # [left, bottom, width, height]

    # Create colorbar
    norm = plt.Normalize(vmin=depth_min, vmax=depth_max)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=16, width=2, length=6)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(2)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=16)
    cbar.set_label('Depth', color='white', fontsize=18, labelpad=15)

    # Convert figure to image using buffer_rgba
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    colorbar_img = np.asarray(buf)[:, :, :3]  # Remove alpha channel
    plt.close(fig)

    # Resize to match target height
    colorbar_img = cv2.resize(colorbar_img, (width, height))

    return colorbar_img


def create_overlay_colorbar(height, width=40, depth_min=0, depth_max=1, colormap='turbo', margin=10):
    """Create a vertical colorbar image for overlay with transparent-friendly design."""
    # Create gradient
    gradient = np.linspace(1, 0, height - 2 * margin).reshape(-1, 1)
    gradient = np.repeat(gradient, width - 10, axis=1)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colorbar = cmap(gradient)[:, :, :3]
    colorbar = (colorbar * 255).astype(np.uint8)

    # Create background with more padding for labels (increased from 60 to 100)
    label_space = 100
    bg = np.zeros((height, width + label_space, 3), dtype=np.uint8)
    bg[:] = (0, 0, 0)  # Black background

    # Place colorbar
    bg[margin:margin + colorbar.shape[0], 5:5 + colorbar.shape[1]] = colorbar

    # Add border to colorbar
    cv2.rectangle(bg, (5, margin), (5 + colorbar.shape[1], margin + colorbar.shape[0]), (255, 255, 255), 1)

    # Add tick labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    label_x = width  # Position labels right after the colorbar

    # Top label (max depth)
    max_text = f"{depth_max:.1f}"
    cv2.putText(bg, max_text, (label_x, margin + 20), font, font_scale, (255, 255, 255), thickness)

    # Middle label
    mid_val = (depth_min + depth_max) / 2
    mid_text = f"{mid_val:.1f}"
    cv2.putText(bg, mid_text, (label_x, height // 2 + 8), font, font_scale, (255, 255, 255), thickness)

    # Bottom label (min depth)
    min_text = f"{depth_min:.1f}"
    cv2.putText(bg, min_text, (label_x, height - margin), font, font_scale, (255, 255, 255), thickness)

    return bg


def extract_center_depths(depth_maps):
    """Extract center pixel depth from each depth map."""
    center_depths = []
    for depth in depth_maps:
        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy().squeeze()
        else:
            depth_np = depth.squeeze() if hasattr(depth, 'squeeze') else depth

        h, w = depth_np.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Use small window for robustness
        window_size = max(1, min(h, w) // 40)
        y_start = max(0, center_y - window_size)
        y_end = min(h, center_y + window_size + 1)
        x_start = max(0, center_x - window_size)
        x_end = min(w, center_x + window_size + 1)

        center_region = depth_np[y_start:y_end, x_start:x_end]
        center_depth = np.median(center_region)
        center_depths.append(center_depth)

    return center_depths


def create_live_plot_frame(time_values, canonical_depths_center, current_idx, total_time,
                           target_height, target_width,
                           metric_depths_center=None):
    """
    Create a live 2D plot with dual y-axes: canonical depth (left) and metric depth (right).

    Args:
        time_values: List of time values (x-axis)
        canonical_depths_center: List of canonical center depth values (left y-axis)
        current_idx: Current frame index (plot up to this point)
        total_time: Total video duration in seconds (fixed x-axis limit)
        target_height: Target height for the output image
        target_width: Target width for the output image
        metric_depths_center: List of metric center depth values (right y-axis), or None

    Returns:
        numpy array of the plot image [H, W, 3]
    """
    fig, ax_canon = plt.subplots(figsize=(target_width / 100, target_height / 100), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    ax_canon.set_facecolor('#1a1a1a')

    plot_times = time_values[:current_idx + 1]

    # --- Left axis: canonical depth (blue) ---------------------------------
    canon_color = '#4da6ff'
    plot_canon = canonical_depths_center[:current_idx + 1]

    if len(plot_times) > 1:
        ax_canon.plot(plot_times, plot_canon, color=canon_color, linewidth=2, alpha=0.9, label='Canonical')
    if len(plot_times) > 0:
        ax_canon.scatter([plot_times[-1]], [plot_canon[-1]], color=canon_color, s=60, zorder=5)

    ax_canon.set_xlim(0, total_time)
    all_canon = np.array(canonical_depths_center)
    margin_c = (all_canon.max() - all_canon.min()) * 0.1 + 1e-6
    ax_canon.set_ylim(all_canon.min() - margin_c, all_canon.max() + margin_c)

    ax_canon.set_xlabel('Time (s)', color='white', fontsize=12)
    ax_canon.set_ylabel('Canonical Depth', color=canon_color, fontsize=12)
    ax_canon.tick_params(axis='y', colors=canon_color, labelsize=10)
    ax_canon.tick_params(axis='x', colors='white', labelsize=10)
    ax_canon.spines['left'].set_color(canon_color)
    ax_canon.spines['bottom'].set_color('white')
    ax_canon.spines['top'].set_color('#333333')
    ax_canon.spines['right'].set_color('#333333')
    ax_canon.grid(True, alpha=0.3, color='#444444')

    # --- Right axis: metric depth (red) ------------------------------------
    if metric_depths_center is not None:
        metric_color = '#ff4d4d'
        ax_metric = ax_canon.twinx()
        plot_metric = metric_depths_center[:current_idx + 1]

        if len(plot_times) > 1:
            ax_metric.plot(plot_times, plot_metric, color=metric_color, linewidth=2, alpha=0.9,
                           linestyle='--', label='Metric (m)')
        if len(plot_times) > 0:
            ax_metric.scatter([plot_times[-1]], [plot_metric[-1]], color=metric_color, s=60, zorder=5)

        all_metric = np.array(metric_depths_center)
        margin_m = (all_metric.max() - all_metric.min()) * 0.1 + 1e-6
        ax_metric.set_ylim(all_metric.min() - margin_m, all_metric.max() + margin_m)

        ax_metric.set_ylabel('Metric Depth (m)', color=metric_color, fontsize=12)
        ax_metric.tick_params(axis='y', colors=metric_color, labelsize=10)
        ax_metric.spines['right'].set_color(metric_color)

        # Combined legend
        lines_canon, labels_canon = ax_canon.get_legend_handles_labels()
        lines_metric, labels_metric = ax_metric.get_legend_handles_labels()
        ax_canon.legend(lines_canon + lines_metric, labels_canon + labels_metric,
                        loc='upper left', fontsize=9, facecolor='#333333', edgecolor='#555555',
                        labelcolor='white')

        title = 'Center Depth: Canonical vs Metric'
    else:
        title = 'Center Depth (Canonical)'

    ax_canon.set_title(title, color='white', fontsize=14, pad=10)

    # Annotation for current values
    if len(plot_times) > 0:
        ann_text = f't={plot_times[-1]:.2f}s\nc={plot_canon[-1]:.3f}'
        if metric_depths_center is not None:
            ann_text += f'\nm={plot_metric[-1]:.1f}m'
        ax_canon.annotate(ann_text,
                          xy=(plot_times[-1], plot_canon[-1]),
                          xytext=(10, 10), textcoords='offset points',
                          color='white', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    plot_img = np.asarray(buf)[:, :, :3]
    plt.close(fig)

    plot_img = cv2.resize(plot_img, (target_width, target_height))
    return plot_img


def create_quadrant_frame(rgb_frame, canonical_depth, metric_depth,
                          time_values, canonical_center_depths, current_idx,
                          total_time, colormap='turbo',
                          canonical_range=None, metric_range=None,
                          global_scale_factor=None,
                          metric_center_depths=None):
    """
    Create a quadrant video frame with:
    - Top-left: Input RGB
    - Top-right: Canonical depth (VGGT)
    - Bottom-left: Live plot (canonical + metric center depth, dual y-axes)
    - Bottom-right: Metric depth

    Args:
        rgb_frame: Original RGB frame [H, W, 3]
        canonical_depth: Canonical depth map from VGGT [H, W]
        metric_depth: Metric depth map in meters [H, W] (or None)
        time_values: List of all time values
        canonical_center_depths: List of canonical center depth values
        current_idx: Current frame index
        total_time: Total video duration
        colormap: Matplotlib colormap name
        canonical_range: (min, max) for canonical depth colorbar
        metric_range: (min, max) for metric depth colorbar
        global_scale_factor: Global scale factor used (for display)
        metric_center_depths: List of metric center depth values (or None)

    Returns:
        Combined quadrant frame [2H, 2W, 3]
    """
    h, w = rgb_frame.shape[:2]

    # Resize depth maps to match RGB dimensions
    if canonical_depth.shape[:2] != (h, w):
        canonical_depth = cv2.resize(canonical_depth, (w, h), interpolation=cv2.INTER_LINEAR)

    if metric_depth is not None and metric_depth.shape[:2] != (h, w):
        metric_depth = cv2.resize(metric_depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Get global ranges for consistent colormap normalization
    c_min, c_max = canonical_range if canonical_range else (None, None)
    m_min, m_max = metric_range if metric_range else (None, None)

    # Create colorized depth images using global ranges
    canonical_colored, c_min, c_max = create_depth_colormap(
        canonical_depth, colormap, vmin=c_min, vmax=c_max
    )

    if metric_depth is not None:
        metric_colored, m_min, m_max = create_depth_colormap(
            metric_depth, colormap, vmin=m_min, vmax=m_max
        )
    else:
        # If no metric depth, show placeholder
        metric_colored = np.zeros((h, w, 3), dtype=np.uint8)
        metric_colored[:] = (50, 50, 50)
        m_min, m_max = 0, 1

    # Overlay colorbars
    canonical_cb = create_overlay_colorbar(h, width=40, depth_min=c_min, depth_max=c_max, colormap=colormap)
    metric_cb = create_overlay_colorbar(h, width=40, depth_min=m_min, depth_max=m_max, colormap=colormap)

    # Overlay on depth images
    cb_h, cb_w = canonical_cb.shape[:2]
    x_offset = w - cb_w - 10

    # Canonical depth colorbar
    roi = canonical_colored[0:cb_h, x_offset:x_offset + cb_w]
    mask = (canonical_cb > 0).any(axis=2)
    alpha = 0.85
    roi[mask] = (alpha * canonical_cb[mask] + (1 - alpha) * roi[mask]).astype(np.uint8)

    # Metric depth colorbar
    if metric_depth is not None:
        roi = metric_colored[0:cb_h, x_offset:x_offset + cb_w]
        mask = (metric_cb > 0).any(axis=2)
        roi[mask] = (alpha * metric_cb[mask] + (1 - alpha) * roi[mask]).astype(np.uint8)

    # Create live plot (dual y-axes: canonical left, metric right)
    plot_frame = create_live_plot_frame(
        time_values, canonical_center_depths, current_idx, total_time,
        target_height=h, target_width=w,
        metric_depths_center=metric_center_depths
    )

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    rgb_labeled = rgb_frame.copy()
    cv2.putText(rgb_labeled, 'Input RGB', (10, 30), font, font_scale, (255, 255, 255), thickness)

    canonical_labeled = canonical_colored.copy()
    cv2.putText(canonical_labeled, 'Canonical Depth', (10, 30), font, font_scale, (255, 255, 255), thickness)

    metric_labeled = metric_colored.copy()
    if metric_depth is not None:
        label = 'Metric Depth (m)'
        if global_scale_factor is not None:
            label += f' [scale={global_scale_factor:.2f}]'
        cv2.putText(metric_labeled, label, (10, 30), font, font_scale, (255, 255, 255), thickness)
    else:
        cv2.putText(metric_labeled, 'Metric Depth (N/A)', (10, 30), font, font_scale, (128, 128, 128), thickness)

    # Combine into quadrant layout
    top_row = np.hstack([rgb_labeled, canonical_labeled])
    bottom_row = np.hstack([plot_frame, metric_labeled])
    quadrant = np.vstack([top_row, bottom_row])

    return quadrant


def encode_quadrant_video(frames, canonical_depths, metric_depths, output_path,
                          fps=10, colormap='turbo', global_scale_factor=None):
    """
    Encode quadrant analysis video with live plot.

    Args:
        frames: List of RGB frames
        canonical_depths: List of canonical depth maps from VGGT
        metric_depths: List of metric depth maps (or None)
        output_path: Output video file path
        fps: Frames per second
        colormap: Matplotlib colormap name
        global_scale_factor: Global scale factor used (for display)
    """
    import subprocess
    import shutil

    if len(frames) == 0 or len(canonical_depths) == 0:
        raise ValueError("No frames or depth maps to encode")

    # Extract center depths for dual y-axis plot
    print("  Extracting center pixel depths...")
    canonical_center_depths = extract_center_depths(canonical_depths)
    metric_center_depths = extract_center_depths(metric_depths) if metric_depths is not None else None

    # Calculate time values
    total_time = len(frames) / fps
    time_values = [i / fps for i in range(len(frames))]

    # Calculate global ranges for consistent colorbars
    all_canonical = np.array([d.squeeze() if hasattr(d, 'squeeze') else d for d in canonical_depths])
    canonical_range = (float(all_canonical.min()), float(all_canonical.max()))

    metric_range = None
    if metric_depths is not None:
        all_metric = np.array([d.squeeze() if hasattr(d, 'squeeze') else d for d in metric_depths])
        metric_range = (float(all_metric.min()), float(all_metric.max()))

    # Get dimensions from first quadrant frame
    first_canonical = canonical_depths[0]
    if isinstance(first_canonical, torch.Tensor):
        first_canonical = first_canonical.cpu().numpy().squeeze()
    first_metric = metric_depths[0] if metric_depths else None

    first_frame = create_quadrant_frame(
        frames[0], first_canonical, first_metric,
        time_values, canonical_center_depths, 0, total_time,
        colormap, canonical_range, metric_range, global_scale_factor,
        metric_center_depths=metric_center_depths
    )
    h, w = first_frame.shape[:2]

    # Check ffmpeg availability
    ffmpeg_available = shutil.which('ffmpeg') is not None

    if ffmpeg_available:
        print(f"Encoding quadrant video with H.264 (ffmpeg): {len(frames)} frames at {fps} fps")
        print(f"  Output resolution: {w}x{h}")

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for i in range(len(frames)):
            canonical = canonical_depths[i]
            if isinstance(canonical, torch.Tensor):
                canonical = canonical.cpu().numpy().squeeze()

            metric = None
            if metric_depths:
                metric = metric_depths[i]
                if isinstance(metric, torch.Tensor):
                    metric = metric.cpu().numpy().squeeze()

            quadrant = create_quadrant_frame(
                frames[i], canonical, metric,
                time_values, canonical_center_depths, i, total_time,
                colormap, canonical_range, metric_range, global_scale_factor,
                metric_center_depths=metric_center_depths
            )

            quadrant_bgr = cv2.cvtColor(quadrant, cv2.COLOR_RGB2BGR)
            process.stdin.write(quadrant_bgr.tobytes())

            if (i + 1) % 10 == 0:
                print(f"    Encoded {i + 1}/{len(frames)} frames")

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            print(f"  FFmpeg error: {process.stderr.read().decode()}")
        else:
            print(f"Quadrant video saved to: {output_path}")
    else:
        print(f"Encoding quadrant video with OpenCV: {len(frames)} frames at {fps} fps")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for i in range(len(frames)):
            canonical = canonical_depths[i]
            if isinstance(canonical, torch.Tensor):
                canonical = canonical.cpu().numpy().squeeze()

            metric = None
            if metric_depths:
                metric = metric_depths[i]
                if isinstance(metric, torch.Tensor):
                    metric = metric.cpu().numpy().squeeze()

            quadrant = create_quadrant_frame(
                frames[i], canonical, metric,
                time_values, canonical_center_depths, i, total_time,
                colormap, canonical_range, metric_range, global_scale_factor,
                metric_center_depths=metric_center_depths
            )

            quadrant_bgr = cv2.cvtColor(quadrant, cv2.COLOR_RGB2BGR)
            out.write(quadrant_bgr)

            if (i + 1) % 10 == 0:
                print(f"    Encoded {i + 1}/{len(frames)} frames")

        out.release()
        print(f"Quadrant video saved to: {output_path}")
        print("  Note: Video encoded as MPEG-4. Install ffmpeg for H.264 support.")


def create_side_by_side_frame(original_frame, depth_map, colormap='turbo', add_colorbar=True,
                               overlay_colorbar=True, vmin=None, vmax=None):
    """Create a side-by-side visualization of original frame and depth map.

    Args:
        original_frame: RGB frame [H, W, 3]
        depth_map: Depth map [H, W]
        colormap: Matplotlib colormap name
        add_colorbar: Whether to add colorbar
        overlay_colorbar: Whether to overlay colorbar on depth image
        vmin: Optional global minimum for normalization
        vmax: Optional global maximum for normalization
    """
    h, w = original_frame.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]

    # Resize depth to match original frame size
    if (depth_h, depth_w) != (h, w):
        depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_map

    # Create colored depth visualization with global normalization
    depth_colored, depth_min, depth_max = create_depth_colormap(
        depth_resized, colormap, vmin=vmin, vmax=vmax
    )

    # Overlay colorbar on depth map
    if add_colorbar and overlay_colorbar:
        colorbar = create_overlay_colorbar(h, width=40, depth_min=depth_min, depth_max=depth_max, colormap=colormap)
        # Overlay on right side of depth image
        cb_h, cb_w = colorbar.shape[:2]
        x_offset = w - cb_w - 10  # 10px from right edge
        y_offset = 0

        # Blend colorbar onto depth (semi-transparent background)
        roi = depth_colored[y_offset:y_offset + cb_h, x_offset:x_offset + cb_w]
        mask = (colorbar > 0).any(axis=2)
        alpha = 0.8
        roi[mask] = (alpha * colorbar[mask] + (1 - alpha) * roi[mask]).astype(np.uint8)

    # Create side-by-side image
    combined = np.hstack([original_frame, depth_colored])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Depth (m)', (w + 10, 30), font, 1, (255, 255, 255), 2)

    return combined


def encode_depth_video(frames, depth_maps, output_path, fps=10, colormap='turbo'):
    """Encode original frames and depth maps into a side-by-side video with colorbar."""
    import subprocess
    import shutil

    if len(frames) == 0 or len(depth_maps) == 0:
        raise ValueError("No frames or depth maps to encode")

    # Calculate global min/max for consistent normalization across all frames
    print("  Calculating global depth range for consistent colormap...")
    all_depths = np.array([d.squeeze() if hasattr(d, 'squeeze') else d for d in depth_maps])
    global_min = float(all_depths.min())
    global_max = float(all_depths.max())
    print(f"  Global depth range: [{global_min:.2f}, {global_max:.2f}] m")

    # Get dimensions from first combined frame
    first_combined = create_side_by_side_frame(frames[0], depth_maps[0], colormap,
                                                vmin=global_min, vmax=global_max)
    h, w = first_combined.shape[:2]

    # Check if ffmpeg is available for H.264 encoding
    ffmpeg_available = shutil.which('ffmpeg') is not None

    if ffmpeg_available:
        # Use ffmpeg pipe for H.264 encoding
        print(f"Encoding video with H.264 (ffmpeg): {len(frames)} frames at {fps} fps")

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
            combined = create_side_by_side_frame(frame, depth, colormap,
                                                  vmin=global_min, vmax=global_max)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            process.stdin.write(combined_bgr.tobytes())

            if (i + 1) % 10 == 0:
                print(f"  Encoded {i + 1}/{len(frames)} frames")

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            print(f"  FFmpeg error: {process.stderr.read().decode()}")
        else:
            print(f"Video saved to: {output_path}")
    else:
        # Fallback to OpenCV
        print(f"Encoding video with OpenCV: {len(frames)} frames at {fps} fps")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
            combined = create_side_by_side_frame(frame, depth, colormap,
                                                  vmin=global_min, vmax=global_max)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            out.write(combined_bgr)

            if (i + 1) % 10 == 0:
                print(f"  Encoded {i + 1}/{len(frames)} frames")

        out.release()
        print(f"Video saved to: {output_path}")
        print("  Note: Video encoded as MPEG-4. Install ffmpeg for H.264 support.")


# ===========================================================================
# Phase 1: segment – extract frames from video, write PNGs to segment dirs
# ===========================================================================

def _output_dir_for_video(output_dir, video_path):
    """Append video stem to output_dir: outputs/ + video.MP4 → outputs/video/"""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, stem)


def phase_segment(video_path, output_dir, fps=10.0, segment_size=600, overlap_frames=10):
    """Extract frames from video, organize into overlapping segment directories.

    Communication contract: writes segments_meta.json under {output_dir}/segments/.
    Each segment dir contains self-contained PNGs (overlap frames duplicated).
    """
    # Auto-derive output subdir from video name
    output_dir = _output_dir_for_video(output_dir, video_path)
    print(f"Output directory: {output_dir}")

    # Skip if segments already exist
    meta_file = _meta_path(output_dir)
    if os.path.exists(meta_file):
        meta = _load_meta(output_dir)
        print(f"  Segments already exist ({meta['num_segments']} segment(s), "
              f"{meta['num_extracted_frames']} frames). Skipping phase 1.")
        return meta

    total_start = time.time()
    segments_dir = os.path.join(output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    if overlap_frames >= segment_size:
        raise ValueError(f"overlap_frames ({overlap_frames}) must be < segment_size ({segment_size})")
    stride = segment_size - overlap_frames

    # --- Stream frames from video and write to a flat temp area first ------
    # We need total extracted count to compute segment ranges, but we also
    # want to stream. Strategy: stream once → write PNGs to a flat staging
    # area, then hard-link/copy into segment dirs.
    print(f"\nExtracting frames from: {video_path}")

    flat_dir = os.path.join(segments_dir, "_flat_frames")
    os.makedirs(flat_dir, exist_ok=True)

    extracted_count = 0
    for extracted_idx, frame_rgb in extract_frames_from_video_streaming(video_path, fps=fps):
        frame_path = os.path.join(flat_dir, f"{extracted_idx:06d}.png")
        Image.fromarray(frame_rgb).save(frame_path)
        extracted_count += 1
        if extracted_count % 50 == 0:
            print(f"  Written {extracted_count} frames to disk...")

    num_frames = extracted_count
    print(f"  Total extracted frames: {num_frames}")

    if num_frames == 0:
        raise ValueError("No frames extracted from video")

    # --- Compute segment ranges --------------------------------------------
    segment_ranges = []
    start = 0
    while start < num_frames:
        end = min(start + segment_size, num_frames)
        segment_ranges.append((start, end))
        if end == num_frames:
            break
        start += stride

    print(f"\n  Segment plan: {len(segment_ranges)} segment(s), "
          f"segment_size={segment_size}, overlap_frames={overlap_frames}, stride={stride}")
    for i, (s, e) in enumerate(segment_ranges):
        print(f"    Segment {i}: frames [{s}, {e})  ({e - s} frames)")

    # --- Populate segment dirs (copy from flat staging) --------------------
    # Build frame_to_segments mapping
    frame_to_segments = {}
    for seg_idx, (seg_start, seg_end) in enumerate(segment_ranges):
        seg_dir = os.path.join(segments_dir, f"seg{seg_idx:05d}", "frames")
        os.makedirs(seg_dir, exist_ok=True)

        for local_idx, global_idx in enumerate(range(seg_start, seg_end)):
            src = os.path.join(flat_dir, f"{global_idx:06d}.png")
            dst = os.path.join(seg_dir, f"{local_idx:06d}.png")
            # Use hard link if possible, else copy
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)

            frame_to_segments.setdefault(global_idx, []).append(
                [seg_idx, local_idx]
            )

    # --- Write meta.json ---------------------------------------------------
    meta = {
        "version": 1,
        "created": datetime.datetime.now().isoformat(),
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "segment_size": segment_size,
        "overlap_frames": overlap_frames,
        "stride": stride,
        "num_extracted_frames": num_frames,
        "num_segments": len(segment_ranges),
        "segments": [
            {
                "seg_idx": i,
                "start": s,
                "end": e,
                "num_frames": e - s,
                "dir": f"seg{i:05d}",
                "inference": None,   # filled by phase_infer
            }
            for i, (s, e) in enumerate(segment_ranges)
        ],
        "frame_to_segments": {str(k): v for k, v in frame_to_segments.items()},
    }
    _save_meta(output_dir, meta)

    elapsed = time.time() - total_start
    print(f"\n  Phase 1 (segment) complete: {num_frames} frames, "
          f"{len(segment_ranges)} segments in {elapsed:.1f}s")
    print(f"  Meta written to: {_meta_path(output_dir)}")
    return meta


# ===========================================================================
# Phase 2: infer – load model, run VGGT on each segment, save .npy outputs
# ===========================================================================

def _run_segment_subprocess(seg_dir, model_name="facebook/VGGT-1B",
                            batch_size=20, save_world_points=False):
    """Spawn a subprocess to run VGGT inference on a single segment.

    Uses sys.executable so the subprocess runs in the same Python/venv.
    Returns True on success, False on failure.
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "scripts", "infer_segment.py")

    cmd = [sys.executable, script_path,
           "--seg-dir", seg_dir,
           "--model-name", model_name,
           "--batch-size", str(batch_size)]
    if save_world_points:
        cmd.append("--save-world-points")

    print(f"    Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=None,              # real-time stdout
            stderr=subprocess.PIPE,   # capture stderr for diagnostics
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: subprocess exited with code {e.returncode}")
        if e.stderr:
            # Show last few lines of stderr (skip traceback boilerplate)
            lines = e.stderr.strip().splitlines()
            for line in lines[-5:]:
                print(f"    | {line}")
        return False


def phase_infer(output_dir, model_name="facebook/VGGT-1B",
                device=None, dtype=None, reload_model=False,
                no_subprocess=False, batch_size=20):
    """Load model and run inference on each segment discovered via meta.json.

    Saves depth.npy, pose_enc.npy (and optionally world_points.npy) per segment.
    Supports resume: skips segments already marked as inferred.

    By default, each segment is inferred in a fresh subprocess so that GPU
    memory is fully reclaimed between segments (no fragmentation / leaks).
    Use no_subprocess=True for the original in-process behavior.
    """
    meta = _load_meta(output_dir)
    segments_dir = os.path.join(output_dir, "segments")
    num_segments = meta["num_segments"]

    mode_label = "in-process" if no_subprocess else "subprocess"
    print(f"\nPhase 2 (infer): {num_segments} segment(s), mode={mode_label}")

    # --- Extra resume check: outputs exist but meta not updated (crash recovery)
    recovered = False
    for seg_info in meta["segments"]:
        if seg_info.get("inference") is None:
            seg_dir = os.path.join(segments_dir, seg_info["dir"])
            depth_path = os.path.join(seg_dir, "depth.npy")
            pose_path = os.path.join(seg_dir, "pose_enc.npy")
            if os.path.exists(depth_path) and os.path.exists(pose_path):
                print(f"  Segment {seg_info['seg_idx']}: outputs exist but meta not updated — marking done.")
                seg_info["inference"] = "done"
                recovered = True
    if recovered:
        _save_meta(output_dir, meta)

    # ----- Subprocess mode (default) ----------------------------------------
    if not no_subprocess:
        for seg_info in meta["segments"]:
            seg_idx = seg_info["seg_idx"]
            seg_dir = os.path.join(segments_dir, seg_info["dir"])

            if seg_info.get("inference") == "done":
                print(f"  Segment {seg_idx}: already inferred, skipping.")
                continue

            print(f"\n  Segment {seg_idx}/{num_segments - 1}: "
                  f"frames [{seg_info['start']}, {seg_info['end']})  "
                  f"({seg_info['num_frames']} frames)")

            ok = _run_segment_subprocess(seg_dir, model_name=model_name,
                                         batch_size=batch_size)
            if not ok:
                seg_info["inference"] = "failed"
                _save_meta(output_dir, meta)
                print(f"    Segment {seg_idx}: marked as FAILED")
                continue

            # Verify outputs exist
            depth_path = os.path.join(seg_dir, "depth.npy")
            pose_path = os.path.join(seg_dir, "pose_enc.npy")
            if not os.path.exists(depth_path) or not os.path.exists(pose_path):
                seg_info["inference"] = "failed"
                _save_meta(output_dir, meta)
                print(f"    Segment {seg_idx}: output files missing — marked as FAILED")
                continue

            seg_info["inference"] = "done"
            _save_meta(output_dir, meta)

        print(f"\n  Phase 2 (infer) complete: all {num_segments} segment(s) processed.")
        return

    # ----- In-process mode (--no-subprocess) --------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = (torch.bfloat16
                 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)

    print(f"  In-process mode: device={device}, dtype={dtype}")

    # Load model once (unless reload_model is set)
    model = None
    if not reload_model:
        print(f"  Loading model to {device}...")
        tic_model = time.time()
        model = VGGT.from_pretrained(model_name).to(device)
        model.eval()
        print(f"  Model loaded in {time.time() - tic_model:.1f}s")

    for seg_info in meta["segments"]:
        seg_idx = seg_info["seg_idx"]
        seg_dir_name = seg_info["dir"]
        seg_dir = os.path.join(segments_dir, seg_dir_name)

        # Resume support: skip already-inferred segments
        if seg_info.get("inference") == "done":
            print(f"  Segment {seg_idx}: already inferred, skipping.")
            continue

        print(f"\n  Segment {seg_idx}/{num_segments - 1}: "
              f"frames [{seg_info['start']}, {seg_info['end']})  ({seg_info['num_frames']} frames)")

        # Collect sorted frame PNGs
        frame_dir = os.path.join(seg_dir, "frames")
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if len(frame_paths) == 0:
            raise ValueError(f"No PNGs found in {frame_dir}")

        # Load model per segment if reload_model
        if reload_model:
            print(f"    Loading model to {device}...")
            model = VGGT.from_pretrained(model_name).to(device)
            model.eval()

        # Preprocess (keep on CPU) & infer in mini-batches
        seg_images = load_and_preprocess_images(frame_paths)
        print(f"    Preprocessed: {seg_images.shape}")

        num_frames_seg = seg_images.shape[0]
        num_batches = (num_frames_seg + batch_size - 1) // batch_size
        all_depths = []
        all_pose_encs = []

        for bi in range(num_batches):
            s = bi * batch_size
            e = min(s + batch_size, num_frames_seg)
            batch_imgs = seg_images[s:e].to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    preds = model(batch_imgs)

            all_depths.append(preds["depth"].cpu())
            all_pose_encs.append(preds["pose_enc"].cpu())
            del preds, batch_imgs
            if device == "cuda":
                torch.cuda.empty_cache()

        depth_seg = torch.cat(all_depths, dim=1).squeeze(0).numpy().squeeze(-1)  # [S,H,W]
        pose_enc_seg = torch.cat(all_pose_encs, dim=1).squeeze(0).numpy()        # [S,9]

        np.save(os.path.join(seg_dir, "depth.npy"), depth_seg)
        np.save(os.path.join(seg_dir, "pose_enc.npy"), pose_enc_seg)

        del seg_images

        # Unload model if reload_model
        if reload_model:
            del model
            model = None
            if device == "cuda":
                torch.cuda.empty_cache()

        # Update meta
        seg_info["inference"] = "done"
        _save_meta(output_dir, meta)
        print(f"    Saved depth.npy {depth_seg.shape}, pose_enc.npy {pose_enc_seg.shape}")

    # Cleanup model
    if model is not None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n  Phase 2 (infer) complete: all {num_segments} segment(s) inferred.")


# ===========================================================================
# Phase 3: globalise – align segments, metric scale, encode videos
# ===========================================================================

def phase_globalise(output_dir, flight_record=None, fps=10.0,
                    num_reference_frames=10, save_points=False,
                    save_cameras_flag=False, save_video=True,
                    quadrant_video=False, colormap="turbo",
                    use_gimbal_yaw=False, no_video=False,
                    gimbal_pitch_override=None):
    """Align per-segment depth maps, compute metric scale, encode output videos.

    Reads segments_meta.json and per-segment .npy files written by phase_infer.
    Writes final outputs (depth/, depth_metric/, cameras/, videos) to output_dir.
    """
    total_start = time.time()
    meta = _load_meta(output_dir)
    segments_dir = os.path.join(output_dir, "segments")
    num_segments = meta["num_segments"]
    overlap_frames = meta["overlap_frames"]

    # Use fps from meta (authoritative, set during extraction) over CLI arg
    fps = meta.get("fps", fps)

    print(f"\nPhase 3 (globalise): {num_segments} segment(s), overlap={overlap_frames}, fps={fps}")

    # --- Overlap alignment (streaming: 2 segments at a time) ---------------
    stitched_depth = []       # list of [H,W] arrays
    stitched_pose_enc = []    # list of [9] arrays

    global_scale = 1.0
    prev_overlap_depth = None  # [O,H,W]

    # Guard: refuse to globalise if any segment failed inference
    failed = [s for s in meta["segments"] if s.get("inference") == "failed"]
    if failed:
        idxs = ", ".join(str(s["seg_idx"]) for s in failed)
        raise RuntimeError(
            f"Cannot globalise: segment(s) {idxs} have inference='failed'. "
            f"Re-run phase 2 (infer) to retry, or remove the failed segments.")

    for seg_info in meta["segments"]:
        seg_idx = seg_info["seg_idx"]
        seg_dir = os.path.join(segments_dir, seg_info["dir"])

        depth_seg = np.load(os.path.join(seg_dir, "depth.npy"))    # [S,H,W]
        pose_enc_seg = np.load(os.path.join(seg_dir, "pose_enc.npy"))  # [S,9]

        # --- Scale alignment -------------------------------------------------
        if seg_idx == 0:
            scaled_depth = depth_seg.copy()
        else:
            actual_overlap = min(overlap_frames, depth_seg.shape[0])
            overlap_curr = depth_seg[:actual_overlap]

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = prev_overlap_depth / overlap_curr

            valid = (
                np.isfinite(ratio) &
                (ratio > 0) &
                (overlap_curr > 1e-6) &
                (prev_overlap_depth > 1e-6)
            )

            if valid.sum() > 0:
                lam = float(np.nanmedian(ratio[valid]))
            else:
                print(f"    WARNING: no valid overlap pixels – using lambda=1.0")
                lam = 1.0

            # lam already IS the absolute scale because prev_overlap_depth
            # was stored in scaled (global) coordinates.  Do NOT compound.
            global_scale = lam
            scaled_depth = depth_seg * global_scale
            print(f"    Segment {seg_idx}: lambda={lam:.6f}, global_scale={global_scale:.6f}, "
                  f"valid_pixels={valid.sum()}/{ratio.size}")

        # --- Determine kept frames (de-duplicate overlap) --------------------
        keep_start = 0 if seg_idx == 0 else overlap_frames

        for f in range(keep_start, scaled_depth.shape[0]):
            stitched_depth.append(scaled_depth[f])
        for f in range(keep_start, pose_enc_seg.shape[0]):
            stitched_pose_enc.append(pose_enc_seg[f])

        # Save overlap for next segment
        if overlap_frames > 0 and seg_idx < num_segments - 1:
            prev_overlap_depth = scaled_depth[-overlap_frames:].copy()

        print(f"    Segment {seg_idx}: kept {scaled_depth.shape[0] - keep_start} frames "
              f"(total: {len(stitched_depth)})")

    # Stack into arrays
    all_depth = np.stack(stitched_depth, axis=0)       # [N,H,W]
    all_pose_enc = np.stack(stitched_pose_enc, axis=0) # [N,9]
    print(f"\n  Stitching complete: {all_depth.shape[0]} frames, global_scale={global_scale:.6f}")

    # --- Save aligned canonical depth maps ---------------------------------
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for i in range(all_depth.shape[0]):
        np.save(os.path.join(depth_dir, f"depth_{i:04d}.npy"), all_depth[i])
        # Visualization
        d = all_depth[i]
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d_img = (d_norm * 255).astype(np.uint8)
        Image.fromarray(d_img).save(os.path.join(depth_dir, f"depth_{i:04d}.png"))
    print(f"  Saved {all_depth.shape[0]} depth maps to {depth_dir}/")

    # --- Pose decoding -----------------------------------------------------
    pose_enc_t = torch.from_numpy(all_pose_enc).unsqueeze(0)   # [1,N,9]
    image_size_hw = (all_depth.shape[1], all_depth.shape[2])
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc_t, image_size_hw=image_size_hw, pose_encoding_type="absT_quaR_FoV"
    )
    extrinsics = extrinsics.squeeze(0)  # [N,3,4]
    intrinsics = intrinsics.squeeze(0)  # [N,3,3]
    print(f"  Decoded camera params: extrinsics {extrinsics.shape}, intrinsics {intrinsics.shape}")

    # --- Save cameras if requested -----------------------------------------
    if save_cameras_flag:
        cam_dir = os.path.join(output_dir, "cameras")
        os.makedirs(cam_dir, exist_ok=True)
        np.save(os.path.join(cam_dir, "extrinsics.npy"), extrinsics.cpu().numpy())
        np.save(os.path.join(cam_dir, "intrinsics.npy"), intrinsics.cpu().numpy())
        print(f"  Saved camera parameters to {cam_dir}/")

    # --- Metric scaling (if flight record) ---------------------------------
    global_scale_factor = None
    metric_depth_maps = None
    if flight_record:
        print("\n" + "=" * 60)
        print("GLOBAL SCALE FACTOR COMPUTATION")
        print("=" * 60)

        video_duration = all_depth.shape[0] / fps
        telemetry = parse_dji_flight_record(
            flight_record,
            video_duration_sec=video_duration,
            target_fps=fps,
            use_gimbal_yaw=use_gimbal_yaw,
            gimbal_pitch_override=gimbal_pitch_override
        )

        if telemetry is not None:
            depth_maps_list = [all_depth[i] for i in range(all_depth.shape[0])]

            global_scale_factor = compute_global_scale_factor(
                canonical_depths=depth_maps_list,
                altitudes=telemetry['altitudes'],
                pitch_angles=telemetry['pitch_below_horizontal'],
                num_reference_frames=num_reference_frames
            )

            metric_depth_maps, per_frame_scales = scale_depth_sequence_with_global_factor(
                canonical_depths=depth_maps_list,
                global_scale_factor=global_scale_factor,
                altitudes=telemetry['altitudes'],
                pitch_angles=telemetry['pitch_below_horizontal']
            )

            # Save metric depth
            metric_dir = os.path.join(output_dir, "depth_metric")
            os.makedirs(metric_dir, exist_ok=True)
            for i, md in enumerate(metric_depth_maps):
                np.save(os.path.join(metric_dir, f"depth_metric_{i:04d}.npy"), md)
            print(f"  Saved {len(metric_depth_maps)} metric depth maps to {metric_dir}/")

            scale_info = {
                'global_scale_factor': global_scale_factor,
                'num_reference_frames': min(num_reference_frames, all_depth.shape[0])
            }
            np.save(os.path.join(metric_dir, "global_scale_factor.npy"), scale_info)
            print(f"  Global scale factor: {global_scale_factor:.4f}")
            print("=" * 60)
        else:
            print("  Skipping metric scaling due to invalid gimbal data.")

    # --- Load RGB frames for video encoding --------------------------------
    # We need de-duplicated RGB frames in order. Use frame_to_segments mapping
    # to load each global frame exactly once from its first segment.
    rgb_frames = None
    need_rgb = (save_video and not no_video) or quadrant_video
    if need_rgb:
        print("\n  Loading RGB frames for video encoding...")
        num_total = meta["num_extracted_frames"]
        frame_to_seg = meta["frame_to_segments"]
        rgb_frames = []
        for global_idx in range(num_total):
            mappings = frame_to_seg[str(global_idx)]
            # Use first segment that contains this frame
            seg_idx_first, local_idx = mappings[0]
            seg_dir_name = meta["segments"][seg_idx_first]["dir"]
            png_path = os.path.join(segments_dir, seg_dir_name, "frames", f"{local_idx:06d}.png")
            img = np.array(Image.open(png_path))
            rgb_frames.append(img)

        # But we only keep de-duplicated count (same as stitched_depth)
        # The frame_to_segments includes ALL extracted frames, but stitching
        # de-duplicates overlap. We need exactly all_depth.shape[0] frames.
        # Reconstruct the de-duplicated global indices:
        dedup_indices = []
        for seg_info_item in meta["segments"]:
            si = seg_info_item["seg_idx"]
            s, e = seg_info_item["start"], seg_info_item["end"]
            keep_start_idx = 0 if si == 0 else overlap_frames
            for gi in range(s + keep_start_idx, e):
                dedup_indices.append(gi)

        rgb_frames = [rgb_frames[gi] for gi in dedup_indices]
        assert len(rgb_frames) == all_depth.shape[0], (
            f"RGB frame count ({len(rgb_frames)}) != depth count ({all_depth.shape[0]})"
        )
        print(f"  Loaded {len(rgb_frames)} de-duplicated RGB frames")

    # --- Encode depth video ------------------------------------------------
    if save_video and not no_video and rgb_frames is not None:
        print("\nEncoding depth video...")
        tic = time.time()
        video_output_path = os.path.join(output_dir, "depth_video.mp4")

        if metric_depth_maps is not None:
            depth_for_video = metric_depth_maps
            print("  Using metric depth values (meters) with GLOBAL scale factor")
        else:
            depth_for_video = [all_depth[i] for i in range(all_depth.shape[0])]
            print("  Using canonical depth values (no flight record)")

        encode_depth_video(rgb_frames, depth_for_video, video_output_path,
                           fps=fps, colormap=colormap)
        print(f"  Video encoding time: {time.time() - tic:.2f}s")

    # --- Encode quadrant video ---------------------------------------------
    if quadrant_video and rgb_frames is not None:
        print("\nEncoding quadrant analysis video...")
        tic = time.time()
        quadrant_output_path = os.path.join(output_dir, "quadrant_analysis.mp4")

        canonical_depths_list = [all_depth[i] for i in range(all_depth.shape[0])]

        encode_quadrant_video(
            rgb_frames, canonical_depths_list, metric_depth_maps,
            quadrant_output_path, fps=fps, colormap=colormap,
            global_scale_factor=global_scale_factor
        )
        print(f"  Quadrant video encoding time: {time.time() - tic:.2f}s")

    # --- Summary -----------------------------------------------------------
    elapsed = time.time() - total_start
    print(f"\n{'=' * 50}")
    print(f"Phase 3 (globalise) complete in {elapsed:.1f}s")
    if global_scale_factor is not None:
        print(f"Global scale factor: {global_scale_factor:.4f}")
    print(f"{'=' * 50}")


def phase_state(output_dir, tracking_json, flight_record, fps,
                source_res=(1280, 720),
                kf_sigma_a=0.5, kf_sigma_meas_h=5.0, kf_sigma_meas_v=2.0,
                yaw_offset=0.0, magnetic_declination=0.0,
                add_drone_yaw=False, use_gimbal_yaw=False,
                timezone="UTC",
                hdbscan_min_cluster_size=10, hdbscan_min_samples=3,
                hdbscan_coherence_weight=10.0, hdbscan_max_speed_mps=2.0,
                hdbscan_cluster_selection_epsilon=2.0,
                hdbscan_max_match_dist=20.0, hdbscan_ema_alpha=0.4,
                hdbscan_memory_frames=15,
                tracking_fps=None,
                output_filename="state_estimation.json"):
    """Phase 4: State estimation + HDBSCAN crowd analysis on v5 pipeline outputs.

    Requires depth_metric/ and cameras/intrinsics.npy from phase 3 (globalise).
    """
    from main_v2_state_est import load_flight_record, process_tracks

    depth_dir = os.path.join(output_dir, "depth_metric")
    intrinsics_path = os.path.join(output_dir, "cameras", "intrinsics.npy")

    # Validate prerequisites
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(
            f"depth_metric/ not found in {output_dir}. Run phase 3 (globalise) with --save-cameras first."
        )
    if not os.path.isfile(intrinsics_path):
        raise FileNotFoundError(
            f"cameras/intrinsics.npy not found in {output_dir}. Run phase 3 (globalise) with --save-cameras first."
        )

    # Load tracking JSON
    print(f"Loading tracking JSON: {tracking_json}")
    with open(tracking_json, 'r') as f:
        tracking_data = json.load(f)

    tracks = tracking_data.get('tracks', [])
    if not tracks:
        raise ValueError("No tracks found in tracking JSON")
    print(f"  {len(tracks)} track entries loaded")

    # Resample tracking frame IDs if tracking was done at a different FPS
    if tracking_fps is not None and tracking_fps != fps:
        ratio = tracking_fps / fps
        print(f"  Resampling tracking frame IDs: {tracking_fps} fps -> {fps} fps (ratio {ratio:.2f})")
        for t in tracks:
            t['frame_id'] = round(t['frame_id'] / ratio)
        unique_frames = len(set(t['frame_id'] for t in tracks))
        print(f"  Remapped to {unique_frames} unique frames")

    # Load flight record
    flight_data, home_position = load_flight_record(flight_record, timezone=timezone)

    # Load intrinsics
    intrinsics = np.load(intrinsics_path)

    # Get depth resolution from first depth map
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "depth_metric_*.npy")))
    if not depth_files:
        raise ValueError("No depth maps found in depth_metric/")

    first_depth = np.load(depth_files[0])
    depth_res = first_depth.shape  # (H, W)
    num_depth_frames = len(depth_files)

    # Resample flight data to match depth frame count
    if len(flight_data) != num_depth_frames:
        print(f"  Resampling flight data: {len(flight_data)} entries -> {num_depth_frames} frames")
        indices = np.linspace(0, len(flight_data) - 1, num_depth_frames, dtype=int)
        flight_data = flight_data.iloc[indices].reset_index(drop=True)

    print(f"  Source resolution: {source_res}, Depth resolution: {depth_res}")
    print(f"  {num_depth_frames} depth frames, {len(flight_data)} flight data entries")

    # Process tracks with state estimation + HDBSCAN crowd analysis
    processed_tracks = process_tracks(
        tracks=tracks,
        flight_data=flight_data,
        home_position=home_position,
        depth_dir=depth_dir,
        intrinsics=intrinsics,
        source_res=source_res,
        depth_res=depth_res,
        fps=fps,
        velocity_window=5,
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

    print(f"  Processed {len(processed_tracks)} track entries")

    # Build output JSON
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
            },
            "pipeline": "v5-global-scale"
        },
        "tracks": processed_tracks
    }

    # Save output
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  State estimation saved to: {output_path}")
    print(f"  Tracks: {len(processed_tracks)}")

    return output_json


# ===========================================================================
# CLI entry point
# ===========================================================================

def _add_common_args(p):
    """Add arguments shared across all modes / subcommands."""
    p.add_argument("--output", "-o", type=str, default="outputs",
                   help="Output directory (default: outputs)")
    p.add_argument("--fps", type=float, default=10.0,
                   help="Frames per second to extract (default: 10.0)")


def main():
    parser = argparse.ArgumentParser(
        description="VGGT Depth Estimation – 3-phase pipeline with global scale factor"
    )
    subparsers = parser.add_subparsers(dest="phase")

    # --- Subcommand: segment -----------------------------------------------
    sp_seg = subparsers.add_parser("segment", help="Phase 1: extract frames into segment dirs")
    sp_seg.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input video file")
    _add_common_args(sp_seg)
    sp_seg.add_argument("--segment-size", type=int, default=600,
                        help="Frames per segment (default: 600)")
    sp_seg.add_argument("--overlap-frames", type=int, default=10,
                        help="Overlap between segments (default: 10)")

    # --- Subcommand: infer -------------------------------------------------
    sp_inf = subparsers.add_parser("infer", help="Phase 2: run VGGT inference per segment")
    sp_inf.add_argument("--output", "-o", type=str, default="outputs",
                        help="Output directory (must contain segments/ from phase 1)")
    sp_inf.add_argument("--reload-model", action="store_true",
                        help="Unload/reload model per segment (in-process mode only)")
    sp_inf.add_argument("--no-subprocess", action="store_true",
                        help="Run inference in-process instead of spawning subprocesses")
    sp_inf.add_argument("--model-name", type=str, default="facebook/VGGT-1B",
                        help="HuggingFace model ID (default: facebook/VGGT-1B)")
    sp_inf.add_argument("--batch-size", type=int, default=20,
                        help="Frames per VGGT inference batch (default: 20)")

    # --- Subcommand: globalise ---------------------------------------------
    sp_glob = subparsers.add_parser("globalise", help="Phase 3: align, scale, encode videos")
    _add_common_args(sp_glob)
    sp_glob.add_argument("--flight-record", "-f", type=str, default=None,
                         help="DJI flight record CSV for metric scaling")
    sp_glob.add_argument("--num-reference-frames", type=int, default=10,
                         help="Frames for global scale factor (default: 10)")
    sp_glob.add_argument("--save-points", action="store_true",
                         help="Save world point maps")
    sp_glob.add_argument("--save-cameras", action="store_true",
                         help="Save camera parameters")
    sp_glob.add_argument("--save-video", action="store_true", default=True,
                         help="Save depth video (default: True)")
    sp_glob.add_argument("--no-video", action="store_true",
                         help="Disable video output")
    sp_glob.add_argument("--colormap", type=str, default="turbo",
                         help="Colormap for depth visualization (default: turbo)")
    sp_glob.add_argument("--quadrant-video", action="store_true",
                         help="Save quadrant analysis video")
    sp_glob.add_argument("--use-gimbal-yaw", action="store_true",
                         help="Use GIMBAL.yaw instead of OSD.yaw for yaw angle")
    sp_glob.add_argument("--gimbal-pitch", type=float, default=None,
                         help="Manual gimbal pitch override in DJI degrees (e.g. -90 for nadir). "
                              "Used when GIMBAL.pitch is missing from flight record.")

    # --- Subcommand: state -------------------------------------------------
    sp_state = subparsers.add_parser("state", help="Phase 4: state estimation + crowd analysis")
    sp_state.add_argument("--output", "-o", type=str, default="outputs",
                          help="Output directory (must contain depth_metric/ and cameras/ from phase 3)")
    sp_state.add_argument("--tracking", "-t", type=str, required=True,
                          help="Path to tracking JSON file")
    sp_state.add_argument("--flight-record", "-f", type=str, required=True,
                          help="DJI flight record CSV")
    sp_state.add_argument("--fps", type=float, default=10.0,
                          help="Extraction FPS used in phases 1-3 (default: 10.0)")
    sp_state.add_argument("--source-res", type=int, nargs=2, default=[1280, 720],
                          metavar=("W", "H"),
                          help="Source video resolution (default: 1280 720)")
    sp_state.add_argument("--output-json", type=str, default="state_estimation.json",
                          help="Output JSON filename (default: state_estimation.json)")
    # KF params
    sp_state.add_argument("--kf-sigma-a", type=float, default=0.5)
    sp_state.add_argument("--kf-sigma-meas-h", type=float, default=5.0)
    sp_state.add_argument("--kf-sigma-meas-v", type=float, default=2.0)
    # Yaw params
    sp_state.add_argument("--yaw-offset", type=float, default=0.0)
    sp_state.add_argument("--magnetic-declination", type=float, default=0.0)
    sp_state.add_argument("--add-drone-yaw", action="store_true")
    sp_state.add_argument("--use-gimbal-yaw", action="store_true", help="Use GIMBAL.yaw instead of OSD.yaw for yaw angle. Default: OSD.yaw")
    sp_state.add_argument("--timezone", default="UTC")
    # HDBSCAN params
    sp_state.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    sp_state.add_argument("--hdbscan-min-samples", type=int, default=3)
    sp_state.add_argument("--hdbscan-coherence-weight", type=float, default=10.0)
    sp_state.add_argument("--hdbscan-max-speed-mps", type=float, default=2.0)
    sp_state.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=2.0)
    sp_state.add_argument("--hdbscan-max-match-dist", type=float, default=20.0)
    sp_state.add_argument("--hdbscan-ema-alpha", type=float, default=0.4)
    sp_state.add_argument("--hdbscan-memory-frames", type=int, default=15)
    sp_state.add_argument("--tracking-fps", type=float, default=None)

    # --- Top-level args (backward-compatible "run all") --------------------
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to input video file")
    parser.add_argument("--flight-record", "-f", type=str, default=None,
                        help="DJI flight record CSV for metric scaling")
    _add_common_args(parser)
    parser.add_argument("--segment-size", type=int, default=600,
                        help="Frames per segment (default: 600)")
    parser.add_argument("--overlap-frames", type=int, default=10,
                        help="Overlap between segments (default: 10)")
    parser.add_argument("--num-reference-frames", type=int, default=10,
                        help="Frames for global scale factor (default: 10)")
    parser.add_argument("--save-points", action="store_true",
                        help="Save world point maps")
    parser.add_argument("--save-cameras", action="store_true",
                        help="Save camera parameters")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save depth video (default: True)")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video output")
    parser.add_argument("--colormap", type=str, default="turbo",
                        help="Colormap for depth visualization (default: turbo)")
    parser.add_argument("--quadrant-video", action="store_true",
                        help="Save quadrant analysis video")
    parser.add_argument("--use-gimbal-yaw", action="store_true",
                        help="Use GIMBAL.yaw instead of OSD.yaw for yaw angle")
    parser.add_argument("--gimbal-pitch", type=float, default=None,
                        help="Manual gimbal pitch override in DJI degrees (e.g. -90 for nadir). "
                             "Used when GIMBAL.pitch is missing from flight record.")
    parser.add_argument("--reload-model", action="store_true",
                        help="Unload/reload model per segment (in-process mode only)")
    parser.add_argument("--no-subprocess", action="store_true",
                        help="Run inference in-process instead of spawning subprocesses")
    parser.add_argument("--model-name", type=str, default="facebook/VGGT-1B",
                        help="HuggingFace model ID (default: facebook/VGGT-1B)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Frames per VGGT inference batch (default: 20)")
    # Phase 4 (state estimation) — only runs if --tracking is provided
    parser.add_argument("--tracking", "-t", type=str, default=None,
                        help="Path to tracking JSON (enables phase 4: state estimation)")
    parser.add_argument("--source-res", type=int, nargs=2, default=[1280, 720],
                        metavar=("W", "H"),
                        help="Source video resolution for state estimation (default: 1280 720)")
    parser.add_argument("--output-json", type=str, default="state_estimation.json",
                        help="State estimation output JSON filename (default: state_estimation.json)")
    # KF params
    parser.add_argument("--kf-sigma-a", type=float, default=0.5)
    parser.add_argument("--kf-sigma-meas-h", type=float, default=5.0)
    parser.add_argument("--kf-sigma-meas-v", type=float, default=2.0)
    # Yaw params
    parser.add_argument("--yaw-offset", type=float, default=0.0)
    parser.add_argument("--magnetic-declination", type=float, default=0.0)
    parser.add_argument("--add-drone-yaw", action="store_true")
    # Note: --use-gimbal-yaw is already defined above (shared with phases 1-3)
    parser.add_argument("--timezone", default="UTC")
    # HDBSCAN params
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10)
    parser.add_argument("--hdbscan-min-samples", type=int, default=3)
    parser.add_argument("--hdbscan-coherence-weight", type=float, default=10.0)
    parser.add_argument("--hdbscan-max-speed-mps", type=float, default=2.0)
    parser.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=2.0)
    parser.add_argument("--hdbscan-max-match-dist", type=float, default=20.0)
    parser.add_argument("--hdbscan-ema-alpha", type=float, default=0.4)
    parser.add_argument("--hdbscan-memory-frames", type=int, default=15)
    parser.add_argument("--tracking-fps", type=float, default=None)

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------
    if args.phase == "segment":
        phase_segment(
            video_path=args.input,
            output_dir=args.output,
            fps=args.fps,
            segment_size=args.segment_size,
            overlap_frames=args.overlap_frames,
        )

    elif args.phase == "infer":
        phase_infer(
            output_dir=args.output,
            model_name=args.model_name,
            reload_model=args.reload_model,
            no_subprocess=args.no_subprocess,
            batch_size=args.batch_size,
        )

    elif args.phase == "globalise":
        phase_globalise(
            output_dir=args.output,
            flight_record=args.flight_record,
            fps=args.fps,
            num_reference_frames=args.num_reference_frames,
            save_points=args.save_points,
            save_cameras_flag=args.save_cameras,
            save_video=args.save_video,
            quadrant_video=args.quadrant_video,
            colormap=args.colormap,
            use_gimbal_yaw=args.use_gimbal_yaw,
            no_video=args.no_video,
            gimbal_pitch_override=args.gimbal_pitch,
        )

    elif args.phase == "state":
        phase_state(
            output_dir=args.output,
            tracking_json=args.tracking,
            flight_record=args.flight_record,
            fps=args.fps,
            source_res=tuple(args.source_res),
            kf_sigma_a=args.kf_sigma_a,
            kf_sigma_meas_h=args.kf_sigma_meas_h,
            kf_sigma_meas_v=args.kf_sigma_meas_v,
            yaw_offset=args.yaw_offset,
            magnetic_declination=args.magnetic_declination,
            add_drone_yaw=args.add_drone_yaw,
            use_gimbal_yaw=args.use_gimbal_yaw,
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
            output_filename=args.output_json,
        )

    else:
        # No subcommand → run all 3 phases sequentially (backward compatible)
        if args.input is None:
            parser.error("--input / -i is required when running all phases")

        # Derive output dir with video stem (phase_segment does this too,
        # but we need the resolved path for phases 2 & 3).
        resolved_output = _output_dir_for_video(args.output, args.input)

        total_start = time.time()
        os.makedirs(resolved_output, exist_ok=True)

        print("=" * 60)
        print("PHASE 1: SEGMENT")
        print("=" * 60)
        phase_segment(
            video_path=args.input,
            output_dir=args.output,
            fps=args.fps,
            segment_size=args.segment_size,
            overlap_frames=args.overlap_frames,
        )

        print("\n" + "=" * 60)
        print("PHASE 2: INFER")
        print("=" * 60)
        phase_infer(
            output_dir=resolved_output,
            model_name=args.model_name,
            reload_model=args.reload_model,
            no_subprocess=args.no_subprocess,
            batch_size=args.batch_size,
        )

        print("\n" + "=" * 60)
        print("PHASE 3: GLOBALISE")
        print("=" * 60)
        phase_globalise(
            output_dir=resolved_output,
            flight_record=args.flight_record,
            fps=args.fps,
            num_reference_frames=args.num_reference_frames,
            save_points=args.save_points,
            save_cameras_flag=args.save_cameras or (args.tracking is not None),
            save_video=args.save_video,
            quadrant_video=args.quadrant_video,
            colormap=args.colormap,
            use_gimbal_yaw=args.use_gimbal_yaw,
            no_video=args.no_video,
            gimbal_pitch_override=args.gimbal_pitch,
        )

        # --- Phase 4 (optional): State estimation + crowd analysis -----------
        if args.tracking is not None:
            print("\n" + "=" * 60)
            print("PHASE 4: STATE ESTIMATION + CROWD ANALYSIS")
            print("=" * 60)
            phase_state(
                output_dir=resolved_output,
                tracking_json=args.tracking,
                flight_record=args.flight_record,
                fps=args.fps,
                source_res=tuple(args.source_res),
                kf_sigma_a=args.kf_sigma_a,
                kf_sigma_meas_h=args.kf_sigma_meas_h,
                kf_sigma_meas_v=args.kf_sigma_meas_v,
                yaw_offset=args.yaw_offset,
                magnetic_declination=args.magnetic_declination,
                add_drone_yaw=args.add_drone_yaw,
                use_gimbal_yaw=args.use_gimbal_yaw,
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
                output_filename=args.output_json,
            )

        total_time = time.time() - total_start
        print(f"\n{'=' * 50}")
        print(f"Total processing time: {total_time:.2f}s ({total_time / 60:.1f} min)")
        print(f"{'=' * 50}")
        print(f"Done! Outputs saved to: {resolved_output}/")


if __name__ == "__main__":
    main()
