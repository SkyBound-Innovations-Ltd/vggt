#!/usr/bin/env python3
"""
VGGT Robust Depth Estimation Pipeline for UAV Video (v2)

This script provides a production-ready depth estimation pipeline with:
1. RANSAC-based metric scaling via ground plane fitting
2. Global color consistency across video frames
3. Robust handling of camera tilt without explicit gimbal angles

Key Improvements over v1:
-------------------------
- Metric scaling uses RANSAC plane fitting instead of single center point
- Video colormap uses fixed global min/max for temporal consistency
- Automatic pitch/roll compensation through 3D geometry

Geometry Notes:
---------------
The RANSAC approach fits a ground plane Z = αX + βY + γ to the 3D point cloud.
The perpendicular distance from camera origin (0,0,0) to this plane gives the
"canonical altitude". Combined with real AGL altitude from telemetry, we compute:
    scale_factor = real_altitude_agl / canonical_altitude

This method is robust to:
- Obstacles (buildings/trees) at image center
- Camera tilt (pitch/roll) - handled implicitly by plane fitting
- Outliers - rejected by RANSAC

Author: Senior Computer Vision Engineer
"""

import torch
import cv2
import os
import argparse
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional, List, Dict
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# =============================================================================
# TELEMETRY PARSING
# =============================================================================

def parse_dji_flight_record(csv_path: str, video_duration_sec: float, target_fps: float) -> Optional[Dict]:
    """
    Parse DJI flight record CSV and extract telemetry synchronized with video frames.

    Note: This function extracts altitude for metric scaling. Gimbal pitch is NOT
    required when using RANSAC plane fitting, as tilt is handled implicitly.

    Args:
        csv_path: Path to DJI flight record CSV file
        video_duration_sec: Duration of the input video in seconds
        target_fps: Target frames per second for extraction

    Returns:
        dict with telemetry data, or None if parsing fails
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Find altitude column (priority: OSD.height > OSD.vpsHeight)
    altitude_col = None
    for col in ['OSD.height [m]', 'OSD.vpsHeight [m]']:
        if col in df.columns:
            altitude_col = col
            break

    if altitude_col is None:
        print(f"  WARNING: Could not find altitude column in {csv_path}")
        return None

    print(f"  Using altitude column: {altitude_col}")

    # Check for video recording column
    if 'CAMERA.isVideo' not in df.columns:
        print(f"  WARNING: CAMERA.isVideo column not found")
        return None

    # Filter for video recording segments
    df['is_video_bool'] = df['CAMERA.isVideo'].apply(
        lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else False
    )
    video_df = df[df['is_video_bool']].copy()

    if len(video_df) == 0:
        print("  WARNING: No rows found where CAMERA.isVideo is TRUE")
        return None

    print(f"  Flight record: {len(df)} total rows, {len(video_df)} during video recording")

    # Extract altitude
    altitudes_raw = video_df[altitude_col].values
    altitudes_raw = np.nan_to_num(altitudes_raw, nan=np.nanmean(altitudes_raw))

    # Resample to match target frame count
    target_frame_count = int(video_duration_sec * target_fps)

    if len(altitudes_raw) != target_frame_count:
        old_indices = np.linspace(0, len(altitudes_raw) - 1, len(altitudes_raw))
        new_indices = np.linspace(0, len(altitudes_raw) - 1, target_frame_count)
        altitudes = np.interp(new_indices, old_indices, altitudes_raw)
        print(f"  Resampled {len(altitudes_raw)} → {target_frame_count} altitude samples")
    else:
        altitudes = altitudes_raw

    print(f"  Altitude range: [{altitudes.min():.1f}, {altitudes.max():.1f}] m (mean: {altitudes.mean():.1f} m)")

    result = {
        'altitudes': list(altitudes),
        'mean_altitude': float(np.mean(altitudes))
    }

    # Try to extract gimbal pitch (if available and not all zeros)
    if 'GIMBAL.pitch' in video_df.columns:
        gimbal_pitch_raw = video_df['GIMBAL.pitch'].values
        gimbal_pitch_raw = np.nan_to_num(gimbal_pitch_raw, nan=0.0)

        # Check if gimbal data is valid (not all zeros)
        if not np.all(gimbal_pitch_raw == 0):
            # Resample gimbal pitch
            if len(gimbal_pitch_raw) != target_frame_count:
                gimbal_pitch = np.interp(new_indices, old_indices, gimbal_pitch_raw)
            else:
                gimbal_pitch = gimbal_pitch_raw

            # Convert DJI convention to pitch_below_horizontal
            # DJI: 0° = horizontal, -90° = nadir (negative = down)
            # Our convention: 0° = horizon, 90° = nadir (positive = below horizon)
            pitch_below_horizontal = -gimbal_pitch  # e.g., -45° → 45° below horizontal
            pitch_below_horizontal = np.clip(pitch_below_horizontal, 0, 90)

            result['pitch_below_horizontal'] = list(pitch_below_horizontal)
            result['gimbal_pitch'] = list(gimbal_pitch)
            print(f"  Gimbal pitch range: [{gimbal_pitch.min():.1f}, {gimbal_pitch.max():.1f}]° (DJI convention)")
            print(f"  Pitch below horizontal: [{pitch_below_horizontal.min():.1f}, {pitch_below_horizontal.max():.1f}]°")
        else:
            print("  WARNING: GIMBAL.pitch is all zeros (data not logged)")
            print("  Please provide --pitch argument for telemetry validation")

    return result


# =============================================================================
# RANSAC GROUND PLANE FITTING
# =============================================================================

def filter_and_downsample_points(point_map: np.ndarray,
                                  downsample_factor: int = 10,
                                  min_z: float = 0.01) -> np.ndarray:
    """
    Reshape point map to (N, 3), filter invalid points, and downsample.

    Args:
        point_map: Array of shape (H, W, 3) with (X, Y, Z) coordinates
        downsample_factor: Take every Nth point for RANSAC
        min_z: Minimum valid Z value (points must be in front of camera)

    Returns:
        points: Filtered and downsampled array of shape (M, 3)
    """
    points = point_map.reshape(-1, 3)

    # Filter invalid points
    valid_mask = (
        (points[:, 2] > min_z) &
        np.isfinite(points[:, 0]) &
        np.isfinite(points[:, 1]) &
        np.isfinite(points[:, 2])
    )

    valid_points = points[valid_mask]

    # Downsample
    if downsample_factor > 1 and len(valid_points) > downsample_factor:
        indices = np.arange(0, len(valid_points), downsample_factor)
        return valid_points[indices]

    return valid_points


def fit_ground_plane_ransac(points: np.ndarray,
                            residual_threshold: float = 0.1,
                            min_samples: int = 3,
                            max_trials: int = 1000) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Fit a ground plane to 3D points using RANSAC.

    Plane model: Z = αX + βY + γ
    This implicitly handles camera pitch/roll by fitting the actual plane orientation.

    Note: This function does NOT reject tilted planes. Validation against
    telemetry should be done separately using validate_plane_against_telemetry().

    Args:
        points: Array of shape (N, 3) with (X, Y, Z) coordinates
        residual_threshold: RANSAC inlier threshold
        min_samples: Minimum samples for plane fitting
        max_trials: Maximum RANSAC iterations

    Returns:
        plane_coeffs: Array [α, β, γ] or None if fitting fails
        inlier_ratio: Fraction of inlier points
        tilt_deg: Plane tilt from horizontal in degrees
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression

    if len(points) < min_samples:
        return None, 0.0, 90.0

    X_features = points[:, :2]  # (X, Y)
    Z_target = points[:, 2]     # Z

    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=42
        )
        ransac.fit(X_features, Z_target)

        alpha = ransac.estimator_.coef_[0]
        beta = ransac.estimator_.coef_[1]
        gamma = ransac.estimator_.intercept_

        inlier_ratio = np.sum(ransac.inlier_mask_) / len(ransac.inlier_mask_)

        # Calculate plane tilt from horizontal
        # Normal vector: (α, β, -1) / ||(α, β, -1)||
        normal = np.array([alpha, beta, -1.0])
        normal = normal / np.linalg.norm(normal)
        tilt_deg = np.degrees(np.arccos(abs(normal[2])))

        return np.array([alpha, beta, gamma]), inlier_ratio, tilt_deg

    except Exception as e:
        print(f"    RANSAC failed: {e}")
        return None, 0.0, 90.0


def validate_plane_against_telemetry(ransac_tilt_deg: float,
                                      pitch_below_horizontal: float,
                                      tolerance_deg: float = 15.0) -> Tuple[bool, float, float]:
    """
    Validate RANSAC plane tilt against expected tilt from telemetry.

    For a camera pointing at angle θ below horizontal at a flat ground:
    - Expected ground plane tilt = 90° - θ
    - Example: θ = 20° (shallow oblique) → expected tilt = 70°
    - Example: θ = 90° (nadir) → expected tilt = 0°

    Args:
        ransac_tilt_deg: Tilt of RANSAC-fitted plane from horizontal
        pitch_below_horizontal: Camera pitch angle below horizontal (0=horizon, 90=nadir)
        tolerance_deg: Acceptable deviation between expected and actual tilt

    Returns:
        is_valid: True if RANSAC plane matches expected ground orientation
        expected_tilt: The expected ground tilt based on telemetry
        tilt_error: Absolute difference between expected and actual tilt
    """
    # Expected ground tilt from camera pitch
    # If camera points 20° below horizontal, ground appears tilted 70° in camera frame
    expected_tilt = 90.0 - pitch_below_horizontal

    # Check if RANSAC tilt matches expected
    tilt_error = abs(ransac_tilt_deg - expected_tilt)
    is_valid = tilt_error <= tolerance_deg

    return is_valid, expected_tilt, tilt_error


def geometric_center_scaling(canonical_depth: np.ndarray,
                              altitude_agl: float,
                              pitch_below_horizontal: float) -> Tuple[float, float, float]:
    """
    Fallback scaling using geometric slant range calculation.

    When RANSAC fails validation (e.g., finds a wall instead of ground),
    use the telemetry-based geometric approach:
        slant_range = altitude / sin(pitch)
        scale = slant_range / center_depth

    This is the safe fallback that uses known telemetry rather than
    potentially incorrect RANSAC results.

    Geometry:
              -------- Horizontal --------
                      *  UAV (camera)
                     /|
                   θ/ |
         slant    /  | h (altitude)
         range   /   |
                /    |
               /_____|____________ Ground

        sin(θ) = h / slant_range
        slant_range = h / sin(θ)

    Args:
        canonical_depth: Depth map (H, W)
        altitude_agl: Real altitude above ground level in meters
        pitch_below_horizontal: Camera pitch below horizontal (0=horizon, 90=nadir)

    Returns:
        scale_factor: Computed scale factor
        slant_range: Calculated slant range in meters
        center_depth: Median canonical depth at center
    """
    # Clamp pitch to avoid division by zero (minimum 5° below horizontal)
    pitch_clamped = max(pitch_below_horizontal, 5.0)
    pitch_rad = np.deg2rad(pitch_clamped)

    # Slant range = altitude / sin(pitch)
    slant_range = altitude_agl / np.sin(pitch_rad)

    # Get center depth (median of center region for robustness)
    h, w = canonical_depth.shape
    center_region = canonical_depth[h//3:2*h//3, w//3:2*w//3]
    center_depth = np.median(center_region)

    # Scale factor
    scale_factor = slant_range / max(center_depth, 1e-6)

    return scale_factor, slant_range, center_depth


def compute_canonical_altitude(plane_coeffs: np.ndarray) -> float:
    """
    Compute perpendicular distance from camera origin (0,0,0) to ground plane.

    Plane: αX + βY - Z + γ = 0
    Distance = |γ| / sqrt(α² + β² + 1)

    Args:
        plane_coeffs: Array [α, β, γ]

    Returns:
        Perpendicular distance from origin to plane
    """
    alpha, beta, gamma = plane_coeffs
    denominator = np.sqrt(alpha**2 + beta**2 + 1.0)
    return abs(gamma) / denominator


def scale_depth_ransac(canonical_depth: np.ndarray,
                       world_points: np.ndarray,
                       altitude_agl: float,
                       pitch_below_horizontal: float,
                       downsample_factor: int = 10,
                       tilt_tolerance_deg: float = 15.0) -> Tuple[np.ndarray, float, Dict]:
    """
    Convert canonical depth to metric using RANSAC ground plane fitting
    with telemetry validation.

    Strategy:
    1. Fit plane using RANSAC
    2. Validate plane tilt against expected tilt from telemetry
    3. If valid (within tolerance), use RANSAC plane
    4. If invalid (likely found a wall), fallback to geometric center scaling

    This method is robust to:
    - Obstacles at image center (trees, buildings)
    - Camera tilt (pitch/roll) - validated against telemetry
    - Outliers - rejected by RANSAC
    - Walls/vertical surfaces - rejected by telemetry validation

    Args:
        canonical_depth: Depth map (H, W)
        world_points: Point cloud (H, W, 3)
        altitude_agl: Real altitude above ground level in meters
        pitch_below_horizontal: Camera pitch below horizontal (0=horizon, 90=nadir)
        downsample_factor: Downsampling for RANSAC efficiency
        tilt_tolerance_deg: Acceptable deviation from expected tilt (default: ±15°)

    Returns:
        metric_depth: Scaled depth map in meters
        scale_factor: Computed scale factor
        info: Dictionary with fitting details
    """
    info = {
        'altitude_agl': altitude_agl,
        'pitch_below_horizontal': pitch_below_horizontal,
    }

    # Filter and downsample points
    points = filter_and_downsample_points(world_points, downsample_factor)
    info['num_points'] = len(points)

    # Calculate expected tilt for reference
    expected_tilt = 90.0 - pitch_below_horizontal
    info['expected_tilt'] = expected_tilt

    # If too few points, use geometric fallback
    if len(points) < 100:
        print(f"    Too few valid points ({len(points)}), using geometric fallback")
        scale_factor, slant_range, center_depth = geometric_center_scaling(
            canonical_depth, altitude_agl, pitch_below_horizontal
        )
        info['method'] = 'geometric_fallback'
        info['slant_range'] = slant_range
        info['center_depth'] = center_depth
        info['inlier_ratio'] = 0.0
        info['tilt_deg'] = expected_tilt
        return canonical_depth * scale_factor, scale_factor, info

    # Fit ground plane using RANSAC
    plane_coeffs, inlier_ratio, ransac_tilt = fit_ground_plane_ransac(points)
    info['inlier_ratio'] = inlier_ratio
    info['tilt_deg'] = ransac_tilt

    # Validate RANSAC plane against telemetry
    if plane_coeffs is not None and inlier_ratio >= 0.1:
        is_valid, expected_tilt, tilt_error = validate_plane_against_telemetry(
            ransac_tilt, pitch_below_horizontal, tilt_tolerance_deg
        )
        info['tilt_error'] = tilt_error
        info['tilt_valid'] = is_valid

        if is_valid:
            # RANSAC plane matches expected ground - use it
            canonical_altitude = compute_canonical_altitude(plane_coeffs)
            info['canonical_altitude'] = canonical_altitude
            info['plane_coeffs'] = plane_coeffs

            if canonical_altitude > 1e-6:
                scale_factor = altitude_agl / canonical_altitude
                info['method'] = 'ransac'
                metric_depth = canonical_depth * scale_factor
                print(f"    RANSAC valid: tilt={ransac_tilt:.1f}° (expected={expected_tilt:.1f}°, error={tilt_error:.1f}°)")
                return metric_depth, scale_factor, info
            else:
                print(f"    RANSAC altitude invalid ({canonical_altitude}), using geometric fallback")
        else:
            print(f"    RANSAC tilt mismatch: {ransac_tilt:.1f}° vs expected {expected_tilt:.1f}° (error={tilt_error:.1f}° > {tilt_tolerance_deg}°)")
            print(f"    Likely found a wall/vertical surface, using geometric fallback")
    else:
        print(f"    RANSAC failed (coeffs={plane_coeffs is not None}, inliers={inlier_ratio:.1%})")
        info['tilt_error'] = 0.0
        info['tilt_valid'] = False

    # Fallback: Geometric center scaling using telemetry
    # This uses: scale = (altitude / sin(pitch)) / center_depth
    scale_factor, slant_range, center_depth = geometric_center_scaling(
        canonical_depth, altitude_agl, pitch_below_horizontal
    )
    info['method'] = 'geometric_fallback'
    info['slant_range'] = slant_range
    info['center_depth'] = center_depth

    metric_depth = canonical_depth * scale_factor
    print(f"    Geometric fallback: slant_range={slant_range:.1f}m, center_depth={center_depth:.4f}")

    return metric_depth, scale_factor, info


def scale_depth_sequence_ransac(canonical_depths: List[np.ndarray],
                                 world_points_list: List[np.ndarray],
                                 altitudes: List[float],
                                 pitch_angles: List[float],
                                 downsample_factor: int = 10,
                                 tilt_tolerance_deg: float = 15.0) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Scale a sequence of depth maps using RANSAC with telemetry validation.

    Args:
        canonical_depths: List of depth maps
        world_points_list: List of point clouds
        altitudes: List of AGL altitudes per frame
        pitch_angles: List of pitch angles below horizontal per frame (0=horizon, 90=nadir)
        downsample_factor: Downsampling for RANSAC
        tilt_tolerance_deg: Acceptable tilt deviation from expected (default: ±15°)

    Returns:
        metric_depths: List of metric depth maps
        infos: List of per-frame fitting info
    """
    metric_depths = []
    infos = []

    for i, (depth, points, alt, pitch) in enumerate(zip(canonical_depths, world_points_list, altitudes, pitch_angles)):
        print(f"  Frame {i+1}/{len(canonical_depths)}:")

        metric_depth, scale_factor, info = scale_depth_ransac(
            depth, points, alt, pitch, downsample_factor, tilt_tolerance_deg
        )

        print(f"    Method: {info['method']}, Scale: {scale_factor:.2f}, "
              f"Inliers: {info.get('inlier_ratio', 0):.1%}")

        metric_depths.append(metric_depth)
        infos.append(info)

    return metric_depths, infos


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def extract_frames_from_video(video_path: str, fps: float = 1.0) -> Tuple[List[np.ndarray], float]:
    """
    Extract frames from video at specified fps.

    Returns:
        frames: List of RGB frame arrays
        video_duration: Duration of video in seconds
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    target_frame_count = int(video_duration * fps)
    frame_interval = max(1, total_frames // target_frame_count) if target_frame_count > 0 else 1

    print(f"Video: {video_fps:.1f} fps, {total_frames} frames, {video_duration:.1f}s duration")
    print(f"Extracting ~{target_frame_count} frames at {fps} fps")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames")

    return frames, video_duration


def save_frames_and_load(frames: List[np.ndarray], tmp_dir: str) -> torch.Tensor:
    """Save frames to temp directory and load with VGGT preprocessing."""
    os.makedirs(tmp_dir, exist_ok=True)
    frame_paths = []

    for i, frame in enumerate(frames):
        frame_path = os.path.join(tmp_dir, f"{i:06d}.png")
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)

    images = load_and_preprocess_images(frame_paths)
    return images


# =============================================================================
# GLOBAL COLOR CONSISTENCY
# =============================================================================

def compute_global_depth_range(metric_depths: List[np.ndarray],
                                percentile_low: float = 1.0,
                                percentile_high: float = 99.0) -> Tuple[float, float]:
    """
    Compute global min/max depth across all frames using percentiles.

    Using percentiles (e.g., 1st and 99th) instead of absolute min/max
    avoids outliers from affecting the color range.

    Args:
        metric_depths: List of metric depth maps
        percentile_low: Lower percentile for min (default: 1%)
        percentile_high: Upper percentile for max (default: 99%)

    Returns:
        global_min: Global minimum depth (at percentile_low)
        global_max: Global maximum depth (at percentile_high)
    """
    # Collect all valid depth values
    all_depths = []
    for depth in metric_depths:
        valid = depth[np.isfinite(depth) & (depth > 0)]
        all_depths.append(valid.flatten())

    all_depths = np.concatenate(all_depths)

    global_min = np.percentile(all_depths, percentile_low)
    global_max = np.percentile(all_depths, percentile_high)

    print(f"Global depth range (p{percentile_low:.0f}-p{percentile_high:.0f}): "
          f"[{global_min:.2f}, {global_max:.2f}] m")

    return global_min, global_max


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_depth_colormap_fixed(depth: np.ndarray,
                                 vmin: float,
                                 vmax: float,
                                 colormap: str = 'turbo') -> np.ndarray:
    """
    Convert depth map to colored visualization with FIXED color range.

    Args:
        depth: Depth map (H, W)
        vmin: Fixed minimum value for colormap
        vmax: Fixed maximum value for colormap
        colormap: Matplotlib colormap name

    Returns:
        Colored depth image (H, W, 3) as uint8
    """
    # Normalize to [0, 1] using fixed range
    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)

    return depth_colored


def create_overlay_colorbar(height: int,
                            width: int = 40,
                            depth_min: float = 0,
                            depth_max: float = 1,
                            colormap: str = 'turbo',
                            margin: int = 10) -> np.ndarray:
    """
    Create a vertical colorbar image for overlay with fixed range labels.

    Args:
        height: Image height
        width: Colorbar width
        depth_min: Fixed minimum depth value
        depth_max: Fixed maximum depth value
        colormap: Matplotlib colormap name
        margin: Margin in pixels

    Returns:
        Colorbar image (height, width+60, 3) as uint8
    """
    # Create gradient
    gradient = np.linspace(1, 0, height - 2 * margin).reshape(-1, 1)
    gradient = np.repeat(gradient, width - 10, axis=1)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colorbar = cmap(gradient)[:, :, :3]
    colorbar = (colorbar * 255).astype(np.uint8)

    # Create background
    bg = np.zeros((height, width + 60, 3), dtype=np.uint8)

    # Place colorbar
    bg[margin:margin + colorbar.shape[0], 5:5 + colorbar.shape[1]] = colorbar

    # Add border
    cv2.rectangle(bg, (5, margin), (5 + colorbar.shape[1], margin + colorbar.shape[0]),
                  (255, 255, 255), 1)

    # Add tick labels (doubled font size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Format labels based on range
    if depth_max > 100:
        fmt = "{:.0f}m"
    elif depth_max > 10:
        fmt = "{:.1f}m"
    else:
        fmt = "{:.2f}m"

    # Top label (max)
    cv2.putText(bg, fmt.format(depth_max), (width - 5, margin + 25),
                font, font_scale, (255, 255, 255), thickness)

    # Middle label
    mid_val = (depth_min + depth_max) / 2
    cv2.putText(bg, fmt.format(mid_val), (width - 5, height // 2 + 10),
                font, font_scale, (255, 255, 255), thickness)

    # Bottom label (min)
    cv2.putText(bg, fmt.format(depth_min), (width - 5, height - margin + 5),
                font, font_scale, (255, 255, 255), thickness)

    return bg


def create_side_by_side_frame_fixed(original_frame: np.ndarray,
                                     depth_map: np.ndarray,
                                     global_min: float,
                                     global_max: float,
                                     colormap: str = 'turbo') -> np.ndarray:
    """
    Create side-by-side visualization with FIXED color range.

    Args:
        original_frame: RGB frame (H, W, 3)
        depth_map: Metric depth map (H, W)
        global_min: Fixed minimum depth for colormap
        global_max: Fixed maximum depth for colormap
        colormap: Matplotlib colormap name

    Returns:
        Combined image (H, 2*W + colorbar, 3)
    """
    h, w = original_frame.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]

    # Resize depth to match frame
    if (depth_h, depth_w) != (h, w):
        depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_map

    # Create colored depth with FIXED range
    depth_colored = create_depth_colormap_fixed(depth_resized, global_min, global_max, colormap)

    # Create and overlay colorbar
    colorbar = create_overlay_colorbar(h, width=40, depth_min=global_min,
                                        depth_max=global_max, colormap=colormap)
    cb_h, cb_w = colorbar.shape[:2]
    x_offset = w - cb_w - 10
    y_offset = 0

    # Blend colorbar
    roi = depth_colored[y_offset:y_offset + cb_h, x_offset:x_offset + cb_w]
    mask = (colorbar > 0).any(axis=2)
    alpha = 0.85
    roi[mask] = (alpha * colorbar[mask] + (1 - alpha) * roi[mask]).astype(np.uint8)

    # Combine side-by-side
    combined = np.hstack([original_frame, depth_colored])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Metric Depth', (w + 10, 30), font, 1, (255, 255, 255), 2)

    return combined


def encode_depth_video_fixed(frames: List[np.ndarray],
                              depth_maps: List[np.ndarray],
                              output_path: str,
                              global_min: float,
                              global_max: float,
                              fps: float = 10,
                              colormap: str = 'turbo'):
    """
    Encode video with FIXED color range for temporal consistency.

    Args:
        frames: List of RGB frames
        depth_maps: List of metric depth maps
        output_path: Output video path
        global_min: Fixed minimum depth for colormap
        global_max: Fixed maximum depth for colormap
        fps: Output frame rate
        colormap: Matplotlib colormap name
    """
    import subprocess
    import shutil

    if len(frames) == 0 or len(depth_maps) == 0:
        raise ValueError("No frames or depth maps to encode")

    # Get dimensions
    first_combined = create_side_by_side_frame_fixed(
        frames[0], depth_maps[0], global_min, global_max, colormap
    )
    h, w = first_combined.shape[:2]

    # Check for ffmpeg
    ffmpeg_available = shutil.which('ffmpeg') is not None

    print(f"Encoding video with FIXED depth range [{global_min:.1f}, {global_max:.1f}] m")

    if ffmpeg_available:
        print(f"Using H.264 (ffmpeg): {len(frames)} frames at {fps} fps")

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
            combined = create_side_by_side_frame_fixed(frame, depth, global_min, global_max, colormap)
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
        print(f"Using OpenCV (MPEG-4): {len(frames)} frames at {fps} fps")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
            combined = create_side_by_side_frame_fixed(frame, depth, global_min, global_max, colormap)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            out.write(combined_bgr)

            if (i + 1) % 10 == 0:
                print(f"  Encoded {i + 1}/{len(frames)} frames")

        out.release()
        print(f"Video saved to: {output_path}")
        print("  Note: Install ffmpeg for H.264 support")


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_depth_maps(depth_maps: List[np.ndarray], output_dir: str, prefix: str = "depth"):
    """Save depth maps as .npy and visualization .png files."""
    depth_dir = os.path.join(output_dir, prefix)
    os.makedirs(depth_dir, exist_ok=True)

    for i, depth in enumerate(depth_maps):
        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy().squeeze()
        else:
            depth_np = depth

        # Save raw numpy
        np.save(os.path.join(depth_dir, f"{prefix}_{i:04d}.npy"), depth_np)

        # Save normalized visualization
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        depth_img = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(os.path.join(depth_dir, f"{prefix}_{i:04d}.png"))

    print(f"Saved {len(depth_maps)} depth maps to {depth_dir}/")


def save_point_maps(point_maps: List[np.ndarray], output_dir: str):
    """Save world point maps as .npy files."""
    points_dir = os.path.join(output_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i, points in enumerate(point_maps):
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
        np.save(os.path.join(points_dir, f"points_{i:04d}.npy"), points_np)

    print(f"Saved {len(point_maps)} point maps to {points_dir}/")


def save_scaling_info(infos: List[Dict], output_dir: str):
    """Save per-frame scaling information as JSON."""
    import json

    info_path = os.path.join(output_dir, "scaling_info.json")

    def convert_to_native(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    json_infos = [convert_to_native(info) for info in infos]

    with open(info_path, 'w') as f:
        json.dump(json_infos, f, indent=2)

    print(f"Saved scaling info to {info_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VGGT Robust Depth Estimation Pipeline (v2) - RANSAC + Global Color Consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with flight record
  python main_v2_robust.py -i video.mp4 -f flight_record.csv

  # With custom altitude (no flight record)
  python main_v2_robust.py -i video.mp4 --altitude 50.0

  # Adjust parameters
  python main_v2_robust.py -i video.mp4 -f record.csv --fps 5 --batch-size 20
        """
    )

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("-f", "--flight-record", type=str, default=None,
                        help="Path to DJI flight record CSV")
    parser.add_argument("-a", "--altitude", type=float, default=None,
                        help="Manual altitude AGL in meters (overrides flight record)")
    parser.add_argument("-p", "--pitch", type=float, default=None,
                        help="Manual pitch below horizontal in degrees (0=horizon, 90=nadir). "
                             "Required when using manual altitude or when gimbal data is missing.")
    parser.add_argument("-o", "--output", type=str, default="outputs_v2",
                        help="Output directory (default: outputs_v2)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Batch size for inference (default: 20)")
    parser.add_argument("--downsample", type=int, default=10,
                        help="Point cloud downsampling for RANSAC (default: 10)")
    parser.add_argument("--tilt-tolerance", type=float, default=15.0,
                        help="RANSAC tilt tolerance in degrees (default: 15)")
    parser.add_argument("--colormap", type=str, default="turbo",
                        help="Colormap for visualization (default: turbo)")
    parser.add_argument("--save-points", action="store_true",
                        help="Save world point maps")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video encoding")

    args = parser.parse_args()

    # Validate inputs
    if args.flight_record is None and args.altitude is None:
        parser.error("Either --flight-record or --altitude must be provided for metric scaling")

    if args.altitude is not None and args.pitch is None:
        parser.error("--pitch is required when using manual --altitude")

    total_start = time.time()

    print("=" * 70)
    print("VGGT Robust Depth Estimation Pipeline (v2)")
    print("  - RANSAC Ground Plane Fitting")
    print("  - Global Color Consistency")
    print("=" * 70)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"\nDevice: {device}, dtype: {dtype}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # ==========================================================================
    # STEP 1: Extract frames
    # ==========================================================================
    print(f"\n[1/6] Extracting frames from: {args.input}")
    tic = time.time()
    frames, video_duration = extract_frames_from_video(args.input, fps=args.fps)
    print(f"  Time: {time.time() - tic:.2f}s")

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # ==========================================================================
    # STEP 2: Parse telemetry
    # ==========================================================================
    print("\n[2/6] Parsing telemetry...")

    # Get altitude values
    if args.altitude is not None:
        altitudes = [args.altitude] * len(frames)
        print(f"  Using manual altitude: {args.altitude} m")
    elif args.flight_record:
        telemetry = parse_dji_flight_record(args.flight_record, video_duration, args.fps)
        if telemetry is not None:
            altitudes = telemetry['altitudes']
            if len(altitudes) < len(frames):
                altitudes = altitudes + [altitudes[-1]] * (len(frames) - len(altitudes))
            elif len(altitudes) > len(frames):
                altitudes = altitudes[:len(frames)]
        else:
            raise ValueError("Failed to parse flight record and no manual altitude provided")
    else:
        raise ValueError("No altitude source available")

    # Get pitch angles (pitch below horizontal: 0=horizon, 90=nadir)
    if args.pitch is not None:
        # Use manual pitch
        pitch_angles = [args.pitch] * len(frames)
        print(f"  Using manual pitch: {args.pitch}° below horizontal")
    elif args.flight_record and telemetry is not None and 'pitch_below_horizontal' in telemetry:
        # Use pitch from flight record (if gimbal data was logged)
        pitch_angles = telemetry['pitch_below_horizontal']
        if len(pitch_angles) < len(frames):
            pitch_angles = pitch_angles + [pitch_angles[-1]] * (len(frames) - len(pitch_angles))
        elif len(pitch_angles) > len(frames):
            pitch_angles = pitch_angles[:len(frames)]
        print(f"  Pitch from flight record: [{min(pitch_angles):.1f}, {max(pitch_angles):.1f}]° below horizontal")
    else:
        # Default: assume nadir (90°) if no pitch info available
        print("  WARNING: No pitch data available. Please provide --pitch argument.")
        print("  Using default pitch of 45° below horizontal (oblique assumption)")
        pitch_angles = [45.0] * len(frames)

    print(f"  Expected ground tilt: {90.0 - pitch_angles[0]:.1f}° from horizontal")

    # ==========================================================================
    # STEP 3: Preprocess and run VGGT inference
    # ==========================================================================
    print(f"\n[3/6] Running VGGT inference...")

    # Save frames temporarily
    tmp_dir = os.path.join(args.output, "frames")
    images = save_frames_and_load(frames, tmp_dir)
    images = images.to(device)
    print(f"  Input tensor: {images.shape}")

    # Load model
    print("  Loading VGGT model...")
    tic = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print(f"  Model loaded in {time.time() - tic:.2f}s")

    # Run batched inference
    print(f"  Inference (batch size: {args.batch_size})...")
    tic = time.time()

    all_depths = []
    all_world_points = []

    num_frames = images.shape[0]
    num_batches = (num_frames + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_frames)
        batch_images = images[start_idx:end_idx]

        print(f"    Batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx - 1})")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(batch_images)

        if "depth" in predictions:
            all_depths.append(predictions["depth"].cpu())
        if "world_points" in predictions:
            all_world_points.append(predictions["world_points"].cpu())

        del predictions
        torch.cuda.empty_cache()

    print(f"  Inference time: {time.time() - tic:.2f}s")

    # Combine batches
    depths_tensor = torch.cat(all_depths, dim=1).squeeze(0)  # [N, H, W, 1]
    points_tensor = torch.cat(all_world_points, dim=1).squeeze(0)  # [N, H, W, 3]

    # Convert to numpy lists
    canonical_depths = [depths_tensor[i].numpy().squeeze() for i in range(depths_tensor.shape[0])]
    world_points = [points_tensor[i].numpy() for i in range(points_tensor.shape[0])]

    # Save canonical depths
    save_depth_maps(canonical_depths, args.output, prefix="depth_canonical")

    # Optionally save point maps
    if args.save_points:
        save_point_maps(world_points, args.output)

    # ==========================================================================
    # STEP 4: RANSAC metric scaling
    # ==========================================================================
    print(f"\n[4/6] RANSAC metric scaling...")
    tic = time.time()

    metric_depths, scaling_infos = scale_depth_sequence_ransac(
        canonical_depths,
        world_points,
        altitudes,
        pitch_angles,
        downsample_factor=args.downsample,
        tilt_tolerance_deg=args.tilt_tolerance
    )

    print(f"  Scaling time: {time.time() - tic:.2f}s")

    # Save metric depths
    save_depth_maps(metric_depths, args.output, prefix="depth_metric")
    save_scaling_info(scaling_infos, args.output)

    # ==========================================================================
    # STEP 5: Compute global depth range
    # ==========================================================================
    print(f"\n[5/6] Computing global depth range...")
    global_min, global_max = compute_global_depth_range(metric_depths)

    # ==========================================================================
    # STEP 6: Encode video with fixed color range
    # ==========================================================================
    if not args.no_video:
        print(f"\n[6/6] Encoding video with global color consistency...")
        tic = time.time()

        video_path = os.path.join(args.output, "depth_video.mp4")
        encode_depth_video_fixed(
            frames,
            metric_depths,
            video_path,
            global_min,
            global_max,
            fps=args.fps,
            colormap=args.colormap
        )

        print(f"  Encoding time: {time.time() - tic:.2f}s")
    else:
        print("\n[6/6] Skipping video encoding (--no-video)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input video:          {args.input}")
    print(f"  Frames processed:     {len(frames)}")
    print(f"  Altitude range:       [{min(altitudes):.1f}, {max(altitudes):.1f}] m")
    print(f"  Metric depth range:   [{global_min:.1f}, {global_max:.1f}] m")
    print(f"  Total time:           {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output directory:     {args.output}/")
    print("=" * 70)

    # Count RANSAC successes
    ransac_count = sum(1 for info in scaling_infos if info.get('method') == 'ransac')
    print(f"\nRANSAC Statistics:")
    print(f"  Successful fits:      {ransac_count}/{len(scaling_infos)} frames")
    print(f"  Fallback frames:      {len(scaling_infos) - ransac_count}")

    return 0


if __name__ == "__main__":
    exit(main())
