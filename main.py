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
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_dji_flight_record(csv_path: str, video_duration_sec: float, target_fps: float) -> dict:
    """
    Parse DJI flight record CSV and extract telemetry synchronized with video frames.

    The function:
    1. Reads the CSV file
    2. Filters rows where CAMERA.isVideo is TRUE (video recording active)
    3. Extracts altitude (AGL) and gimbal pitch angle
    4. Resamples to match the target frame count for the video

    Note: Only gimbal pitch is used for metric scaling (not drone pitch),
    because the gimbal is stabilized and independent of drone body orientation.
    If gimbal pitch data is all zeros (corrupted/missing), returns None to skip metric scaling.

    Args:
        csv_path: Path to DJI flight record CSV file
        video_duration_sec: Duration of the input video in seconds
        target_fps: Target frames per second for extraction

    Returns:
        dict with keys, or None if gimbal data is invalid:
            - 'altitudes': List of altitude values (meters) per frame
            - 'pitch_gimbal': List of gimbal pitch angles (degrees, DJI convention) per frame
            - 'pitch_below_horizontal': List of pitch angles below horizontal (0=horizon, 90=nadir) per frame
            - 'timestamps': List of timestamps per frame
    """
    import pandas as pd

    # Read CSV
    df = pd.read_csv(csv_path)

    # Find relevant columns (handle potential naming variations)
    # Priority: OSD.height (height above takeoff) > OSD.vpsHeight (VPS has limited range ~20-30m)
    # Note: OSD.altitude is GPS altitude (above sea level), NOT suitable for AGL
    altitude_col = None
    for col in ['OSD.height [m]', 'OSD.vpsHeight [m]']:
        if col in df.columns:
            altitude_col = col
            break

    if altitude_col is None:
        raise ValueError(f"Could not find altitude column in {csv_path}")

    print(f"  Using altitude column: {altitude_col}")

    # Required columns
    required_cols = {
        'altitude': altitude_col,
        'pitch_gimbal': 'GIMBAL.pitch',
        'is_video': 'CAMERA.isVideo',
        'timestamp': 'CUSTOM.updateTime [local]'
    }

    # Check all required columns exist
    for name, col in required_cols.items():
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available: {list(df.columns)[:20]}...")

    # Filter for video recording segments
    # Convert isVideo to boolean (handles 'TRUE'/'FALSE' strings or True/False)
    df['is_video_bool'] = df[required_cols['is_video']].apply(
        lambda x: str(x).upper() == 'TRUE' if pd.notna(x) else False
    )

    video_df = df[df['is_video_bool']].copy()

    if len(video_df) == 0:
        raise ValueError("No rows found where CAMERA.isVideo is TRUE")

    print(f"  Flight record: {len(df)} total rows, {len(video_df)} during video recording")

    # Extract telemetry from video segment
    altitudes_raw = video_df[required_cols['altitude']].values
    pitch_gimbal_raw = video_df[required_cols['pitch_gimbal']].values
    timestamps_raw = video_df[required_cols['timestamp']].values

    # Check if gimbal pitch data is valid (not all zeros)
    if np.all(pitch_gimbal_raw == 0):
        print("  WARNING: GIMBAL.pitch is all zeros (data not logged or corrupted)")
        print("  Metric depth scaling will be skipped.")
        return None

    # Handle NaN values
    altitudes_raw = np.nan_to_num(altitudes_raw, nan=np.nanmean(altitudes_raw))
    pitch_gimbal_raw = np.nan_to_num(pitch_gimbal_raw, nan=0.0)

    # Calculate target frame count
    target_frame_count = int(video_duration_sec * target_fps)

    print(f"  Video duration: {video_duration_sec:.1f}s, Target frames: {target_frame_count}")
    print(f"  Flight record video segment: {len(video_df)} samples")

    # Resample telemetry to match target frame count
    if len(altitudes_raw) != target_frame_count:
        # Linear interpolation to match frame count
        old_indices = np.linspace(0, len(altitudes_raw) - 1, len(altitudes_raw))
        new_indices = np.linspace(0, len(altitudes_raw) - 1, target_frame_count)

        altitudes = np.interp(new_indices, old_indices, altitudes_raw)
        pitch_gimbal = np.interp(new_indices, old_indices, pitch_gimbal_raw)

        # For timestamps, sample at indices
        timestamp_indices = np.linspace(0, len(timestamps_raw) - 1, target_frame_count).astype(int)
        timestamps = [timestamps_raw[i] for i in timestamp_indices]

        print(f"  Resampled {len(altitudes_raw)} → {target_frame_count} samples")
    else:
        altitudes = altitudes_raw
        pitch_gimbal = pitch_gimbal_raw
        timestamps = list(timestamps_raw)

    # Convert gimbal pitch to "pitch below horizontal" for scale_depth_to_metric
    # DJI GIMBAL.pitch convention: 0° = horizontal, negative = pointing down
    # Our convention: 0° = horizon, 90° = nadir (straight down)
    # Conversion: pitch_below_horizontal = -gimbal_pitch = |gimbal_pitch|
    pitch_below_horizontal = -pitch_gimbal  # e.g., -45° gimbal → 45° below horizontal

    # Clamp to valid range [5, 90] degrees (avoid near-horizon angles)
    pitch_below_horizontal = np.clip(pitch_below_horizontal, 5, 90)

    # Summary statistics
    print(f"  Altitude range: [{altitudes.min():.1f}, {altitudes.max():.1f}] m (mean: {altitudes.mean():.1f} m)")
    print(f"  Gimbal pitch range: [{pitch_gimbal.min():.1f}, {pitch_gimbal.max():.1f}]° (DJI convention)")
    print(f"  Pitch below horizontal: [{pitch_below_horizontal.min():.1f}, {pitch_below_horizontal.max():.1f}]°")

    return {
        'altitudes': list(altitudes),
        'pitch_gimbal': list(pitch_gimbal),
        'pitch_below_horizontal': list(pitch_below_horizontal),
        'timestamps': timestamps
    }


def scale_depth_to_metric(canonical_depth: np.ndarray,
                          altitude_meters: float,
                          pitch_angle_deg: float,
                          min_pitch_from_horizon: float = 5.0) -> np.ndarray:
    """
    Convert VGGT canonical depth map to metric depth (meters) using UAV telemetry.

    Geometric Derivation (Pinhole Camera Model):
    ============================================

    Consider a UAV at altitude h looking at a flat ground plane:

              -------- Horizontal --------
                      *  UAV (camera)
                     /|
                   θ/ |
         slant    /  | h (altitude)
         range   /   |
                /    |
               /     |
              /______|____________ Ground Plane

    Where:
        - h = altitude (vertical distance to ground)
        - θ (theta) = pitch angle below horizontal (0° = horizon, 90° = nadir)
                      measured between the optical axis and the horizontal plane
        - slant_range = distance along optical axis to ground

    Trigonometric Relationship:
        sin(θ) = h / slant_range

    Therefore:
        slant_range = h / sin(θ)

    DJI Gimbal Convention:
        - GIMBAL.pitch = 0° → horizontal (θ = 0°, but invalid for ground viewing)
        - GIMBAL.pitch = -45° → 45° below horizontal (θ = 45°)
        - GIMBAL.pitch = -90° → nadir/straight down (θ = 90°)
        - Conversion: θ = |GIMBAL.pitch| = -GIMBAL.pitch

    The VGGT model outputs "canonical" depth (scale-invariant, normalized).
    The depth at the principal point (image center) corresponds to the slant range.

    Scale Factor Calculation:
        s = real_slant_range / canonical_depth_at_center
        s = (h / sin(θ)) / d_center

    Metric Depth:
        D_metric = D_canonical × s

    Args:
        canonical_depth: 2D numpy array of canonical depth values from VGGT [H, W]
        altitude_meters: UAV height above ground in meters (h)
        pitch_angle_deg: Camera pitch angle below horizontal in degrees (θ)
                         0° = horizon (invalid for ground viewing)
                         90° = nadir (looking straight down)
                         This is the absolute value of DJI GIMBAL.pitch
        min_pitch_from_horizon: Minimum angle below horizon to avoid numerical
                                instability (default: 5°). If pitch < this,
                                the function will raise an error.

    Returns:
        metric_depth: 2D numpy array of depth values in meters [H, W]

    Raises:
        ValueError: If pitch angle is too close to horizon (0°) or invalid inputs
    """
    # Input validation
    if canonical_depth is None or canonical_depth.size == 0:
        raise ValueError("canonical_depth cannot be None or empty")

    if altitude_meters <= 0:
        raise ValueError(f"altitude_meters must be positive, got {altitude_meters}")

    if not 0 < pitch_angle_deg <= 90:
        raise ValueError(f"pitch_angle_deg must be in (0, 90], got {pitch_angle_deg}")

    # Check for near-horizon pitch (causes division by near-zero)
    if pitch_angle_deg < min_pitch_from_horizon:
        raise ValueError(
            f"pitch_angle_deg ({pitch_angle_deg}°) too close to horizon. "
            f"Minimum allowed: {min_pitch_from_horizon}° (sin → 0 causes instability)"
        )

    # Convert pitch angle to radians
    pitch_rad = np.deg2rad(pitch_angle_deg)

    # Calculate sin(pitch) - this is the projection factor
    # At nadir (90°): sin(90) = 1.0 → slant_range = altitude
    # At 45° below horizon: sin(45) ≈ 0.707 → slant_range ≈ 1.41 × altitude
    # At 5° below horizon: sin(5) ≈ 0.087 → slant_range ≈ 11.5 × altitude
    sin_pitch = np.sin(pitch_rad)

    # Calculate the real slant range along the optical axis (meters)
    # This is the actual distance from camera to ground along the viewing direction
    real_slant_range = altitude_meters / sin_pitch

    # Get canonical depth at principal point (image center)
    # The principal point corresponds to the optical axis
    h, w = canonical_depth.shape[:2]
    center_y, center_x = h // 2, w // 2

    # Use a small window around center for robustness against noise
    window_size = max(1, min(h, w) // 20)  # 5% of smaller dimension
    y_start = max(0, center_y - window_size)
    y_end = min(h, center_y + window_size + 1)
    x_start = max(0, center_x - window_size)
    x_end = min(w, center_x + window_size + 1)

    center_region = canonical_depth[y_start:y_end, x_start:x_end]
    d_center = np.median(center_region)  # Median is robust to outliers

    # Validate center depth
    if d_center <= 0 or not np.isfinite(d_center):
        raise ValueError(
            f"Invalid depth at principal point: {d_center}. "
            "The depth map may be corrupted or the scene has no valid depth at center."
        )

    # Calculate scale factor
    # s = real_world_distance / canonical_distance
    scale_factor = real_slant_range / d_center

    # Apply scale factor to entire depth map
    metric_depth = canonical_depth * scale_factor

    # Log useful debugging information
    print(f"  Metric depth scaling:")
    print(f"    Altitude: {altitude_meters:.2f} m")
    print(f"    Pitch below horizontal: {pitch_angle_deg:.1f}° (sin={sin_pitch:.4f})")
    print(f"    Real slant range: {real_slant_range:.2f} m")
    print(f"    Canonical depth at center: {d_center:.4f}")
    print(f"    Scale factor: {scale_factor:.4f}")
    print(f"    Metric depth range: [{metric_depth.min():.2f}, {metric_depth.max():.2f}] m")

    return metric_depth


def scale_depth_sequence_to_metric(canonical_depths: list,
                                   altitudes: list,
                                   pitch_angles: list) -> list:
    """
    Convert a sequence of canonical depth maps to metric depth using per-frame telemetry.

    Args:
        canonical_depths: List of 2D numpy arrays [N × [H, W]]
        altitudes: List of altitude values in meters [N]
        pitch_angles: List of pitch angles below horizontal in degrees [N]
                      (0° = horizon, 90° = nadir/straight down)

    Returns:
        List of metric depth maps [N × [H, W]]
    """
    if len(canonical_depths) != len(altitudes) or len(canonical_depths) != len(pitch_angles):
        raise ValueError(
            f"Length mismatch: depths({len(canonical_depths)}), "
            f"altitudes({len(altitudes)}), pitch_angles({len(pitch_angles)})"
        )

    metric_depths = []
    for i, (depth, alt, pitch) in enumerate(zip(canonical_depths, altitudes, pitch_angles)):
        print(f"  Frame {i+1}/{len(canonical_depths)}:")
        try:
            metric_depth = scale_depth_to_metric(depth, alt, pitch)
            metric_depths.append(metric_depth)
        except ValueError as e:
            print(f"    Warning: {e}. Using unscaled depth for frame {i+1}.")
            metric_depths.append(depth)

    return metric_depths


def extract_frames_from_video(video_path, fps=1):
    """Extract frames from video at specified fps and return list of frame arrays."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate duration from metadata
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate how many frames we want to extract based on desired fps and duration
    # This ensures output duration matches input duration
    target_frame_count = int(video_duration * fps)
    frame_interval = max(1, total_frames // target_frame_count) if target_frame_count > 0 else 1

    print(f"Video FPS: {video_fps}, Total frames: {total_frames}, Duration: {video_duration:.1f}s")
    print(f"Target: {fps} fps → extracting ~{target_frame_count} frames (1 every {frame_interval} frames)")
    print(f"Output video will be: {target_frame_count / fps:.1f}s at {fps} fps")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def save_frames_and_load(frames, tmp_dir):
    """Save frames to temp directory and load with VGGT preprocessing."""
    os.makedirs(tmp_dir, exist_ok=True)
    frame_paths = []

    for i, frame in enumerate(frames):
        frame_path = os.path.join(tmp_dir, f"{i:06d}.png")
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)

    images = load_and_preprocess_images(frame_paths)
    return images


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


def create_live_plot_frame(time_values, depth_values, current_idx, total_time,
                           target_height, target_width, y_min=None, y_max=None):
    """
    Create a live 2D plot frame showing accumulated center depth values over time.

    Args:
        time_values: List of time values (x-axis)
        depth_values: List of depth values (y-axis)
        current_idx: Current frame index (plot up to this point)
        total_time: Total video duration in seconds (fixed x-axis limit)
        target_height: Target height for the output image
        target_width: Target width for the output image
        y_min: Optional fixed y-axis minimum
        y_max: Optional fixed y-axis maximum

    Returns:
        numpy array of the plot image [H, W, 3]
    """
    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(target_width / 100, target_height / 100), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # Plot accumulated data up to current frame
    plot_times = time_values[:current_idx + 1]
    plot_depths = depth_values[:current_idx + 1]

    # Plot line with gradient effect
    if len(plot_times) > 1:
        ax.plot(plot_times, plot_depths, color='#00ff88', linewidth=2, alpha=0.9)

    # Mark current point
    if len(plot_times) > 0:
        ax.scatter([plot_times[-1]], [plot_depths[-1]], color='#ff4444', s=80, zorder=5)

    # Fixed x-axis limit
    ax.set_xlim(0, total_time)

    # Y-axis limits (use provided or auto-calculate from all data)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    else:
        all_depths = np.array(depth_values)
        margin = (all_depths.max() - all_depths.min()) * 0.1
        ax.set_ylim(all_depths.min() - margin, all_depths.max() + margin)

    # Styling
    ax.set_xlabel('Time (s)', color='white', fontsize=12)
    ax.set_ylabel('Center Depth (canonical)', color='white', fontsize=12)
    ax.set_title('Center Pixel Depth Over Time', color='white', fontsize=14, pad=10)

    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('#333333')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('#333333')
    ax.grid(True, alpha=0.3, color='#444444')

    # Add current time annotation
    if len(plot_times) > 0:
        ax.annotate(f't={plot_times[-1]:.2f}s\nd={plot_depths[-1]:.3f}',
                    xy=(plot_times[-1], plot_depths[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    color='white', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#333333', alpha=0.8))

    plt.tight_layout()

    # Convert to image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    plot_img = np.asarray(buf)[:, :, :3]  # Remove alpha
    plt.close(fig)

    # Resize to exact target dimensions
    plot_img = cv2.resize(plot_img, (target_width, target_height))

    return plot_img


def create_quadrant_frame(rgb_frame, canonical_depth, metric_depth,
                          time_values, center_depths, current_idx,
                          total_time, colormap='turbo',
                          canonical_range=None, metric_range=None):
    """
    Create a quadrant video frame with:
    - Top-left: Input RGB
    - Top-right: Canonical depth (VGGT)
    - Bottom-left: Live plot (time vs center depth)
    - Bottom-right: Metric depth

    Args:
        rgb_frame: Original RGB frame [H, W, 3]
        canonical_depth: Canonical depth map from VGGT [H, W]
        metric_depth: Metric depth map in meters [H, W] (or None)
        time_values: List of all time values
        center_depths: List of all center depth values
        current_idx: Current frame index
        total_time: Total video duration
        colormap: Matplotlib colormap name
        canonical_range: (min, max) for canonical depth colorbar
        metric_range: (min, max) for metric depth colorbar

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

    # Create live plot
    plot_frame = create_live_plot_frame(
        time_values, center_depths, current_idx, total_time,
        target_height=h, target_width=w
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
        cv2.putText(metric_labeled, 'Metric Depth (m)', (10, 30), font, font_scale, (255, 255, 255), thickness)
    else:
        cv2.putText(metric_labeled, 'Metric Depth (N/A)', (10, 30), font, font_scale, (128, 128, 128), thickness)

    # Combine into quadrant layout
    top_row = np.hstack([rgb_labeled, canonical_labeled])
    bottom_row = np.hstack([plot_frame, metric_labeled])
    quadrant = np.vstack([top_row, bottom_row])

    return quadrant


def encode_quadrant_video(frames, canonical_depths, metric_depths, output_path,
                          fps=10, colormap='turbo'):
    """
    Encode quadrant analysis video with live plot.

    Args:
        frames: List of RGB frames
        canonical_depths: List of canonical depth maps from VGGT
        metric_depths: List of metric depth maps (or None)
        output_path: Output video file path
        fps: Frames per second
        colormap: Matplotlib colormap name
    """
    import subprocess
    import shutil

    if len(frames) == 0 or len(canonical_depths) == 0:
        raise ValueError("No frames or depth maps to encode")

    # Extract center depths from canonical depth maps
    print("  Extracting center pixel depths...")
    center_depths = extract_center_depths(canonical_depths)

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
        time_values, center_depths, 0, total_time,
        colormap, canonical_range, metric_range
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
                time_values, center_depths, i, total_time,
                colormap, canonical_range, metric_range
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
                time_values, center_depths, i, total_time,
                colormap, canonical_range, metric_range
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
    cv2.putText(combined, 'Depth', (w + 10, 30), font, 1, (255, 255, 255), 2)

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
    print(f"  Global depth range: [{global_min:.2f}, {global_max:.2f}]")

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


def main():
    parser = argparse.ArgumentParser(description="VGGT Depth Estimation from Video")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--flight-record", "-f", type=str, default=None,
                        help="Path to DJI flight record CSV for metric depth scaling")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of frames to process at once (default: 20, reduce if OOM)")
    parser.add_argument("--save-points", action="store_true",
                        help="Also save world point maps")
    parser.add_argument("--save-cameras", action="store_true",
                        help="Also save camera parameters")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save side-by-side depth video (default: True)")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video output")
    parser.add_argument("--colormap", type=str, default="turbo",
                        help="Colormap for depth visualization (default: turbo)")
    parser.add_argument("--quadrant-video", action="store_true",
                        help="Save quadrant analysis video (RGB, canonical depth, live plot, metric depth)")
    args = parser.parse_args()

    # Start total timer
    total_start = time.time()

    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Extract frames from video
    print(f"\nExtracting frames from: {args.input}")
    tic = time.time()
    frames = extract_frames_from_video(args.input, fps=args.fps)
    extraction_time = time.time() - tic
    print(f"  Frame extraction time: {extraction_time:.2f}s")

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Save frames temporarily and load with preprocessing
    tmp_dir = os.path.join(args.output, "frames")
    print(f"\nPreprocessing {len(frames)} frames...")
    images = save_frames_and_load(frames, tmp_dir)
    images = images.to(device)
    print(f"Input tensor shape: {images.shape}")

    # Load model
    print("\nLoading VGGT model...")
    tic = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    model_load_time = time.time() - tic
    print(f"  Model loading time: {model_load_time:.2f}s")

    # Run inference in batches to avoid OOM
    print(f"\nRunning inference (batch size: {args.batch_size})...")
    tic = time.time()
    all_depths = []
    all_world_points = []
    all_pose_encs = []

    num_frames = images.shape[0]
    num_batches = (num_frames + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_frames)
        batch_images = images[start_idx:end_idx]

        print(f"  Processing batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx - 1})...")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                batch_predictions = model(batch_images)

        # Collect results
        if "depth" in batch_predictions:
            all_depths.append(batch_predictions["depth"].cpu())
        if "world_points" in batch_predictions:
            all_world_points.append(batch_predictions["world_points"].cpu())
        if "pose_enc" in batch_predictions:
            all_pose_encs.append(batch_predictions["pose_enc"].cpu())

        # Clear GPU memory
        del batch_predictions
        torch.cuda.empty_cache()

    inference_time = time.time() - tic
    print(f"  Inference time: {inference_time:.2f}s")

    # Combine batch results
    # Model outputs shape: [1, S, H, W, C] where S is sequence length
    # Concatenate along dim=1 (sequence), then squeeze dim=0
    predictions = {}
    if all_depths:
        predictions["depth"] = torch.cat(all_depths, dim=1).squeeze(0)  # [S, H, W, 1]
    if all_world_points:
        predictions["world_points"] = torch.cat(all_world_points, dim=1).squeeze(0)  # [S, H, W, 3]
    if all_pose_encs:
        pose_enc = torch.cat(all_pose_encs, dim=1)  # [1, S, 9]
        predictions["pose_enc"] = pose_enc.squeeze(0)  # [S, 9]
        # Decode pose encoding to extrinsics and intrinsics
        # Get image size from depth output (H, W) or from input images
        if all_depths:
            depth_shape = all_depths[0].shape  # [1, S, H, W, 1]
            image_size_hw = (depth_shape[2], depth_shape[3])
        else:
            # Fallback to input image size (after preprocessing)
            image_size_hw = (images.shape[2], images.shape[3])
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=image_size_hw, pose_encoding_type="absT_quaR_FoV"
        )
        predictions["extrinsic"] = extrinsics.squeeze(0)  # [S, 3, 4]
        predictions["intrinsic"] = intrinsics.squeeze(0)  # [S, 3, 3]
        print(f"  Decoded camera parameters: extrinsics {predictions['extrinsic'].shape}, intrinsics {predictions['intrinsic'].shape}")

    # Apply metric scaling if flight record is provided
    if args.flight_record and "depth" in predictions:
        print("\nApplying metric depth scaling from flight record...")
        tic = time.time()

        # Get video duration from extracted frames
        video_duration = len(frames) / args.fps

        # Parse flight record
        telemetry = parse_dji_flight_record(
            args.flight_record,
            video_duration_sec=video_duration,
            target_fps=args.fps
        )

        # Check if telemetry is valid (gimbal data was logged)
        if telemetry is not None:
            # Convert canonical depth to metric depth
            depth_maps_np = [d.cpu().numpy().squeeze() for d in predictions["depth"]]
            metric_depth_maps = scale_depth_sequence_to_metric(
                canonical_depths=depth_maps_np,
                altitudes=telemetry['altitudes'],
                pitch_angles=telemetry['pitch_below_horizontal']
            )

            # Update predictions with metric depth (convert back to tensor format)
            predictions["depth_metric"] = metric_depth_maps
            predictions["telemetry"] = telemetry

            metric_time = time.time() - tic
            print(f"  Metric scaling time: {metric_time:.2f}s")
        else:
            print("  Skipping metric scaling due to invalid gimbal data.")

    # Save outputs
    print("\nSaving outputs...")

    # Always save depth maps
    if "depth" in predictions:
        save_depth_maps(predictions["depth"], args.output)

    # Save metric depth maps if available
    if "depth_metric" in predictions:
        metric_depth_dir = os.path.join(args.output, "depth_metric")
        os.makedirs(metric_depth_dir, exist_ok=True)
        for i, depth in enumerate(predictions["depth_metric"]):
            np.save(os.path.join(metric_depth_dir, f"depth_metric_{i:04d}.npy"), depth)
        print(f"Saved {len(predictions['depth_metric'])} metric depth maps to {metric_depth_dir}/")

    # Optionally save point maps
    if args.save_points and "world_points" in predictions:
        save_point_maps(predictions["world_points"], args.output)

    # Optionally save camera parameters
    if args.save_cameras:
        save_cameras(predictions, args.output)

    # Encode depth video with colorbar
    if args.save_video and not args.no_video:
        print("\nEncoding depth video...")
        tic = time.time()
        video_output_path = os.path.join(args.output, "depth_video.mp4")

        # Use metric depth if available, otherwise use canonical depth
        if "depth_metric" in predictions:
            depth_maps_np = predictions["depth_metric"]
            print("  Using metric depth values (meters)")
        else:
            depth_maps_np = [d.cpu().numpy().squeeze() for d in predictions["depth"]]
            print("  Using canonical depth values (no flight record)")

        encode_depth_video(frames, depth_maps_np, video_output_path,
                           fps=args.fps, colormap=args.colormap)
        encoding_time = time.time() - tic
        print(f"  Video encoding time: {encoding_time:.2f}s")

    # Encode quadrant analysis video if requested
    if args.quadrant_video and "depth" in predictions:
        print("\nEncoding quadrant analysis video...")
        tic = time.time()
        quadrant_output_path = os.path.join(args.output, "quadrant_analysis.mp4")

        # Get canonical depth maps
        canonical_depths = [d.cpu().numpy().squeeze() for d in predictions["depth"]]

        # Get metric depth maps if available
        metric_depths = predictions.get("depth_metric", None)

        encode_quadrant_video(
            frames, canonical_depths, metric_depths,
            quadrant_output_path, fps=args.fps, colormap=args.colormap
        )
        quadrant_time = time.time() - tic
        print(f"  Quadrant video encoding time: {quadrant_time:.2f}s")

    # Print total time
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"Total processing time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"{'='*50}")
    print(f"Done! Outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
