#!/usr/bin/env python3
"""
Object State Estimation Without Flight Log

Estimates object 3D positions and velocities using VGGT depth estimation
when no flight log is available. Outputs are in drone-relative NED coordinates
with compatible JSON format (lat/lon = null).

Coordinate System (Drone-Relative NED):
  - The drone's viewing direction (when looking down) defines the reference frame
  - North (0°) = Drone's forward direction = Top of image
  - East (90°) = Drone's right direction = Right side of image
  - Down = Into the scene (depth direction)

  This means:
  - Object moving toward top of image → heading ≈ 0° (North/forward)
  - Object moving toward right of image → heading ≈ 90° (East/right)
  - Object moving toward bottom of image → heading ≈ 180° (South/backward)
  - Object moving toward left of image → heading ≈ 270° (West/left)

  NOTE: This is NOT true geographic North! It's relative to the drone's
  orientation at the first frame. Useful for understanding motion patterns
  relative to the drone's viewpoint.

Inputs:
  - Video file (MP4)
  - Object tracking JSON (bounding boxes, track IDs)

Outputs:
  - JSON with state estimates (pos_vggt=[N,E,D], vel_vggt=[vN,vE,vD], depth_units)
  - 2D bird's-eye visualization video (North-East plane)

Author: Claude Code Assistant
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

# For visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Warning: matplotlib not installed. Visualization disabled.")

# Class colors consistent with main_v2_state_est.py and main_v3_service.py
CLASS_COLORS = {
    'car': '#FF0000',       # Red
    'vehicle': '#0000FF',   # Blue
    'person': '#00FF00',    # Green
    'cycle': '#00FFFF',     # Cyan
    'bus': '#FF00FF',       # Magenta
    'UAV': '#FFA500',       # Orange (for drone/camera)
}
FALLBACK_COLORS = ['#FFD700', '#FF8C00', '#9400D3', '#00CED1', '#DC143C']


class KalmanFilter:
    """
    Kalman Filter for object tracking in VGGT coordinate system.

    State vector: x = [p_X, p_Y, p_Z, v_X, v_Y, v_Z]
    - Position: [p_X, p_Y, p_Z] in VGGT units
    - Velocity: [v_X, v_Y, v_Z] in VGGT units/frame

    Measurement: z = [z_X, z_Y, z_Z] (position only)

    Based on constant velocity motion model with discrete white noise acceleration.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        fps: float,
        initial_velocity: Optional[np.ndarray] = None,
        sigma_a: float = 0.1,  # Lower for VGGT units (arbitrary scale)
        sigma_meas: float = 0.5,  # Measurement noise (VGGT units)
        initial_pos_uncertainty: float = 1.0,
        initial_vel_uncertainty: float = 0.5
    ):
        """
        Initialize Kalman Filter.

        Args:
            initial_position: Initial position [X, Y, Z] in VGGT coords
            fps: Frames per second (dt = 1/fps)
            initial_velocity: Initial velocity (default: zero)
            sigma_a: Process noise std (acceleration) in VGGT units/frame²
            sigma_meas: Measurement noise std in VGGT units
            initial_pos_uncertainty: Initial position uncertainty std
            initial_vel_uncertainty: Initial velocity uncertainty std
        """
        self.dt = 1.0 / fps
        self.sigma_a = sigma_a

        # Initialize state vector [p_X, p_Y, p_Z, v_X, v_Y, v_Z]
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        self.x = np.concatenate([initial_position, initial_velocity])  # 6x1

        # Initialize state covariance matrix P
        p_var = initial_pos_uncertainty ** 2
        v_var = initial_vel_uncertainty ** 2
        self.P = np.diag([p_var, p_var, p_var, v_var, v_var, v_var])  # 6x6

        # State transition matrix F
        dt = self.dt
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])  # 6x6

        # Process noise covariance Q
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        q = sigma_a ** 2
        self.Q = q * np.array([
            [dt4/4, 0,     0,     dt3/2, 0,     0    ],
            [0,     dt4/4, 0,     0,     dt3/2, 0    ],
            [0,     0,     dt4/4, 0,     0,     dt3/2],
            [dt3/2, 0,     0,     dt2,   0,     0    ],
            [0,     dt3/2, 0,     0,     dt2,   0    ],
            [0,     0,     dt3/2, 0,     0,     dt2  ]
        ])  # 6x6

        # Observation matrix H (measures position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])  # 3x6

        # Measurement noise covariance R
        self.R = np.diag([sigma_meas ** 2] * 3)  # 3x3

        # Identity matrix for update step
        self.I = np.eye(6)

    def predict(self):
        """Prediction step: propagate state and covariance forward."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        """Update step: correct state estimate with measurement."""
        z = measurement
        nu = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ nu  # State update
        self.P = (self.I - K @ self.H) @ self.P  # Covariance update

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate (position, velocity)."""
        return self.x[:3].copy(), self.x[3:].copy()

    def get_position(self) -> np.ndarray:
        """Get current position estimate [X, Y, Z]."""
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate [vX, vY, vZ]."""
        return self.x[3:].copy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Object state estimation without flight log using VGGT'
    )

    # Required arguments
    parser.add_argument('-i', '--input-video', required=True,
                        help='Input video file (MP4)')
    parser.add_argument('-t', '--tracking-json', required=True,
                        help='Object tracking JSON file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output JSON file path')

    # Processing parameters
    parser.add_argument('--fps', type=float, default=5.0,
                        help='Frame extraction rate (default: 5.0)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='VGGT batch size (default: 5)')
    parser.add_argument('--source-res', type=int, nargs=2, default=[1920, 1080],
                        help='Source video resolution [W H] (default: 1920 1080)')

    # Kalman Filter parameters
    parser.add_argument('--no-kf', action='store_true',
                        help='Disable Kalman Filter (use raw measurements)')
    parser.add_argument('--kf-sigma-a', type=float, default=0.1,
                        help='KF process noise (acceleration) std (default: 0.1)')
    parser.add_argument('--kf-sigma-meas', type=float, default=0.5,
                        help='KF measurement noise std (default: 0.5)')

    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualization and depth maps')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization video generation')
    parser.add_argument('--save-depth', action='store_true',
                        help='Save depth maps to output directory')

    # Scale hint (optional)
    parser.add_argument('--scale-hint', type=float, default=None,
                        help='Optional scale factor (meters per VGGT unit). '
                             'If provided, outputs will be in approximate meters.')

    return parser.parse_args()


def extract_frames_from_video(video_path: str, fps: float = 1.0) -> Tuple[List[np.ndarray], float]:
    """
    Extract frames from video at specified fps.

    Returns:
        Tuple of (frames list, video duration in seconds)
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

    print(f"Video: {video_fps:.1f} fps, {total_frames} frames, {video_duration:.1f}s")
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


def run_vggt_inference(
    frames: List[np.ndarray],
    batch_size: int = 5,
    device: str = "cuda"
) -> Dict:
    """
    Run VGGT inference on frames.

    Returns:
        Dictionary with depth maps, extrinsics, intrinsics, world_points
    """
    print("\n" + "=" * 50)
    print("Running VGGT Inference")
    print("=" * 50)

    # Determine dtype based on GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability >= 8 else torch.float16
        print(f"Using device: {device}, dtype: {dtype}")
    else:
        dtype = torch.float32
        device = "cpu"
        print("Using CPU (no CUDA available)")

    # Load model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Prepare frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        images = save_frames_and_load(frames, tmp_dir)

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Processing {num_frames} frames in batches of {batch_size}")

    # Process in batches
    all_depths = []
    all_depth_conf = []
    all_world_points = []
    all_world_points_conf = []
    all_pose_enc = []

    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx + batch_size, num_frames)
        batch = images[start_idx:end_idx].unsqueeze(0)  # Add batch dim

        print(f"  Processing frames {start_idx+1}-{end_idx}...")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(batch)

        # Extract outputs
        all_depths.append(predictions["depth"].squeeze(0).cpu())
        all_depth_conf.append(predictions["depth_conf"].squeeze(0).cpu())
        all_world_points.append(predictions["world_points"].squeeze(0).cpu())
        all_world_points_conf.append(predictions["world_points_conf"].squeeze(0).cpu())
        all_pose_enc.append(predictions["pose_enc"].squeeze(0).cpu())

        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatenate results
    depths = torch.cat(all_depths, dim=0)  # [N, H, W, 1]
    depth_conf = torch.cat(all_depth_conf, dim=0)  # [N, H, W]
    world_points = torch.cat(all_world_points, dim=0)  # [N, H, W, 3]
    world_points_conf = torch.cat(all_world_points_conf, dim=0)  # [N, H, W]
    pose_enc = torch.cat(all_pose_enc, dim=0)  # [N, 9]

    # Convert pose encoding to extrinsics and intrinsics
    H, W = depths.shape[1:3]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc.unsqueeze(0),
        image_size_hw=(H, W)
    )
    extrinsics = extrinsics.squeeze(0)  # [N, 3, 4]
    intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]

    print(f"VGGT inference complete. Depth shape: {depths.shape}")

    return {
        "depth": depths.numpy(),
        "depth_conf": depth_conf.numpy(),
        "world_points": world_points.numpy(),
        "world_points_conf": world_points_conf.numpy(),
        "extrinsics": extrinsics.numpy(),
        "intrinsics": intrinsics.numpy(),
        "image_size": (H, W)
    }


def load_tracks(json_path: str) -> List[Dict]:
    """Load object tracks from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle nested structure with "tracks" key
    if isinstance(data, dict) and 'tracks' in data:
        tracks = data['tracks']
    elif isinstance(data, list):
        tracks = data
    else:
        raise ValueError("Unexpected JSON structure")

    print(f"Loaded {len(tracks)} object detections")
    return tracks


def rescale_bbox(bbox: List[float], sx: float, sy: float) -> List[float]:
    """Rescale bounding box from source resolution to depth map resolution."""
    x1, y1, x2, y2 = bbox
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def extract_robust_depth(
    depth_map: np.ndarray,
    bbox: List[float],
    percentile_threshold: float = 70.0
) -> Optional[float]:
    """
    Extract robust foreground depth using median with background rejection.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    h, w = depth_map.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    depth_roi = depth_map[y1:y2, x1:x2].flatten()
    valid_mask = (depth_roi > 0) & np.isfinite(depth_roi)
    valid_depths = depth_roi[valid_mask]

    if len(valid_depths) == 0:
        return None

    # Background rejection
    threshold = np.percentile(valid_depths, percentile_threshold)
    foreground_depths = valid_depths[valid_depths <= threshold]

    if len(foreground_depths) == 0:
        return None

    return float(np.median(foreground_depths))


def pixel_to_camera_frame(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray
) -> np.ndarray:
    """
    Unproject pixel to camera frame coordinates.

    Args:
        u, v: Pixel coordinates
        depth: Depth value
        K: 3x3 intrinsic matrix

    Returns:
        3D point [X, Y, Z] in camera frame (OpenCV convention: X=right, Y=down, Z=forward)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    X_cam = (u - cx) * depth / fx
    Y_cam = (v - cy) * depth / fy
    Z_cam = depth

    return np.array([X_cam, Y_cam, Z_cam])


def camera_to_ned(point_cam: np.ndarray) -> np.ndarray:
    """
    Transform point from camera frame to drone-relative NED frame.

    Assumes drone is looking down, so:
    - Camera X (right in image) → East
    - Camera -Y (up in image) → North (drone's forward)
    - Camera Z (depth, into scene) → Down

    Args:
        point_cam: 3D point [X, Y, Z] in camera frame

    Returns:
        3D point [N, E, D] in drone-relative NED frame
    """
    N = -point_cam[1]  # Up in image = North = drone forward
    E = point_cam[0]   # Right in image = East = drone right
    D = point_cam[2]   # Depth = Down

    return np.array([N, E, D])


def pixel_to_ned(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
    extrinsic: np.ndarray,
    reference_extrinsic: np.ndarray
) -> np.ndarray:
    """
    Unproject pixel to 3D point in drone-relative NED coordinates.

    The coordinate system uses the first frame's camera orientation as reference:
    - North (0°) = drone's forward direction (top of image when looking down)
    - East (90°) = drone's right direction (right side of image)
    - Down = into the scene (depth direction)

    Args:
        u, v: Pixel coordinates
        depth: Depth value (VGGT canonical)
        K: 3x3 intrinsic matrix
        extrinsic: 3x4 extrinsic matrix for current frame (camera-from-world)
        reference_extrinsic: 3x4 extrinsic matrix for first frame (defines NED axes)

    Returns:
        3D point [N, E, D] in drone-relative NED coordinates
    """
    # Unproject to camera frame
    point_cam = pixel_to_camera_frame(u, v, depth, K)

    # Transform to world coordinates
    cam_to_world = closed_form_inverse_se3(extrinsic[np.newaxis])[0]
    R_cam = cam_to_world[:3, :3]
    t_cam = cam_to_world[:3, 3]
    point_world = R_cam @ point_cam + t_cam

    # Transform to reference (first) camera frame
    # This makes all points consistent relative to first camera's orientation
    R_ref = reference_extrinsic[:3, :3]
    t_ref = reference_extrinsic[:3, 3]
    point_ref_cam = R_ref @ point_world + t_ref

    # Apply NED convention
    point_ned = camera_to_ned(point_ref_cam)

    return point_ned


def pixel_to_world_vggt(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
    extrinsic: np.ndarray
) -> np.ndarray:
    """
    Unproject pixel to 3D world point using VGGT camera parameters.
    (Legacy function - now use pixel_to_ned for NED coordinates)

    Args:
        u, v: Pixel coordinates
        depth: Depth value (VGGT canonical)
        K: 3x3 intrinsic matrix
        extrinsic: 3x4 extrinsic matrix (camera-from-world)

    Returns:
        3D point [X, Y, Z] in VGGT world coordinates
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Unproject to camera frame
    X_cam = (u - cx) * depth / fx
    Y_cam = (v - cy) * depth / fy
    Z_cam = depth
    point_cam = np.array([X_cam, Y_cam, Z_cam])

    # Get camera-to-world transform
    cam_to_world = closed_form_inverse_se3(extrinsic[np.newaxis])[0]
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]

    # Transform to world coordinates
    point_world = R @ point_cam + t

    return point_world


def compute_heading_ned(velocity: np.ndarray) -> float:
    """
    Compute heading from velocity vector in drone-relative NED frame.

    The heading uses the drone's forward direction (top of image when looking down)
    as the reference "North" (0°). This is NOT true geographic North, but a
    drone-relative reference that makes the visualization intuitive:

    - If object moves toward top of image → heading ≈ 0° (North/forward)
    - If object moves toward right of image → heading ≈ 90° (East/right)
    - If object moves toward bottom of image → heading ≈ 180° (South/backward)
    - If object moves toward left of image → heading ≈ 270° (West/left)

    Convention (NED):
    - 0° = North (drone forward, top of image)
    - 90° = East (drone right, right of image)
    - 180° = South (drone backward, bottom of image)
    - 270° = West (drone left, left of image)

    Args:
        velocity: Velocity vector [vN, vE, vD] in NED frame

    Returns:
        Heading in degrees [0, 360)
    """
    vN, vE = velocity[0], velocity[1]
    speed_h = np.sqrt(vN**2 + vE**2)

    if speed_h < 1e-6:
        return 0.0

    # atan2(vE, vN) gives angle from North (clockwise positive)
    heading = np.degrees(np.arctan2(vE, vN))

    # Normalize to [0, 360)
    if heading < 0:
        heading += 360.0

    return heading


def compute_heading(velocity: np.ndarray) -> float:
    """
    Legacy function - now calls compute_heading_ned.
    Velocity should be in NED frame [vN, vE, vD].
    """
    return compute_heading_ned(velocity)


def process_tracks_no_flight(
    tracks: List[Dict],
    vggt_outputs: Dict,
    source_res: Tuple[int, int],
    fps: float,
    use_kf: bool = True,
    kf_sigma_a: float = 0.1,
    kf_sigma_meas: float = 0.5,
    scale_hint: Optional[float] = None
) -> List[Dict]:
    """
    Process tracks and estimate state in drone-relative NED coordinate system.

    The coordinate system uses the first frame's camera orientation as reference:
    - North (0°) = drone's forward direction (top of image when looking down)
    - East (90°) = drone's right direction (right side of image)
    - Down = into the scene (depth direction)

    Args:
        tracks: List of track dictionaries
        vggt_outputs: VGGT inference outputs
        source_res: Source video resolution (W, H)
        fps: Frames per second
        use_kf: Use Kalman Filter for smoothing
        kf_sigma_a: KF process noise
        kf_sigma_meas: KF measurement noise
        scale_hint: Optional scale factor (meters per VGGT unit)

    Returns:
        Processed tracks with state estimates in NED coordinates
    """
    depth_maps = vggt_outputs["depth"]
    extrinsics = vggt_outputs["extrinsics"]
    intrinsics = vggt_outputs["intrinsics"]
    depth_h, depth_w = vggt_outputs["image_size"]

    source_w, source_h = source_res
    sx = depth_w / source_w
    sy = depth_h / source_h
    print(f"Scale factors: sx={sx:.4f}, sy={sy:.4f}")

    # Use first frame's extrinsic as reference for NED coordinates
    reference_extrinsic = extrinsics[0]
    print("Using first frame as NED reference (North = drone forward direction)")

    # Group tracks by track_id
    tracks_by_id = {}
    for track in tracks:
        obj_id = track.get('track_id', track.get('id', id(track)))
        if obj_id not in tracks_by_id:
            tracks_by_id[obj_id] = []
        tracks_by_id[obj_id].append(track)

    # Sort each group by frame_id
    for obj_id in tracks_by_id:
        tracks_by_id[obj_id].sort(key=lambda x: x['frame_id'])

    print(f"Processing {len(tracks_by_id)} unique tracks")

    processed_tracks = []

    if use_kf:
        print(f"Using Kalman Filter (sigma_a={kf_sigma_a}, sigma_meas={kf_sigma_meas})")

    for obj_id, obj_tracks in tracks_by_id.items():
        kf = None
        track_data = []

        # First pass: collect measurements in NED coordinates
        for track in obj_tracks:
            frame_id = track['frame_id']
            bbox = track['bbox']

            if frame_id >= len(depth_maps):
                continue

            depth_map = depth_maps[frame_id].squeeze()  # [H, W]
            K = intrinsics[min(frame_id, len(intrinsics) - 1)]
            ext = extrinsics[min(frame_id, len(extrinsics) - 1)]

            # Rescale bbox
            bbox_scaled = rescale_bbox(bbox, sx, sy)

            # Get depth
            depth = extract_robust_depth(depth_map, bbox_scaled)
            if depth is None or depth <= 0:
                continue

            # Get bbox center
            u, v = get_bbox_center(bbox_scaled)

            # Unproject to NED coordinates (using first frame as reference)
            point_ned = pixel_to_ned(u, v, depth, K, ext, reference_extrinsic)

            track_data.append((frame_id, track.copy(), point_ned, depth))

        if not track_data:
            continue

        # Second pass: apply Kalman Filter or raw output
        prev_frame_id = None

        for frame_id, track, measurement, depth in track_data:
            if use_kf:
                if kf is None:
                    kf = KalmanFilter(
                        initial_position=measurement,
                        fps=fps,
                        sigma_a=kf_sigma_a,
                        sigma_meas=kf_sigma_meas
                    )
                    filtered_pos = measurement
                    filtered_vel = np.zeros(3)
                else:
                    # Handle frame gaps
                    if prev_frame_id is not None:
                        frame_gap = frame_id - prev_frame_id
                        for _ in range(frame_gap):
                            kf.predict()

                    kf.update(measurement)
                    filtered_pos, filtered_vel = kf.get_state()

                prev_frame_id = frame_id
            else:
                filtered_pos = measurement
                filtered_vel = np.zeros(3)

            # Compute speed and heading in NED frame
            # Horizontal speed uses North and East components
            speed = np.sqrt(filtered_vel[0]**2 + filtered_vel[1]**2)
            heading = compute_heading_ned(filtered_vel)

            # Apply scale hint if provided
            if scale_hint is not None:
                pos_out = (filtered_pos * scale_hint).tolist()
                vel_out = (filtered_vel * scale_hint * fps).tolist()
                speed_out = float(speed * scale_hint * fps)
                depth_out = float(depth * scale_hint)
                coord_system = "ned_metric_approx"
            else:
                pos_out = filtered_pos.tolist()
                vel_out = (filtered_vel * fps).tolist()  # Convert to units/second
                speed_out = float(speed * fps)
                depth_out = float(depth)
                coord_system = "ned_drone_relative"

            # Create output entry (compatible with main_v3 format)
            track_processed = {
                'frame_id': frame_id,
                'track_id': track.get('track_id', track.get('id')),
                'bbox': track['bbox'],
                'confidence': track.get('confidence'),
                'class_name': track.get('class_name', track.get('class', 'unknown')),
                # Position - lat/lon null for compatibility
                'lat': None,
                'lon': None,
                # Depth in arbitrary units (no metric reference without flight log)
                'depth_units': depth_out,
                # Position [North, East, Down] in drone-relative frame
                'pos_vggt': pos_out,
                # Velocity [vN, vE, vD] in drone-relative frame
                'vel_vggt': vel_out,
                # Derived quantities
                'speed_units': speed_out,
                'heading_deg': float(heading),
                # Metadata
                'coordinate_system': coord_system
            }

            processed_tracks.append(track_processed)

    # Sort by frame_id, then track_id
    processed_tracks.sort(key=lambda x: (x['frame_id'], x['track_id']))

    return processed_tracks


def create_camera_tracks(
    vggt_outputs: Dict,
    fps: float,
    scale_hint: Optional[float] = None
) -> List[Dict]:
    """
    Create camera/UAV track entries from VGGT extrinsics in NED coordinates.

    The first frame defines the origin of the NED coordinate system.
    - Camera position at frame 0 is approximately [0, 0, 0]
    - Subsequent positions are relative to frame 0

    Returns track entries for track_id=0 (camera position each frame).
    """
    extrinsics = vggt_outputs["extrinsics"]
    num_frames = len(extrinsics)

    # Use first frame as reference
    reference_extrinsic = extrinsics[0]

    camera_tracks = []
    prev_pos = None

    for frame_id in range(num_frames):
        ext = extrinsics[frame_id]

        # Get camera position in world coordinates
        cam_to_world = closed_form_inverse_se3(ext[np.newaxis])[0]
        pos_world = cam_to_world[:3, 3]

        # Transform to first camera's frame
        R_ref = reference_extrinsic[:3, :3]
        t_ref = reference_extrinsic[:3, 3]
        pos_ref_cam = R_ref @ pos_world + t_ref

        # Apply NED convention
        pos_ned = camera_to_ned(pos_ref_cam)

        # Compute velocity from position change in NED
        if prev_pos is not None:
            vel = (pos_ned - prev_pos) * fps
        else:
            vel = np.zeros(3)
        prev_pos = pos_ned.copy()

        # Horizontal speed and heading in NED
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        heading = compute_heading_ned(vel)

        # Apply scale hint
        if scale_hint is not None:
            pos_out = (pos_ned * scale_hint).tolist()
            vel_out = (vel * scale_hint).tolist()
            speed_out = float(speed * scale_hint)
            coord_system = "ned_metric_approx"
        else:
            pos_out = pos_ned.tolist()
            vel_out = vel.tolist()
            speed_out = float(speed)
            coord_system = "ned_drone_relative"

        camera_track = {
            'frame_id': frame_id,
            'track_id': 0,  # Reserved for camera/UAV
            'bbox': None,
            'confidence': None,
            'class_name': 'UAV',
            'lat': None,
            'lon': None,
            'depth_units': None,  # N/A for camera itself
            'pos_vggt': pos_out,
            'vel_vggt': vel_out,
            'speed_units': speed_out,
            'heading_deg': float(heading),
            'coordinate_system': coord_system
        }

        camera_tracks.append(camera_track)

    return camera_tracks


def get_class_color(class_name: str, fallback_idx: int = 0) -> str:
    """
    Get color for a class name, matching main_v2_state_est.py and main_v3_service.py.

    Args:
        class_name: Object class name
        fallback_idx: Index for fallback color if class not in predefined list

    Returns:
        Color string (hex or named color)
    """
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


def create_visualization_video(
    processed_tracks: List[Dict],
    output_path: str,
    fps: float = 5.0,
    num_frames: Optional[int] = None
):
    """
    Create 2D bird's-eye view video visualization in North-East plane.

    The visualization shows objects from a top-down perspective:
    - X-axis = East (drone's right direction)
    - Y-axis = North (drone's forward direction)
    - North arrow indicator in corner
    - Velocity vectors show direction of movement
    """
    if not HAS_VIZ:
        print("Visualization dependencies not available, skipping")
        return

    print("Creating visualization video (North-East plane)...")

    # Group tracks by frame_id
    tracks_by_frame = {}
    for track in processed_tracks:
        frame_id = track.get('frame_id')
        if frame_id is not None:
            if frame_id not in tracks_by_frame:
                tracks_by_frame[frame_id] = []
            tracks_by_frame[frame_id].append(track)

    if not tracks_by_frame:
        print("No tracks to visualize")
        return

    # Get unique classes and assign colors (consistent with v2/v3)
    classes = set()
    for track in processed_tracks:
        cls = track.get('class_name', 'unknown')
        classes.add(cls)

    class_colors = {}
    fallback_idx = 0
    for cls in sorted(classes):
        class_colors[cls] = get_class_color(cls, fallback_idx)
        if cls not in CLASS_COLORS:
            fallback_idx += 1

    # Compute bounds from all positions (pos_vggt = [N, E, D])
    # For visualization: X-axis = East, Y-axis = North
    all_east, all_north = [], []
    for track in processed_tracks:
        pos = track.get('pos_vggt')
        if pos is not None and len(pos) >= 2:
            all_north.append(pos[0])  # N
            all_east.append(pos[1])   # E

    if not all_east:
        print("No valid positions to visualize")
        return

    east_min, east_max = min(all_east), max(all_east)
    north_min, north_max = min(all_north), max(all_north)

    # Add padding
    east_range = max(east_max - east_min, 1.0)
    north_range = max(north_max - north_min, 1.0)
    padding = 0.1 * max(east_range, north_range)

    east_min -= padding
    east_max += padding
    north_min -= padding
    north_max += padding

    # Determine frame range
    if num_frames is None:
        max_frame = max(tracks_by_frame.keys())
    else:
        max_frame = num_frames - 1

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    # Camera trajectory
    camera_positions = []

    for frame_idx in range(max_frame + 1):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('#f0f0f0')

        # Set axis limits (X = East, Y = North)
        ax.set_xlim(east_min, east_max)
        ax.set_ylim(north_min, north_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Collect camera position for trajectory
        frame_tracks = tracks_by_frame.get(frame_idx, [])
        for track in frame_tracks:
            if track['track_id'] == 0:  # UAV
                pos = track.get('pos_vggt')
                if pos is not None and len(pos) >= 2:
                    camera_positions.append((pos[1], pos[0]))  # (East, North) for plot

        # Draw camera trajectory
        if len(camera_positions) > 1:
            traj_east = [p[0] for p in camera_positions]
            traj_north = [p[1] for p in camera_positions]
            ax.plot(traj_east, traj_north, color=CLASS_COLORS['UAV'], linestyle='-',
                   linewidth=2, alpha=0.5, label='Drone path')

        # Draw objects
        for track in frame_tracks:
            pos = track.get('pos_vggt')
            if pos is None or len(pos) < 2:
                continue

            # pos = [N, E, D], plot X=E, Y=N
            north, east = pos[0], pos[1]
            cls = track.get('class_name', 'unknown')
            color = class_colors.get(cls, '#FFFFFF')

            # Draw point
            if track['track_id'] == 0:  # UAV
                ax.scatter(east, north, c=CLASS_COLORS['UAV'], s=150, marker='^',
                          edgecolors='white', linewidth=2, zorder=5)
            else:
                ax.scatter(east, north, c=color, s=100, edgecolors='white',
                          linewidth=1.5, zorder=4, alpha=0.9)

                # Draw velocity vector
                vel = track.get('vel_vggt')
                if vel is not None and len(vel) >= 2:
                    scale = 0.2  # Scale factor for velocity arrows
                    vel_north, vel_east = vel[0], vel[1]
                    d_east = vel_east * scale
                    d_north = vel_north * scale
                    if abs(d_east) > 0.01 or abs(d_north) > 0.01:
                        ax.annotate('', xy=(east + d_east, north + d_north),
                                   xytext=(east, north),
                                   arrowprops=dict(arrowstyle='->', color=color,
                                                 lw=2, alpha=0.8),
                                   zorder=4)

        # Draw North arrow indicator in top-left corner
        arrow_x = east_min + 0.08 * (east_max - east_min)
        arrow_y = north_max - 0.08 * (north_max - north_min)
        arrow_len = 0.05 * max(east_max - east_min, north_max - north_min)
        ax.annotate('', xy=(arrow_x, arrow_y),
                   xytext=(arrow_x, arrow_y - arrow_len),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(arrow_x + arrow_len * 0.3, arrow_y, 'N', fontsize=12,
               fontweight='bold', ha='left', va='center')

        # Frame info
        ax.text(0.02, 0.98, f'Frame: {frame_idx}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Legend (compact, at upper right)
        legend_handles = []
        for cls in sorted(classes):
            color = class_colors[cls]
            marker = '^' if cls == 'UAV' else 'o'
            handle = plt.scatter([], [], c=color, s=60, marker=marker,
                                edgecolors='white', linewidth=1, label=cls)
            legend_handles.append(handle)

        if legend_handles:
            legend = ax.legend(handles=legend_handles, loc='upper right',
                             framealpha=0.9, fontsize=9)
            legend.get_frame().set_facecolor('white')

        ax.set_xlabel('East (drone-relative units)', fontsize=10)
        ax.set_ylabel('North (drone-relative units)', fontsize=10)
        ax.set_title('Object Tracking (Drone-Relative NED Frame)\n'
                    'North = Drone Forward | East = Drone Right',
                    fontsize=11, fontweight='bold')

        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, facecolor='white')
        plt.close(fig)
        frame_paths.append(frame_path)

        if (frame_idx + 1) % 20 == 0 or frame_idx == max_frame:
            print(f"    Generated {frame_idx + 1}/{max_frame + 1} frames")

    # Compile video
    print("  Compiling video...")

    if not output_path.endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'

    first_frame = cv2.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"Video saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Object State Estimation (No Flight Log)")
    print("=" * 60)

    # Load tracking data
    tracks = load_tracks(args.tracking_json)

    # Extract frames from video
    frames, video_duration = extract_frames_from_video(args.input_video, args.fps)

    # Run VGGT inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vggt_outputs = run_vggt_inference(frames, args.batch_size, device)

    # Save depth maps if requested
    if args.save_depth and args.output_dir:
        depth_dir = Path(args.output_dir) / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        for i, depth in enumerate(vggt_outputs["depth"]):
            np.save(depth_dir / f"depth_{i:04d}.npy", depth)
        print(f"Saved depth maps to {depth_dir}")

    # Process tracks
    source_res = tuple(args.source_res)
    print(f"\nSource resolution: {source_res[0]}x{source_res[1]}")
    print(f"FPS: {args.fps}")

    processed_tracks = process_tracks_no_flight(
        tracks=tracks,
        vggt_outputs=vggt_outputs,
        source_res=source_res,
        fps=args.fps,
        use_kf=not args.no_kf,
        kf_sigma_a=args.kf_sigma_a,
        kf_sigma_meas=args.kf_sigma_meas,
        scale_hint=args.scale_hint
    )

    # Create camera tracks
    camera_tracks = create_camera_tracks(
        vggt_outputs=vggt_outputs,
        fps=args.fps,
        scale_hint=args.scale_hint
    )

    # Combine and sort
    all_tracks = camera_tracks + processed_tracks
    all_tracks.sort(key=lambda x: (x['frame_id'], x['track_id']))

    print(f"\nProcessed {len(processed_tracks)} object entries + {len(camera_tracks)} camera entries")

    # Save output JSON with metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_json = {
        "metadata": {
            "source_resolution": {"width": source_res[0], "height": source_res[1]},
            "depth_resolution": {"height": vggt_outputs["image_size"][0],
                                "width": vggt_outputs["image_size"][1]},
            "fps": args.fps,
            "total_frames": len(vggt_outputs["depth"]),
            "coordinate_system": {
                "type": "ned_drone_relative" if args.scale_hint is None else "ned_metric_approx",
                "description": "Drone-relative NED frame (North=drone forward, East=drone right)",
                "north_reference": "Drone forward direction (top of image when looking down)",
                "east_reference": "Drone right direction (right side of image)",
                "down_reference": "Depth into scene",
                "scale_hint": args.scale_hint
            },
            "kalman_filter": {
                "enabled": not args.no_kf,
                "sigma_a": args.kf_sigma_a if not args.no_kf else None,
                "sigma_meas": args.kf_sigma_meas if not args.no_kf else None
            }
        },
        "tracks": all_tracks
    }

    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)

    print(f"Saved state estimates to: {output_path}")

    # Create visualization video
    if not args.no_viz and HAS_VIZ:
        viz_dir = args.output_dir or output_path.parent
        viz_path = Path(viz_dir) / f"{output_path.stem}_map.mp4"
        num_frames = len(vggt_outputs["depth"])
        create_visualization_video(
            all_tracks,
            str(viz_path),
            fps=args.fps,
            num_frames=num_frames
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
