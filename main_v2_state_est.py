#!/usr/bin/env python3
"""
Object Geolocalization and State Estimation Script

Fuses computer vision data (depth maps, object tracks) with UAV telemetry
to estimate object positions in global coordinates (LLA) and velocities.

Author: Claude Code Assistant
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# For geodetic conversions
try:
    import pymap3d as pm
except ImportError:
    pm = None
    print("Warning: pymap3d not installed. Install with: pip install pymap3d")

# For visualization
try:
    import matplotlib.pyplot as plt
    import contextily as ctx
    from matplotlib.colors import to_rgba
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Warning: matplotlib/contextily not installed. Visualization disabled.")


class KalmanFilter:
    """
    Kalman Filter for object tracking in NED frame.

    State vector: x = [p_N, p_E, p_D, v_N, v_E, v_D]
    - Position: [p_N, p_E, p_D] in meters (NED frame)
    - Velocity: [v_N, v_E, v_D] in m/s (NED frame)

    Measurement: z = [z_N, z_E, z_D] (position only)

    Based on constant velocity motion model with discrete white noise acceleration.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        fps: float,
        initial_velocity: Optional[np.ndarray] = None,
        sigma_a: float = 0.5,
        sigma_meas_horizontal: float = 5.0,
        sigma_meas_vertical: float = 2.0,
        initial_pos_uncertainty: float = 5.0,
        initial_vel_uncertainty: float = 2.0
    ):
        """
        Initialize Kalman Filter.

        Args:
            initial_position: Initial position [p_N, p_E, p_D] in meters
            fps: Frames per second (dt = 1/fps)
            initial_velocity: Initial velocity [v_N, v_E, v_D] in m/s (default: zero)
            sigma_a: Process noise standard deviation (acceleration) in m/s²
            sigma_meas_horizontal: Measurement noise std for N, E in meters
            sigma_meas_vertical: Measurement noise std for D in meters
            initial_pos_uncertainty: Initial position uncertainty std in meters
            initial_vel_uncertainty: Initial velocity uncertainty std in m/s
        """
        self.dt = 1.0 / fps
        self.sigma_a = sigma_a

        # Initialize state vector [p_N, p_E, p_D, v_N, v_E, v_D]
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

        # Process noise covariance Q (discrete white noise acceleration model)
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
        self.R = np.diag([
            sigma_meas_horizontal ** 2,
            sigma_meas_horizontal ** 2,
            sigma_meas_vertical ** 2
        ])  # 3x3

        # Identity matrix for update step
        self.I = np.eye(6)

    def predict(self):
        """
        Prediction step: propagate state and covariance forward in time.

        x_{k|k-1} = F * x_{k-1|k-1}
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        # State prediction
        self.x = self.F @ self.x

        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray):
        """
        Update step: correct state estimate with measurement.

        Args:
            measurement: Position measurement [z_N, z_E, z_D] in meters

        Steps:
        1. Innovation: ν = z - H * x_{k|k-1}
        2. Innovation covariance: S = H * P_{k|k-1} * H^T + R
        3. Kalman gain: K = P_{k|k-1} * H^T * S^{-1}
        4. State update: x_{k|k} = x_{k|k-1} + K * ν
        5. Covariance update: P_{k|k} = (I - K * H) * P_{k|k-1}
        """
        # Innovation (measurement residual)
        z = measurement
        nu = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ nu

        # Covariance update (Joseph form for numerical stability)
        self.P = (self.I - K @ self.H) @ self.P

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate.

        Returns:
            Tuple of (position, velocity)
            - position: [p_N, p_E, p_D] in meters
            - velocity: [v_N, v_E, v_D] in m/s
        """
        position = self.x[:3]
        velocity = self.x[3:]
        return position, velocity

    def get_position(self) -> np.ndarray:
        """Get current position estimate [p_N, p_E, p_D]."""
        return self.x[:3]

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate [v_N, v_E, v_D]."""
        return self.x[3:]

    def get_covariance(self) -> np.ndarray:
        """Get current state covariance matrix P (6x6)."""
        return self.P.copy()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Object geolocalization and state estimation from UAV data"
    )
    parser.add_argument(
        "--input-json", "-i",
        type=str,
        required=True,
        help="Path to input object tracks JSON file"
    )
    parser.add_argument(
        "--flight-record", "-f",
        type=str,
        required=True,
        help="Path to DJI flight record CSV file"
    )
    parser.add_argument(
        "--depth-dir", "-d",
        type=str,
        required=True,
        help="Directory containing metric depth maps (*.npy)"
    )
    parser.add_argument(
        "--intrinsics", "-k",
        type=str,
        required=True,
        help="Path to camera intrinsics .npy file [N, 3, 3]"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tracks_state_estimated.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--source-res",
        type=int,
        nargs=2,
        default=[1920, 1080],
        metavar=("WIDTH", "HEIGHT"),
        help="Source video resolution for bbox scaling (default: 1920 1080)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for velocity calculation"
    )
    parser.add_argument(
        "--velocity-window",
        type=int,
        default=5,
        help="Moving average window size (only used with --no-kf, deprecated)"
    )
    parser.add_argument(
        "--no-kf",
        action="store_true",
        help="Disable Kalman Filter (use simple moving average instead). KF is enabled by default."
    )
    parser.add_argument(
        "--kf-sigma-a",
        type=float,
        default=0.5,
        help="[KEY TUNING PARAM] KF process noise: acceleration std in m/s². Lower=smoother/slower, Higher=responsive/noisier. (default: 0.5)"
    )
    parser.add_argument(
        "--kf-sigma-meas-h",
        type=float,
        default=5.0,
        help="KF measurement noise: horizontal (N,E) position std in meters. Main jitter source. (default: 5.0)"
    )
    parser.add_argument(
        "--kf-sigma-meas-v",
        type=float,
        default=2.0,
        help="KF measurement noise: vertical (D) position std in meters. (default: 2.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualization (default: same as output JSON)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization output"
    )
    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
        help="Yaw offset in degrees for calibration (added to computed absolute yaw)"
    )
    parser.add_argument(
        "--magnetic-declination",
        type=float,
        default=0.0,
        help="Magnetic declination in degrees (positive=East, negative=West). Check https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml"
    )
    parser.add_argument(
        "--add-drone-yaw",
        action="store_true",
        help="Add drone heading (OSD.yaw) to gimbal yaw. Default: OFF (gimbal yaw is typically already absolute in DJI logs)"
    )
    parser.add_argument(
        "--use-osd-yaw",
        action="store_true",
        help="Use OSD.yaw (drone heading) instead of GIMBAL.yaw for yaw angle"
    )
    return parser.parse_args()


def load_flight_record(csv_path: str) -> Tuple[pd.DataFrame, Tuple[float, float, float]]:
    """
    Load and parse DJI flight record CSV.

    Filters to only include rows where CAMERA.isVideo is true.

    Returns:
        Tuple of (DataFrame, HOME position)
        - DataFrame with columns: timestamp, latitude, longitude, altitude,
                                 gimbal_pitch, gimbal_yaw, gimbal_roll,
                                 vel_north, vel_east, vel_down
        - HOME position: (latitude, longitude, height) in degrees and meters
    """
    print(f"Loading flight record: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # Find relevant columns (DJI naming conventions vary)
    col_mapping = {}
    df_cols = df.columns.tolist()

    def find_col(candidates, df_cols):
        """Find first matching column, supporting partial matches for bracketed units."""
        for c in candidates:
            if c in df_cols:
                return c
        # Try partial matching for columns with units like [m], [deg]
        for c in candidates:
            for df_c in df_cols:
                if df_c.startswith(c + ' [') or df_c == c:
                    return df_c
        return None

    # Try different column naming conventions
    lat_cols = ['OSD.latitude']
    lon_cols = ['OSD.longitude']
    alt_cols = ['OSD.height', 'OSD.altitude']  # Will match OSD.height [m] etc.
    pitch_cols = ['GIMBAL.pitch']
    yaw_cols = ['GIMBAL.yaw']
    roll_cols = ['GIMBAL.roll']
    drone_yaw_cols = ['OSD.yaw']  # Drone heading (absolute)
    drone_roll_cols = ['OSD.roll']  # Drone body roll (for FPV mode)
    drone_pitch_cols = ['OSD.pitch']  # Drone body pitch (for FPV mode)
    gimbal_mode_cols = ['GIMBAL.mode']  # Gimbal mode (Follow Yaw, Free, etc.)
    video_cols = ['CAMERA.isVideo', 'isVideo']
    time_cols = ['OSD.flyTime', 'time', 'Time', 'timestamp']
    home_lat_cols = ['HOME.latitude']
    home_lon_cols = ['HOME.longitude']
    home_height_cols = ['HOME.height']
    xspeed_cols = ['OSD.xSpeed']  # North velocity (NED)
    yspeed_cols = ['OSD.ySpeed']  # East velocity (NED)
    zspeed_cols = ['OSD.zSpeed']  # Down velocity (NED)

    col_mapping['latitude'] = find_col(lat_cols, df_cols)
    col_mapping['longitude'] = find_col(lon_cols, df_cols)
    col_mapping['altitude'] = find_col(alt_cols, df_cols)
    col_mapping['gimbal_pitch'] = find_col(pitch_cols, df_cols)
    col_mapping['gimbal_yaw'] = find_col(yaw_cols, df_cols)
    col_mapping['gimbal_roll'] = find_col(roll_cols, df_cols)
    col_mapping['drone_yaw'] = find_col(drone_yaw_cols, df_cols)  # Drone heading
    col_mapping['drone_roll'] = find_col(drone_roll_cols, df_cols)  # Drone body roll
    col_mapping['drone_pitch'] = find_col(drone_pitch_cols, df_cols)  # Drone body pitch
    col_mapping['gimbal_mode'] = find_col(gimbal_mode_cols, df_cols)  # Gimbal mode
    col_mapping['is_video'] = find_col(video_cols, df_cols)
    col_mapping['time'] = find_col(time_cols, df_cols)
    col_mapping['home_lat'] = find_col(home_lat_cols, df_cols)
    col_mapping['home_lon'] = find_col(home_lon_cols, df_cols)
    col_mapping['home_height'] = find_col(home_height_cols, df_cols)
    col_mapping['xspeed'] = find_col(xspeed_cols, df_cols)
    col_mapping['yspeed'] = find_col(yspeed_cols, df_cols)
    col_mapping['zspeed'] = find_col(zspeed_cols, df_cols)

    print(f"  Column mapping: {col_mapping}")

    # Check required columns
    required = ['latitude', 'longitude', 'altitude', 'gimbal_pitch', 'gimbal_yaw', 'gimbal_roll']
    missing = [k for k in required if col_mapping[k] is None]
    if missing:
        print(f"Available columns: {df_cols[:30]}...")  # First 30 columns
        raise ValueError(f"Missing required columns: {missing}")

    # Extract HOME position (before filtering by video)
    home_lat = None
    home_lon = None
    home_height = None
    if col_mapping['home_lat'] is not None and col_mapping['home_lon'] is not None:
        # Get first non-NaN HOME position
        home_lat_series = pd.to_numeric(df[col_mapping['home_lat']], errors='coerce')
        home_lon_series = pd.to_numeric(df[col_mapping['home_lon']], errors='coerce')

        valid_home = home_lat_series.notna() & home_lon_series.notna()
        if valid_home.any():
            first_valid_idx = valid_home.idxmax()
            home_lat = float(home_lat_series.iloc[first_valid_idx])
            home_lon = float(home_lon_series.iloc[first_valid_idx])

            if col_mapping['home_height'] is not None:
                home_height_series = pd.to_numeric(df[col_mapping['home_height']], errors='coerce')
                home_height = float(home_height_series.iloc[first_valid_idx]) if pd.notna(home_height_series.iloc[first_valid_idx]) else 0.0
            else:
                home_height = 0.0

            print(f"  HOME position: lat={home_lat:.6f}, lon={home_lon:.6f}, height={home_height:.2f}m")
        else:
            print("  Warning: No valid HOME position found in flight record")
    else:
        print("  Warning: HOME.latitude/longitude columns not found")

    # If HOME not found, use first telemetry position as fallback
    if home_lat is None or home_lon is None:
        print("  Using first telemetry position as HOME fallback")
        home_lat = float(pd.to_numeric(df[col_mapping['latitude']].iloc[0], errors='coerce'))
        home_lon = float(pd.to_numeric(df[col_mapping['longitude']].iloc[0], errors='coerce'))
        home_height = 0.0

    # Filter by video recording if column exists
    if col_mapping['is_video'] is not None:
        video_col = col_mapping['is_video']
        # Handle different formats: True/False, 1/0, "true"/"false", TRUE/FALSE
        video_series = df[video_col]
        if video_series.dtype == bool:
            mask = video_series == True
        elif video_series.dtype in [np.int64, np.float64]:
            mask = video_series == 1
        else:
            mask = video_series.astype(str).str.upper() == 'TRUE'
        df = df[mask].copy()
        print(f"  Filtered to {len(df)} rows where video is recording")

    if len(df) == 0:
        raise ValueError("No rows found with video recording enabled")

    # Extract relevant columns with robust conversion
    result = pd.DataFrame()

    for key in ['latitude', 'longitude', 'altitude', 'gimbal_pitch', 'gimbal_yaw', 'gimbal_roll']:
        col_name = col_mapping[key]
        values = df[col_name].copy()
        # Convert to numeric, replacing non-numeric with NaN
        result[key] = pd.to_numeric(values, errors='coerce')
        valid_count = result[key].notna().sum()
        print(f"    {key} ({col_name}): {valid_count}/{len(df)} valid values")

    # Load drone yaw (heading) if available
    if col_mapping['drone_yaw'] is not None:
        col_name = col_mapping['drone_yaw']
        result['drone_yaw'] = pd.to_numeric(df[col_name].copy(), errors='coerce')
        valid_count = result['drone_yaw'].notna().sum()
        print(f"    drone_yaw ({col_name}): {valid_count}/{len(df)} valid values")
    else:
        result['drone_yaw'] = 0.0
        print("    drone_yaw: Not found, defaulting to 0")

    # Load drone roll (body roll for FPV mode) if available
    if col_mapping['drone_roll'] is not None:
        col_name = col_mapping['drone_roll']
        result['drone_roll'] = pd.to_numeric(df[col_name].copy(), errors='coerce')
        valid_count = result['drone_roll'].notna().sum()
        print(f"    drone_roll ({col_name}): {valid_count}/{len(df)} valid values")
    else:
        result['drone_roll'] = 0.0
        print("    drone_roll: Not found, defaulting to 0")

    # Load drone pitch (body pitch for FPV mode) if available
    if col_mapping['drone_pitch'] is not None:
        col_name = col_mapping['drone_pitch']
        result['drone_pitch'] = pd.to_numeric(df[col_name].copy(), errors='coerce')
        valid_count = result['drone_pitch'].notna().sum()
        print(f"    drone_pitch ({col_name}): {valid_count}/{len(df)} valid values")
    else:
        result['drone_pitch'] = 0.0
        print("    drone_pitch: Not found, defaulting to 0")

    # Load gimbal mode if available
    if col_mapping['gimbal_mode'] is not None:
        col_name = col_mapping['gimbal_mode']
        result['gimbal_mode'] = df[col_name].copy().astype(str)
        print(f"    gimbal_mode ({col_name}): loaded")
    else:
        result['gimbal_mode'] = 'Follow Yaw'  # Default assumption
        print("    gimbal_mode: Not found, defaulting to 'Follow Yaw'")

    # Load OSD velocities (NED frame: North, East, Down)
    if col_mapping['xspeed'] is not None:
        col_name = col_mapping['xspeed']
        result['vel_north'] = pd.to_numeric(df[col_name].copy(), errors='coerce')
        valid_count = result['vel_north'].notna().sum()
        print(f"    vel_north ({col_name}): {valid_count}/{len(df)} valid values")
    else:
        result['vel_north'] = 0.0
        print("    vel_north: Not found, defaulting to 0")

    if col_mapping['yspeed'] is not None:
        col_name = col_mapping['yspeed']
        result['vel_east'] = pd.to_numeric(df[col_name].copy(), errors='coerce')
        valid_count = result['vel_east'].notna().sum()
        print(f"    vel_east ({col_name}): {valid_count}/{len(df)} valid values")
    else:
        result['vel_east'] = 0.0
        print("    vel_east: Not found, defaulting to 0")

    if col_mapping['zspeed'] is not None:
        col_name = col_mapping['zspeed']
        result['vel_down'] = pd.to_numeric(df[col_name].copy(), errors='coerce')
        valid_count = result['vel_down'].notna().sum()
        print(f"    vel_down ({col_name}): {valid_count}/{len(df)} valid values")
    else:
        result['vel_down'] = 0.0
        print("    vel_down: Not found, defaulting to 0")

    # Handle time column
    if col_mapping['time'] is not None:
        time_col = col_mapping['time']
        time_values = df[time_col].copy()
        # Try to parse time strings like "9m 15.0s"
        if time_values.dtype == object:
            def parse_time(t):
                if pd.isna(t):
                    return np.nan
                t = str(t)
                try:
                    # Try direct numeric conversion first
                    return float(t)
                except:
                    pass
                # Parse "Xm Ys" format
                import re
                match = re.match(r'(\d+)m\s*([\d.]+)s', t)
                if match:
                    minutes = int(match.group(1))
                    seconds = float(match.group(2))
                    return minutes * 60 + seconds
                return np.nan
            result['time'] = time_values.apply(parse_time)
        else:
            result['time'] = pd.to_numeric(time_values, errors='coerce')
    else:
        result['time'] = np.arange(len(result))

    # Only drop rows where essential columns (lat/lon/alt) are NaN
    essential_cols = ['latitude', 'longitude', 'altitude']
    before_count = len(result)
    result = result.dropna(subset=essential_cols).reset_index(drop=True)

    # Fill gimbal NaN with 0 (neutral position)
    result['gimbal_pitch'] = result['gimbal_pitch'].fillna(0)
    result['gimbal_yaw'] = result['gimbal_yaw'].fillna(0)
    result['gimbal_roll'] = result['gimbal_roll'].fillna(0)
    result['drone_yaw'] = result['drone_yaw'].fillna(0)
    result['drone_roll'] = result['drone_roll'].fillna(0)
    result['drone_pitch'] = result['drone_pitch'].fillna(0)
    result['gimbal_mode'] = result['gimbal_mode'].fillna('Follow Yaw')

    # Fill velocity NaN with 0
    result['vel_north'] = result['vel_north'].fillna(0)
    result['vel_east'] = result['vel_east'].fillna(0)
    result['vel_down'] = result['vel_down'].fillna(0)

    print(f"  Loaded {len(result)} valid telemetry records (dropped {before_count - len(result)} with missing lat/lon/alt)")

    home_position = (home_lat, home_lon, home_height)
    return result, home_position


def load_depth_map(depth_dir: str, frame_id: int) -> Optional[np.ndarray]:
    """
    Load depth map for a specific frame.

    Tries multiple naming conventions:
    - depth_metric_XXXX.npy
    - depth_XXXX.npy
    - frame_XXXX_depth.npy
    """
    patterns = [
        f"depth_metric_{frame_id:04d}.npy",
        f"depth_{frame_id:04d}.npy",
        f"frame_{frame_id:04d}_depth.npy",
        f"{frame_id:04d}.npy",
    ]

    for pattern in patterns:
        path = os.path.join(depth_dir, pattern)
        if os.path.exists(path):
            return np.load(path)

    # Try without zero padding
    for pattern in [f"depth_metric_{frame_id}.npy", f"depth_{frame_id}.npy"]:
        path = os.path.join(depth_dir, pattern)
        if os.path.exists(path):
            return np.load(path)

    return None


def get_depth_resolution(depth_dir: str) -> Tuple[int, int]:
    """Get depth map resolution from first available file."""
    depth_files = sorted(Path(depth_dir).glob("*.npy"))
    if not depth_files:
        raise ValueError(f"No .npy files found in {depth_dir}")

    first_depth = np.load(depth_files[0])
    h, w = first_depth.shape[:2]
    print(f"Depth map resolution: {w}x{h}")
    return h, w


def load_intrinsics(intrinsics_path: str) -> np.ndarray:
    """
    Load camera intrinsics.

    Returns:
        Array of shape [N, 3, 3] or [3, 3]
    """
    intrinsics = np.load(intrinsics_path)
    print(f"Loaded intrinsics with shape: {intrinsics.shape}")
    return intrinsics


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
        raise ValueError(f"Unexpected JSON structure. Expected list or dict with 'tracks' key.")

    print(f"Loaded {len(tracks)} object tracks")
    return tracks


def rescale_bbox(bbox: List[float], sx: float, sy: float) -> List[float]:
    """
    Rescale bounding box from source resolution to depth map resolution.

    Args:
        bbox: [x1, y1, x2, y2] in source resolution
        sx: scale factor for x (depth_w / source_w)
        sy: scale factor for y (depth_h / source_h)

    Returns:
        Rescaled bbox [x1, y1, x2, y2]
    """
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

    Args:
        depth_map: 2D depth array [H, W] in meters
        bbox: [x1, y1, x2, y2] in depth map coordinates
        percentile_threshold: Reject pixels above this percentile (background)

    Returns:
        Robust median depth in meters, or None if invalid
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    # Clamp to image bounds
    h, w = depth_map.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    # Extract depth region
    depth_roi = depth_map[y1:y2, x1:x2].flatten()

    # Filter invalid depths
    valid_mask = (depth_roi > 0) & np.isfinite(depth_roi)
    valid_depths = depth_roi[valid_mask]

    if len(valid_depths) == 0:
        return None

    # Background rejection: keep only depths below 70th percentile
    threshold = np.percentile(valid_depths, percentile_threshold)
    foreground_depths = valid_depths[valid_depths <= threshold]

    if len(foreground_depths) == 0:
        return None

    # Return median of foreground depths
    return float(np.median(foreground_depths))


def pixel_to_camera_frame(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray
) -> np.ndarray:
    """
    Unproject pixel coordinates to 3D point in camera frame.

    Camera frame convention (OpenCV):
    - X: right
    - Y: down
    - Z: forward (into the scene)

    Args:
        u, v: Pixel coordinates
        depth: Depth in meters
        K: 3x3 intrinsic matrix

    Returns:
        3D point [X, Y, Z] in camera frame
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Unproject
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.array([X, Y, Z])


def gimbal_to_rotation_matrix(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    drone_pitch_deg: float = 0.0,
    gimbal_mode: str = 'Follow Yaw',
    yaw_offset_deg: float = 0.0,
    magnetic_declination_deg: float = 0.0,
    add_drone_yaw: bool = False,
    use_osd_yaw: bool = False
) -> np.ndarray:
    """
    Convert DJI gimbal angles to rotation matrix (Camera -> NED).

    DJI Gimbal Convention:
    - Pitch: Positive = looking up, Negative = looking down
    - Yaw: Positive = rotate right (clockwise from above)
    - Roll: Positive = right wing down

    Gimbal Modes:
    1. 'Follow Yaw' (Default): Pitch and Roll are absolute (horizon-stabilized).
       Yaw is absolute (North-relative) = drone_yaw + gimbal_yaw.
    2. 'FPV': Gimbal Roll locks to drone body roll (not horizon-stabilized).
       Must account for drone's banking angle.
    3. 'Free'/'Lock': All three angles (Pitch, Roll, Yaw) are independent of drone heading.
       Gimbal maintains its own absolute orientation.

    Camera frame (OpenCV): X-right, Y-down, Z-forward
    NED frame: X-North, Y-East, Z-Down

    Args:
        pitch_deg: Gimbal pitch in degrees (horizon-relative in Follow/Free modes)
        yaw_deg: Gimbal yaw in degrees
        roll_deg: Gimbal roll in degrees
        drone_yaw_deg: Drone heading in degrees (0=North, positive=clockwise)
        drone_roll_deg: Drone body roll in degrees (for FPV mode)
        drone_pitch_deg: Drone body pitch in degrees (for FPV mode)
        gimbal_mode: Gimbal mode string ('Follow Yaw', 'Free', 'FPV', etc.)
        yaw_offset_deg: Additional yaw offset for calibration (degrees)
        magnetic_declination_deg: Magnetic declination (positive=East, negative=West)
        add_drone_yaw: If True, add drone yaw to gimbal yaw. Default False since DJI gimbal yaw is typically already absolute.
        use_osd_yaw: If True, use OSD.yaw (drone heading) instead of GIMBAL.yaw for yaw angle.

    Returns:
        3x3 rotation matrix R_cam_to_ned
    """
    gimbal_mode_lower = gimbal_mode.lower() if isinstance(gimbal_mode, str) else 'follow'

    # Option to use OSD.yaw (drone heading) instead of GIMBAL.yaw
    if use_osd_yaw:
        # Use drone heading directly as the yaw angle
        base_yaw_deg = drone_yaw_deg
    else:
        # Use gimbal yaw (default - typically already absolute in DJI logs)
        base_yaw_deg = yaw_deg

    # Determine absolute angles based on gimbal mode
    if 'fpv' in gimbal_mode_lower:
        # FPV Mode: Roll is locked to drone body, not horizon-stabilized
        if use_osd_yaw:
            absolute_yaw_deg = base_yaw_deg  # Already using OSD.yaw
        elif add_drone_yaw:
            absolute_yaw_deg = drone_yaw_deg + yaw_deg
        else:
            absolute_yaw_deg = base_yaw_deg  # Already absolute
        absolute_pitch_deg = pitch_deg
        absolute_roll_deg = drone_roll_deg + roll_deg  # Locked to drone body roll

    elif 'free' in gimbal_mode_lower or 'lock' in gimbal_mode_lower:
        # Free/Lock Mode: All angles are independent of drone heading
        absolute_yaw_deg = base_yaw_deg  # Use selected yaw source
        absolute_pitch_deg = pitch_deg
        absolute_roll_deg = roll_deg

    else:
        # Follow Yaw Mode (Default): Pitch/Roll are horizon-stabilized
        if use_osd_yaw:
            absolute_yaw_deg = base_yaw_deg  # Already using OSD.yaw
        elif add_drone_yaw:
            absolute_yaw_deg = drone_yaw_deg + yaw_deg
        else:
            absolute_yaw_deg = base_yaw_deg  # Already absolute
        absolute_pitch_deg = pitch_deg
        absolute_roll_deg = roll_deg

    # Apply magnetic declination correction (True North = Magnetic North + Declination)
    absolute_yaw_deg += magnetic_declination_deg

    # Apply additional calibration offset if needed
    absolute_yaw_deg += yaw_offset_deg

    # Convert to radians
    pitch = np.radians(absolute_pitch_deg)
    yaw = np.radians(absolute_yaw_deg)
    roll = np.radians(absolute_roll_deg)

    # Camera axes in body frame (when gimbal at neutral pointing forward):
    # Cam X (right) -> Body Y (right)
    # Cam Y (down) -> Body Z (down)
    # Cam Z (forward) -> Body X (forward)
    R_cam_to_body_neutral = np.array([
        [0, 0, 1],   # Body X = Cam Z
        [1, 0, 0],   # Body Y = Cam X
        [0, 1, 0],   # Body Z = Cam Y
    ])

    # Create rotation from gimbal angles
    # Using intrinsic rotation: Yaw-Pitch-Roll (ZYX)
    # Yaw rotates around down/Z axis (NED)
    # Pitch rotates around lateral/Y axis
    # Roll rotates around forward/X axis
    R_gimbal = R.from_euler('ZYX', [yaw, pitch, roll], degrees=False).as_matrix()

    # Full rotation: Camera -> NED
    R_cam_to_ned = R_gimbal @ R_cam_to_body_neutral

    return R_cam_to_ned


def get_camera_heading_ne(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    drone_yaw_deg: float = 0.0,
    drone_roll_deg: float = 0.0,
    drone_pitch_deg: float = 0.0,
    gimbal_mode: str = 'Follow Yaw',
    yaw_offset_deg: float = 0.0,
    magnetic_declination_deg: float = 0.0,
    add_drone_yaw: bool = False,
    use_osd_yaw: bool = False
) -> Tuple[float, float]:
    """
    Get camera look direction projected onto NE (North-East) plane.

    Returns unit vector (N, E) indicating where camera is pointing horizontally.

    Args:
        Same as gimbal_to_rotation_matrix

    Returns:
        Tuple (north_component, east_component) of unit heading vector
    """
    R_cam_to_ned = gimbal_to_rotation_matrix(
        pitch_deg, yaw_deg, roll_deg,
        drone_yaw_deg, drone_roll_deg, drone_pitch_deg,
        gimbal_mode, yaw_offset_deg, magnetic_declination_deg, add_drone_yaw, use_osd_yaw
    )

    # Camera look direction is +Z in camera frame (forward)
    cam_forward = np.array([0, 0, 1])

    # Transform to NED
    look_ned = R_cam_to_ned @ cam_forward

    # Project onto NE plane (ignore Down component)
    north = look_ned[0]
    east = look_ned[1]

    # Normalize to unit vector
    magnitude = np.sqrt(north**2 + east**2)
    if magnitude > 1e-6:
        north /= magnitude
        east /= magnitude
    else:
        # Camera pointing straight down, default to drone heading
        north = np.cos(np.radians(drone_yaw_deg))
        east = np.sin(np.radians(drone_yaw_deg))

    return north, east


def camera_to_ned(
    point_cam: np.ndarray,
    R_cam_to_ned: np.ndarray,
    uav_ned: np.ndarray
) -> np.ndarray:
    """
    Transform point from camera frame to NED frame.

    Args:
        point_cam: 3D point in camera frame [X, Y, Z]
        R_cam_to_ned: Rotation matrix from camera to NED
        uav_ned: UAV position in NED frame [N, E, D]

    Returns:
        3D point in NED frame [N, E, D]
    """
    # Rotate point to NED frame
    point_ned_relative = R_cam_to_ned @ point_cam

    # Translate by UAV position
    point_ned_absolute = point_ned_relative + uav_ned

    return point_ned_absolute


def lla_to_ned(
    lat: float,
    lon: float,
    alt: float,
    lat0: float,
    lon0: float,
    alt0: float
) -> np.ndarray:
    """
    Convert LLA (Lat, Lon, Alt) to NED coordinates.

    Args:
        lat, lon: Target latitude/longitude in degrees
        alt: Target altitude in meters
        lat0, lon0: Reference (home) latitude/longitude in degrees
        alt0: Reference (home) altitude in meters

    Returns:
        NED coordinates [N, E, D] in meters
    """
    if pm is not None:
        n, e, d = pm.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
        return np.array([n, e, d])
    else:
        # Fallback: simple approximation for small distances
        # 1 degree lat ≈ 111320 m
        # 1 degree lon ≈ 111320 * cos(lat) m
        lat_rad = np.radians(lat0)
        n = (lat - lat0) * 111320
        e = (lon - lon0) * 111320 * np.cos(lat_rad)
        d = alt0 - alt  # Down is positive
        return np.array([n, e, d])


def ned_to_lla(
    n: float,
    e: float,
    d: float,
    lat0: float,
    lon0: float,
    alt0: float
) -> Tuple[float, float, float]:
    """
    Convert NED coordinates to LLA.

    Args:
        n, e, d: NED coordinates in meters
        lat0, lon0: Reference (home) latitude/longitude in degrees
        alt0: Reference (home) altitude in meters

    Returns:
        (latitude, longitude, altitude) in degrees and meters
    """
    if pm is not None:
        lat, lon, alt = pm.ned2geodetic(n, e, d, lat0, lon0, alt0)
        return lat, lon, alt
    else:
        # Fallback: simple approximation
        lat_rad = np.radians(lat0)
        lat = lat0 + n / 111320
        lon = lon0 + e / (111320 * np.cos(lat_rad))
        alt = alt0 - d
        return lat, lon, alt


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Apply moving average filter to data."""
    if len(data) < window:
        return data

    result = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        result[i] = np.mean(data[start:end], axis=0)

    return result


def estimate_velocity(
    positions_ned: List[np.ndarray],
    frame_ids: List[int],
    fps: float,
    window: int = 5
) -> List[Tuple[np.ndarray, float]]:
    """
    Estimate velocity from position history.

    Args:
        positions_ned: List of NED positions
        frame_ids: Corresponding frame IDs
        fps: Frames per second
        window: Moving average window size

    Returns:
        List of (velocity_ned [Vn, Ve, Vd], speed_mps) tuples
    """
    if len(positions_ned) < 2:
        return [(np.array([0.0, 0.0, 0.0]), 0.0)] * len(positions_ned)

    positions = np.array(positions_ned)
    frames = np.array(frame_ids)

    # Calculate velocities via differentiation
    velocities = []
    for i in range(len(positions)):
        if i == 0:
            # Forward difference
            if frames[i+1] != frames[i]:
                dt = (frames[i+1] - frames[i]) / fps
                vel = (positions[i+1] - positions[i]) / dt
            else:
                vel = np.array([0.0, 0.0, 0.0])
        elif i == len(positions) - 1:
            # Backward difference
            if frames[i] != frames[i-1]:
                dt = (frames[i] - frames[i-1]) / fps
                vel = (positions[i] - positions[i-1]) / dt
            else:
                vel = np.array([0.0, 0.0, 0.0])
        else:
            # Central difference
            if frames[i+1] != frames[i-1]:
                dt = (frames[i+1] - frames[i-1]) / fps
                vel = (positions[i+1] - positions[i-1]) / dt
            else:
                vel = np.array([0.0, 0.0, 0.0])
        velocities.append(vel)

    velocities = np.array(velocities)

    # Apply moving average
    smoothed = moving_average(velocities, window)

    # Calculate speed (horizontal magnitude)
    results = []
    for vel in smoothed:
        speed = np.sqrt(vel[0]**2 + vel[1]**2)  # Horizontal speed
        results.append((vel, float(speed)))

    return results


def process_tracks(
    tracks: List[Dict],
    flight_data: pd.DataFrame,
    home_position: Tuple[float, float, float],
    depth_dir: str,
    intrinsics: np.ndarray,
    source_res: Tuple[int, int],
    depth_res: Tuple[int, int],
    fps: float,
    velocity_window: int,
    use_kf: bool = True,
    kf_sigma_a: float = 0.5,
    kf_sigma_meas_h: float = 5.0,
    kf_sigma_meas_v: float = 2.0,
    yaw_offset_deg: float = 0.0,
    magnetic_declination_deg: float = 0.0,
    add_drone_yaw: bool = False,
    use_osd_yaw: bool = False
) -> List[Dict]:
    """
    Process all tracks and estimate their states.

    Args:
        tracks: List of track dictionaries
        flight_data: UAV telemetry DataFrame
        home_position: (latitude, longitude, height) of HOME position
        depth_dir: Directory with depth maps
        intrinsics: Camera intrinsics [N, 3, 3] or [3, 3]
        source_res: Source video resolution (W, H)
        depth_res: Depth map resolution (H, W)
        fps: Frames per second
        velocity_window: Window size for velocity smoothing (if use_kf=False)
        use_kf: Use Kalman Filter for state estimation (default: True)
        kf_sigma_a: KF process noise std (acceleration) in m/s²
        kf_sigma_meas_h: KF measurement noise std (horizontal) in meters
        kf_sigma_meas_v: KF measurement noise std (vertical) in meters

    Returns:
        Processed tracks with added state information
    """
    source_w, source_h = source_res
    depth_h, depth_w = depth_res

    # Scale factors
    sx = depth_w / source_w
    sy = depth_h / source_h
    print(f"Scale factors: sx={sx:.4f}, sy={sy:.4f}")

    # Home position (NED origin)
    home_lat, home_lon, home_alt = home_position
    print(f"Home position (NED origin): ({home_lat:.6f}, {home_lon:.6f}), altitude: {home_alt:.2f}m")

    # Group tracks by object ID for velocity estimation
    tracks_by_id = {}
    for track in tracks:
        obj_id = track.get('track_id', track.get('id', id(track)))
        if obj_id not in tracks_by_id:
            tracks_by_id[obj_id] = []
        tracks_by_id[obj_id].append(track)

    # Sort each group by frame_id
    for obj_id in tracks_by_id:
        tracks_by_id[obj_id].sort(key=lambda x: x['frame_id'])

    # Process each track
    processed_tracks = []

    if use_kf:
        print(f"Using Kalman Filter for state estimation (sigma_a={kf_sigma_a}, sigma_meas_h={kf_sigma_meas_h}, sigma_meas_v={kf_sigma_meas_v})")

        # Kalman Filter approach: process frame-by-frame with predict/update
        for obj_id, obj_tracks in tracks_by_id.items():
            kf = None  # Will be initialized on first measurement
            track_data = []  # Store (frame_id, track_dict, measurement_ned) tuples

            # First pass: collect measurements
            for track in obj_tracks:
                frame_id = track['frame_id']
                bbox = track['bbox']

                # Get telemetry for this frame
                if frame_id >= len(flight_data):
                    print(f"Warning: Frame {frame_id} exceeds telemetry data, skipping")
                    continue

                telem = flight_data.iloc[frame_id]

                # Load depth map
                depth_map = load_depth_map(depth_dir, frame_id)
                if depth_map is None:
                    print(f"Warning: No depth map for frame {frame_id}, skipping")
                    continue

                # Get intrinsics for this frame
                if len(intrinsics.shape) == 3:
                    K = intrinsics[min(frame_id, len(intrinsics) - 1)]
                else:
                    K = intrinsics

                # Rescale bbox
                bbox_scaled = rescale_bbox(bbox, sx, sy)

                # Get robust depth
                depth = extract_robust_depth(depth_map, bbox_scaled)
                if depth is None or depth <= 0:
                    print(f"Warning: Invalid depth for frame {frame_id}, skipping")
                    continue

                # Get bbox center in depth map coordinates
                u, v = get_bbox_center(bbox_scaled)

                # Unproject to camera frame
                point_cam = pixel_to_camera_frame(u, v, depth, K)

                # Get rotation matrix from gimbal angles
                R_cam_to_ned = gimbal_to_rotation_matrix(
                    telem['gimbal_pitch'],
                    telem['gimbal_yaw'],
                    telem['gimbal_roll'],
                    drone_yaw_deg=telem.get('drone_yaw', 0.0),
                    drone_roll_deg=telem.get('drone_roll', 0.0),
                    drone_pitch_deg=telem.get('drone_pitch', 0.0),
                    gimbal_mode=telem.get('gimbal_mode', 'Follow Yaw'),
                    yaw_offset_deg=yaw_offset_deg,
                    magnetic_declination_deg=magnetic_declination_deg,
                    add_drone_yaw=add_drone_yaw,
                    use_osd_yaw=use_osd_yaw
                )

                # Get UAV position in NED
                uav_ned = lla_to_ned(
                    telem['latitude'],
                    telem['longitude'],
                    telem['altitude'],
                    home_lat,
                    home_lon,
                    home_alt
                )

                # Transform to NED (raw measurement, noisy)
                measurement_ned = camera_to_ned(point_cam, R_cam_to_ned, uav_ned)

                # Store track data for KF processing
                track_data.append((frame_id, track.copy(), measurement_ned, depth))

            if not track_data:
                continue

            # Second pass: apply Kalman Filter
            prev_frame_id = None
            for frame_id, track, measurement_ned, depth in track_data:
                if kf is None:
                    # Initialize KF with first measurement
                    kf = KalmanFilter(
                        initial_position=measurement_ned,
                        fps=fps,
                        initial_velocity=None,  # Start with zero velocity
                        sigma_a=kf_sigma_a,
                        sigma_meas_horizontal=kf_sigma_meas_h,
                        sigma_meas_vertical=kf_sigma_meas_v
                    )
                    filtered_pos = measurement_ned
                    filtered_vel = np.zeros(3)
                else:
                    # Predict forward (handle missing frames)
                    if prev_frame_id is not None:
                        frame_gap = frame_id - prev_frame_id
                        for _ in range(frame_gap):
                            kf.predict()
                    else:
                        kf.predict()

                    # Update with measurement
                    kf.update(measurement_ned)

                    # Get filtered state
                    filtered_pos, filtered_vel = kf.get_state()

                prev_frame_id = frame_id

                # Convert filtered position to LLA
                lat, lon, alt = ned_to_lla(
                    filtered_pos[0],
                    filtered_pos[1],
                    filtered_pos[2],
                    home_lat,
                    home_lon,
                    home_alt
                )

                # Calculate horizontal speed
                speed_mps = float(np.sqrt(filtered_vel[0]**2 + filtered_vel[1]**2))

                # Create processed track with filtered state
                track_processed = track
                track_processed['lat'] = float(lat)
                track_processed['lon'] = float(lon)
                track_processed['depth_m'] = float(depth)
                track_processed['pos_ned'] = filtered_pos.tolist()
                track_processed['vel_ned'] = filtered_vel.tolist()
                track_processed['speed_mps'] = speed_mps

                processed_tracks.append(track_processed)

    else:
        print("Using moving average for velocity estimation (legacy mode)")

        # Legacy approach: collect all positions then estimate velocities
        for obj_id, obj_tracks in tracks_by_id.items():
            positions_ned = []
            frame_ids = []

            for track in obj_tracks:
                frame_id = track['frame_id']
                bbox = track['bbox']

                # Get telemetry for this frame
                if frame_id >= len(flight_data):
                    print(f"Warning: Frame {frame_id} exceeds telemetry data, skipping")
                    continue

                telem = flight_data.iloc[frame_id]

                # Load depth map
                depth_map = load_depth_map(depth_dir, frame_id)
                if depth_map is None:
                    print(f"Warning: No depth map for frame {frame_id}, skipping")
                    continue

                # Get intrinsics for this frame
                if len(intrinsics.shape) == 3:
                    K = intrinsics[min(frame_id, len(intrinsics) - 1)]
                else:
                    K = intrinsics

                # Rescale bbox
                bbox_scaled = rescale_bbox(bbox, sx, sy)

                # Get robust depth
                depth = extract_robust_depth(depth_map, bbox_scaled)
                if depth is None or depth <= 0:
                    print(f"Warning: Invalid depth for frame {frame_id}, skipping")
                    continue

                # Get bbox center in depth map coordinates
                u, v = get_bbox_center(bbox_scaled)

                # Unproject to camera frame
                point_cam = pixel_to_camera_frame(u, v, depth, K)

                # Get rotation matrix from gimbal angles
                R_cam_to_ned = gimbal_to_rotation_matrix(
                    telem['gimbal_pitch'],
                    telem['gimbal_yaw'],
                    telem['gimbal_roll'],
                    drone_yaw_deg=telem.get('drone_yaw', 0.0),
                    drone_roll_deg=telem.get('drone_roll', 0.0),
                    drone_pitch_deg=telem.get('drone_pitch', 0.0),
                    gimbal_mode=telem.get('gimbal_mode', 'Follow Yaw'),
                    yaw_offset_deg=yaw_offset_deg,
                    magnetic_declination_deg=magnetic_declination_deg,
                    add_drone_yaw=add_drone_yaw,
                    use_osd_yaw=use_osd_yaw
                )

                # Get UAV position in NED
                uav_ned = lla_to_ned(
                    telem['latitude'],
                    telem['longitude'],
                    telem['altitude'],
                    home_lat,
                    home_lon,
                    home_alt
                )

                # Transform to NED
                point_ned = camera_to_ned(point_cam, R_cam_to_ned, uav_ned)

                # Convert to LLA
                lat, lon, alt = ned_to_lla(
                    point_ned[0],
                    point_ned[1],
                    point_ned[2],
                    home_lat,
                    home_lon,
                    home_alt
                )

                # Store for velocity estimation
                positions_ned.append(point_ned)
                frame_ids.append(frame_id)

                # Add basic state info
                track_processed = track.copy()
                track_processed['lat'] = float(lat)
                track_processed['lon'] = float(lon)
                track_processed['depth_m'] = float(depth)
                track_processed['pos_ned'] = point_ned.tolist()

                processed_tracks.append(track_processed)

            # Estimate velocities for this object
            if len(positions_ned) >= 2:
                velocities = estimate_velocity(positions_ned, frame_ids, fps, velocity_window)

                # Update processed tracks with velocity info
                track_idx = 0
                for i, track in enumerate(processed_tracks):
                    if track.get('track_id', track.get('id')) == obj_id:
                        if track_idx < len(velocities):
                            vel, speed = velocities[track_idx]
                            track['vel_ned'] = vel.tolist()
                            track['speed_mps'] = speed
                            track_idx += 1
            else:
                # No velocity available
                for track in processed_tracks:
                    if track.get('track_id', track.get('id')) == obj_id:
                        track['vel_ned'] = [0.0, 0.0, 0.0]
                        track['speed_mps'] = 0.0

    # Add heading_deg for all tracks (North=0°, East=90°, clockwise)
    for track in processed_tracks:
        vel = track.get('vel_ned', [0.0, 0.0, 0.0])
        v_north, v_east, v_down = vel[0], vel[1], vel[2]

        # Calculate heading from velocity (atan2(East, North))
        # atan2(y, x) gives angle from x-axis, so atan2(East, North) gives angle from North
        if abs(v_north) > 0.01 or abs(v_east) > 0.01:  # Threshold to avoid noise
            heading_rad = np.arctan2(v_east, v_north)
            heading_deg = float(np.degrees(heading_rad))
            # Normalize to [0, 360)
            if heading_deg < 0:
                heading_deg += 360.0
        else:
            heading_deg = None  # No valid heading for stationary objects

        track['heading_deg'] = heading_deg

    # Calculate density for each track (objects within 10m radius at same frame)
    for track in processed_tracks:
        frame_id = track.get('frame_id')
        track_id = track.get('track_id')
        pos = np.array(track.get('pos_ned', [0, 0, 0]))

        # Count other objects at same frame within 10m radius
        count = 0
        for other_track in processed_tracks:
            other_frame_id = other_track.get('frame_id')
            other_track_id = other_track.get('track_id')

            # Same frame, different track, not UAV
            if (other_frame_id == frame_id and
                other_track_id != track_id and
                other_track_id != 0):
                other_pos = np.array(other_track.get('pos_ned', [0, 0, 0]))
                distance = np.linalg.norm(pos - other_pos)
                if distance <= 10.0:
                    count += 1

        track['density'] = count

    # Create UAV state entries (track_id=0) for all frames
    print("Creating UAV state entries (track_id=0)...")
    uav_tracks = []
    for frame_id in range(len(flight_data)):
        telem = flight_data.iloc[frame_id]

        # Get UAV position in NED
        uav_ned = lla_to_ned(
            telem['latitude'],
            telem['longitude'],
            telem['altitude'],
            home_lat,
            home_lon,
            home_alt
        )

        # Convert back to LLA (should match original)
        lat, lon, alt = ned_to_lla(
            uav_ned[0],
            uav_ned[1],
            uav_ned[2],
            home_lat,
            home_lon,
            home_alt
        )

        # Get UAV velocity from telemetry (NED frame)
        vel_north = float(telem.get('vel_north', 0.0))
        vel_east = float(telem.get('vel_east', 0.0))
        vel_down = float(telem.get('vel_down', 0.0))
        vel_ned = [vel_north, vel_east, vel_down]

        # Calculate horizontal speed
        speed_mps = float(np.sqrt(vel_north**2 + vel_east**2))

        # Calculate UAV heading from velocity
        if abs(vel_north) > 0.01 or abs(vel_east) > 0.01:
            heading_rad = np.arctan2(vel_east, vel_north)
            heading_deg = float(np.degrees(heading_rad))
            if heading_deg < 0:
                heading_deg += 360.0
        else:
            # Use drone yaw as fallback
            heading_deg = float(telem.get('drone_yaw', 0.0))

        # UAV track entry
        uav_track = {
            'frame_id': frame_id,
            'track_id': 0,  # Reserved for UAV
            'bbox': None,
            'confidence': None,
            'class_name': 'UAV',
            'lat': float(lat),
            'lon': float(lon),
            'depth_m': None,
            'pos_ned': uav_ned.tolist(),
            'vel_ned': vel_ned,
            'speed_mps': speed_mps,
            'heading_deg': heading_deg,
            'density': None  # Not applicable for UAV
        }

        uav_tracks.append(uav_track)

    print(f"  Created {len(uav_tracks)} UAV state entries")

    # Combine UAV tracks with object tracks and sort by frame_id, then track_id
    all_tracks = uav_tracks + processed_tracks
    all_tracks.sort(key=lambda x: (x['frame_id'], x['track_id']))

    return all_tracks


def create_visualization_video(
    processed_tracks: List[Dict],
    flight_data: pd.DataFrame,
    output_path: str,
    fps: float = 5.0,
    num_frames: int = None,
    yaw_offset_deg: float = 0.0,
    magnetic_declination_deg: float = 0.0,
    add_drone_yaw: bool = False,
    use_osd_yaw: bool = False
):
    """
    Create 2D map video visualization with UAV trajectory and object positions.

    Features:
    - Grayscale satellite/map background
    - Objects colored by class
    - UAV trajectory overlay (progressive)
    - Velocity vectors
    - Frame-by-frame animation at specified FPS

    Args:
        processed_tracks: Processed track data
        flight_data: UAV telemetry
        output_path: Path to save video (should end with .mp4)
        fps: Frames per second for output video
        num_frames: Total number of video frames (if None, inferred from tracks)
    """
    if not HAS_VIZ:
        print("Visualization dependencies not available, skipping")
        return

    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False
        print("Warning: OpenCV not installed. Trying ffmpeg fallback.")

    print("Creating visualization video...")

    # Convert to Web Mercator (EPSG:3857)
    def latlon_to_mercator(lat, lon):
        """Convert lat/lon to Web Mercator coordinates."""
        x = np.asarray(lon) * 20037508.34 / 180
        y = np.log(np.tan((90 + np.asarray(lat)) * np.pi / 360)) / (np.pi / 180)
        y = y * 20037508.34 / 180
        return x, y

    # Convert UAV trajectory to Web Mercator
    uav_lats = flight_data['latitude'].values
    uav_lons = flight_data['longitude'].values
    uav_x, uav_y = latlon_to_mercator(uav_lats, uav_lons)

    # Group tracks by frame_id
    tracks_by_frame = {}
    for track in processed_tracks:
        frame_id = track.get('frame_id')
        if frame_id is not None:
            if frame_id not in tracks_by_frame:
                tracks_by_frame[frame_id] = []
            tracks_by_frame[frame_id].append(track)

    # Get unique classes and assign specific colors
    classes = set()
    for track in processed_tracks:
        cls = track.get('class_name', track.get('class', 'unknown'))
        classes.add(cls)

    # Specific high-contrast colors for each class
    specific_class_colors = {
        'car': '#FF0000',       # Red
        'vehicle': '#0000FF',   # Blue
        'person': '#00FF00',    # Green
        'cycle': '#00FFFF',     # Cyan
        'bus': '#FF00FF',       # Magenta
    }
    # Fallback colors for other classes
    fallback_colors = ['#FFD700', '#FF8C00', '#9400D3', '#00CED1', '#DC143C']

    class_list = sorted(classes)
    class_colors = {}
    fallback_idx = 0
    for cls in class_list:
        if cls in specific_class_colors:
            class_colors[cls] = specific_class_colors[cls]
        else:
            class_colors[cls] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1

    # Determine frame range from num_frames or tracks
    if num_frames is not None:
        max_frame = num_frames - 1
    elif tracks_by_frame:
        max_frame = max(tracks_by_frame.keys())
    else:
        max_frame = len(flight_data) - 1

    # Calculate telemetry to video frame ratio
    telem_per_frame = len(flight_data) / (max_frame + 1)

    # Calculate fixed map bounds (for consistent view across frames)
    all_x = list(uav_x)
    all_y = list(uav_y)
    for track in processed_tracks:
        lat, lon = track.get('lat'), track.get('lon')
        if lat is not None and lon is not None:
            x, y = latlon_to_mercator(lat, lon)
            all_x.append(x)
            all_y.append(y)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add padding (10%)
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    # Maintain aspect ratio (figure is 14x10, so target ratio is 1.4:1)
    # Ensure equal scaling in both directions (1:1 aspect for map)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Use 1:1 aspect ratio for proper map scaling
    if x_range > y_range:
        # Expand y to match x
        y_center = (y_min + y_max) / 2
        y_min = y_center - x_range / 2
        y_max = y_center + x_range / 2
    else:
        # Expand x to match y
        x_center = (x_min + x_max) / 2
        x_min = x_center - y_range / 2
        x_max = x_center + y_range / 2

    # Recalculate after aspect ratio fix
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Determine basemap provider
    print("  Setting up basemap...")
    basemap_provider = None
    grayscale_providers = [
        ctx.providers.CartoDB.Positron,
        ctx.providers.CartoDB.DarkMatter,
        ctx.providers.OpenStreetMap.Mapnik,
    ]

    # Test which provider works
    for provider in grayscale_providers:
        try:
            fig_test, ax_test = plt.subplots(figsize=(4, 3))
            ax_test.set_xlim(x_min, x_max)
            ax_test.set_ylim(y_min, y_max)
            ctx.add_basemap(ax_test, source=provider, zoom=16)
            plt.close(fig_test)
            basemap_provider = provider
            print(f"  Using basemap provider: {provider.get('name', 'grayscale')}")
            break
        except Exception as e:
            continue

    if basemap_provider is None:
        print("  Warning: No basemap provider available, using plain background")

    # Create temporary directory for frames
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix='state_est_viz_')
    print(f"  Generating {max_frame + 1} frames...")

    # Generate frames
    frame_paths = []
    for frame_idx in range(max_frame + 1):
        fig, ax = plt.subplots(figsize=(14, 10))

        # Set fixed bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')

        # Add basemap (grayscale background)
        if basemap_provider is not None:
            try:
                ctx.add_basemap(ax, source=basemap_provider, zoom=17)
            except Exception:
                ax.set_facecolor('#E0E0E0')  # Light gray fallback
        else:
            ax.set_facecolor('#E0E0E0')  # Light gray fallback

        # Plot full UAV trajectory (faded)
        ax.plot(uav_x, uav_y, color='gray', linewidth=1, alpha=0.3, zorder=1)

        # Map video frame to telemetry index
        current_telem_idx = min(int(frame_idx * telem_per_frame), len(uav_x) - 1)

        # Plot UAV trajectory up to current frame
        if current_telem_idx > 0:
            ax.plot(uav_x[:current_telem_idx+1], uav_y[:current_telem_idx+1],
                   color='white', linewidth=3, alpha=0.9, zorder=2)
            ax.plot(uav_x[:current_telem_idx+1], uav_y[:current_telem_idx+1],
                   color='dodgerblue', linewidth=2, alpha=0.9, zorder=3)

        # Plot current UAV position (black)
        uav_current_x = uav_x[current_telem_idx]
        uav_current_y = uav_y[current_telem_idx]
        ax.scatter(uav_current_x, uav_current_y,
                  c='black', s=150, marker='o', zorder=7,
                  edgecolors='white', linewidth=2)

        # Plot start marker
        ax.scatter(uav_x[0], uav_y[0], c='lime', s=100, marker='^',
                  zorder=6, edgecolors='white', linewidth=1.5)

        # Draw UAV heading vector (camera look direction projected to NE plane)
        if current_telem_idx < len(flight_data):
            telem = flight_data.iloc[current_telem_idx]
            heading_n, heading_e = get_camera_heading_ne(
                telem['gimbal_pitch'],
                telem['gimbal_yaw'],
                telem['gimbal_roll'],
                drone_yaw_deg=telem.get('drone_yaw', 0.0),
                drone_roll_deg=telem.get('drone_roll', 0.0),
                drone_pitch_deg=telem.get('drone_pitch', 0.0),
                gimbal_mode=telem.get('gimbal_mode', 'Follow Yaw'),
                yaw_offset_deg=yaw_offset_deg,
                magnetic_declination_deg=magnetic_declination_deg,
                add_drone_yaw=add_drone_yaw,
                use_osd_yaw=use_osd_yaw
            )
            # Scale heading vector for visualization (in mercator units)
            heading_scale = (x_max - x_min) * 0.08  # 8% of map width
            # In mercator: dx ~ East, dy ~ North
            heading_dx = heading_e * heading_scale
            heading_dy = heading_n * heading_scale
            # Draw heading arrow (orange for visibility)
            ax.annotate('', xy=(uav_current_x + heading_dx, uav_current_y + heading_dy),
                       xytext=(uav_current_x, uav_current_y),
                       arrowprops=dict(arrowstyle='->', color='orange',
                                     lw=2.5, alpha=0.9),
                       zorder=8)

        # Plot objects at current frame
        current_tracks = tracks_by_frame.get(frame_idx, [])

        for track in current_tracks:
            lat = track.get('lat')
            lon = track.get('lon')

            if lat is None or lon is None:
                continue

            x, y = latlon_to_mercator(lat, lon)
            cls = track.get('class_name', track.get('class', 'unknown'))
            color = class_colors.get(cls, '#FFFFFF')

            # Plot object point
            ax.scatter(x, y, c=color, s=100, alpha=0.9, edgecolors='white',
                      linewidth=2, zorder=5)

            # Draw velocity arrow (scale = 10, reduced to 1/3)
            vel = track.get('velocity_ned')
            if vel is not None and (abs(vel[0]) > 0.1 or abs(vel[1]) > 0.1):
                scale = 10
                dx = vel[1] * scale
                dy = vel[0] * scale
                ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                           arrowprops=dict(arrowstyle='->', color=color,
                                         lw=2, alpha=0.8),
                           zorder=4)

        # Add frame info
        ax.text(0.02, 0.98, f'Frame: {frame_idx}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add legend for classes (compact)
        legend_handles = []
        for cls in class_list:
            color = class_colors[cls]
            handle = plt.scatter([], [], c=color, s=60, edgecolors='white',
                                linewidth=1, label=cls)
            legend_handles.append(handle)

        if legend_handles:
            legend = ax.legend(handles=legend_handles, loc='upper right',
                             framealpha=0.9, fontsize=9)
            legend.get_frame().set_facecolor('white')

        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        ax.set_title('UAV Trajectory and Object Geolocalization', fontsize=12, fontweight='bold')

        # Save frame
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:05d}.png')
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, facecolor='white')
        plt.close(fig)
        frame_paths.append(frame_path)

        # Progress indicator
        if (frame_idx + 1) % 50 == 0 or frame_idx == max_frame:
            print(f"    Generated {frame_idx + 1}/{max_frame + 1} frames")

    # Compile frames into video
    print("  Compiling video...")

    # Ensure output path ends with .mp4
    if not output_path.endswith('.mp4'):
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'

    if HAS_CV2:
        # Use OpenCV
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            video_writer.write(frame)

        video_writer.release()
    else:
        # Use ffmpeg
        import subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)

    # Cleanup temp directory
    shutil.rmtree(temp_dir)

    print(f"Video saved to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Object Geolocalization and State Estimation")
    print("=" * 60)

    # Load data
    flight_data, home_position = load_flight_record(args.flight_record)
    tracks = load_tracks(args.input_json)
    intrinsics = load_intrinsics(args.intrinsics)

    # Get depth resolution
    depth_res = get_depth_resolution(args.depth_dir)
    source_res = tuple(args.source_res)

    print(f"Source resolution: {source_res[0]}x{source_res[1]}")
    print(f"FPS: {args.fps}")
    print(f"Velocity smoothing window: {args.velocity_window}")

    # Process tracks
    processed_tracks = process_tracks(
        tracks=tracks,
        flight_data=flight_data,
        home_position=home_position,
        depth_dir=args.depth_dir,
        intrinsics=intrinsics,
        source_res=source_res,
        depth_res=depth_res,
        fps=args.fps,
        velocity_window=args.velocity_window,
        use_kf=not args.no_kf,
        kf_sigma_a=args.kf_sigma_a,
        kf_sigma_meas_h=args.kf_sigma_meas_h,
        kf_sigma_meas_v=args.kf_sigma_meas_v,
        yaw_offset_deg=args.yaw_offset,
        magnetic_declination_deg=args.magnetic_declination,
        add_drone_yaw=args.add_drone_yaw,
        use_osd_yaw=args.use_osd_yaw
    )

    print(f"\nProcessed {len(processed_tracks)} track entries")

    # Save output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write JSON manually for high-precision float output
        f.write('[\n')
        for i, track in enumerate(processed_tracks):
            f.write('  {\n')
            items = list(track.items())
            for j, (key, value) in enumerate(items):
                comma = ',' if j < len(items) - 1 else ''
                if value is None:
                    # Handle None values
                    f.write(f'    "{key}": null{comma}\n')
                elif key in ('lat', 'lon') and isinstance(value, float):
                    # 10 decimal places for lat/lon (~0.01mm precision)
                    f.write(f'    "{key}": {value:.10f}{comma}\n')
                elif key in ('depth_m', 'speed_mps') and isinstance(value, float):
                    # 4 decimal places for depth and speed
                    f.write(f'    "{key}": {value:.4f}{comma}\n')
                elif key in ('heading_deg',) and isinstance(value, float):
                    # 2 decimal places for heading
                    f.write(f'    "{key}": {value:.2f}{comma}\n')
                elif isinstance(value, list):
                    # Format list values with precision
                    formatted = [f'{v:.6f}' if isinstance(v, float) else json.dumps(v) for v in value]
                    f.write(f'    "{key}": [{", ".join(formatted)}]{comma}\n')
                elif isinstance(value, float):
                    f.write(f'    "{key}": {value:.6f}{comma}\n')
                else:
                    f.write(f'    "{key}": {json.dumps(value)}{comma}\n')
            f.write('  }')
            if i < len(processed_tracks) - 1:
                f.write(',')
            f.write('\n')
        f.write(']\n')

    print(f"Saved state estimates to: {output_path}")

    # Create visualization video
    if not args.no_viz and HAS_VIZ:
        viz_dir = args.output_dir or output_path.parent
        viz_path = Path(viz_dir) / f"{output_path.stem}_map.mp4"
        # Get number of frames from intrinsics
        num_frames = intrinsics.shape[0] if len(intrinsics.shape) == 3 else None
        create_visualization_video(processed_tracks, flight_data, str(viz_path),
                                   fps=args.fps, num_frames=num_frames,
                                   yaw_offset_deg=args.yaw_offset,
                                   magnetic_declination_deg=args.magnetic_declination,
                                   add_drone_yaw=args.add_drone_yaw,
                                   use_osd_yaw=args.use_osd_yaw)

    print("\nDone!")


if __name__ == "__main__":
    main()
