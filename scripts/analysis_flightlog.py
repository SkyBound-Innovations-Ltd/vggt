#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJI Flight Log Analysis Script with State Estimate Comparison
Extracts and visualizes 6 DoF (Degrees of Freedom) of UAV from DJI flight logs
and compares with state estimates from ViPE-X processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy import interpolate
import argparse
import sys


def parse_timestamp(date_str, time_str):
    """Parse DJI timestamp from separate date and time columns"""
    datetime_str = f"{date_str},{time_str}"
    return datetime.strptime(datetime_str, '%m/%d/%Y,%I:%M:%S.%f %p')


def gps_to_local_meters(lat, lon, lat_ref, lon_ref):
    """
    Convert GPS coordinates to local meters relative to reference point
    Returns: (east, north) in meters
    """
    # Earth radius in meters
    R = 6371000

    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    # Calculate differences
    dlat = lat_rad - lat_ref_rad
    dlon = lon_rad - lon_ref_rad

    # Convert to meters (approximate for small distances)
    north = dlat * R
    east = dlon * R * np.cos(lat_ref_rad)

    return east, north


def rotate_to_world_frame(east, north, height, yaw_initial_deg):
    """
    Rotate local ENU coordinates to camera world frame
    X^W: Right when camera started
    Y^W: Down when camera started (positive downward)
    Z^W: Forward when camera started

    Args:
        east, north: ENU coordinates in meters
        height: Height in meters (positive up)
        yaw_initial_deg: Initial yaw angle in degrees (0 = North, +CW)

    Returns: 
        X_world, Y_world, Z_world
    """
    # Convert yaw to radians (DJI yaw: 0 = North, clockwise positive)
    yaw_rad = np.radians(yaw_initial_deg)

    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    # Corrected rotation matrix:
    # Forward direction in ENU: [sin(ψ), cos(ψ)]
    # Right direction in ENU: [cos(ψ), -sin(ψ)]
    X_world = east * cos_yaw - north * sin_yaw  # Right
    Z_world = east * sin_yaw + north * cos_yaw  # Forward
    Y_world = -height                           # Down (CV convention)

    return X_world, Y_world, Z_world


def find_video_sessions(df):
    """
    Find all video recording sessions where CAMERA.isVideo = True
    Returns list of (start_idx, end_idx) tuples
    """
    is_video = df['CAMERA.isVideo'].fillna(False)

    # Find transitions
    sessions = []
    in_session = False
    start_idx = None

    for idx, recording in enumerate(is_video):
        if recording and not in_session:
            # Start of session
            start_idx = idx
            in_session = True
        elif not recording and in_session:
            # End of session
            sessions.append((start_idx, idx - 1))
            in_session = False

    # Handle case where recording continues to end
    if in_session:
        sessions.append((start_idx, len(df) - 1))

    return sessions


def load_flight_log_6dof(csv_path):
    """
    Load and calculate 6 DoF from DJI flight log
    Returns DataFrame with time_seconds, X_world, Y_world, Z_world, pitch_total, roll_total, yaw_total
    """
    print(f"Loading flight log: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total records: {len(df)}")

    # Find video sessions
    sessions = find_video_sessions(df)
    num_sessions = len(sessions)

    if num_sessions == 0:
        print("No video recording sessions found in the flight log!")
        return None

    print(f"\nFound {num_sessions} video recording session(s)")

    # Alert if multiple sessions detected
    if num_sessions >= 2:
        print("\n" + "="*60)
        print("⚠️  WARNING: MULTIPLE VIDEO SESSIONS DETECTED!")
        print("="*60)
        print(f"The flight log contains {num_sessions} separate video recording sessions.")
        print("THE FLIGHT LOG MUST BE MODIFIED to contain only ONE video session.")
        print(f"Only the FIRST session will be processed (sessions 2-{num_sessions} will be ignored).")
        print("="*60 + "\n")

    print(f"Processing video session 1/{num_sessions}")

    # Extract FIRST session only
    start_idx, end_idx = sessions[0]
    session_df = df.iloc[start_idx:end_idx + 1].copy()
    print(f"Session duration: {len(session_df)} frames")

    # Parse timestamps and calculate relative time
    session_df['timestamp'] = session_df.apply(
        lambda row: parse_timestamp(row['CUSTOM.date [local]'], row['CUSTOM.updateTime [local]']),
        axis=1
    )
    time_start = session_df['timestamp'].iloc[0]
    session_df['time_seconds'] = (session_df['timestamp'] - time_start).dt.total_seconds()

    # Extract initial reference values
    lat_ref = session_df['OSD.latitude'].iloc[0]
    lon_ref = session_df['OSD.longitude'].iloc[0]
    height_ref = session_df['OSD.height [m]'].iloc[0]
    yaw_initial = session_df['OSD.yaw'].iloc[0]

    print(f"\nInitial position:")
    print(f"  Latitude: {lat_ref:.8f}")
    print(f"  Longitude: {lon_ref:.8f}")
    print(f"  Height: {height_ref:.2f} m")
    print(f"  Yaw: {yaw_initial:.2f}°")

    # Calculate world frame coordinates
    X_world = []
    Y_world = []
    Z_world = []

    for _, row in session_df.iterrows():
        lat = row['OSD.latitude']
        lon = row['OSD.longitude']
        height = row['OSD.height [m]']

        # Convert GPS to local meters
        east, north = gps_to_local_meters(lat, lon, lat_ref, lon_ref)

        # Transform to world frame
        x, y, z = rotate_to_world_frame(east, north, height - height_ref, yaw_initial)

        X_world.append(x)
        Y_world.append(y)
        Z_world.append(z)

    session_df['X_world'] = X_world
    session_df['Y_world'] = Y_world
    session_df['Z_world'] = Z_world

    # Calculate total rotation (OSD + GIMBAL) for absolute camera orientation
    # Get initial angles
    pitch_initial = session_df['OSD.pitch'].iloc[0] + session_df['GIMBAL.pitch'].iloc[0]
    roll_initial = session_df['OSD.roll'].iloc[0] + session_df['GIMBAL.roll'].iloc[0]
    yaw_initial_total = session_df['OSD.yaw'].iloc[0] + session_df['GIMBAL.yaw'].iloc[0]

    print(f"\nInitial camera orientation (OSD + GIMBAL):")
    print(f"  Pitch: {pitch_initial:.2f} deg")
    print(f"  Roll: {roll_initial:.2f} deg")
    print(f"  Yaw: {yaw_initial_total:.2f} deg")

    # Calculate relative pitch and roll (OSD + GIMBAL)
    session_df['pitch_total'] = (session_df['OSD.pitch'] + session_df['GIMBAL.pitch']) - pitch_initial
    session_df['roll_total'] = (session_df['OSD.roll'] + session_df['GIMBAL.roll']) - roll_initial

    # Handle yaw angle wrapping and make relative
    osd_yaw = session_df['OSD.yaw'].values
    gimbal_yaw = session_df['GIMBAL.yaw'].values
    yaw_total = osd_yaw + gimbal_yaw

    # Make relative to initial yaw
    yaw_relative = yaw_total - yaw_initial_total

    # Normalize to [-180, 180]
    yaw_relative = np.arctan2(np.sin(np.radians(yaw_relative)),
                              np.cos(np.radians(yaw_relative)))
    yaw_relative = np.degrees(yaw_relative)
    session_df['yaw_total'] = yaw_relative

    # Store raw OSD angles (absolute, not relative)
    session_df['osd_pitch'] = session_df['OSD.pitch']
    session_df['osd_roll'] = session_df['OSD.roll']
    session_df['osd_yaw'] = session_df['OSD.yaw']

    # Store raw GIMBAL angles (absolute, not relative)
    session_df['gimbal_pitch'] = session_df['GIMBAL.pitch']
    session_df['gimbal_roll'] = session_df['GIMBAL.roll']
    session_df['gimbal_yaw'] = session_df['GIMBAL.yaw']

    return session_df


def load_pose_estimates(npz_path):
    """
    Load pose estimates from ViPE output .npz file
    Returns DataFrame with frame, X_world, Y_world, Z_world, pitch_total, roll_total, yaw_total
    """
    print(f"\nLoading pose estimates: {npz_path}")

    # Load NPZ file
    data = np.load(npz_path)

    print(f"NPZ keys: {list(data.keys())}")

    # Extract poses (typically stored as 4x4 transformation matrices)
    # Common formats: 'poses', 'camera_poses', or 'transformations'
    if 'poses' in data:
        poses = data['poses']
    elif 'camera_poses' in data:
        poses = data['camera_poses']
    else:
        # Try to find the main array
        main_key = [k for k in data.keys() if not k.startswith('_')][0]
        poses = data[main_key]
        print(f"Using key: {main_key}")

    print(f"Pose shape: {poses.shape}")
    num_frames = poses.shape[0]
    print(f"Number of frames: {num_frames}")

    # Extract translation and rotation from poses
    # Assuming poses are 4x4 transformation matrices [R|t]
    translations = []
    rotations = []

    for i in range(num_frames):
        pose = poses[i]

        # Extract translation (last column, first 3 rows)
        if pose.shape == (4, 4):
            t = pose[:3, 3]
            R = pose[:3, :3]
        elif pose.shape == (3, 4):
            t = pose[:, 3]
            R = pose[:, :3]
        else:
            raise ValueError(f"Unexpected pose shape: {pose.shape}")

        translations.append(t)

        # Convert rotation matrix to Euler angles (ZYX convention)
        # pitch (x-axis), roll (y-axis), yaw (z-axis)
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

        rotations.append([np.degrees(pitch), np.degrees(roll), np.degrees(yaw)])

    translations = np.array(translations)
    rotations = np.array(rotations)

    # Use absolute poses in ViPE's world frame with actual rotation values
    pose_df = pd.DataFrame({
        'frame': np.arange(num_frames),
        'X_world': translations[:, 0],
        'Y_world': translations[:, 1],
        'Z_world': translations[:, 2],
        'pitch_total': rotations[:, 0],
        'roll_total': rotations[:, 1],
        'yaw_total': rotations[:, 2]
    })

    print(f"\nPose estimate statistics:")
    print(f"  X: [{pose_df['X_world'].min():.2f}, {pose_df['X_world'].max():.2f}] m")
    print(f"  Y: [{pose_df['Y_world'].min():.2f}, {pose_df['Y_world'].max():.2f}] m")
    print(f"  Z: [{pose_df['Z_world'].min():.2f}, {pose_df['Z_world'].max():.2f}] m")
    print(f"  Pitch: [{pose_df['pitch_total'].min():.2f}, {pose_df['pitch_total'].max():.2f}] deg")
    print(f"  Roll: [{pose_df['roll_total'].min():.2f}, {pose_df['roll_total'].max():.2f}] deg")
    print(f"  Yaw: [{pose_df['yaw_total'].min():.2f}, {pose_df['yaw_total'].max():.2f}] deg")

    return pose_df


def load_da3_estimates(npz_path):
    """
    Load DA3 pose estimates from da3_results.npz file
    The extrinsics array has shape Nx3x4 (rotation + translation)
    Returns DataFrame with frame, X_world, Y_world, Z_world, pitch_total, roll_total, yaw_total
    """
    print(f"\nLoading DA3 estimates: {npz_path}")

    # Load NPZ file
    data = np.load(npz_path)

    print(f"DA3 NPZ keys: {list(data.keys())}")

    # Extract extrinsics (Nx3x4 format)
    if 'extrinsics' not in data:
        print("Warning: 'extrinsics' key not found in DA3 results")
        return None

    extrinsics = data['extrinsics']
    print(f"Extrinsics shape: {extrinsics.shape}")

    num_frames = extrinsics.shape[0]
    print(f"Number of frames: {num_frames}")

    # Extract translation and rotation from extrinsics (3x4 matrices: [R|t])
    translations = []
    rotations = []

    for i in range(num_frames):
        extrinsic = extrinsics[i]  # 3x4 matrix (world-to-camera)

        # Extract extrinsic rotation (first 3 columns) and translation (last column)
        R_extrinsic = extrinsic[:, :3]
        t_extrinsic = extrinsic[:, 3]

        # Convert extrinsic (world-to-camera) to pose (camera-to-world)
        # R_pose = R_extrinsic^T
        # t_pose = -R_extrinsic^T @ t_extrinsic
        R = R_extrinsic.T
        t = -R @ t_extrinsic

        translations.append(t)

        # Convert rotation matrix to Euler angles (ZYX convention)
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

        rotations.append([np.degrees(pitch), np.degrees(roll), np.degrees(yaw)])

    translations = np.array(translations)
    rotations = np.array(rotations)

    # Create DataFrame
    da3_df = pd.DataFrame({
        'frame': np.arange(num_frames),
        'X_world': translations[:, 0],
        'Y_world': translations[:, 1],
        'Z_world': translations[:, 2],
        'pitch_total': rotations[:, 0],
        'roll_total': rotations[:, 1],
        'yaw_total': rotations[:, 2]
    })

    print(f"\nDA3 estimate statistics:")
    print(f"  X: [{da3_df['X_world'].min():.2f}, {da3_df['X_world'].max():.2f}] m")
    print(f"  Y: [{da3_df['Y_world'].min():.2f}, {da3_df['Y_world'].max():.2f}] m")
    print(f"  Z: [{da3_df['Z_world'].min():.2f}, {da3_df['Z_world'].max():.2f}] m")
    print(f"  Pitch: [{da3_df['pitch_total'].min():.2f}, {da3_df['pitch_total'].max():.2f}] deg")
    print(f"  Roll: [{da3_df['roll_total'].min():.2f}, {da3_df['roll_total'].max():.2f}] deg")
    print(f"  Yaw: [{da3_df['yaw_total'].min():.2f}, {da3_df['yaw_total'].max():.2f}] deg")

    return da3_df


def align_trajectories(flight_log_df, state_estimates_df, target_fps=5.0):
    """
    Align flight log to state estimate timestamps by:
    1. Truncating flight log to match estimate duration
    2. Downsampling flight log to match estimate sampling rate

    Returns: (aligned_gt_df, aligned_est_df) with matching time_seconds
    """
    print(f"\n{'='*60}")
    print("Aligning trajectories...")
    print(f"{'='*60}")

    # Calculate time from frame numbers (assuming target_fps)
    state_estimates_df = state_estimates_df.copy()
    state_estimates_df['time_seconds'] = state_estimates_df['frame'] / target_fps

    max_time = state_estimates_df['time_seconds'].max()
    print(f"State estimate duration: {max_time:.2f} seconds ({max_time/60:.2f} minutes)")

    # Truncate flight log to match duration
    flight_log_truncated = flight_log_df[flight_log_df['time_seconds'] <= max_time].copy()
    print(f"Flight log truncated from {len(flight_log_df)} to {len(flight_log_truncated)} samples")

    # Get target timestamps from state estimates
    target_times = state_estimates_df['time_seconds'].values
    print(f"Downsampling flight log to {len(target_times)} timestamps (matching estimates)")

    # Interpolate flight log to match estimate timestamps
    aligned_gt = pd.DataFrame({'time_seconds': target_times})

    # Interpolate translation
    for col in ['X_world', 'Y_world', 'Z_world']:
        f = interpolate.interp1d(flight_log_truncated['time_seconds'],
                                  flight_log_truncated[col],
                                  kind='linear', fill_value='extrapolate')
        aligned_gt[col] = f(target_times)

    # Interpolate rotation (handle angle wrapping for yaw)
    for col in ['pitch_total', 'roll_total']:
        f = interpolate.interp1d(flight_log_truncated['time_seconds'],
                                  flight_log_truncated[col],
                                  kind='linear', fill_value='extrapolate')
        aligned_gt[col] = f(target_times)

    # Special handling for yaw (circular interpolation)
    yaw_rad = np.radians(flight_log_truncated['yaw_total'])
    yaw_sin = np.sin(yaw_rad)
    yaw_cos = np.cos(yaw_rad)

    f_sin = interpolate.interp1d(flight_log_truncated['time_seconds'], yaw_sin,
                                   kind='linear', fill_value='extrapolate')
    f_cos = interpolate.interp1d(flight_log_truncated['time_seconds'], yaw_cos,
                                   kind='linear', fill_value='extrapolate')

    yaw_interp = np.degrees(np.arctan2(f_sin(target_times), f_cos(target_times)))
    aligned_gt['yaw_total'] = yaw_interp

    # Prepare estimates DataFrame (already has X_world, Y_world, Z_world, pitch, roll, yaw)
    aligned_est = state_estimates_df[['time_seconds', 'frame', 'X_world', 'Y_world', 'Z_world',
                                      'pitch_total', 'roll_total', 'yaw_total']].copy()

    print(f"✅ Alignment complete: {len(aligned_gt)} samples")
    print(f"   Time range: 0.00 to {max_time:.2f} seconds")

    return aligned_gt, aligned_est


def calculate_errors(gt_df, est_df):
    """
    Calculate errors: estimate - ground truth
    Returns DataFrame with error values
    """
    error_df = pd.DataFrame()
    error_df['time_seconds'] = gt_df['time_seconds']

    # Translation errors
    error_df['X_error'] = est_df['X_world'] - gt_df['X_world']
    error_df['Y_error'] = est_df['Y_world'] - gt_df['Y_world']
    error_df['Z_error'] = est_df['Z_world'] - gt_df['Z_world']

    # Rotation errors (only if estimates available)
    error_df['pitch_error'] = est_df['pitch_total'] - gt_df['pitch_total']
    error_df['roll_error'] = est_df['roll_total'] - gt_df['roll_total']
    error_df['yaw_error'] = est_df['yaw_total'] - gt_df['yaw_total']

    # Calculate RMS errors
    rms_x = np.sqrt(np.mean(error_df['X_error']**2))
    rms_y = np.sqrt(np.mean(error_df['Y_error']**2))
    rms_z = np.sqrt(np.mean(error_df['Z_error']**2))
    rms_pitch = np.sqrt(np.mean(error_df['pitch_error']**2))
    rms_roll = np.sqrt(np.mean(error_df['roll_error']**2))
    rms_yaw = np.sqrt(np.mean(error_df['yaw_error']**2))

    print(f"\n{'='*60}")
    print("RMS Errors:")
    print(f"{'='*60}")
    print("Translation:")
    print(f"  X (Right):   {rms_x:.3f} m")
    print(f"  Y (Down):    {rms_y:.3f} m")
    print(f"  Z (Forward): {rms_z:.3f} m")
    print(f"  3D:          {np.sqrt(rms_x**2 + rms_y**2 + rms_z**2):.3f} m")
    print("\nRotation:")
    print(f"  Pitch:       {rms_pitch:.3f} deg")
    print(f"  Roll:        {rms_roll:.3f} deg")
    print(f"  Yaw:         {rms_yaw:.3f} deg")

    return error_df


def format_time_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def plot_comparison(gt_df, est_df, output_path, da3_df=None):
    """
    Create 2x3 subplot visualization comparing ground truth and estimates
    Top row: Translational (X, Y, Z)
    Bottom row: Rotational (Pitch, Roll, Yaw)

    Args:
        gt_df: Ground truth DataFrame
        est_df: Estimate DataFrame (ViPE-X)
        output_path: Path to save the figure
        da3_df: Optional DA3 estimate DataFrame
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('UAV 6 DoF - Ground Truth vs Estimates', fontsize=16, fontweight='bold')

    time_seconds = gt_df['time_seconds'].values

    # Column names
    pos_cols = ['X_world', 'Y_world', 'Z_world']
    rot_cols = ['pitch_total', 'roll_total', 'yaw_total']
    pos_labels = ['X^W (Right)', 'Y^W (Down)', 'Z^W (Forward)']
    rot_labels = ['Pitch (Relative)', 'Roll (Relative)', 'Yaw (Relative)']

    # Calculate unified y-axis limits for translation (all X, Y, Z data)
    pos_values_gt = np.concatenate([gt_df[col].values for col in pos_cols])
    pos_values_est = np.concatenate([est_df[col].values for col in pos_cols])
    pos_all = [pos_values_gt, pos_values_est]
    if da3_df is not None:
        pos_values_da3 = np.concatenate([da3_df[col].values for col in pos_cols])
        pos_all.append(pos_values_da3)
    pos_min = min([v.min() for v in pos_all])
    pos_max = max([v.max() for v in pos_all])
    pos_margin = (pos_max - pos_min) * 0.05  # 5% margin
    pos_ylim = (pos_min - pos_margin, pos_max + pos_margin)

    # Calculate unified y-axis limits for rotation (all pitch, roll, yaw data)
    rot_values_gt = np.concatenate([gt_df[col].values for col in rot_cols])
    rot_values_est = np.concatenate([est_df[col].values for col in rot_cols])
    rot_all = [rot_values_gt, rot_values_est]
    if da3_df is not None:
        rot_values_da3 = np.concatenate([da3_df[col].values for col in rot_cols])
        rot_all.append(rot_values_da3)
    rot_min = min([v.min() for v in rot_all])
    rot_max = max([v.max() for v in rot_all])
    rot_margin = (rot_max - rot_min) * 0.05  # 5% margin
    rot_ylim = (rot_min - rot_margin, rot_max + rot_margin)

    # Translational plots
    for idx, (col, label) in enumerate(zip(pos_cols, pos_labels)):
        ax = axes[0, idx]

        # Plot ground truth (red dotted)
        ax.plot(time_seconds, gt_df[col].values, 'r:', linewidth=2, label='Ground Truth', alpha=0.8)

        # Plot ViPE-X estimates (black solid)
        ax.plot(time_seconds, est_df[col].values, 'k-', linewidth=1.5, label='ViPE-X')

        # Plot DA3 estimates (blue dash-dot) if available
        if da3_df is not None:
            ax.plot(time_seconds, da3_df[col].values, 'b-.', linewidth=1.5, label='DA3')

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Position (m)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Set unified y-axis limits for translation
        ax.set_ylim(pos_ylim)

        # Format x-axis as MM:SS
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    # Rotational plots
    for idx, (col, label) in enumerate(zip(rot_cols, rot_labels)):
        ax = axes[1, idx]

        # Plot ground truth (red dotted)
        ax.plot(time_seconds, gt_df[col].values, 'r:', linewidth=2, label='Ground Truth', alpha=0.8)

        # Plot ViPE-X estimates (black solid)
        ax.plot(time_seconds, est_df[col].values, 'k-', linewidth=1.5, label='ViPE-X')

        # Plot DA3 estimates (blue dash-dot) if available
        if da3_df is not None:
            ax.plot(time_seconds, da3_df[col].values, 'b-.', linewidth=1.5, label='DA3')

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.legend(loc='best', fontsize=9)

        # Set unified y-axis limits for rotation
        ax.set_ylim(rot_ylim)

        # Format x-axis as MM:SS
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def plot_errors(error_df, output_path):
    """
    Create 2x3 subplot visualization of errors
    Top row: Translational errors (X, Y, Z)
    Bottom row: Rotational errors (Pitch, Roll, Yaw)

    Args:
        error_df: Error DataFrame
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('UAV 6 DoF - Errors (Estimate - Ground Truth)', fontsize=16, fontweight='bold')

    time_seconds = error_df['time_seconds'].values

    # Column names
    pos_cols = ['X_error', 'Y_error', 'Z_error']
    rot_cols = ['pitch_error', 'roll_error', 'yaw_error']
    pos_labels = ['X Error (Right)', 'Y Error (Down)', 'Z Error (Forward)']
    rot_labels = ['Pitch Error', 'Roll Error', 'Yaw Error']

    # Calculate unified y-axis limits for translation errors (all X, Y, Z errors)
    pos_error_values = np.concatenate([error_df[col].values for col in pos_cols])
    pos_error_min = pos_error_values.min()
    pos_error_max = pos_error_values.max()
    pos_error_margin = (pos_error_max - pos_error_min) * 0.05  # 5% margin
    pos_error_ylim = (pos_error_min - pos_error_margin, pos_error_max + pos_error_margin)

    # Calculate unified y-axis limits for rotation errors (all pitch, roll, yaw errors)
    rot_error_values = np.concatenate([error_df[col].values for col in rot_cols])
    rot_error_min = rot_error_values.min()
    rot_error_max = rot_error_values.max()
    rot_error_margin = (rot_error_max - rot_error_min) * 0.05  # 5% margin
    rot_error_ylim = (rot_error_min - rot_error_margin, rot_error_max + rot_error_margin)

    # Translational error plots
    for idx, (col, label) in enumerate(zip(pos_cols, pos_labels)):
        ax = axes[0, idx]
        data = error_df[col].values

        ax.plot(time_seconds, data, 'r-', linewidth=1.5)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Error (m)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

        # Set unified y-axis limits for translation errors
        ax.set_ylim(pos_error_ylim)

        # Format x-axis as MM:SS
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    # Rotational error plots
    for idx, (col, label) in enumerate(zip(rot_cols, rot_labels)):
        ax = axes[1, idx]
        data = error_df[col].values

        ax.plot(time_seconds, data, 'r-', linewidth=1.5)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Error (degrees)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

        # Set unified y-axis limits for rotation errors
        ax.set_ylim(rot_error_ylim)

        # Format x-axis as MM:SS
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")

    plt.close()


def plot_ground_truth_only(gt_df, output_path):
    """
    Create 3x3 subplot visualization of ground truth from flight log
    Row 1: Translational motion (X, Y, Z) in world frame
    Row 2: OSD rotational angles (Pitch, Roll, Yaw) - absolute
    Row 3: GIMBAL rotational angles (Pitch, Roll, Yaw) - absolute

    Args:
        gt_df: Ground truth DataFrame with time_seconds, X_world, Y_world, Z_world,
               osd_pitch, osd_roll, osd_yaw, gimbal_pitch, gimbal_roll, gimbal_yaw
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('UAV Ground Truth from Flight Log', fontsize=16, fontweight='bold')

    time_seconds = gt_df['time_seconds'].values

    # Column names and labels
    pos_cols = ['X_world', 'Y_world', 'Z_world']
    osd_cols = ['osd_pitch', 'osd_roll', 'osd_yaw']
    gimbal_cols = ['gimbal_pitch', 'gimbal_roll', 'gimbal_yaw']

    pos_labels = ['X^W (Right)', 'Y^W (Down)', 'Z^W (Forward)']
    osd_labels = ['OSD Pitch', 'OSD Roll', 'OSD Yaw']
    gimbal_labels = ['Gimbal Pitch', 'Gimbal Roll', 'Gimbal Yaw']

    # Calculate unified y-axis limits for translation
    pos_values = np.concatenate([gt_df[col].values for col in pos_cols])
    pos_min, pos_max = pos_values.min(), pos_values.max()
    pos_margin = (pos_max - pos_min) * 0.05
    pos_ylim = (pos_min - pos_margin, pos_max + pos_margin)

    # Calculate unified y-axis limits for OSD angles
    osd_values = np.concatenate([gt_df[col].values for col in osd_cols])
    osd_ylim = (osd_values.min() - 5, osd_values.max() + 5)

    # Calculate unified y-axis limits for GIMBAL angles
    gimbal_values = np.concatenate([gt_df[col].values for col in gimbal_cols])
    gimbal_ylim = (gimbal_values.min() - 5, gimbal_values.max() + 5)

    # Row 1: Translational plots
    for idx, (col, label) in enumerate(zip(pos_cols, pos_labels)):
        ax = axes[0, idx]
        ax.plot(time_seconds, gt_df[col].values, 'r-', linewidth=1.5)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Position (m)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(pos_ylim)
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    # Row 2: OSD rotational plots
    for idx, (col, label) in enumerate(zip(osd_cols, osd_labels)):
        ax = axes[1, idx]
        ax.plot(time_seconds, gt_df[col].values, 'b-', linewidth=1.5)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(osd_ylim)
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    # Row 3: GIMBAL rotational plots
    for idx, (col, label) in enumerate(zip(gimbal_cols, gimbal_labels)):
        ax = axes[2, idx]
        ax.plot(time_seconds, gt_df[col].values, 'g-', linewidth=1.5)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax.set_ylim(gimbal_ylim)
        ax.set_xlim(0, time_seconds[-1])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_time_mmss(x)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze DJI flight log and compare with ViPE pose estimates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/analysis_flightlog.py data/DJIFlightRecord.csv
  python scripts/analysis_flightlog.py data/DJIFlightRecord.csv output/sDJI_0910_W/pose/sDJI_0910_W.npz
  python scripts/analysis_flightlog.py data/DJIFlightRecord.csv output/sDJI_0910_W/pose/sDJI_0910_W.npz --fps 30
        """
    )
    parser.add_argument('flight_log', type=str, help='Path to DJI flight log CSV file')
    parser.add_argument('pose_estimates', type=str, nargs='?', default=None,
                        help='Path to ViPE pose estimates NPZ file (optional - if not provided, only GT is plotted)')
    parser.add_argument('--fps', type=float, default=5.0, help='Frame rate of pose estimates (default: 5.0)')
    parser.add_argument('--start', type=float, default=None, metavar='SECONDS',
                        help='Crop: only include data from this time offset (seconds from recording start)')
    parser.add_argument('--end', type=float, default=None, metavar='SECONDS',
                        help='Crop: only include data up to this time offset (seconds from recording start)')

    args = parser.parse_args()

    # Get paths from arguments
    flight_log_path = Path(args.flight_log)
    pose_estimates_path = Path(args.pose_estimates) if args.pose_estimates else None
    target_fps = args.fps

    # Verify flight log exists
    if not flight_log_path.exists():
        print(f"Error: Flight log not found: {flight_log_path}")
        sys.exit(1)

    # Verify pose estimates exist (if provided)
    if pose_estimates_path is not None and not pose_estimates_path.exists():
        print(f"Error: Pose estimates not found: {pose_estimates_path}")
        sys.exit(1)

    print(f"{'='*60}")
    if pose_estimates_path:
        print("UAV Trajectory Analysis: Ground Truth vs Estimates")
    else:
        print("UAV Trajectory Analysis: Ground Truth Only")
    print(f"{'='*60}")
    print(f"Flight log: {flight_log_path}")
    if pose_estimates_path:
        print(f"Pose estimates: {pose_estimates_path}")
        print(f"Target FPS: {target_fps}")
    else:
        print("Mode: Ground Truth visualization only (no pose estimates provided)")

    # Load flight log and calculate 6 DoF
    flight_log_df = load_flight_log_6dof(flight_log_path)
    if flight_log_df is None:
        sys.exit(1)

    # Apply time crop if requested
    if args.start is not None or args.end is not None:
        total_duration = flight_log_df['time_seconds'].iloc[-1]
        start_s = args.start if args.start is not None else 0.0
        end_s = args.end if args.end is not None else total_duration
        print(f"\nCropping to [{start_s:.1f}s, {end_s:.1f}s] (total: {total_duration:.1f}s)")
        flight_log_df = flight_log_df[
            (flight_log_df['time_seconds'] >= start_s) &
            (flight_log_df['time_seconds'] <= end_s)
        ].copy()
        # Re-zero time_seconds so plots start from 0
        flight_log_df['time_seconds'] -= start_s
        flight_log_df = flight_log_df.reset_index(drop=True)
        print(f"  {len(flight_log_df)} rows remain after crop")
        if len(flight_log_df) == 0:
            print("Error: No data in the specified time range.")
            sys.exit(1)

    # Build crop suffix for output filenames (e.g. "_10s_60s", empty if no crop)
    crop_suffix = ""
    if args.start is not None or args.end is not None:
        s = int(args.start) if args.start is not None else 0
        e = int(args.end) if args.end is not None else int(flight_log_df['time_seconds'].iloc[-1] + (args.start or 0))
        crop_suffix = f"_{s}s_{e}s"

    # Ground truth only mode
    if pose_estimates_path is None:
        # Determine output directory based on flight log name
        flight_log_stem = flight_log_path.stem
        output_dir = Path("outputs") / "flight_log_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating ground truth plot...")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")

        # Generate ground truth only plot
        output_file = output_dir / f"{flight_log_stem}_6dof_gt{crop_suffix}.png"
        plot_ground_truth_only(flight_log_df, output_file)

        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  - {flight_log_stem}_6dof_gt{crop_suffix}.png (Ground Truth: red solid)")
        sys.exit(0)

    # Comparison mode: Load pose estimates (full 6 DoF from ViPE)
    pose_estimates_df = load_pose_estimates(pose_estimates_path)
    if pose_estimates_df is None:
        sys.exit(1)

    # Check for DA3 results in the same pose directory
    da3_path = pose_estimates_path.parent / "da3_results.npz"
    da3_estimates_df = None
    if da3_path.exists():
        print(f"\nDA3 results found: {da3_path}")
        da3_estimates_df = load_da3_estimates(da3_path)
    else:
        print(f"\nNo DA3 results found at: {da3_path}")

    # Align trajectories
    gt_aligned, est_aligned = align_trajectories(flight_log_df, pose_estimates_df, target_fps=target_fps)

    # Align DA3 if available (using same alignment)
    da3_aligned = None
    if da3_estimates_df is not None:
        # DA3 uses same frame indices, so apply same time calculation
        da3_estimates_df = da3_estimates_df.copy()
        da3_estimates_df['time_seconds'] = da3_estimates_df['frame'] / target_fps

        # Interpolate DA3 to match aligned timestamps
        target_times = gt_aligned['time_seconds'].values
        da3_aligned = pd.DataFrame({'time_seconds': target_times})

        for col in ['X_world', 'Y_world', 'Z_world', 'pitch_total', 'roll_total', 'yaw_total']:
            f = interpolate.interp1d(da3_estimates_df['time_seconds'],
                                      da3_estimates_df[col],
                                      kind='linear', fill_value='extrapolate')
            da3_aligned[col] = f(target_times)

        print(f"DA3 alignment complete: {len(da3_aligned)} samples")

    # Calculate errors
    error_df = calculate_errors(gt_aligned, est_aligned)

    # Determine output directory: output/<video_folder_name>/validation/
    video_folder_name = pose_estimates_path.parent.parent.name
    validation_dir = Path("outputs") / video_folder_name / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating plots...")
    print(f"{'='*60}")
    print(f"Output directory: {validation_dir}")

    # Generate two plots
    plot_comparison(gt_aligned, est_aligned, validation_dir / f"gt_vs_estimate_6dof{crop_suffix}.png", da3_df=da3_aligned)
    plot_errors(error_df, validation_dir / f"error_6dof{crop_suffix}.png")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {validation_dir}")
    print("\nGenerated files:")
    if da3_aligned is not None:
        print(f"  - gt_vs_estimate_6dof{crop_suffix}.png (GT: red dotted, ViPE-X: black solid, DA3: blue dash-dot)")
    else:
        print(f"  - gt_vs_estimate_6dof{crop_suffix}.png (GT: red dotted, ViPE-X: black solid)")
    print(f"  - error_6dof{crop_suffix}.png (Errors: red solid)")


if __name__ == "__main__":
    main()
