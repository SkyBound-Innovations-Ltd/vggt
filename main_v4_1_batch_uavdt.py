#!/usr/bin/env python3
"""
Batch UAVDT State Estimation

Processes UAVDT dataset videos in batch mode using VGGT for depth estimation
and Kalman filtering for state estimation.

Input:
  - JSON file with tracking data (contains "id" field linking to video names)
  - Video folder with {id}.mp4 files

Output:
  - Updated JSON file with state estimation fields added to each entry
  - Depth visualization videos saved as {id}_depth.mp4

Usage:
    python main_v4_1_batch_uavdt.py \
        --input-json Tiancheng/uavdt_q2_vggt_input_format.json \
        --video-dir Tiancheng/uavdt \
        --fps 30

Author: Claude Code Assistant
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

# Import shared functions from main_v4_no_flight
from main_v4_no_flight import (
    KalmanFilter,
    rescale_bbox,
    get_bbox_center,
    extract_robust_depth,
    pixel_to_ned,
    compute_heading_ned,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch UAVDT state estimation using VGGT'
    )

    # Required arguments
    parser.add_argument('--input-json', required=True,
                        help='Input JSON file with tracking data')
    parser.add_argument('--video-dir', required=True,
                        help='Directory containing video files')

    # Processing parameters
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Video frame rate (default: 30.0)')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='VGGT batch size (default: 5)')

    # Kalman Filter parameters
    parser.add_argument('--no-kf', action='store_true',
                        help='Disable Kalman Filter (use raw measurements)')
    parser.add_argument('--kf-sigma-a', type=float, default=0.1,
                        help='KF process noise (acceleration) std (default: 0.1)')
    parser.add_argument('--kf-sigma-meas', type=float, default=0.5,
                        help='KF measurement noise std (default: 0.5)')

    # Output options
    parser.add_argument('--output-json', type=str, default=None,
                        help='Output JSON file (default: overwrite input)')
    parser.add_argument('--no-depth-video', action='store_true',
                        help='Skip depth video generation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing (skip videos already processed)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of videos to process (for testing)')

    return parser.parse_args()


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Auto-detect video resolution."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return width, height


def extract_all_frames(video_path: str) -> Tuple[List[np.ndarray], int, int]:
    """
    Extract all frames from video at native frame rate.

    Returns:
        Tuple of (frames list, width, height)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Video: {width}x{height}, {video_fps:.1f} fps, {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    print(f"  Extracted {len(frames)} frames")
    return frames, width, height


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
    model: VGGT,
    batch_size: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Dict:
    """
    Run VGGT inference on frames using pre-loaded model.

    Returns:
        Dictionary with depth maps, extrinsics, intrinsics
    """
    # Prepare frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        images = save_frames_and_load(frames, tmp_dir)

    images = images.to(device)
    num_frames = images.shape[0]

    # Process in batches
    all_depths = []
    all_pose_enc = []

    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx + batch_size, num_frames)
        batch = images[start_idx:end_idx].unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(batch)

        all_depths.append(predictions["depth"].squeeze(0).cpu())
        all_pose_enc.append(predictions["pose_enc"].squeeze(0).cpu())

        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatenate results
    depths = torch.cat(all_depths, dim=0)  # [N, H, W, 1]
    pose_enc = torch.cat(all_pose_enc, dim=0)  # [N, 9]

    # Convert pose encoding to extrinsics and intrinsics
    H, W = depths.shape[1:3]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc.unsqueeze(0),
        image_size_hw=(H, W)
    )
    extrinsics = extrinsics.squeeze(0)  # [N, 3, 4]
    intrinsics = intrinsics.squeeze(0)  # [N, 3, 3]

    return {
        "depth": depths.numpy(),
        "extrinsics": extrinsics.numpy(),
        "intrinsics": intrinsics.numpy(),
        "image_size": (H, W)
    }


def create_depth_video(
    frames: List[np.ndarray],
    depth_maps: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    colormap: int = cv2.COLORMAP_TURBO
):
    """Create side-by-side depth visualization video."""
    if len(frames) != len(depth_maps):
        print(f"  Warning: frame count mismatch ({len(frames)} vs {len(depth_maps)})")
        min_len = min(len(frames), len(depth_maps))
        frames = frames[:min_len]
        depth_maps = depth_maps[:min_len]

    # Get dimensions
    h, w = frames[0].shape[:2]
    depth_h, depth_w = depth_maps[0].shape[:2]

    # Output will be side-by-side
    out_w = w * 2
    out_h = h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
        # Process depth map
        depth_squeezed = depth.squeeze()

        # Normalize depth for visualization
        valid_mask = (depth_squeezed > 0) & np.isfinite(depth_squeezed)
        if valid_mask.any():
            d_min = np.percentile(depth_squeezed[valid_mask], 2)
            d_max = np.percentile(depth_squeezed[valid_mask], 98)
            depth_norm = np.clip((depth_squeezed - d_min) / (d_max - d_min + 1e-6), 0, 1)
        else:
            depth_norm = np.zeros_like(depth_squeezed)

        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)

        # Resize depth to match frame
        depth_resized = cv2.resize(depth_colored, (w, h))

        # Convert RGB frame to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Combine side-by-side
        combined = np.hstack([frame_bgr, depth_resized])

        writer.write(combined)

    writer.release()
    print(f"  Saved depth video: {output_path}")


def process_single_video(
    video_id: str,
    tracks: List[Dict],
    video_path: str,
    model: VGGT,
    fps: float,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    use_kf: bool = True,
    kf_sigma_a: float = 0.1,
    kf_sigma_meas: float = 0.5,
    save_depth_video: bool = True
) -> List[Dict]:
    """
    Process a single video and return updated track entries.

    Args:
        video_id: Video identifier
        tracks: List of track entries for this video
        video_path: Path to video file
        model: Pre-loaded VGGT model
        fps: Video frame rate
        batch_size: VGGT batch size
        device: Compute device
        dtype: Model dtype
        use_kf: Use Kalman Filter
        kf_sigma_a: KF process noise
        kf_sigma_meas: KF measurement noise
        save_depth_video: Save depth visualization

    Returns:
        Updated track entries with state estimation fields
    """
    print(f"\nProcessing: {video_id}")

    # Extract frames and get resolution
    frames, width, height = extract_all_frames(video_path)
    source_res = (width, height)

    if len(frames) == 0:
        print(f"  WARNING: No frames extracted from {video_path}")
        return tracks

    # Run VGGT inference
    print(f"  Running VGGT inference ({len(frames)} frames)...")
    vggt_outputs = run_vggt_inference(frames, model, batch_size, device, dtype)

    depth_h, depth_w = vggt_outputs["image_size"]
    depth_maps = vggt_outputs["depth"]
    extrinsics = vggt_outputs["extrinsics"]
    intrinsics = vggt_outputs["intrinsics"]

    # Scale factors
    sx = depth_w / width
    sy = depth_h / height

    # Reference extrinsic (first frame)
    reference_extrinsic = extrinsics[0]

    # Group tracks by track_id for Kalman filtering
    tracks_by_id = {}
    for track in tracks:
        tid = track.get('track_id', track.get('id'))
        if tid not in tracks_by_id:
            tracks_by_id[tid] = []
        tracks_by_id[tid].append(track)

    # Sort each group by frame_id
    for tid in tracks_by_id:
        tracks_by_id[tid].sort(key=lambda x: x['frame_id'])

    # Normalize frame_ids to 0-indexed (some videos have absolute frame numbers)
    all_frame_ids = [t['frame_id'] for t in tracks]
    min_frame_id = min(all_frame_ids) if all_frame_ids else 0
    if min_frame_id > 0:
        print(f"  Normalizing frame_ids: {min_frame_id} -> 0")

    print(f"  Processing {len(tracks_by_id)} tracks with {len(tracks)} detections...")

    # Process each track
    updated_tracks = []

    for tid, track_group in tracks_by_id.items():
        kf = None
        track_data = []

        # First pass: collect measurements
        for track in track_group:
            frame_id = track['frame_id']
            frame_idx = frame_id - min_frame_id  # Normalize to 0-indexed
            bbox = track['bbox']

            if frame_idx < 0 or frame_idx >= len(depth_maps):
                # Frame out of range, keep original entry
                updated_tracks.append(track.copy())
                continue

            depth_map = depth_maps[frame_idx].squeeze()
            K = intrinsics[min(frame_idx, len(intrinsics) - 1)]
            ext = extrinsics[min(frame_idx, len(extrinsics) - 1)]

            # Rescale bbox
            bbox_scaled = rescale_bbox(bbox, sx, sy)

            # Get depth
            depth = extract_robust_depth(depth_map, bbox_scaled)
            if depth is None or depth <= 0:
                # Invalid depth, keep original entry without state
                track_copy = track.copy()
                track_copy['pos_vggt'] = None
                track_copy['vel_vggt'] = None
                track_copy['speed_units'] = None
                track_copy['heading_deg'] = None
                track_copy['depth_units'] = None
                updated_tracks.append(track_copy)
                continue

            # Get bbox center
            u, v = get_bbox_center(bbox_scaled)

            # Unproject to NED coordinates
            point_ned = pixel_to_ned(u, v, depth, K, ext, reference_extrinsic)

            track_data.append((frame_id, track.copy(), point_ned, depth))

        if not track_data:
            continue

        # Second pass: apply Kalman Filter
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

            # Compute speed and heading
            speed = np.sqrt(filtered_vel[0]**2 + filtered_vel[1]**2)
            heading = compute_heading_ned(filtered_vel)

            # Convert velocity to units/second
            vel_per_sec = filtered_vel * fps
            speed_per_sec = speed * fps

            # Update track entry with state information
            track['pos_vggt'] = filtered_pos.tolist()
            track['vel_vggt'] = vel_per_sec.tolist()
            track['speed_units'] = float(speed_per_sec)
            track['heading_deg'] = float(heading)
            track['depth_units'] = float(depth)

            updated_tracks.append(track)

    # Save depth video
    if save_depth_video:
        video_dir = os.path.dirname(video_path)
        depth_video_path = os.path.join(video_dir, f"{video_id}_depth.mp4")
        create_depth_video(frames, depth_maps, depth_video_path, fps)

    print(f"  Completed: {len(updated_tracks)} entries processed")

    return updated_tracks


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Batch UAVDT State Estimation")
    print("=" * 60)

    # Load input JSON
    print(f"\nLoading: {args.input_json}")
    with open(args.input_json, 'r') as f:
        all_tracks = json.load(f)

    print(f"Total entries: {len(all_tracks)}")

    # Group by video ID
    tracks_by_video = {}
    for track in all_tracks:
        vid = track['id']
        if vid not in tracks_by_video:
            tracks_by_video[vid] = []
        tracks_by_video[vid].append(track)

    print(f"Unique videos: {len(tracks_by_video)}")

    # Check which videos exist
    video_dir = Path(args.video_dir)
    available_videos = set(p.stem for p in video_dir.glob("*.mp4"))

    videos_to_process = []
    videos_missing = []

    for vid in tracks_by_video:
        if vid in available_videos:
            videos_to_process.append(vid)
        else:
            videos_missing.append(vid)

    print(f"Videos available: {len(videos_to_process)}")
    print(f"Videos missing: {len(videos_missing)}")

    if videos_missing:
        print(f"  Missing: {', '.join(videos_missing[:5])}{'...' if len(videos_missing) > 5 else ''}")

    # Apply limit if specified
    if args.limit:
        videos_to_process = videos_to_process[:args.limit]
        print(f"Limited to: {len(videos_to_process)} videos")

    # Initialize VGGT model
    print("\nLoading VGGT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability >= 8 else torch.float16
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: {gpu_name}, dtype: {dtype}")
    else:
        dtype = torch.float32
        print("Using CPU (no CUDA available)")

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("Model loaded successfully")

    # Check for resume
    processed_videos = set()
    if args.resume:
        # Check which videos already have state info
        for track in all_tracks:
            if track.get('pos_vggt') is not None:
                processed_videos.add(track['id'])

        if processed_videos:
            print(f"\nResuming: {len(processed_videos)} videos already processed")
            videos_to_process = [v for v in videos_to_process if v not in processed_videos]
            print(f"Remaining: {len(videos_to_process)} videos")

    # Process each video
    print(f"\n{'='*60}")
    print(f"Processing {len(videos_to_process)} videos")
    print(f"{'='*60}")

    updated_tracks_by_video = {}
    start_time = time.time()

    for i, vid in enumerate(videos_to_process):
        video_path = video_dir / f"{vid}.mp4"

        print(f"\n[{i+1}/{len(videos_to_process)}] {vid}")

        try:
            updated = process_single_video(
                video_id=vid,
                tracks=tracks_by_video[vid],
                video_path=str(video_path),
                model=model,
                fps=args.fps,
                batch_size=args.batch_size,
                device=device,
                dtype=dtype,
                use_kf=not args.no_kf,
                kf_sigma_a=args.kf_sigma_a,
                kf_sigma_meas=args.kf_sigma_meas,
                save_depth_video=not args.no_depth_video
            )
            updated_tracks_by_video[vid] = updated

        except Exception as e:
            print(f"  ERROR: {e}")
            # Keep original tracks on error
            updated_tracks_by_video[vid] = tracks_by_video[vid]

        # Periodic save
        if (i + 1) % 10 == 0:
            print(f"\n  Checkpoint: processed {i+1} videos, saving...")
            save_output(
                args, all_tracks, tracks_by_video, updated_tracks_by_video,
                processed_videos, videos_missing
            )

    # Final save
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed {len(videos_to_process)} videos in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    save_output(
        args, all_tracks, tracks_by_video, updated_tracks_by_video,
        processed_videos, videos_missing
    )

    print("\nDone!")


def save_output(
    args,
    all_tracks: List[Dict],
    tracks_by_video: Dict[str, List[Dict]],
    updated_tracks_by_video: Dict[str, List[Dict]],
    processed_videos: set,
    videos_missing: List[str]
):
    """Save updated JSON output."""
    # Combine all tracks
    final_tracks = []

    for vid, tracks in tracks_by_video.items():
        if vid in updated_tracks_by_video:
            final_tracks.extend(updated_tracks_by_video[vid])
        elif vid in processed_videos:
            # Already processed in previous run (resume mode)
            final_tracks.extend(tracks)
        elif vid in videos_missing:
            # Missing video, keep original
            final_tracks.extend(tracks)
        else:
            # Not processed yet
            final_tracks.extend(tracks)

    # Determine output path
    output_path = args.output_json if args.output_json else args.input_json

    # Save
    with open(output_path, 'w') as f:
        json.dump(final_tracks, f, indent=2)

    print(f"Saved: {output_path}")
    print(f"  Total entries: {len(final_tracks)}")

    # Count entries with state info
    with_state = sum(1 for t in final_tracks if t.get('pos_vggt') is not None)
    print(f"  Entries with state: {with_state}")


if __name__ == "__main__":
    main()
