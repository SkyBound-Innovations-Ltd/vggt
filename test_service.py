"""
Test script for VGGT-P Service API.

Usage:
    python test_service.py --video <video> --flight-log <csv> --tracking <json>
"""

import requests
import argparse
import json
import time
from pathlib import Path


def test_api(
    video_path: str,
    flight_log_path: str,
    tracking_json_path: str,
    output_path: str = "output_api.json",
    base_url: str = "http://localhost:8000",
    fps: float = 5.0,
    batch_size: int = 5
):
    """Test the VGGT-P service API."""

    # Check files exist
    for path in [video_path, flight_log_path, tracking_json_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    print("=" * 60)
    print("VGGT-P Service API Test")
    print("=" * 60)

    # 1. Health check
    print("\n1. Checking service health...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Device: {health['device']}")
        print(f"   GPU available: {health['gpu_available']}")
    except Exception as e:
        print(f"   ERROR: Service not available - {e}")
        print(f"   Make sure service is running: python main_v3_service.py")
        return

    # 2. Process video
    print(f"\n2. Submitting processing request...")
    print(f"   Video: {video_path}")
    print(f"   Flight log: {flight_log_path}")
    print(f"   Tracking JSON: {tracking_json_path}")

    start_time = time.time()

    with open(video_path, 'rb') as video_file, \
         open(flight_log_path, 'rb') as flight_file, \
         open(tracking_json_path, 'rb') as tracking_file:

        files = {
            'video': ('video.mp4', video_file, 'video/mp4'),
            'flight_log': ('flight_log.csv', flight_file, 'text/csv'),
            'tracking_json': ('tracking.json', tracking_file, 'application/json')
        }

        data = {
            'fps': fps,
            'batch_size': batch_size,
            'kf_sigma_a': 0.5,
            'kf_sigma_meas_h': 5.0,
            'kf_sigma_meas_v': 2.0,
            'yaw_offset': 0.0,
            'magnetic_declination': 0.0,
            'add_drone_yaw': False,
            'use_osd_yaw': False
        }

        print(f"\n   Processing (this may take several minutes)...")
        try:
            response = requests.post(
                f"{base_url}/process",
                files=files,
                data=data,
                timeout=600  # 10 minute timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            print("   ERROR: Request timed out (>10 minutes)")
            return
        except requests.exceptions.RequestException as e:
            print(f"   ERROR: Request failed - {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            return

    elapsed_time = time.time() - start_time

    # 3. Save result
    print(f"\n3. Processing complete! ({elapsed_time:.1f}s)")

    result = response.json()

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"   Output saved to: {output_path}")

    # 4. Summary
    print(f"\n4. Result summary:")
    if 'metadata' in result:
        meta = result['metadata']
        print(f"   Source resolution: {meta['source_resolution']['width']}x{meta['source_resolution']['height']}")
        print(f"   Depth resolution: {meta['depth_resolution']['width']}x{meta['depth_resolution']['height']}")
        print(f"   FPS: {meta['fps']}")
        print(f"   Total frames: {meta['total_frames']}")
        if 'kalman_filter' in meta:
            kf = meta['kalman_filter']
            print(f"   Kalman Filter: enabled={kf['enabled']}, sigma_a={kf['sigma_a']}")

    if 'tracks' in result:
        tracks = result['tracks']
        print(f"   Total track entries: {len(tracks)}")
        if tracks:
            print(f"\n   Sample track entry:")
            sample = tracks[0]
            for key in ['frame_id', 'track_id', 'lat', 'lon', 'depth_m', 'speed_mps', 'heading_deg']:
                if key in sample:
                    print(f"     {key}: {sample[key]}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test VGGT-P Service API")
    parser.add_argument("--video", "-v", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--flight-log", "-f", type=str, required=True,
                        help="Path to DJI flight record CSV")
    parser.add_argument("--tracking", "-t", type=str, required=True,
                        help="Path to tracking JSON file")
    parser.add_argument("--output", "-o", type=str, default="output_api.json",
                        help="Output JSON path (default: output_api.json)")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="Service base URL (default: http://localhost:8000)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second (default: 5.0)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for VGGT inference (default: 5)")
    args = parser.parse_args()

    test_api(
        video_path=args.video,
        flight_log_path=args.flight_log,
        tracking_json_path=args.tracking,
        output_path=args.output,
        base_url=args.url,
        fps=args.fps,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
