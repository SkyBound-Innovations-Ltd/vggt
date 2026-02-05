"""
Test script for VGGT-P Service API (URL-based).

Usage:
    python test_service_url.py \
        --video-url <url> \
        --flight-log-url <url> \
        --tracking-url <url> \
        --output <json>
"""

import requests
import argparse
import json
import time


def test_api_url(
    video_url: str,
    flight_log_url: str,
    tracking_json_url: str,
    output_path: str = "output_api.json",
    base_url: str = "http://localhost:8000",
    job_id: int = 123456,
    file_id: str = "test-file-id"
):
    """Test the VGGT-P service API with URL inputs."""

    print("=" * 60)
    print("VGGT-P Service API Test (URL-based)")
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
    print(f"   Video URL: {video_url}")
    print(f"   Flight log URL: {flight_log_url}")
    print(f"   Tracking JSON URL: {tracking_json_url}")

    payload = {
        "job_id": job_id,
        "input_file_url": video_url,
        "flight_log_url": flight_log_url,
        "tracking_json_url": tracking_json_url,
        "file_id": file_id,
        "parameters": {
            "fps": 5.0,
            "batch_size": 5,
            "kf_sigma_a": 0.5,
            "kf_sigma_meas_h": 5.0,
            "kf_sigma_meas_v": 2.0,
            "yaw_offset": 0.0,
            "magnetic_declination": 0.0,
            "add_drone_yaw": False,
            "use_osd_yaw": False
        }
    }

    start_time = time.time()

    print(f"\n   Processing (this may take several minutes)...")
    try:
        response = requests.post(
            f"{base_url}/process",
            json=payload,
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
    parser = argparse.ArgumentParser(description="Test VGGT-P Service API (URL-based)")
    parser.add_argument("--video-url", type=str, required=True,
                        help="URL to input video file")
    parser.add_argument("--flight-log-url", type=str, required=True,
                        help="URL to DJI flight record CSV")
    parser.add_argument("--tracking-url", type=str, required=True,
                        help="URL to tracking JSON file")
    parser.add_argument("--output", "-o", type=str, default="output_api.json",
                        help="Output JSON path (default: output_api.json)")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="Service base URL (default: http://localhost:8000)")
    parser.add_argument("--job-id", type=int, default=123456,
                        help="Job ID (default: 123456)")
    parser.add_argument("--file-id", type=str, default="test-file-id",
                        help="File ID (default: test-file-id)")
    args = parser.parse_args()

    test_api_url(
        video_url=args.video_url,
        flight_log_url=args.flight_log_url,
        tracking_json_url=args.tracking_url,
        output_path=args.output,
        base_url=args.url,
        job_id=args.job_id,
        file_id=args.file_id
    )


if __name__ == "__main__":
    main()
