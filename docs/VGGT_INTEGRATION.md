# VGGT Integration Guide

**Status:** Ready for integration
**Service URL:** `http://<VGGT_EC2_IP>:8000`

---

## Quick Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VGGT Service                                │
│                                                                     │
│   INPUTS (3 files):                 OUTPUT:                         │
│   ─────────────────                 ───────                         │
│   1. Video URL (.mp4)               JSON with state estimation:     │
│   2. Flight Log URL (.csv)    →     - lat, lon (GPS coordinates)    │
│   3. MOT Tracking URL (.json)       - depth_m (meters)              │
│                                     - vel_ned (velocity NED)        │
│                                     - speed_mps (m/s)               │
│                                     - heading_deg (degrees)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## API Specification

### Endpoint: `POST /process-upload` (Recommended)

**Content-Type:** `multipart/form-data`

**Request:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | File | Yes | Preprocessed video (.mp4) |
| `flight_log` | File | Yes | DJI flight record (.csv) |
| `tracking_json` | File | Yes | MOT tracking result (.json) from det-track |
| `fps` | float | No | Frame rate (default: 5.0) |
| `batch_size` | int | No | Inference batch size (default: 5) |
| `kf_sigma_a` | float | No | Kalman filter process noise (default: 0.5) |
| `kf_sigma_meas_h` | float | No | KF horizontal measurement noise (default: 5.0) |
| `kf_sigma_meas_v` | float | No | KF vertical measurement noise (default: 2.0) |

**Response:** JSON with geolocalized tracks (see below)

---

### Alternative Endpoint: `POST /process` (URL-based)

**Content-Type:** `application/json`

**Request:**
```json
{
  "job_id": 123456,
  "input_file_url": "http://backend:8011/api/video/video.mp4",
  "flight_log_url": "http://backend:8011/api/flight-log/flight.csv",
  "tracking_json_url": "http://backend:8011/api/tracking/mot.json",
  "file_id": "uuid-string",
  "parameters": {
    "fps": 5.0,
    "batch_size": 5,
    "kf_sigma_a": 0.5,
    "kf_sigma_meas_h": 5.0,
    "kf_sigma_meas_v": 2.0
  }
}
```

---

## Response Format

```json
{
  "metadata": {
    "source_resolution": {"width": 1280, "height": 720},
    "depth_resolution": {"width": 596, "height": 336},
    "fps": 5.0,
    "total_frames": 225,
    "home_position": {
      "latitude": 51.476322,
      "longitude": -3.189584,
      "altitude": 13.9
    },
    "kalman_filter": {
      "enabled": true,
      "sigma_a": 0.5,
      "sigma_meas_h": 5.0,
      "sigma_meas_v": 2.0
    }
  },
  "tracks": [
    {
      "frame_id": 0,
      "track_id": 1,
      "bbox": [100, 200, 150, 250],
      "lat": 51.476123,
      "lon": -3.189234,
      "depth_m": 45.2,
      "pos_ned": [12.3, -8.5, -45.2],
      "vel_ned": [2.1, 0.5, -0.1],
      "speed_mps": 2.16,
      "heading_deg": 13.5
    },
    ...
  ]
}
```

### Track Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Frame number (0-indexed) |
| `track_id` | int | Object ID from MOT |
| `bbox` | [x1,y1,x2,y2] | Bounding box in pixels |
| `lat` | float | Latitude (WGS84) |
| `lon` | float | Longitude (WGS84) |
| `depth_m` | float | Depth from camera in meters |
| `pos_ned` | [N,E,D] | Position in NED frame (meters from home) |
| `vel_ned` | [vN,vE,vD] | Velocity in NED frame (m/s) |
| `speed_mps` | float | Horizontal speed (m/s) |
| `heading_deg` | float | Heading (0=North, 90=East, clockwise) |

---

## Backend Integration Code

### Option 1: File Upload (Recommended)

```python
# services/vggt.py
import httpx
from config import VGGT_URL  # e.g., "http://vggt-host:8000"

async def call_vggt(
    video_path: str,
    flight_log_path: str,
    tracking_json_path: str,
    fps: float = 5.0,
    kf_sigma_a: float = 0.5,
    timeout: float = 600.0
) -> dict:
    """
    Call VGGT service with file uploads.

    Returns JSON with geolocalized tracks.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(video_path, "rb") as vf, \
             open(flight_log_path, "rb") as ff, \
             open(tracking_json_path, "rb") as tf:

            files = {
                "video": ("video.mp4", vf, "video/mp4"),
                "flight_log": ("flight.csv", ff, "text/csv"),
                "tracking_json": ("tracking.json", tf, "application/json")
            }
            data = {"fps": fps, "kf_sigma_a": kf_sigma_a}

            response = await client.post(
                f"{VGGT_URL}/process-upload",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
```

### Usage in Pipeline

```python
async def run_full_pipeline(video_path, flight_log_path):
    # Step 1: Run det-track
    det_track_result = await call_det_track(video_path)
    mot_json_path = det_track_result["mot_json_path"]

    # Step 2: Run VGGT
    vggt_result = await call_vggt(
        video_path=video_path,
        flight_log_path=flight_log_path,
        tracking_json_path=mot_json_path
    )

    # Step 3: Return combined result
    return {
        "det_track": det_track_result,
        "vggt": vggt_result  # Contains tracks with lat/lon/velocity
    }
```

---

## Pipeline Flow

```
User Upload                      Backend                         VGGT Service
    │                               │                                 │
    ├─── video.mp4 ────────────────►│                                 │
    ├─── flight_log.csv ───────────►│                                 │
    │                               │                                 │
    │                         [Preprocess]                            │
    │                               │                                 │
    │                         [Det-Track]                             │
    │                               │                                 │
    │                               ├── video.mp4 ───────────────────►│
    │                               ├── flight_log.csv ──────────────►│
    │                               ├── mot_tracking.json ───────────►│
    │                               │                                 │
    │                               │                           [VGGT Depth]
    │                               │                           [State Est.]
    │                               │                                 │
    │                               │◄─── tracks JSON ────────────────┤
    │                               │     (lat, lon, vel, heading)    │
    │                               │                                 │
    │◄──── Display Results ─────────┤                                 │
    │      (map + tracks)           │                                 │
```

---

## Health Check

```bash
curl http://<VGGT_HOST>:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true
}
```

---

## Configuration

### Backend `.env`
```bash
VGGT_URL=http://<VGGT_EC2_IP>:8000
VGGT_TIMEOUT=600  # 10 minutes
```

---

## Testing

### cURL Test (File Upload)
```bash
curl -X POST http://localhost:8000/process-upload \
  -F "video=@video.mp4" \
  -F "flight_log=@flight_log.csv" \
  -F "tracking_json=@tracking.json" \
  -F "fps=5.0" \
  -o result.json
```

### Python Test Script
```bash
python test_service.py \
  --video inputs/video.mp4 \
  --flight-log inputs/flight_log.csv \
  --tracking inputs/tracking.json \
  --output result.json
```

---

## Error Handling

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Invalid request (missing files, bad params) |
| 500 | Processing error (check logs) |
| 503 | Model not loaded (service starting up) |

---

## Notes

- Processing time: ~2-3 minutes for 45-second video
- Fixed resolution: 720p (1280x720)
- Kalman Filter smooths position/velocity estimates
- Output tracks are in same frame order as input MOT JSON
