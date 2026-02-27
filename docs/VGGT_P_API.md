# VGGT-P API Documentation

Base URL: `http://<host>:8003/vggt_p`

## Table of Contents

- [Data Flow](#data-flow)
- [Endpoints Overview](#endpoints-overview)
- [GET /](#get-)
- [POST /process](#post-process)
  - [Request](#request)
  - [Parameters](#parameters)
  - [Response](#response)
  - [Errors](#errors)
  - [Examples: CURL, Python and Node.js](#examples)
- [POST /process-upload](#post-process-upload)
- [Supporting Endpoints](#supporting-endpoints)

---

## Data Flow

VGGT-P is the second stage in a two-service pipeline orchestrated by the backend.

```
Orchestrator / Backend                 Det-Track (:8002)          VGGT-P (:8003)
        │                                    │                          │
        │  1. User uploads video             │                          │
        │     + flight log                   │                          │
        │                                    │                          │
        ├── POST /process ──────────────────►│                          │
        │   (video URL)                      │                          │
        │                                    │                          │
        │◄── tracking JSON (MOT) ────────────┤                          │
        │    + output_json_url               │                          │
        │                                    │                          │
        ├── POST /vggt_p/process ───────────────────────────────────────►│
        │   (video URL + flight log URL + tracking JSON URL)            │
        │                                                               │
        │◄── geolocalized tracks JSON ──────────────────────────────────┤
        │    (lat, lon, depth, velocity, heading)                       │
        │                                                               │
   [Store result + serve to frontend]                                   │
```

### Input Sources

| Request Field | Source | Description |
|---------------|--------|-------------|
| `input_file_url` | Backend / S3 | Preprocessed video uploaded by the user (720p MP4) |
| `flight_log_url` | Backend / S3 | DJI flight record CSV uploaded by the user |
| `tracking_json_url` | **Det-Track service** | MOT tracking JSON — the output of `POST /process` on the det-track service (port 8002). Contains per-frame bounding boxes with `track_id`, `class_name`, and `confidence` |

### Output Destination

The response JSON is returned **directly in the HTTP response body** to the orchestrator. There is no secondary upload — the orchestrator is responsible for:

1. Storing the result (database, S3, etc.)
2. Serving it to the frontend for map display (tracks with lat/lon coordinates)

---

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vggt_p/` | GET | Service information and available endpoints |
| `/vggt_p/health` | GET | Health check (GPU status, model loaded, device info) |
| `/vggt_p/process` | POST | Process video via URLs (video, flight log, tracking JSON) |
| `/vggt_p/process-upload` | POST | Process video via file upload |

---

## GET /

Returns service information, status, and available endpoints.

### Response

**Status:** `200 OK`

```json
{
  "service": "VGGT-P Service",
  "version": "1.0",
  "description": "Full pipeline from video to geolocalized object tracks with state estimation",
  "status": "online",
  "model_loaded": true,
  "endpoints": {
    "GET /vggt_p/": "Service information (this page)",
    "GET /vggt_p/health": "Health check endpoint",
    "POST /vggt_p/process": "Process video via URLs (video, flight log, tracking JSON)",
    "POST /vggt_p/process-upload": "Process video via file upload"
  },
  "documentation": "See /docs/VGGT_P_API.md for detailed API documentation"
}
```

### Example

```bash
curl http://<host>:8003/vggt_p/
```

---

## POST /process

Submits a video, flight log, and tracking JSON for depth estimation and state estimation. **Synchronous** — blocks until processing is complete. Returns geolocalized object tracks with Kalman-filtered position, velocity, and heading.

**Pipeline:** Video + Flight Log + Tracking JSON → VGGT Depth Estimation → State Estimation (Kalman Filter) → Geolocalized Tracks

### Request

**Content-Type:** `application/json`

```json
{
  "job_id": 123456,
  "input_file_url": "https://presigned-s3-url.com/video.mp4",
  "flight_log_url": "https://presigned-s3-url.com/flight_log.csv",
  "tracking_json_url": "https://presigned-s3-url.com/tracking.json",
  "file_id": "abc123",
  "parameters": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | int | Yes | Unique job identifier, to be assigned by orchestrator |
| `input_file_url` | string | Yes | URL to MP4 video file |
| `flight_log_url` | string | Yes | URL to DJI flight record CSV |
| `tracking_json_url` | string | Yes | URL to object tracking JSON (from det-track service) |
| `file_id` | string | Yes | File identifier |
| `parameters` | object | No | Processing parameters (see below) |

### Parameters

#### General

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fps` | float | 5.0 | Frame extraction rate from the video |
| `batch_size` | int | 5 | VGGT inference batch size. Adjust based on GPU memory |

#### Kalman Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kf_sigma_a` | float | 0.5 | Process noise standard deviation (acceleration) |
| `kf_sigma_meas_h` | float | 5.0 | Measurement noise standard deviation — horizontal (North, East) |
| `kf_sigma_meas_v` | float | 2.0 | Measurement noise standard deviation — vertical (Down) |

#### Yaw Calibration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yaw_offset` | float | 0.0 | Yaw calibration offset in degrees |
| `magnetic_declination` | float | 0.0 | Magnetic declination in degrees |
| `add_drone_yaw` | bool | false | Add drone yaw to gimbal yaw |
| `use_osd_yaw` | bool | false | Use `OSD.yaw` instead of `GIMBAL.yaw` |
| `timezone` | string | "UTC" | Timezone for flight log local timestamps (e.g., `"UTC"`, `"Europe/London"`) |

#### Crowd Clustering (HDBSCAN)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hdbscan_min_cluster_size` | int | 5 | Minimum number of person detections to form a crowd cluster (≥2) |
| `hdbscan_min_samples` | int | 3 | Controls how conservative clustering is — higher = more conservative (≥1) |
| `hdbscan_coherence_weight` | float | 2.0 | Weight multiplier on velocity features (higher = more emphasis on movement coherence) |
| `hdbscan_max_speed_mps` | float | 2.0 | Max expected walking speed (m/s) for velocity normalisation |

### Response

**Status:** `200 OK`

```json
{
  "metadata": {
    "source_resolution": { "width": 1280, "height": 720 },
    "depth_resolution": { "height": 336, "width": 596 },
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
    },
    "crowd_clustering": {
      "method": "HDBSCAN",
      "min_cluster_size": 5,
      "min_samples": 3,
      "coherence_weight": 2.0,
      "max_speed_mps": 2.0
    }
  },
  "tracks": [
    {
      "timestamp": "2026-01-20T11:36:09.730000+00:00",
      "frame_id": 0,
      "track_id": 1,
      "bbox": [100, 200, 150, 250],
      "class_name": "person",
      "lat": 51.476123,
      "lon": -3.189234,
      "alt": 58.1,
      "depth_m": 45.2,
      "pos_ned": [12.3, -8.5, -45.2],
      "vel_ned": [2.1, 0.5, -0.1],
      "speed_mph": 4.83,
      "heading_deg": 13.5,
      "density": 3,
      "crowd_id": 1,
      "crowd_density": 0.0312
    }
  ]
}
```

#### Metadata Object

| Field | Type | Description |
|-------|------|-------------|
| `source_resolution` | object | Input video resolution (fixed 1280x720) |
| `depth_resolution` | object | VGGT depth map output resolution (~596x336) |
| `fps` | float | Frame extraction rate used |
| `total_frames` | int | Number of frames processed |
| `home_position` | object | NED origin — takeoff location (lat, lon, altitude in meters) |
| `kalman_filter` | object | Kalman Filter parameters used |
| `crowd_clustering` | object | HDBSCAN crowd clustering parameters used |

#### Track Object

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 timestamp with timezone (TIMESTAMPTZ), from flight log `CUSTOM.date` + `CUSTOM.updateTime` |
| `frame_id` | int | Frame number |
| `track_id` | int | Persistent object ID (from det-track tracking) |
| `bbox` | array | Bounding box `[x1, y1, x2, y2]` in source resolution pixels |
| `lat` | float | Latitude (WGS84) |
| `lon` | float | Longitude (WGS84) |
| `alt` | float | Altitude (WGS84) in meters |
| `depth_m` | float | Estimated depth in meters |
| `pos_ned` | array | Position in NED frame `[North, East, Down]` in meters relative to `home_position` |
| `vel_ned` | array | Kalman-filtered velocity `[v_N, v_E, v_D]` in m/s |
| `speed_mph` | float | Horizontal speed in mph (√(v_N² + v_E²) × 2.23694) |
| `heading_deg` | float | Heading in degrees (0°=North, 90°=East, -90°=West, ±180°=South, range [-180, 180]) |
| `density` | int | Count of other objects within 10m radius at the same frame |
| `crowd_id` | int \| null | HDBSCAN crowd cluster ID (1-based). `null` for noise, non-person objects, or UAV |
| `crowd_density` | float \| null | People per m² within the cluster's convex hull area. `null` when `crowd_id` is null |

### Errors

| Status | Cause |
|--------|-------|
| 500 | Processing failed (corrupt video, depth estimation failure, invalid flight log, etc.) |
| 503 | Model not loaded yet (service still starting up) |

### Examples

**curl**

```bash
curl -X POST http://<host>:8003/vggt_p/process \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": 123456,
    "input_file_url": "https://example.com/video.mp4",
    "flight_log_url": "https://example.com/flight_log.csv",
    "tracking_json_url": "https://example.com/tracking.json",
    "file_id": "file_001",
    "parameters": {
      "fps": 5.0,
      "batch_size": 5,
      "kf_sigma_a": 0.5
    }
  }' \
  --max-time 600 \
  -o output.json
```

**Python**

```python
import requests

response = requests.post("http://<host>:8003/vggt_p/process", json={
    "job_id": 123456,
    "input_file_url": "https://example.com/video.mp4",
    "flight_log_url": "https://example.com/flight_log.csv",
    "tracking_json_url": "https://example.com/tracking.json",
    "file_id": "file_001",
    "parameters": {
        "fps": 5.0,
        "batch_size": 5,
        "kf_sigma_a": 0.5,
        "kf_sigma_meas_h": 5.0,
        "kf_sigma_meas_v": 2.0,
    },
}, timeout=600)

result = response.json()
print(f"{result['metadata']['total_frames']} frames processed")
print(f"{len(result['tracks'])} track entries")
```

**Node.js**

```js
const response = await fetch("http://<host>:8003/vggt_p/process", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    job_id: 123456,
    input_file_url: "https://example.com/video.mp4",
    flight_log_url: "https://example.com/flight_log.csv",
    tracking_json_url: "https://example.com/tracking.json",
    file_id: "file_001",
    parameters: {
      fps: 5.0,
      batch_size: 5,
      kf_sigma_a: 0.5,
    },
  }),
});

const result = await response.json();
console.log(result.metadata.total_frames, "frames processed");
console.log(result.tracks.length, "track entries");
```

---

## POST /process-upload

File upload variant of `/process`. Use this endpoint when sending files directly instead of URLs.

**Content-Type:** `multipart/form-data`

### Form Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video` | File | Yes | — | MP4 video file |
| `flight_log` | File | Yes | — | DJI flight record CSV |
| `tracking_json` | File | Yes | — | Object tracking JSON |
| `fps` | float | No | 5.0 | Frame extraction rate |
| `batch_size` | int | No | 5 | VGGT inference batch size |
| `kf_sigma_a` | float | No | 0.5 | KF process noise (acceleration std) |
| `kf_sigma_meas_h` | float | No | 5.0 | KF measurement noise — horizontal |
| `kf_sigma_meas_v` | float | No | 2.0 | KF measurement noise — vertical |
| `yaw_offset` | float | No | 0.0 | Yaw calibration offset in degrees |
| `magnetic_declination` | float | No | 0.0 | Magnetic declination in degrees |
| `add_drone_yaw` | bool | No | false | Add drone yaw to gimbal yaw |
| `use_osd_yaw` | bool | No | false | Use OSD.yaw instead of GIMBAL.yaw |
| `timezone` | string | No | "UTC" | Timezone for flight log local timestamps |
| `hdbscan_min_cluster_size` | int | No | 5 | Min person detections to form a crowd |
| `hdbscan_min_samples` | int | No | 3 | HDBSCAN conservativeness |
| `hdbscan_coherence_weight` | float | No | 2.0 | Velocity feature weight |
| `hdbscan_max_speed_mps` | float | No | 2.0 | Max walking speed for normalisation |

### Response

Same as [POST /process response](#response).

### Example (curl)

```bash
curl -X POST http://<host>:8003/vggt_p/process-upload \
  -F "video=@video.mp4" \
  -F "flight_log=@flight_log.csv" \
  -F "tracking_json=@tracking.json" \
  -F "fps=5.0" \
  -F "batch_size=5" \
  --max-time 600 \
  -o output.json
```

### Example (Python)

```python
import requests

files = {
    "video": open("video.mp4", "rb"),
    "flight_log": open("flight_log.csv", "rb"),
    "tracking_json": open("tracking.json", "rb"),
}
data = {
    "fps": 5.0,
    "batch_size": 5,
    "kf_sigma_a": 0.5,
}

response = requests.post(
    "http://<host>:8003/vggt_p/process-upload",
    files=files,
    data=data,
    timeout=600,
)

result = response.json()
print(f"{len(result['tracks'])} track entries")
```

---

## Supporting Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vggt_p/health` | GET | Health check (GPU status, model loaded, device info) |

### Health Check Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true
}
```
