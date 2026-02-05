# VGGT-P Main Files Version History

This document describes the differences between the five `main_*.py` files in the project, their evolution, and when to use each one.

## Quick Summary

| File | Purpose | Input | Output | Lines |
|------|---------|-------|--------|-------|
| `main.py` | Basic VGGT depth estimation | Video + Flight log | Depth maps, depth video | ~1240 |
| `main_v1_robust.py` | RANSAC-based robust depth | Video + Flight log | Metric depth maps, video | ~1184 |
| `main_v2_state_est.py` | Object state estimation | Depth maps + Tracks + Telemetry | Geolocalized tracks JSON | ~1980 |
| `main_v3_service.py` | FastAPI service (production) | Video + Flight log + Tracks (via API) | Geolocalized tracks JSON | ~488 |
| `main_v4_no_flight.py` | State estimation without flight log | Video + Tracks | State JSON (VGGT coords) + 2D map | ~680 |

---

## Detailed Comparison

### 1. `main.py` - Basic VGGT Pipeline

**Purpose:** Original depth estimation script for extracting metric depth from drone video.

**Key Features:**
- Extracts frames from video at specified FPS
- Runs VGGT model inference (batched)
- Simple center-point metric scaling: `scale = altitude / sin(pitch) / center_depth`
- Outputs depth maps (canonical + metric)
- Generates side-by-side depth visualization video
- Optional quadrant analysis video (RGB, canonical, plot, metric)

**CLI Usage:**
```bash
python main.py -i video.mp4 -f flight_record.csv -o outputs/ --fps 5 --batch-size 20
```

**Metric Scaling Method:**
- Uses single center point depth
- Assumes ground at image center
- Sensitive to obstacles at center (trees, buildings)

**Output Files:**
- `depth/depth_XXXX.npy` - Canonical depth maps
- `depth/depth_XXXX.png` - Depth visualizations
- `depth_metric/depth_metric_XXXX.npy` - Metric depth maps
- `depth_video.mp4` - Side-by-side visualization
- `quadrant_analysis.mp4` - 4-panel analysis video (optional)

---

### 2. `main_v1_robust.py` - RANSAC Robust Depth

**Purpose:** Improved depth estimation with robust metric scaling using RANSAC ground plane fitting.

**Key Improvements over main.py:**
- **RANSAC plane fitting** instead of single center point
- **Telemetry validation** - rejects walls/vertical surfaces
- **Global color consistency** - fixed depth range across all frames
- **Fallback mechanism** - uses geometric scaling if RANSAC fails

**Metric Scaling Method:**
1. Fit ground plane `Z = αX + βY + γ` using RANSAC
2. Validate plane tilt against expected tilt from telemetry
3. If valid: `scale = real_altitude / canonical_altitude`
4. If invalid (found wall): fallback to geometric center scaling

**CLI Usage:**
```bash
python main_v1_robust.py -i video.mp4 -f flight_record.csv -o outputs_v2/ --fps 5
```

**Additional Options:**
- `--tilt-tolerance` - RANSAC validation tolerance (default: 15°)
- `--downsample` - Point cloud downsampling factor (default: 10)

**Output Files:**
- `depth_canonical/` - Raw VGGT depth maps
- `depth_metric/` - RANSAC-scaled metric depth maps
- `scaling_info.json` - Per-frame scaling details (method used, inlier ratio, etc.)
- `depth_video.mp4` - Video with global color consistency

---

### 3. `main_v2_state_est.py` - State Estimation Pipeline

**Purpose:** Geolocalize detected objects and estimate their state (position, velocity, heading).

**Key Features:**
- Does NOT run VGGT inference (takes depth maps as input)
- Fuses multiple data sources:
  - Object detection/tracking JSON (bounding boxes)
  - Metric depth maps from VGGT
  - UAV telemetry (position, gimbal angles)
  - Camera intrinsics
- **Kalman Filter** for smooth state estimation (6-state: position + velocity in NED)
- Coordinate transformations: Pixel → Camera → NED → LLA
- Creates UAV entries (track_id=0) for every frame
- Calculates per-object density (neighbors within 10m)

**State Vector (per object per frame):**
- `lat`, `lon` - WGS84 coordinates
- `depth_m` - Distance from camera
- `pos_ned` - Position in NED frame [N, E, D]
- `vel_ned` - Velocity in NED frame [Vn, Ve, Vd]
- `speed_mps` - Horizontal speed
- `heading_deg` - Movement direction (0°=North, 90°=East)
- `density` - Number of nearby objects

**CLI Usage:**
```bash
python main_v2_state_est.py \
  -i tracking.json \
  -f flight_record.csv \
  -d outputs/depth_metric/ \
  -k outputs/cameras/intrinsics.npy \
  -o tracks_geolocalized.json \
  --source-res 1920 1080 \
  --fps 5
```

**Kalman Filter Tuning:**
- `--kf-sigma-a` - Process noise (acceleration std) [default: 0.5 m/s²]
- `--kf-sigma-meas-h` - Horizontal measurement noise [default: 5.0 m]
- `--kf-sigma-meas-v` - Vertical measurement noise [default: 2.0 m]

**Yaw Calibration:**
- `--yaw-offset` - Manual yaw calibration offset
- `--magnetic-declination` - Magnetic declination correction
- `--add-drone-yaw` - Add OSD.yaw to GIMBAL.yaw
- `--use-osd-yaw` - Use OSD.yaw instead of GIMBAL.yaw

---

### 4. `main_v3_service.py` - FastAPI Service (Production)

**Purpose:** Production-ready REST API combining VGGT inference + state estimation.

**Key Features:**
- **FastAPI** web service running on port 8000
- Model loaded once at startup (singleton)
- Combines `main.py` inference + `main_v2_state_est.py` processing
- Two API endpoints for different input methods
- Automatic temporary file cleanup

**API Endpoints:**

| Endpoint | Method | Input Format |
|----------|--------|--------------|
| `/vggt_p/health` | GET | - |
| `/vggt_p/process` | POST | JSON with URLs |
| `/vggt_p/process-upload` | POST | Multipart form data |

**URL-based Request (`/vggt_p/process`):**
```json
{
  "job_id": 123,
  "input_file_url": "https://..../video.mp4",
  "flight_log_url": "https://..../flight_log.csv",
  "tracking_json_url": "https://..../tracking.json",
  "file_id": "abc123",
  "parameters": {
    "fps": 5.0,
    "batch_size": 5,
    "kf_sigma_a": 0.5,
    "kf_sigma_meas_h": 5.0,
    "kf_sigma_meas_v": 2.0
  }
}
```

**File Upload Request (`/vggt_p/process-upload`):**
- `video` - MP4 file
- `flight_log` - CSV file
- `tracking_json` - JSON file
- Form fields for parameters

**Starting the Service:**
```bash
# Using helper script
./start_service.sh

# Or directly
uvicorn main_v3_service:app --host 0.0.0.0 --port 8000
```

---

### 5. `main_v4_no_flight.py` - State Estimation Without Flight Log

**Purpose:** Estimate object states when no flight log is available, using VGGT's intrinsic coordinate system.

**Key Features:**
- **No flight log required** - works with video + tracking JSON only
- Runs VGGT inference internally (like main_v3)
- Outputs in **VGGT coordinate system** (arbitrary scale, consistent across frames)
- Compatible JSON format with main_v3 (lat/lon = null)
- **Kalman Filter** for smooth state estimation
- Bird's-eye 2D visualization in VGGT XY plane
- Optional `--scale-hint` for approximate metric conversion

**Limitations (without flight log):**
- No metric depth (meters) - uses VGGT arbitrary units
- No geolocalization (lat/lon) - positions in VGGT coords
- No true velocity (m/s) - uses units/frame or units/second
- Heading IS valid (scale-independent)

**State Vector (per object per frame):**
- `lat`, `lon` - Always `null` (no GPS)
- `depth_m` - Depth in VGGT units (or meters if scale_hint)
- `pos_vggt` - Position [X, Y, Z] in VGGT coordinates
- `vel_vggt` - Velocity [vX, vY, vZ] in VGGT units/second
- `speed_units` - Horizontal speed in VGGT units/second
- `heading_deg` - Movement direction relative to VGGT frame (0°=+Y, 90°=+X)
  - **Note:** NOT absolute (no North reference). Useful for relative motion analysis.
- `coordinate_system` - "vggt" or "metric_approx"

**CLI Usage:**
```bash
python main_v4_no_flight.py \
  -i video.mp4 \
  -t tracking.json \
  -o tracks_vggt.json \
  --fps 5 \
  --source-res 1920 1080
```

**Optional Scale Hint:**
```bash
# If you know approximate altitude (e.g., 50m), estimate scale
python main_v4_no_flight.py \
  -i video.mp4 \
  -t tracking.json \
  -o tracks_approx.json \
  --scale-hint 0.5  # meters per VGGT unit (requires calibration)
```

**Kalman Filter Tuning:**
- `--kf-sigma-a` - Process noise (default: 0.1 VGGT units/frame²)
- `--kf-sigma-meas` - Measurement noise (default: 0.5 VGGT units)

**Output Files:**
- `{output}.json` - State estimates in VGGT coordinates
- `{output}_map.mp4` - Bird's-eye visualization video
- `depth/depth_XXXX.npy` - Depth maps (if `--save-depth`)

---

## Evolution Flow

```
main.py (basic depth)
    │
    ├── main_v1_robust.py (RANSAC improvement)
    │       │
    │       └── [Depth maps output]
    │                │
    │                ▼
    └──────► main_v2_state_est.py (state estimation)
                    │
                    ├──────────────────────────┐
                    ▼                          ▼
            main_v3_service.py          main_v4_no_flight.py
            (API with flight log)       (CLI without flight log)
```

---

## When to Use Each File

| Scenario | Recommended File |
|----------|------------------|
| Just need depth maps from video | `main.py` or `main_v1_robust.py` |
| Need robust depth with terrain validation | `main_v1_robust.py` |
| Already have depth maps, need object positions | `main_v2_state_est.py` |
| Production API deployment (with flight log) | `main_v3_service.py` |
| **No flight log available** | `main_v4_no_flight.py` |
| Debugging/development | Individual scripts |
| Backend integration (with GPS) | `main_v3_service.py` |
| Backend integration (no GPS) | `main_v4_no_flight.py` |

---

## Dependencies Between Files

- `main_v3_service.py` imports from:
  - `main.py`: `extract_frames_from_video`, `save_frames_and_load`, `parse_dji_flight_record`, `scale_depth_sequence_to_metric`
  - `main_v2_state_est.py`: `load_flight_record`, `process_tracks`

- `main_v4_no_flight.py` is standalone:
  - Contains own VGGT inference code (derived from `main.py`)
  - Contains own Kalman Filter (derived from `main_v2_state_est.py`)
  - Uses VGGT coordinate system instead of NED/LLA

- `main_v2_state_est.py` is standalone (no imports from other main files)

- `main_v1_robust.py` is standalone (independent implementation)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| main.py | Initial | Basic VGGT pipeline with center-point scaling |
| main_v1_robust.py | v1 | RANSAC plane fitting, telemetry validation, global color consistency |
| main_v2_state_est.py | v2 | State estimation with Kalman Filter, object geolocalization |
| main_v3_service.py | v3 | FastAPI service wrapper for production deployment |
| main_v4_no_flight.py | v4 | State estimation without flight log, VGGT coordinate system |
