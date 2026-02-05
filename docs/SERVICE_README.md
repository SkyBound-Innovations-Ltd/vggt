# VGGT-P Service API

FastAPI service that runs the full VGGT depth estimation + state estimation pipeline.

## Overview

**Pipeline:**
1. Video + Flight Log + Tracking JSON → VGGT Depth Estimation → State Estimation with Kalman Filter → Geolocalized Tracks JSON

**Features:**
- Two endpoints: URL-based (`/process`) and file upload (`/process-upload`)
- Kalman Filter for smooth position/velocity estimation
- Fixed 720p resolution (1280x720)
- JSON output only (no videos)
- Model loaded once at startup (fast subsequent requests)

---

## Service Files

| File | Purpose |
|------|---------|
| `main_v3_service.py` | Main service code |
| `Dockerfile.vggt-service` | Docker image |
| `docker-compose.yml` | Docker compose |
| `start_service.sh` | Local startup script |
| `test_service.py` | Tests (file upload) |
| `test_service_url.py` | Tests (URL-based) |
| `requirements_service.txt` | Dependencies |

---

## Installation

```bash
# Install service dependencies
pip install -r requirements_service.txt

# Required for state estimation (scipy)
pip install scipy
```

---

## Starting the Service

### Local Development
```bash
./start_service.sh
# or
uvicorn main_v3_service:app --host 0.0.0.0 --port 8000 --reload
```

Service will run on: `http://localhost:8000`

### Production (with Uvicorn)
```bash
uvicorn main_v3_service:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note:** Use `--workers 1` because GPU model is loaded as a singleton.

### Docker (Recommended for Production)
```bash
# Build Docker image
docker build -f Dockerfile.vggt-service -t vggt-service:latest .

# Run container (Option 1: --gpus all)
docker run -d \
  --name vggt-service \
  --gpus all \
  -p 8011:8000 \
  -v $(pwd):/workspace \
  vggt-service:latest

# Run container (Option 2: --runtime=nvidia, if --gpus all fails)
docker run -d \
  --name vggt-service \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -p 8011:8000 \
  -v $(pwd):/workspace \
  vggt-service:latest
```

**Note:** The `-v $(pwd):/workspace` mount is needed because the Docker image requires local files (`main.py`, `main_v2_state_est.py`).

---

## API Endpoints

All endpoints are prefixed with `/vggt_p/`.

### Health Check
```bash
GET /vggt_p/health
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

### Process Video (URL-based)
```bash
POST /vggt_p/process
Content-Type: application/json
```

**Request Body (JSON):**
```json
{
  "job_id": 123456,
  "input_file_url": "http://example.com/video.mp4",
  "flight_log_url": "http://example.com/flight_log.csv",
  "tracking_json_url": "http://example.com/tracking.json",
  "file_id": "uuid-string",
  "parameters": {
    "fps": 5.0,
    "batch_size": 5,
    "kf_sigma_a": 0.5,
    "kf_sigma_meas_h": 5.0,
    "kf_sigma_meas_v": 2.0,
    "yaw_offset": 0.0,
    "magnetic_declination": 0.0,
    "add_drone_yaw": false,
    "use_osd_yaw": false
  }
}
```

**Required Fields:**
- `job_id` (int): Job ID from backend
- `input_file_url` (string): URL to MP4 video file
- `flight_log_url` (string): URL to DJI flight record CSV
- `tracking_json_url` (string): URL to object tracking JSON
- `file_id` (string): File identifier

---

### Process Video (File Upload)
```bash
POST /vggt_p/process-upload
Content-Type: multipart/form-data
```

**Form Fields:**

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `video` | File | Yes | - |
| `flight_log` | File | Yes | - |
| `tracking_json` | File | Yes | - |
| `fps` | float | No | 5.0 |
| `batch_size` | int | No | 5 |
| `kf_sigma_a` | float | No | 0.5 |
| `kf_sigma_meas_h` | float | No | 5.0 |
| `kf_sigma_meas_v` | float | No | 2.0 |
| `yaw_offset` | float | No | 0.0 |
| `magnetic_declination` | float | No | 0.0 |
| `add_drone_yaw` | bool | No | false |
| `use_osd_yaw` | bool | No | false |

---

### Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fps` | 5.0 | Frame extraction rate |
| `batch_size` | 5 | VGGT inference batch size (adjust for GPU memory) |
| `kf_sigma_a` | 0.5 | Kalman Filter process noise (acceleration std) |
| `kf_sigma_meas_h` | 5.0 | Measurement noise horizontal (N, E) |
| `kf_sigma_meas_v` | 2.0 | Measurement noise vertical (D) |
| `yaw_offset` | 0.0 | Yaw calibration offset in degrees |
| `magnetic_declination` | 0.0 | Magnetic declination in degrees |
| `add_drone_yaw` | false | Add drone yaw to gimbal yaw |
| `use_osd_yaw` | false | Use OSD.yaw instead of GIMBAL.yaw |

---

### Response Format

```json
{
  "metadata": {
    "source_resolution": {"width": 1280, "height": 720},
    "depth_resolution": {"height": 336, "width": 596},
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
    }
  ]
}
```

---

## Usage Examples

### Using cURL (URL-based)
```bash
curl -X POST http://localhost:8000/vggt_p/process \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": 123456,
    "input_file_url": "http://example.com/video.mp4",
    "flight_log_url": "http://example.com/flight_log.csv",
    "tracking_json_url": "http://example.com/tracking.json",
    "file_id": "test-file-id",
    "parameters": {
      "fps": 5.0,
      "batch_size": 5
    }
  }' \
  -o output.json
```

### Using cURL (File Upload)
```bash
curl -X POST http://localhost:8000/vggt_p/process-upload \
  -F "video=@video.mp4" \
  -F "flight_log=@flight_log.csv" \
  -F "tracking_json=@tracking.json" \
  -F "fps=5.0" \
  -F "batch_size=5" \
  -o output.json
```

### Using Python `requests`
```python
import requests
import json

payload = {
    "job_id": 123456,
    "input_file_url": "http://example.com/video.mp4",
    "flight_log_url": "http://example.com/flight_log.csv",
    "tracking_json_url": "http://example.com/tracking.json",
    "file_id": "test-file-id",
    "parameters": {
        "fps": 5.0,
        "batch_size": 5,
        "kf_sigma_a": 0.5,
        "kf_sigma_meas_h": 5.0,
        "kf_sigma_meas_v": 2.0
    }
}

response = requests.post(
    'http://localhost:8000/vggt_p/process',
    json=payload,
    timeout=600
)
result = response.json()

with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### Using Test Script
```bash
python test_service_url.py \
  --video-url http://example.com/video.mp4 \
  --flight-log-url http://example.com/flight_log.csv \
  --tracking-url http://example.com/tracking.json \
  --output outputs/result.json
```

---

## Performance Notes

### GPU Memory
- **Batch Size 5**: Recommended for Tesla T4 (15GB VRAM)
- **Batch Size 10-20**: For A100/H100 GPUs
- Adjust `batch_size` parameter if OOM occurs

### Processing Time
Approximate times for 45-second video at 5 FPS (225 frames):
- **Frame extraction**: 5-10s
- **VGGT inference**: 60-120s (depends on GPU)
- **State estimation**: 5-10s
- **Total**: ~2-3 minutes

### Concurrent Requests
- Model is loaded as a singleton (shared across requests)
- GPU processing is sequential (one request at a time)
- Use job queue system (Celery/RQ) for production if concurrent requests needed

---

## Output JSON Structure

### Metadata
- `source_resolution`: Original video resolution (fixed 720p)
- `depth_resolution`: VGGT output resolution (~590x330)
- `fps`: Frame extraction rate
- `total_frames`: Number of frames processed
- `home_position`: NED origin (takeoff location)
- `kalman_filter`: KF parameters used

### Tracks
Each track entry contains:
- `frame_id`: Frame number
- `track_id`: Object ID
- `bbox`: Bounding box [x1, y1, x2, y2]
- `lat`, `lon`: GPS coordinates (WGS84)
- `depth_m`: Depth in meters
- `pos_ned`: Position in NED frame [N, E, D] meters
- `vel_ned`: Velocity in NED frame [v_N, v_E, v_D] m/s
- `speed_mps`: Horizontal speed in m/s
- `heading_deg`: Heading in degrees (0=North, 90=East, clockwise)

---

## Troubleshooting

### Service won't start
```bash
# Check if port 8000 is available
lsof -i :8000

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Model loading fails
```bash
# Check VGGT installation
python -c "from vggt.models.vggt import VGGT; print('OK')"

# Try manual model download
python -c "from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"
```

### Docker GPU issues
```bash
# If --gpus all fails, use --runtime=nvidia instead
docker run -d \
  --name vggt-service \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -p 8011:8000 \
  -v $(pwd):/workspace \
  vggt-service:latest

# Verify GPU inside container
docker exec vggt-service nvidia-smi
```

### Request timeout
- Increase timeout in client (default: 600s = 10 minutes)
- Check GPU memory with `nvidia-smi`
- Reduce `batch_size` parameter

### Out of Memory (OOM)
- Reduce `batch_size` to 3 or 2
- Check for memory leaks: `nvidia-smi` during processing
- Restart service to clear cached models

### Missing scipy
```bash
# Install scipy in running container
docker exec vggt-service pip install scipy
```

---

## Deployment Checklist

- [ ] GPU driver installed (CUDA 12.1+)
- [ ] VGGT model pre-downloaded
- [ ] scipy installed
- [ ] Firewall rules configured for port 8000/8011
- [ ] Service auto-restart configured (systemd/supervisor)
- [ ] Logging configured for production
- [ ] Health check monitoring enabled
- [ ] Backup strategy for uploaded files (if persistence needed)

---

## Support

For issues related to:
- **VGGT model**: Check [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)
- **Kalman Filter tuning**: See `docs/KF_SLAM_Principles.md`
- **API errors**: Check service logs with `docker logs vggt-service`
