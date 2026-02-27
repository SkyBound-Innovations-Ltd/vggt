# VGGT-P Service Configuration Guide

This document provides container requirements and front-end configurations for deploying the VGGT-P depth estimation service.

## Service Overview

| Property | Value |
|----------|-------|
| Service Name | VGGT-P (VGGT Pipeline) |
| Base URL | `http://{host}:8011/vggt_p` |
| GPU Required | Tesla T4 (15GB VRAM) or better |
| Model | facebook/VGGT-1B |
| Purpose | Depth estimation for drone footage with flight log telemetry |

---

## Container Configuration

### Docker Run Command (Production)

```bash
docker run -d --name vggt-service \
  --runtime=nvidia \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e SERVICE_PORT=8011 \
  --shm-size=8g \
  --health-cmd="curl -f http://localhost:8011/vggt_p/health || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-start-period=180s \
  --health-retries=3 \
  -v $(pwd)/uploads:/workspace/uploads \
  -v $(pwd)/outputs:/workspace/outputs \
  vggt-t4:latest
```

### Docker Build Command

```bash
docker build -f Dockerfile.vggt-t4 -t vggt-t4:latest .
```

### Critical Configuration Options

| Option | Value | Purpose |
|--------|-------|---------|
| `--runtime=nvidia` | Required | GPU access for CUDA operations |
| `--network host` | Required | Avoids Docker NAT issues with Tailscale/VPN |
| `--shm-size=8g` | Required | Shared memory for PyTorch multiprocessing |
| `--health-start-period=180s` | Required | Model loading takes 30-60 seconds |
| `SERVICE_PORT=8011` | Default | Service port number |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_PORT` | 8011 | Port the service listens on |
| `UPLOAD_DIR` | /workspace/uploads | Directory for uploaded files |
| `OUTPUT_DIR` | /workspace/outputs | Directory for processing outputs |
| `NVIDIA_VISIBLE_DEVICES` | all | GPU visibility |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./uploads` | `/workspace/uploads` | Uploaded video and flight log files |
| `./outputs` | `/workspace/outputs` | Processing output files |
| `./service/static` | `/workspace/service/static` | Frontend static files (dev mode) |

---

## Key Dependencies

### requirements.txt (Service)

```
# FastAPI and web server
fastapi==0.115.0
uvicorn[standard]==0.38.0  # CRITICAL: Must be 0.38.0+ for large file uploads
python-multipart==0.0.9
pydantic==2.9.0
httpx==0.27.0

# Already in vggt requirements
torch==2.3.1
torchvision==0.18.1
numpy==1.26.1
Pillow
huggingface_hub
einops
safetensors

# Video processing
opencv-python
matplotlib
pandas
```

### Critical Version Notes

- **uvicorn >= 0.38.0**: Required for large file uploads over slow/relay connections
- **torch 2.3.1**: Must match CUDA version in base image
- **pydantic 2.9.0**: For FastAPI request/response models

---

## Frontend Configuration

### Design Parameters (index.html)

```javascript
// File size limits
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB hard limit

// Upload timeout
xhr.timeout = 30 * 60 * 1000; // 30 minutes for large files

// API base path
const API_BASE = '/vggt_p';
```

### Processing Parameters (Slider Ranges)

| Parameter | Min | Max | Default | Step | Description |
|-----------|-----|-----|---------|------|-------------|
| FPS | 0.5 | 10 | 1.0 | 0.5 | Frame extraction rate |
| Colormap | - | - | turbo | - | Depth visualization colormap |

### Available Colormaps

| Colormap | Description |
|----------|-------------|
| `turbo` | Rainbow (default, high contrast) |
| `viridis` | Blue-Green-Yellow (perceptually uniform) |
| `plasma` | Purple-Orange |
| `inferno` | Black-Red-Yellow |
| `magma` | Black-Purple-White |

### Supported Input Formats

| Type | Formats | Max Size |
|------|---------|----------|
| Video | MP4, AVI, MOV, MKV | 500MB |
| Flight Log | CSV (DJI format) | 10MB |

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/vggt_p/` | GET | Web UI |
| `/vggt_p/health` | GET | Health check |
| `/vggt_p/upload` | POST | Upload video + flight log |
| `/vggt_p/process` | POST | Start processing job |
| `/vggt_p/jobs/{job_id}` | GET | Job status |
| `/vggt_p/jobs/{job_id}/files` | GET | List output files |
| `/vggt_p/download/{job_id}/{path}` | GET | Download result file |
| `/vggt_p/queue` | GET | Queue status |
| `/docs` | GET | Swagger API documentation |

---

## Processing Parameters

### VGGT Model Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | facebook/VGGT-1B | Pre-trained VGGT model |
| Batch Size | 5 | Frames per inference batch (T4 optimized) |
| dtype | float16 | T4 compute capability 7.5 |
| Device | cuda | GPU acceleration |

### Metric Depth Scaling

When a DJI flight log is provided:

| Flight Log Column | Purpose |
|-------------------|---------|
| `CAMERA.isVideo` | Filter for video recording segments |
| `OSD.height [m]` | Altitude above takeoff (preferred) |
| `OSD.vpsHeight [m]` | VPS height (fallback, limited range) |
| `GIMBAL.pitch` | Gimbal pitch angle for slant range calculation |

---

## Directory Structure

```
/workspace/
├── vggt/                   # VGGT model package
├── service/
│   ├── main.py             # FastAPI application
│   ├── vggt_service.py     # Processing wrapper
│   ├── job_manager.py      # Job queue
│   ├── requirements.txt
│   └── static/
│       └── index.html      # Web UI
├── uploads/                # Uploaded files (volume mount)
│   └── {upload_id}/
│       ├── video.mp4
│       └── flight_log.csv
└── outputs/                # Processing outputs (volume mount)
    └── {job_id}/
        ├── frames/         # Extracted frames
        ├── depth/
        │   ├── depth_0001.npy
        │   ├── depth_0001.png
        │   └── ...
        └── depth_video.mp4
```

---

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `depth_video.mp4` | H.264/MP4 | Side-by-side RGB + depth visualization |
| `depth/depth_XXXX.npy` | NumPy | Raw depth arrays (metric if flight log provided) |
| `depth/depth_XXXX.png` | PNG | Grayscale depth visualization |

---

## Troubleshooting

### "Broken pipe" error during video encoding

**Cause**: FFmpeg failed due to odd video dimensions (H.264 requires even dimensions)

**Solution**: Fixed in vggt_service.py - dimensions are now forced to be even, with automatic fallback to OpenCV

### Container permission denied on stop/remove

**Cause**: NVIDIA runtime lock

**Solution**:
```bash
# Kill the process inside container first
docker exec vggt-service pkill -f uvicorn
# Then remove
docker rm vggt-service
# Or restart Docker daemon
sudo systemctl restart docker
```

### Health check shows "unhealthy" during startup

**Cause**: Model loading takes 30-60 seconds

**Solution**: Wait for `--health-start-period=180s` to pass, or check logs:
```bash
docker logs -f vggt-service
```

### GPU not available in container

**Solution**: Use `--runtime=nvidia` instead of `--gpus all`:
```bash
docker run --runtime=nvidia ...
```

### "Network error - connection reset" on upload

**Cause**: Docker NAT or outdated uvicorn version

**Solutions**:
1. Use `--network host` instead of port mapping
2. Ensure uvicorn >= 0.38.0
3. Use Tailscale for reliable connections

---

## Service URLs

| Environment | URL |
|-------------|-----|
| Local | http://localhost:8011/vggt_p/ |
| AWS Public IP | http://13.40.25.32:8011/vggt_p/ |
| Tailscale | http://100.102.61.99:8011/vggt_p/ |
| Health Check | http://{host}:8011/vggt_p/health |
| API Docs | http://{host}:8011/docs |

---

## Quick Start

```bash
# 1. Build the image
docker build -f Dockerfile.vggt-t4 -t vggt-t4:latest .

# 2. Run the service
./docker-run.sh

# 3. Access via browser (Tailscale recommended)
open http://100.102.61.99:8011/vggt_p/

# 4. Or use curl
curl http://localhost:8011/vggt_p/health
```
