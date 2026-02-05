# Migration Guide: File Upload → URL-based API

This document explains the changes made to [main_v3_service.py](../main_v3_service.py) to support URL-based inputs for backend integration.

---

## What Changed

### Before (File Upload API)
```bash
POST /process
Content-Type: multipart/form-data

Files:
- video (file upload)
- flight_log (file upload)
- tracking_json (file upload)

Form fields:
- fps=5.0
- batch_size=5
- kf_sigma_a=0.5
- ...
```

### After (URL-based API)
```bash
POST /process
Content-Type: application/json

JSON payload:
{
  "job_id": 123456,
  "input_file_url": "http://...",
  "flight_log_url": "http://...",
  "tracking_json_url": "http://...",
  "file_id": "uuid",
  "parameters": { ... }
}
```

---

## Why This Change

1. **Backend Integration**: Your det-track backend expects to send URLs, not files
2. **Microservice Architecture**: Services communicate via URLs, not file uploads
3. **Scalability**: No need to handle large file uploads through the API gateway
4. **Flexibility**: Files can be served from any accessible URL (S3, CDN, local server)

---

## Modified Files

### 1. [main_v3_service.py](../main_v3_service.py)
**Changes:**
- Added `httpx` import for downloading files
- Added Pydantic models: `VGGTRequest`, `VGGTParameters`
- Added `download_file()` async function
- Changed `/process` endpoint from multipart form to JSON payload
- Parameters now nested under `request.parameters.*`

**New Request Models:**
```python
class VGGTParameters(BaseModel):
    fps: float = 5.0
    batch_size: int = 5
    kf_sigma_a: float = 0.5
    kf_sigma_meas_h: float = 5.0
    kf_sigma_meas_v: float = 2.0
    yaw_offset: float = 0.0
    magnetic_declination: float = 0.0
    add_drone_yaw: bool = False
    use_osd_yaw: bool = False

class VGGTRequest(BaseModel):
    job_id: int
    input_file_url: str
    flight_log_url: str
    tracking_json_url: str
    file_id: str
    parameters: VGGTParameters = VGGTParameters()
```

### 2. [requirements_service.txt](../requirements_service.txt)
**Added:**
- `httpx>=0.27.0`

### 3. New Files Created
- [test_service_url.py](../test_service_url.py): Test script for URL-based API
- [docs/BACKEND_INTEGRATION_EXAMPLE.md](BACKEND_INTEGRATION_EXAMPLE.md): Backend integration guide

### 4. Updated Files
- [SERVICE_README.md](../SERVICE_README.md): Updated API documentation

---

## How to Test

### 1. Install Dependencies
```bash
pip install httpx
```

### 2. Start Service
```bash
python main_v3_service.py
```

### 3. Test Health Check
```bash
curl http://localhost:8000/health
```

### 4. Test with URLs

**Option A: Use a simple HTTP server to serve local files**
```bash
# Terminal 1: Serve files
cd /path/to/your/files
python -m http.server 9000

# Terminal 2: Test service
python test_service_url.py \
  --video-url http://localhost:9000/video.mp4 \
  --flight-log-url http://localhost:9000/flight_log.csv \
  --tracking-url http://localhost:9000/tracking.json \
  --output test_output.json
```

**Option B: Use actual URLs from your backend**
```bash
python test_service_url.py \
  --video-url http://18.134.114.2:8011/api/video/preprocessed.mp4 \
  --flight-log-url http://18.134.114.2:8011/api/flight-log.csv \
  --tracking-url http://18.134.114.2:8011/api/tracking.json \
  --output test_output.json
```

---

## Backend Integration Checklist

- [ ] Update `services/vggt.py` to use new JSON payload format
- [ ] Set `VGGT_URL` environment variable in backend `.env`
- [ ] Ensure MOT JSON is accessible via URL (served by backend)
- [ ] Ensure flight log is accessible via URL (served by backend)
- [ ] Test full pipeline: upload → preprocess → det-track → VGGT → display
- [ ] Update frontend to extract coordinates from `result.vggt.tracks`
- [ ] Update frontend map display to show tracks with velocity/heading
- [ ] Configure firewall rules between backend and VGGT service
- [ ] Set appropriate timeout (600s = 10 minutes)

---

## Response Format (Unchanged)

The response format remains the same:

```json
{
  "metadata": { ... },
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

## Troubleshooting

### Service won't start
```bash
# Check if old service is still running
lsof -i :8000
kill <PID>

# Restart
python main_v3_service.py
```

### "httpx not found" error
```bash
pip install httpx
```

### URL download fails
- Check URL is accessible: `curl -I <url>`
- Ensure firewall allows outbound HTTP requests
- Verify URL returns expected content type
- Check timeout settings (default: 300s for download)

### "Model not loaded" error
- Wait for startup to complete (~30-60s for model loading)
- Check GPU availability: `nvidia-smi`
- Check logs for model loading errors

---

## Performance Notes

- **URL Download Time**: Depends on file size and network speed
  - 45s video (720p): ~50-200MB → 5-30s download
  - Flight log CSV: ~1MB → <1s
  - Tracking JSON: ~1MB → <1s

- **Total Processing Time**: Same as before
  - Download: 5-30s
  - Frame extraction: 5-10s
  - VGGT inference: 60-120s (GPU-dependent)
  - State estimation: 5-10s
  - **Total: ~2-3 minutes**

---

## Reverting to File Upload API (if needed)

If you need to keep the old file upload API alongside the URL-based API:

1. Rename current endpoint to `/process-url`
2. Keep old endpoint as `/process`
3. Both can coexist in the same service

However, for backend integration, the URL-based API is recommended.

---

## Next Steps

1. Test the service with local URLs (using Python HTTP server)
2. Update your backend's `services/vggt.py` using the example in [BACKEND_INTEGRATION_EXAMPLE.md](BACKEND_INTEGRATION_EXAMPLE.md)
3. Test full pipeline integration
4. Deploy to production EC2 instance
5. Update frontend to display geolocalized tracks on map
