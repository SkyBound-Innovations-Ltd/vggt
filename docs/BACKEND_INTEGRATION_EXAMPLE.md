# Backend Integration Example

This document shows how to integrate the VGGT-P service with your det-track backend.

---

## Integration Flow

```
User Upload → Preprocess → Det-Track → VGGT-P Service → Display Results
```

1. User uploads video + flight log
2. Backend preprocesses video
3. Det-track service generates MOT JSON
4. Backend calls VGGT-P service with URLs
5. VGGT-P returns geolocalized tracks JSON
6. Backend displays results (tracks on map)

---

## VGGT-P Service Call (from backend)

### Python Example (services/vggt.py)

```python
import httpx
from typing import Dict, Any

VGGT_URL = "http://vggt-service-host:8000"  # Set in .env

async def call_vggt_service(
    job_id: int,
    video_url: str,
    flight_log_url: str,
    tracking_json_url: str,
    file_id: str,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Call VGGT-P service for depth estimation and state estimation.

    Args:
        job_id: Backend job ID
        video_url: URL to preprocessed video
        flight_log_url: URL to flight log CSV
        tracking_json_url: URL to MOT tracking JSON
        file_id: File identifier
        parameters: VGGT parameters (optional)

    Returns:
        Dict containing:
        - metadata: Processing metadata (resolution, fps, etc.)
        - tracks: List of geolocalized tracks with state estimation
    """
    if parameters is None:
        parameters = {
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

    payload = {
        "job_id": job_id,
        "input_file_url": video_url,
        "flight_log_url": flight_log_url,
        "tracking_json_url": tracking_json_url,
        "file_id": file_id,
        "parameters": parameters
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{VGGT_URL}/process",
            json=payload
        )
        response.raise_for_status()
        return response.json()
```

### Usage in Background Task

```python
async def run_full_pipeline(job_id: int, video_url: str, flight_log_url: str, file_id: str):
    """Run det-track + VGGT-P pipeline."""

    # Step 1: Run det-track
    det_track_result = await call_det_track_service(
        job_id=job_id,
        video_url=video_url,
        file_id=file_id
    )

    # Get MOT JSON URL from det-track result
    tracking_json_url = det_track_result.get("mot_json_url")

    # Step 2: Run VGGT-P
    vggt_result = await call_vggt_service(
        job_id=job_id,
        video_url=video_url,
        flight_log_url=flight_log_url,
        tracking_json_url=tracking_json_url,
        file_id=file_id
    )

    # Step 3: Store results
    jobs[job_id].result = {
        "det_track": det_track_result,
        "vggt": vggt_result
    }

    jobs[job_id].status = "completed"
```

---

## Response Format

### VGGT-P Response Structure

```json
{
  "metadata": {
    "source_resolution": {
      "width": 1280,
      "height": 720
    },
    "depth_resolution": {
      "height": 336,
      "width": 596
    },
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

### Field Descriptions

**Metadata:**
- `source_resolution`: Original video resolution (fixed 720p)
- `depth_resolution`: VGGT depth map resolution
- `fps`: Frame extraction rate
- `total_frames`: Number of processed frames
- `home_position`: NED origin (drone takeoff location)
- `kalman_filter`: KF parameters used for state estimation

**Tracks (per entry):**
- `frame_id`: Frame number (0-indexed)
- `track_id`: Object ID
- `bbox`: Bounding box [x1, y1, x2, y2]
- `lat`, `lon`: GPS coordinates (WGS84)
- `depth_m`: Depth in meters
- `pos_ned`: Position in NED frame [North, East, Down] meters
- `vel_ned`: Velocity in NED frame [v_N, v_E, v_D] m/s
- `speed_mps`: Horizontal speed in m/s
- `heading_deg`: Heading in degrees (0=North, 90=East, clockwise)

---

## Frontend Display

### Extract Coordinates for Map

```javascript
// In pollJobStatus() when job completes
if (data.result.vggt) {
    const vggtResult = data.result.vggt;
    const tracks = vggtResult.tracks;

    // Extract unique track IDs
    const trackIds = [...new Set(tracks.map(t => t.track_id))];

    // Group by track_id
    const tracksByID = {};
    trackIds.forEach(id => {
        tracksByID[id] = tracks.filter(t => t.track_id === id);
    });

    // Display on map
    displayTracksOnMap(tracksByID, vggtResult.metadata.home_position);
}

function displayTracksOnMap(tracksByID, homePosition) {
    // Clear existing layers
    map.eachLayer(layer => {
        if (layer !== baseLayer) {
            map.removeLayer(layer);
        }
    });

    // Add home marker
    L.marker([homePosition.latitude, homePosition.longitude], {
        icon: L.icon({
            iconUrl: 'home-icon.png',
            iconSize: [32, 32]
        })
    }).addTo(map).bindPopup("Drone Home");

    // Plot each track
    Object.entries(tracksByID).forEach(([trackId, points]) => {
        const coords = points.map(p => [p.lat, p.lon]);

        // Draw polyline
        L.polyline(coords, {
            color: getTrackColor(trackId),
            weight: 3
        }).addTo(map);

        // Add markers with speed/heading info
        points.forEach((point, idx) => {
            if (idx % 5 === 0) {  // Show every 5th point
                L.circleMarker([point.lat, point.lon], {
                    radius: 5,
                    color: getTrackColor(trackId)
                }).addTo(map).bindPopup(`
                    Track ${trackId}<br>
                    Frame: ${point.frame_id}<br>
                    Speed: ${point.speed_mps.toFixed(2)} m/s<br>
                    Heading: ${point.heading_deg.toFixed(1)}°<br>
                    Depth: ${point.depth_m.toFixed(1)} m
                `);
            }
        });
    });

    // Fit map to track bounds
    const allCoords = Object.values(tracksByID).flat().map(p => [p.lat, p.lon]);
    map.fitBounds(L.latLngBounds(allCoords));
}
```

---

## Environment Configuration

### Backend .env
```bash
# VGGT-P Service
VGGT_URL=http://your-vggt-host:8000

# Service timeout (10 minutes for VGGT processing)
VGGT_TIMEOUT=600
```

---

## Error Handling

```python
try:
    vggt_result = await call_vggt_service(...)
except httpx.TimeoutException:
    logger.error(f"VGGT service timeout for job {job_id}")
    jobs[job_id].status = "failed"
    jobs[job_id].error = "VGGT processing timeout (>10 minutes)"
except httpx.HTTPStatusError as e:
    logger.error(f"VGGT service error: {e.response.status_code}")
    jobs[job_id].status = "failed"
    jobs[job_id].error = f"VGGT service error: {e.response.text}"
except Exception as e:
    logger.error(f"Unexpected error calling VGGT: {str(e)}")
    jobs[job_id].status = "failed"
    jobs[job_id].error = f"VGGT service error: {str(e)}"
```

---

## Testing Integration

### 1. Start VGGT Service
```bash
cd /path/to/vggt-service
python main_v3_service.py
# or with docker:
docker-compose up vggt-service
```

### 2. Test Health Check
```bash
curl http://vggt-host:8000/health
```

### 3. Test Full Pipeline
```bash
python test_service_url.py \
  --video-url http://backend:8011/api/video/preprocessed.mp4 \
  --flight-log-url http://backend:8011/api/flight-log.csv \
  --tracking-url http://backend:8011/api/tracking.json \
  --url http://vggt-host:8000
```

---

## Deployment Notes

- VGGT service should be on same EC2 instance or network as backend for fast file access
- Ensure firewall rules allow traffic between services
- Use persistent volume for model cache (avoid re-downloading on restart)
- Consider nginx reverse proxy for production (not just uvicorn)
- Monitor GPU memory with `nvidia-smi` during processing
