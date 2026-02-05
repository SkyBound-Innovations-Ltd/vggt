# VGGT Pipeline Analysis: Operation Without Flight Log

This document analyzes the feasibility of estimating object state variables when no flight log is available.

---

## 1. What VGGT Provides

The VGGT model outputs the following predictions for a video sequence:

| Output | Shape | Description |
|--------|-------|-------------|
| `depth` | [B, S, H, W, 1] | Depth maps (canonical/relative scale) |
| `depth_conf` | [B, S, H, W] | Depth confidence scores |
| `world_points` | [B, S, H, W, 3] | 3D world coordinates |
| `world_points_conf` | [B, S, H, W] | Point confidence scores |
| `pose_enc` | [B, S, 9] | Camera pose encoding |

### Camera Pose Encoding (9 dimensions)

The `pose_enc` contains:
- **Translation T** (3D): Camera position in VGGT's coordinate system
- **Quaternion** (4D): Camera rotation
- **Field of View** (2D): Horizontal and vertical FoV

From `pose_enc`, we can derive:
- **Extrinsic matrix** [3×4]: Camera-from-world transformation (OpenCV convention)
- **Intrinsic matrix** [3×3]: fx, fy, cx, cy derived from FoV

```python
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

extrinsic, intrinsic = pose_encoding_to_extri_intri(
    pose_enc,
    image_size_hw=(H, W)
)
```

---

## 2. What the Flight Log Provides

The DJI flight log provides:

| Data | Units | Purpose |
|------|-------|---------|
| `OSD.height` | meters | Altitude above takeoff (AGL) |
| `GIMBAL.pitch` | degrees | Camera tilt angle |
| `OSD.latitude` / `OSD.longitude` | degrees | GPS position |
| `GIMBAL.yaw` / `OSD.yaw` | degrees | Heading direction |
| `CAMERA.isVideo` | boolean | Video recording indicator |

---

## 3. The Scale Ambiguity Problem

**VGGT outputs are self-consistent but have arbitrary scale.**

The depth and world_points from VGGT describe the 3D structure correctly, but there's no absolute reference. The model cannot distinguish between:
- A tabletop miniature 1 meter away
- A real landscape 100 meters away

Both scenes would produce **identical relative structure** but at different scales.

### Current Metric Scaling (main.py)

```python
# Calculate real slant range from telemetry
slant_range = altitude / sin(pitch)

# Scale factor: real distance / canonical distance
scale = slant_range / center_depth

# Apply to all depths
metric_depth = canonical_depth * scale
```

**Required inputs from flight log:**
1. `altitude` - Real-world height (meters)
2. `pitch` - Camera angle (to compute slant range)

---

## 4. Feasibility Analysis: Without Flight Log

### What IS Possible ✅

| Feature | How |
|---------|-----|
| **Consistent 3D reconstruction** | VGGT's world_points provide self-consistent 3D structure |
| **Relative depth** | Depth maps show which objects are closer/farther |
| **Object tracking in 3D** | Track positions in VGGT's coordinate system |
| **Relative positions between objects** | Compute distances in arbitrary units |
| **Movement direction (heading)** | Derived from velocity vector direction (relative to VGGT frame, not North) |
| **Relative velocity** | Position change per frame (arbitrary units/frame) |
| **Object density (normalized)** | Count neighbors within X units (not meters) |
| **Camera motion estimation** | Extrinsics show relative camera movement |

**Note on Heading:** Without flight log, heading is relative to VGGT's coordinate frame (established by first camera frame), NOT absolute compass direction. Useful for comparing motion patterns between objects (same heading = moving in same direction).

### What is NOT Possible ❌

| Feature | Why Not |
|---------|---------|
| **Metric depth (meters)** | No absolute scale reference |
| **Real-world coordinates (lat/lon)** | No GPS anchor point |
| **Metric velocity (m/s)** | No scale + no timestamps |
| **Metric distances (meters)** | No scale reference |
| **True altitude estimation** | No ground-truth reference |

---

## 5. Alternative Approaches for Scale Estimation

### Option A: Known Object Size

If any object in the scene has a known real-world size:

```python
# Example: Detect a car (typical length ~4.5m)
known_size_meters = 4.5  # car length
measured_size_vggt = compute_object_size(world_points, bbox)

scale_factor = known_size_meters / measured_size_vggt
metric_depth = vggt_depth * scale_factor
```

**Pros:** Can recover metric scale
**Cons:** Requires object detection + size database

### Option B: Ground Plane Assumption

Assume the ground is at a known altitude (e.g., takeoff = 0m):

```python
# If we know the UAV height from other source (barometer, etc.)
assumed_altitude = 50.0  # meters (user input or external sensor)
assumed_pitch = 45.0     # degrees (estimated or fixed)

scale = (assumed_altitude / sin(pitch)) / center_depth
```

**Pros:** Simple, recovers approximate metric
**Cons:** User must provide altitude estimate

### Option C: Normalized State Estimation

Work entirely in VGGT's coordinate system without conversion:

```python
# Position in VGGT units (not meters)
pos_vggt = world_points[y, x]  # [x, y, z] in VGGT coords

# Velocity in VGGT units per frame
vel_vggt = (pos_frame_t - pos_frame_t_minus_1) / dt_frames

# Heading from velocity direction (valid regardless of scale)
heading = atan2(vel_east, vel_north)  # works in any scale
```

**Pros:** Fully functional, no external data needed
**Cons:** Outputs are not in real-world units

### Option D: Camera Motion as Scale Proxy

If the UAV moves at known speed, use extrinsics to estimate scale:

```python
# VGGT estimates camera translation between frames
T_vggt = extrinsic_t[:, :3, 3] - extrinsic_t_minus_1[:, :3, 3]
translation_vggt = np.linalg.norm(T_vggt)

# If UAV speed is known (e.g., 10 m/s, frame rate 5 fps)
known_displacement = 10.0 / 5.0  # 2 meters per frame
scale = known_displacement / translation_vggt
```

**Pros:** No need for altitude/pitch
**Cons:** Requires known UAV speed

---

## 6. Proposed `main_v4_no_flight.py` Approach

### Recommended: Normalized State Estimation

For a no-flight-log pipeline, I recommend **Option C: Normalized State Estimation** with optional scale override:

```
Inputs:
  - Video (MP4)
  - Tracking JSON (bounding boxes)
  - [Optional] scale_hint (meters per VGGT unit)
  - [Optional] fps (for temporal calculations)

Processing:
  1. Extract frames from video
  2. Run VGGT inference → depth, world_points, extrinsics, intrinsics
  3. For each tracked object:
     a. Sample depth at bounding box center
     b. Unproject to 3D using intrinsics + extrinsics
     c. Track position across frames in VGGT coordinate system
     d. Apply Kalman Filter for smoothing
     e. Estimate velocity and heading from position changes

Outputs:
  - Object positions (x, y, z in VGGT coords, or meters if scale_hint provided)
  - Object velocity (units/frame, or m/s if scale_hint + fps provided)
  - Object heading (degrees, always valid)
  - Relative distances between objects
```

### Output Format

```json
{
  "coordinate_system": "vggt",  // or "metric" if scale_hint provided
  "scale_factor": null,         // or float if scale_hint provided
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_sec": null,    // null without flight log
      "camera_pose": {
        "position": [0, 0, 0],
        "rotation_quat": [1, 0, 0, 0]
      },
      "objects": [
        {
          "track_id": 1,
          "bbox": [100, 200, 150, 250],
          "position_3d": [1.23, 4.56, 7.89],
          "velocity_3d": [0.1, 0.2, 0.0],
          "speed": 0.223,           // units/frame
          "heading_deg": 63.4,      // always valid
          "depth": 45.6             // VGGT canonical depth
        }
      ]
    }
  ]
}
```

---

## 7. Comparison: With vs Without Flight Log

| Capability | With Flight Log | Without Flight Log |
|------------|-----------------|-------------------|
| **Metric depth** | ✅ Meters | ❌ Arbitrary units |
| **Geolocalization** | ✅ Lat/Lon | ❌ Not possible |
| **Metric velocity** | ✅ m/s | ❌ Units/frame |
| **Heading estimation** | ✅ Valid | ✅ Valid |
| **Object tracking** | ✅ In NED frame | ✅ In VGGT frame |
| **Relative distances** | ✅ Meters | ✅ Arbitrary units |
| **Density calculation** | ✅ Within X meters | ⚠️ Within X units |
| **UAV position** | ✅ From GPS | ❌ From VGGT (relative) |

---

## 8. Conclusion

**VGGT CAN estimate depth and 3D structure without a flight log**, but:

1. **Scale is arbitrary** - Outputs are not in real-world units
2. **No geolocalization** - Cannot convert to lat/lon without GPS anchor
3. **Heading IS valid** - Direction of motion works regardless of scale

**Recommended approach for `main_v4_no_flight.py`:**
- Use VGGT's world_points directly for 3D tracking
- Report positions/velocities in normalized units
- Allow optional `scale_hint` parameter for metric conversion
- Always output heading (valid in any coordinate system)

This provides a functional state estimation pipeline that gracefully degrades when flight log is unavailable, while still enabling relative analysis of object behavior.
