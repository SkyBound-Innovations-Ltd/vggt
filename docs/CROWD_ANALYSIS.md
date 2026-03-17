# Crowd Analysis Pipeline

This document describes the crowd density estimation and clustering algorithms
used in `main_v2_state_est.py`, exposed through the `main_v3_service.py` API.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Order](#pipeline-order)
- [1. Per-Person Crowd Density (Voronoi)](#1-per-person-crowd-density-voronoi)
  - [1.1 Algorithm](#11-algorithm)
  - [1.2 Boundary Handling](#12-boundary-handling)
  - [1.3 Edge Cases](#13-edge-cases)
  - [1.4 Previous Method (Fixed-Radius)](#14-previous-method-fixed-radius)
- [2. HDBSCAN Crowd Clustering](#2-hdbscan-crowd-clustering)
  - [2.1 Feature Space](#21-feature-space)
  - [2.2 HDBSCAN Parameters](#22-hdbscan-parameters)
  - [2.3 Temporal Stability (Hungarian Matching)](#23-temporal-stability-hungarian-matching)
  - [2.4 Crossing Crowds Problem](#24-crossing-crowds-problem)
- [3. Output Fields](#3-output-fields)
- [4. Visualisation](#4-visualisation)
- [5. Known Limitations](#5-known-limitations)

---

## Overview

The crowd analysis pipeline runs **after** state estimation (geolocalization +
Kalman filtering) and operates on the NED-frame positions of detected persons.
It produces two independent outputs per person per frame:

| Output | What it measures | Method |
|---|---|---|
| `crowd_density` | Local density around this person (p/m²) | Voronoi tessellation |
| `crowd_id` | Which crowd group this person belongs to | HDBSCAN + Hungarian matching |

These are complementary: `crowd_density` is a continuous per-person metric,
while `crowd_id` is a discrete group label.

---

## Pipeline Order

```
process_tracks()
  ├── Geolocalization (pixel → NED → LLA)
  ├── Kalman Filter (smooth positions, estimate velocities)
  ├── Per-person crowd density (Voronoi tessellation)    ← Section 1
  └── HDBSCAN crowd clustering (per-frame + Hungarian)   ← Section 2
```

Both steps operate per-frame on the horizontal NE plane (`pos_ned[:2]`).

---

## 1. Per-Person Crowd Density (Voronoi)

**File**: `main_v2_state_est.py`, inside `process_tracks()` (after Kalman filtering)

### 1.1 Algorithm

For each frame, extract all person positions on the NE plane and compute
a [Voronoi tessellation](https://en.wikipedia.org/wiki/Voronoi_diagram):

```
density_i = 1 / area(voronoi_cell_i)    [persons/m²]
```

Each person's Voronoi cell is the region of the plane closer to that person
than to any other.  The cell area represents the "personal space" that person
occupies — tightly packed people get small cells (high density), isolated
people get large cells (low density).

**Advantages over fixed-radius counting:**

| Property | Fixed-radius (`count / π·r²`) | Voronoi (`1 / cell_area`) |
|---|---|---|
| Parameters | Radius `r` (arbitrary choice) | None (parameter-free) |
| Adaptivity | Fixed scale everywhere | Adapts to local spacing |
| Boundary bias | Half-empty circles at edges | Handled by clipping |
| Sensitivity to `r` | High — r=2m vs r=5m gives very different results | N/A |

### 1.2 Boundary Handling

Persons on the convex hull of the crowd would get infinite Voronoi cells.
To handle this:

1. Compute the bounding box of all person positions
2. Expand it by `BOUNDARY_BUFFER_M = 5.0` metres on each side
3. Add **mirror points** beyond the bounding box (reflected copies of all
   real points across each edge)
4. Compute Voronoi on the augmented point set
5. Only read cell areas for the original (non-mirror) points

The mirror points ensure all original cells are finite and bounded, with
boundary persons getting cells that extend to the buffer distance rather
than to infinity.

### 1.3 Edge Cases

| Scenario | Behaviour |
|---|---|
| 1 person in frame | `crowd_density = 1 / MAX_CELL_AREA_M2` (= 1/200 = 0.005 p/m²) |
| 2 persons | Each gets area = `dist × BOUNDARY_BUFFER_M`, clamped to [0.01, 200] m² |
| Collinear points | Falls back to `MAX_CELL_AREA_M2` per person |
| Very close persons | Cell area floored at 0.01 m² → max density = 100 p/m² |

**Constants:**

| Constant | Value | Purpose |
|---|---|---|
| `BOUNDARY_BUFFER_M` | 5.0 m | Buffer around convex hull for clipping |
| `MAX_CELL_AREA_M2` | 200.0 m² | Cap for isolated / degenerate cases |

### 1.4 Previous Method (Fixed-Radius)

Before Voronoi, density was computed as:

```
crowd_density = count_within_r / (π × r²)
```

With `r = 2.0 m` (fixed). This was replaced because:
- The 2m radius was close to position noise at typical drone altitudes (50-60m)
- Two people 1.5m apart got density `2/12.57 = 0.16 p/m²` despite being very close
- Edge-of-crowd persons always had half-empty circles

### Legacy `density` Field

The raw neighbour count field `density` (integer) is still computed for
backward compatibility:

```
density = count of non-UAV objects within 10m radius (excluding self)
```

This uses `scipy.spatial.cKDTree` with `COMPAT_RADIUS_M = 10.0`.

---

## 2. HDBSCAN Crowd Clustering

**File**: `main_v2_state_est.py`, function `cluster_crowds_per_frame()`

Groups persons into distinct crowds using density-based clustering on a
4D feature space (position + velocity), with temporal consistency maintained
by Hungarian matching across frames.

### 2.1 Feature Space

Each person detection is represented as a 4D feature vector:

```
features = [p_N, p_E, v_N × w/s, v_E × w/s]
```

Where:
- `p_N, p_E` — NED position (North, East) in metres
- `v_N, v_E` — NED velocity in m/s
- `w` = `coherence_weight` (default 5.0)
- `s` = `max_walking_speed_mps` (default 2.0)

**Effective velocity scaling**: `v × 5.0/2.0 = v × 2.5`

This means a velocity difference of 1 m/s contributes 2.5 units to the
feature-space distance — equivalent to being 2.5 metres apart spatially.

### 2.2 HDBSCAN Parameters

| Parameter | Default | Description | Effect of increasing |
|---|---|---|---|
| `min_cluster_size` | 10 | Minimum persons to form a crowd | Fewer, larger crowds; small groups become noise |
| `min_samples` | 3 | Core point density threshold | More conservative; requires denser local neighbourhood |
| `coherence_weight` | 5.0 | Velocity feature multiplier | Stronger separation of groups moving differently |
| `max_walking_speed_mps` | 2.0 | Velocity normalisation divisor | Lower value = more velocity sensitivity |
| `cluster_selection_epsilon` | 2.0 | Merge threshold (metres) | Prevents over-fragmenting sub-clusters within this distance |
| `max_match_dist` | 20.0 | Max Hungarian match cost (metres) | More lenient matching across frames |
| `ema_alpha` | 0.4 | EMA smoothing for centroid/momentum | Higher = faster response, less smoothing (1.0 = raw) |
| `memory_frames` | 15 | Frames to remember absent clusters | Longer memory bridges larger temporal gaps (15 = 1.5s at 10fps) |

**HDBSCAN** (Hierarchical DBSCAN) is chosen over DBSCAN because:
- No need to specify a fixed `eps` radius — it discovers clusters at varying densities
- `cluster_selection_epsilon` acts as a soft merge threshold rather than a hard cutoff
- Noise points (label = -1) are naturally handled — isolated persons get `crowd_id = None`

### 2.3 Temporal Stability (Hungarian Matching)

HDBSCAN runs independently per frame, producing local cluster labels that
have no correspondence across frames. To maintain stable `crowd_id` values:

**Phase 1 — Per-frame clustering:**
Run HDBSCAN on the current frame's 4D features. For each resulting cluster,
compute its macro-state:
- `centroid` = median position (NE) of all members
- `momentum` = median velocity (NE) of all members

**Phase 2 — Hungarian matching with memory:**
Match current-frame clusters to a **cluster memory** (not just the previous
frame) using `scipy.optimize.linear_sum_assignment` on a cost matrix:

```
cost[i,j] = spatial_dist + coherence_weight × velocity_dist / max_walking_speed_mps
```

Where:
- `spatial_dist` = Euclidean distance between centroids
- `velocity_dist` = Euclidean distance between momentum vectors, normalised
  by `max_walking_speed_mps` (consistent with HDBSCAN feature scaling)

Matches with cost > `max_match_dist` (default 20.0 m) are rejected.
Unmatched current clusters receive a new global `crowd_id`.

**EMA smoothing:** When a cluster matches, its centroid and momentum are
updated using exponential moving average:
```
entry.centroid = α × current + (1 − α) × entry.centroid
entry.momentum = α × current + (1 − α) × entry.momentum
```
With `ema_alpha=0.4` (default), this dampens frame-to-frame jitter in
cluster macro-state, reducing spurious match failures.

**Multi-frame memory:** Clusters that are not matched in a frame are not
immediately discarded. Instead, their `age` counter is incremented. A cluster
remains in memory for up to `memory_frames` frames (default 15 = 1.5s at
10fps). This bridges temporary gaps where a cluster is not detected (e.g.
due to occlusion or momentary HDBSCAN noise). When the cluster reappears,
it can match its memorised entry and recover its original `crowd_id`.

**Phase 3 — Stamp labels:**
Write the resolved `crowd_id` onto every person detection in the frame.
HDBSCAN noise points (label = -1) get `crowd_id = None`.

### 2.4 Crossing Crowds Problem

The `coherence_weight` was increased from 2.0 to 5.0 to address a specific
scenario: two crowds crossing at the same location but moving in opposite
directions.

**With `coherence_weight = 2.0` (old):**
```
Cross-crowd (opposite dirs at 1.5 m/s, 0m apart):  distance ≈ 3.0
Same-crowd  (same direction, 3m apart):             distance ≈ 3.0
→ HDBSCAN cannot distinguish — equal feature-space distance
```

**With `coherence_weight = 5.0` (new):**
```
Cross-crowd (opposite dirs at 1.5 m/s, 0m apart):  distance ≈ 7.5
Same-crowd  (same direction, 3m apart):             distance ≈ 3.0
→ 2.5× separation — HDBSCAN reliably splits crossing crowds
```

---

## 3. Output Fields

Each person detection in the output JSON includes:

| Field | Type | Description |
|---|---|---|
| `density` | int | Legacy: count of non-UAV objects within 10m (backward compat) |
| `crowd_id` | int \| null | HDBSCAN crowd cluster ID (1-based). `null` for noise or non-person |
| `crowd_density` | float \| null | Voronoi-based local density in persons/m². `null` for non-person or missing position |

**Example output:**
```json
{
  "frame_id": 42,
  "track_id": 15,
  "class_name": "person",
  "lat": 51.476325,
  "lon": -3.189590,
  "pos_ned": [12.5, -3.2, -62.0],
  "vel_ned": [0.8, -0.3, 0.0],
  "density": 8,
  "crowd_id": 2,
  "crowd_density": 0.1534
}
```

The `crowd_density` and `crowd_id` are **independent calculations**:
- A person can have high `crowd_density` but `crowd_id = null` (in a dense
  area but the group is too small for HDBSCAN's `min_cluster_size`)
- A person can have low `crowd_density` but a valid `crowd_id` (in a large
  but sparse crowd)

---

## 4. Visualisation

The script `scripts/visualise_vggt_output.py` generates three video types
from a `state_estimation.json`:

| Video | Suffix | What it shows |
|---|---|---|
| Standard map | `_map.mp4` | Class-coloured markers, velocity arrows, crowd hulls |
| Crowd map | `_crowd.mp4` | Persons coloured by `crowd_id` (golden-angle palette) |
| Density map | `_density.mp4` | Persons coloured by `crowd_density` (gray→navy gradient) |

**CLI usage:**
```bash
python scripts/visualise_vggt_output.py \
  -i outputs/<dataset>/state_estimation.json \
  --video density,crowd \
  --no-arrows              # optional: hide velocity arrows
```

**Density colour scale:**
- Light gray (#CCCCCC) = 0 p/m²
- Dark navy (#0A2463) = P99 density across all frames
- Unclustered persons shown as mid-gray (#888888) at 40% alpha

---

## 5. Parameter Tuning

The script `scripts/tune_hdbscan.py` performs automated parameter search
over the HDBSCAN clustering parameters. It re-runs only the clustering
step (no VGGT inference or state estimation needed) on an existing
`state_estimation.json`.

**Usage:**
```bash
python scripts/tune_hdbscan.py \
    -i outputs/sim_260306_.../state_estimation.json \
    --method random --n-iter 200 --top 20
```

**Options:**
| Flag | Default | Description |
|---|---|---|
| `-i` / `--input` | (required) | Path to state_estimation.json |
| `--method` | `random` | `random` or `grid` search |
| `--n-iter` | 200 | Iterations for random search |
| `--top` | 20 | Show top N results |
| `--seed` | 42 | Random seed |

**Proxy metrics** (no ground-truth labels needed):

| Metric | Measures | Lower = better |
|---|---|---|
| `switches` | Total crowd_id changes across all track histories | Yes |
| `noise_ratio` | Fraction of persons with `crowd_id=None` | Yes |
| `unique_ids` | Number of distinct crowd_ids used | Yes |
| `count_std` | Std of crowds-per-frame count | Yes |

These are combined into a composite cost for ranking.

---

## 6. Known Limitations

### Position Accuracy & Drone Rotation Artefacts
Crowd density precision is fundamentally limited by the position estimates
from monocular depth (VGGT). At typical drone altitudes (50-60m), horizontal
position errors of ±5-10m are common. The density values are best interpreted
as **relative** (denser vs sparser regions) rather than absolute measurements.

**Drone yaw/tilt artefacts:** When the drone rotates, stationary objects
appear to move in the geolocalised NED frame because the pixel-to-world
projection shifts with viewing angle. This creates phantom velocity estimates
in the Kalman filter, which propagate into the HDBSCAN velocity features.
The root cause is geolocalisation error (depth estimate changes, yaw
inaccuracy amplified over distance), not the KF or clustering. With perfect
geolocalisation, drone rotation would produce zero apparent motion.

### Voronoi Boundary Effects
The mirror-point approach for boundary handling is an approximation. Persons
at the very edge of the field of view may have slightly distorted cell areas
compared to the true tessellation that would include off-screen persons.

### HDBSCAN Sensitivity
- `min_cluster_size` (default 7) means smaller groups are classified as noise
- The velocity-weighted feature space assumes pedestrian speeds; vehicle
  crowds (if any) would need different `max_walking_speed_mps`
- Stationary crowds (speed ≈ 0) are clustered purely on spatial proximity

### Track ID Flickering
The HDBSCAN clustering does **not** use `track_id` — it operates purely on
per-frame `pos_ned` and `vel_ned` features. Flickering or unstable track IDs
from the upstream MOT tracker do not directly affect clustering results.

However, flickering track IDs **significantly affect the Kalman filter**,
which runs per-`track_id`. When the same physical person receives different
IDs across frames, the KF treats them as separate short-lived tracks:

1. **Short KF lives → poor velocity convergence.** A track that exists for
   only 5-10 frames barely has enough history to estimate velocity reliably.
2. **Gaps cause prediction drift.** If a track_id disappears then reappears,
   the KF predicts forward into empty frames, drifting from the true state.
   The reappearing measurement triggers a large correction jump.
3. **Degraded velocity → noisy HDBSCAN features.** The `vel_ned` fed into
   the clustering becomes unreliable, reducing the usefulness of the
   velocity-weighted feature space.

**Empirical evidence:** On a real-world dataset with median track length of
19 frames (594 IDs for ~50 visible people), the parameter optimizer selects
`coherence_weight=5` to avoid amplifying noisy velocity. On a simulation
dataset with stable track IDs (median track length >100 frames), the
optimizer selects `coherence_weight=20` — a 4× difference, directly
attributable to track ID quality.

Flickering track IDs also affect the **tuning proxy metrics**: the `switches`
metric counts `crowd_id` changes per `track_id`, so fragmented tracks
undercount the true number of crowd assignment switches.

### Frame-Level Independence
Both density and clustering operate per-frame. A person's `crowd_density`
can change significantly between consecutive frames due to position jitter,
even after Kalman filtering. The `crowd_id` is temporally stabilised by
Hungarian matching with EMA smoothing and multi-frame memory, but the
density value is not smoothed across time.

### Crowd ID Semantics
`crowd_id` is a per-frame group label, **not** a permanent identity. The same
person may have different `crowd_id` values across frames as crowds merge,
split, or as the person moves between groups. The Hungarian matching
maintains consistency where possible, but cannot prevent relabelling when
crowd topology changes.

### Parameter Sensitivity Across Scenes
The optimal HDBSCAN parameters are **scene-dependent** and do not transfer
well between datasets. Empirical results from two datasets:

| Parameter | Simulation (sim_260306) | Real-world (DJI_20260215) | Why different |
|---|---|---|---|
| `coherence_weight` | 20 | 5 | Noisy velocity from geolocalisation errors and track flickering |
| `min_cluster_size` | 7 | 8 | Similar crowd density in both scenes |
| `ema_alpha` | 0.5 | 0.2 | Noisier data needs heavier smoothing |
| `memory_frames` | 30 | 20 | Similar temporal behaviour |
| `max_match_dist` | 50 | 30 | Larger clusters in simulation scene |

Key scene factors that influence optimal parameters:
- **Drone altitude** — higher altitude = larger position noise = lower `cw`
- **Tracking quality** — flickering track IDs degrade KF velocity, forcing lower `cw`
- **Crowd density/size** — affects `min_cluster_size` and `epsilon`
- **Scene spatial extent** — affects `max_match_dist`
- **FPS** — affects `memory_frames` (same real-time window needs different frame counts)

Currently, `scripts/tune_hdbscan.py` must be run per-dataset to find optimal
parameters. This takes ~3-5 minutes for 300 iterations.

---

## 7. Future Work

### Scene-Adaptive Parameters
The current approach requires per-dataset parameter tuning. Three strategies
to eliminate or reduce this:

1. **Feature normalisation (most impactful):** Z-score normalise both position
   and velocity features to unit variance per-frame before HDBSCAN. This makes
   `coherence_weight` act as a true position/velocity balance ratio rather than
   a dataset-dependent scale factor. A single `cw` value would then generalise
   across altitudes and scene sizes.

2. **Auto-adaptive `coherence_weight`:** Measure velocity noise in the first
   N frames (e.g. variance of velocity estimates for near-stationary objects)
   and scale `cw` inversely. High noise → low `cw`, clean velocity → high `cw`.

3. **Scene-class presets:** Define profiles (e.g. `dense-outdoor`,
   `sparse-outdoor`, `simulation`) with pre-tuned defaults selected by the
   user or inferred from metadata (altitude, FPS, tracker type).

### Yaw-Compensated Measurement Noise
Scale `kf_sigma_meas_h` by drone yaw rate: `sigma_h * (1 + k * |yaw_rate|)`.
During rotation, the KF would trust predictions over noisy measurements,
reducing phantom velocity estimates from geolocalisation shifts.

### Temporal Density Smoothing
`crowd_density` is currently per-frame with no temporal smoothing. A sliding
window median or EMA on density values per-track would reduce frame-to-frame
jitter without introducing significant lag.

### Track ID Unification
Pre-process MOT tracking output to merge fragmented track IDs before KF
state estimation. Link dying tracks to new births using spatial proximity,
bbox IoU, and trajectory extrapolation. This improves KF velocity convergence
by providing longer observation histories per physical person.
