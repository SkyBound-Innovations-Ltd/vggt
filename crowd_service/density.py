"""
Per-person crowd_density via Voronoi tessellation.

Lifted verbatim (with minor refactoring) from
`main_v2_state_est.py::process_tracks`. Kept as a standalone module so the
crowd_service pipeline can populate crowd_density without calling the
full state-estimation flow.

crowd_density = 1 / voronoi_cell_area   [persons / m²]

The Voronoi diagram is built in-frame on (pos_N, pos_E). Boundary cells
(infinite area) are made finite by reflecting all points across the
scene bounding box before tessellating, then only the original N cells
are kept.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


BOUNDARY_BUFFER_M = 5.0
MAX_CELL_AREA_M2 = 200.0
MIN_CELL_AREA_M2 = 0.01


def _voronoi_cell_areas(points_2d: np.ndarray) -> np.ndarray:
    """Voronoi cell areas (m²) for each of N 2-D points.

    Boundary cells are bounded by a bbox-mirror trick. Returns an
    ndarray of length N clamped to [MIN_CELL_AREA_M2, MAX_CELL_AREA_M2].
    """
    from scipy.spatial import Voronoi, ConvexHull

    n = len(points_2d)
    if n == 1:
        return np.array([MAX_CELL_AREA_M2])
    if n == 2:
        dist = np.linalg.norm(points_2d[0] - points_2d[1])
        area = max(dist * BOUNDARY_BUFFER_M, 1.0)
        return np.array([area, area])

    try:
        hull = ConvexHull(points_2d)
    except Exception:
        return np.full(n, MAX_CELL_AREA_M2)

    hull_pts = points_2d[hull.vertices]
    centroid = hull_pts.mean(axis=0)
    directions = hull_pts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    buffered_hull = hull_pts + directions / norms * BOUNDARY_BUFFER_M

    bbox_min = buffered_hull.min(axis=0) - BOUNDARY_BUFFER_M
    bbox_max = buffered_hull.max(axis=0) + BOUNDARY_BUFFER_M

    mirrors = np.vstack([
        np.column_stack([2 * bbox_min[0] - points_2d[:, 0], points_2d[:, 1]]),
        np.column_stack([2 * bbox_max[0] - points_2d[:, 0], points_2d[:, 1]]),
        np.column_stack([points_2d[:, 0], 2 * bbox_min[1] - points_2d[:, 1]]),
        np.column_stack([points_2d[:, 0], 2 * bbox_max[1] - points_2d[:, 1]]),
    ])
    augmented = np.vstack([points_2d, mirrors])

    try:
        vor = Voronoi(augmented)
    except Exception:
        return np.full(n, MAX_CELL_AREA_M2)

    areas = np.full(n, MAX_CELL_AREA_M2)
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        polygon = vor.vertices[region]
        x, y = polygon[:, 0], polygon[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        areas[i] = min(max(area, MIN_CELL_AREA_M2), MAX_CELL_AREA_M2)
    return areas


def compute_crowd_density_voronoi(tracks) -> int:
    """Annotate every person track with `crowd_density` in-place.

    Iterates per frame_id, extracts person (pos_N, pos_E), runs Voronoi,
    writes 1/cell_area. Non-person rows and persons with missing pos_ned
    get crowd_density = None. Returns the count of rows written.
    """
    frame_idx: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tracks):
        if t.get("class_name") != "person":
            continue
        pos = t.get("pos_ned")
        if not pos or pos[0] is None or pos[1] is None:
            continue
        fid = t.get("frame_id")
        if fid is None:
            continue
        frame_idx[fid].append(i)

    for t in tracks:
        if t.get("class_name") == "person":
            t["crowd_density"] = None

    n_written = 0
    for fid, indices in frame_idx.items():
        if not indices:
            continue
        positions = np.array([tracks[i]["pos_ned"][:2] for i in indices],
                             dtype=np.float64)
        areas = _voronoi_cell_areas(positions)
        for local_j, global_j in enumerate(indices):
            tracks[global_j]["crowd_density"] = round(
                1.0 / float(areas[local_j]), 4
            )
            n_written += 1
    return n_written
