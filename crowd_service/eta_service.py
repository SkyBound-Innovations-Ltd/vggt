"""
FastAPI service for single-track ETA to Cardiff Central Station.

Endpoints:
  GET  /                     — static viewer (index.html)
  POST /api/eta              — {frame_id, track_id} → {eta_s, path_geojson, …}

Startup:
  • Load walking graph (network_type="walk", simplify=False) via osmnx; cached
    under outputs/.osm_cache/ alongside the all-network cache.
  • Pre-compute edge bearings (deg, 0=N clockwise) as edge attribute.
  • Snap Cardiff Central Station to its nearest graph node.

Per-request algorithm:
  1. Look up the track's (lat, lon, vel_ned, heading) at frame_id from the
     active state_estimation.json (selected by ?dataset= or default).
  2. Snap (lat, lon) to nearest walking-graph node.
  3. Filter outgoing edges by heading cone (±90°) against the track's
     current heading to block U-turns.
  4. Virtual-source Dijkstra: inject a synthetic node connected only to
     cone-passing first edges at weight 0, run shortest-path to station.
  5. speed = max(|vel_ned|, SPEED_FLOOR); eta_s = path_length / speed.
  6. Return the path as a GeoJSON LineString (lat/lon pairs).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
import pyproj
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ── Configuration ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE
DATASETS: dict[str, Path] = {
    # Friendly name → path. Client passes "dataset" in request (optional).
    "cardiff_20260416": HERE.parent / "outputs" / "cc_run_20260416_crowd" / "state_estimation.json",
    "cardiff_20260407": HERE.parent / "outputs" / "cc_run_20260407_200138_crowd" / "state_estimation.json",
}
DEFAULT_DATASET = "cardiff_20260416"

STATION_LATLON = (51.47640, -3.17940)        # Cardiff Central Station
STATION_LABEL = "Cardiff Central"
CONE_DEG = 90.0                               # ±90°
SPEED_FLOOR_MPS = 0.5
OSM_CACHE_DIR = HERE.parent / "outputs" / ".osm_cache"


# ── Lazy global state ────────────────────────────────────────────────────
_graph_proj: nx.MultiDiGraph | None = None
_station_node: int | None = None
_transformer_to_graph: pyproj.Transformer | None = None
_transformer_to_wgs: pyproj.Transformer | None = None


def _load_walking_graph() -> nx.MultiDiGraph:
    """Load (cached) or fetch the walking network for the active scene."""
    # Derive bbox from either configured dataset (first available). We use
    # the larger-coverage Cardiff bbox used throughout this project.
    # Hard-coded here to avoid a circular dep on state_estimation.json at
    # import time — it's the same bbox as the all-network cache.
    min_lat, min_lon = 51.47398, -3.18560
    max_lat, max_lon = 51.47965, -3.17605
    margin = 0.0015
    bbox = (
        min_lat - margin,
        min_lon - margin,
        max_lat + margin,
        max_lon + margin,
    )

    import hashlib
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = f"{round(bbox[0], 4)}_{round(bbox[1], 4)}_{round(bbox[2], 4)}_{round(bbox[3], 4)}_walk_simp0"
    key = hashlib.sha1(payload.encode()).hexdigest()[:16]
    cache_path = OSM_CACHE_DIR / f"{key}.graphml"

    if cache_path.exists():
        print(f"  walking-graph cache hit: {cache_path.name}")
        G = ox.load_graphml(cache_path)
    else:
        print(f"  fetching walking graph for bbox {bbox} …")
        G = ox.graph_from_bbox(
            bbox=(bbox[1], bbox[0], bbox[3], bbox[2]),  # (west, south, east, north)
            network_type="walk",
            simplify=False,
        )
        ox.save_graphml(G, cache_path)
        print(f"  cached → {cache_path}")

    G_proj = ox.projection.project_graph(G)
    return G_proj


def _compute_edge_bearings(G: nx.MultiDiGraph) -> None:
    """Annotate every edge with `bearing_deg` (0 = N, clockwise) in the
    graph's projected CRS. Stored in-place."""
    for u, v, k in G.edges(keys=True):
        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
        # In UTM: x=easting, y=northing. Bearing = atan2(Δeasting, Δnorthing).
        dx, dy = x2 - x1, y2 - y1
        bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        G.edges[u, v, k]["bearing_deg"] = bearing


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference (deg) between two bearings."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


# ── Dataset loading (memoised per file) ──────────────────────────────────

@lru_cache(maxsize=4)
def _load_tracks_by_frame(dataset_key: str) -> dict[tuple[int, int], dict]:
    """Return {(frame_id, track_id): row_dict} for fast lookup."""
    path = DATASETS.get(dataset_key)
    if path is None or not path.exists():
        raise FileNotFoundError(f"dataset {dataset_key!r} not found at {path}")
    with open(path) as f:
        data = json.load(f)
    rows = data.get("tracks", []) if isinstance(data, dict) else data
    index: dict[tuple[int, int], dict] = {}
    for r in rows:
        if r.get("class_name") != "person":
            continue
        fid, tid = r.get("frame_id"), r.get("track_id")
        if fid is None or tid is None:
            continue
        index[(int(fid), int(tid))] = r
    return index


# ── Shortest path with heading gate ──────────────────────────────────────

def _snap_latlon_to_node(lat: float, lon: float) -> int:
    """Return the nearest walking-graph node id for a WGS84 point."""
    assert _transformer_to_graph is not None
    assert _graph_proj is not None
    x, y = _transformer_to_graph.transform(lon, lat)
    # osmnx 2.x nearest_nodes
    return int(ox.distance.nearest_nodes(_graph_proj, X=x, Y=y))


def _cone_filtered_edges(node: int, heading_deg: float | None) -> list[tuple[int, int, int]]:
    """Return outgoing edges from `node` whose bearing is within ±CONE_DEG
    of `heading_deg`. If heading is None/nan, all outgoing edges pass."""
    assert _graph_proj is not None
    edges = []
    for u, v, k in _graph_proj.out_edges(node, keys=True):
        if heading_deg is None or not math.isfinite(heading_deg):
            edges.append((u, v, k))
            continue
        b = _graph_proj.edges[u, v, k].get("bearing_deg")
        if b is None:
            edges.append((u, v, k))
            continue
        if _angular_diff(b, heading_deg) <= CONE_DEG:
            edges.append((u, v, k))
    return edges


def _shortest_path_with_cone(
    start_node: int, heading_deg: float | None
) -> tuple[list[int], float]:
    """Run shortest-path on a virtual graph where the first step is gated
    to cone-passing edges. Returns (node_path, length_m)."""
    assert _graph_proj is not None
    assert _station_node is not None

    passing = _cone_filtered_edges(start_node, heading_deg)
    if not passing:
        # Fall back to no gate if cone would leave no options (dead-end kerb)
        passing = list(_graph_proj.out_edges(start_node, keys=True))
    if not passing:
        raise HTTPException(status_code=404, detail="no outgoing edges at start node")

    # Build a virtual source connected at weight 0 to each cone-passing
    # edge's terminal node. Dijkstra from virtual source to station.
    VIRT = -1
    H = _graph_proj.copy()  # cheap: node/edge dicts, not geometry
    H.add_node(VIRT)
    for _, v, k in passing:
        edge_len = _graph_proj.edges[start_node, v, k].get("length", 0.0)
        # Weight 0 for virtual→v but then tacks on the first edge's real length
        # by inserting two hops: VIRT → start_node → v. Simpler: VIRT → v at
        # edge_len so the path length includes the first edge.
        H.add_edge(VIRT, v, key=0, length=float(edge_len))

    try:
        length = nx.shortest_path_length(H, VIRT, _station_node, weight="length")
        node_path = nx.shortest_path(H, VIRT, _station_node, weight="length")
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="no walking path to station")

    # Drop the virtual source; prepend start_node so the visible path begins
    # at the pedestrian's actual location.
    real_path = [start_node] + node_path[1:]
    return real_path, float(length)


def _node_path_to_geojson(node_path: list[int]) -> dict:
    """Convert a list of node ids into a GeoJSON LineString (WGS84 coords)."""
    assert _graph_proj is not None
    assert _transformer_to_wgs is not None
    coords: list[list[float]] = []
    for n in node_path:
        x, y = _graph_proj.nodes[n]["x"], _graph_proj.nodes[n]["y"]
        lon, lat = _transformer_to_wgs.transform(x, y)
        coords.append([lon, lat])
    return {"type": "LineString", "coordinates": coords}


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(title="Crowd ETA service", version="0.1")


class ETARequest(BaseModel):
    frame_id: int
    track_id: int
    dataset: str | None = None


class ETAResponse(BaseModel):
    eta_s: float
    distance_m: float
    speed_mps: float
    heading_deg: float | None
    start_lat_lon: list[float]
    station_lat_lon: list[float]
    station_label: str
    path_geojson: dict
    dataset: str


@app.on_event("startup")
def _startup() -> None:
    global _graph_proj, _station_node, _transformer_to_graph, _transformer_to_wgs
    print("ETA service starting …")
    _graph_proj = _load_walking_graph()
    _compute_edge_bearings(_graph_proj)
    graph_crs = _graph_proj.graph.get("crs", "EPSG:32630")
    _transformer_to_graph = pyproj.Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
    _transformer_to_wgs = pyproj.Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)
    _station_node = _snap_latlon_to_node(*STATION_LATLON)
    print(f"  graph: {len(_graph_proj.nodes)} nodes, {len(_graph_proj.edges)} edges")
    print(f"  station node: {_station_node} @ {STATION_LATLON} ({STATION_LABEL})")


@app.post("/api/eta", response_model=ETAResponse)
def eta(req: ETARequest) -> ETAResponse:
    dataset_key = req.dataset or DEFAULT_DATASET
    try:
        index = _load_tracks_by_frame(dataset_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    row = index.get((req.frame_id, req.track_id))
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"track {req.track_id} not present at frame {req.frame_id}",
        )

    lat = float(row["lat"])
    lon = float(row["lon"])
    vel = row.get("vel_ned") or [0.0, 0.0, 0.0]
    vn, ve = float(vel[0]), float(vel[1])
    speed = max(math.hypot(vn, ve), SPEED_FLOOR_MPS)
    heading = row.get("heading_deg")
    if heading is not None:
        heading = float(heading) % 360.0
    else:
        # Derive from velocity if heading_deg missing
        if abs(vn) + abs(ve) > 1e-6:
            heading = (math.degrees(math.atan2(ve, vn)) + 360.0) % 360.0

    start_node = _snap_latlon_to_node(lat, lon)
    node_path, dist_m = _shortest_path_with_cone(start_node, heading)
    eta_s = dist_m / speed

    return ETAResponse(
        eta_s=round(eta_s, 2),
        distance_m=round(dist_m, 1),
        speed_mps=round(speed, 3),
        heading_deg=round(heading, 1) if heading is not None else None,
        start_lat_lon=[lat, lon],
        station_lat_lon=list(STATION_LATLON),
        station_label=STATION_LABEL,
        path_geojson=_node_path_to_geojson(node_path),
        dataset=dataset_key,
    )


@app.get("/api/datasets")
def list_datasets() -> dict:
    return {
        "default": DEFAULT_DATASET,
        "available": [k for k, p in DATASETS.items() if p.exists()],
    }


# ── Static viewer ────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/{filename:path}")
def static_passthrough(filename: str):
    p = STATIC_DIR / filename
    if not p.exists() or p.is_dir() or filename.startswith("api/"):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p))
