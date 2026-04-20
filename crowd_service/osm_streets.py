"""
OSM street polygon helpers for street-partitioned crowd clustering.

Responsibilities:
- Fetch named pedestrian-area polygons (plazas, squares, etc.) from OSM via
  Overpass, cached as GeoPackage under `outputs/.osm_cache/polys_{sha}.gpkg`.
- Load user-supplied polygons defined either by OSM node id lists (one or
  more closed rings per street) or by auto-buffering an OSM LineString with
  a given name.
- Per-row point-in-polygon snap (`snap_persons_polygon_only`). Each person
  detection is assigned to the smallest polygon containing its (lat, lon)
  — no distance tolerance, no edge fallback.

Uses osmnx >= 2.0 API: graph_from_bbox(bbox=(west, south, east, north)).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import osmnx as ox
    import networkx as nx  # noqa: F401  (imported for user; osmnx also pulls it)
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

UNASSIGNED_KEY = "__unassigned__"


def compute_track_medians(tracks) -> dict[int, tuple[float, float]]:
    """Median (lat, lon) per person track_id across all frames it appears in."""
    lats_by_tid: dict[int, list[float]] = defaultdict(list)
    lons_by_tid: dict[int, list[float]] = defaultdict(list)
    for t in tracks:
        if t.get("class_name") != "person":
            continue
        lat = t.get("lat")
        lon = t.get("lon")
        if lat is None or lon is None:
            continue
        tid = t["track_id"]
        lats_by_tid[tid].append(float(lat))
        lons_by_tid[tid].append(float(lon))
    return {
        tid: (float(np.median(lats_by_tid[tid])), float(np.median(lons_by_tid[tid])))
        for tid in lats_by_tid
    }


def _overpass_fetch_ways_by_name(
    name: str,
    bbox: tuple[float, float, float, float],
    *,
    timeout: int = 60,
):
    """Fetch all ways with the exact `name` tag inside `bbox` as Shapely LineStrings.

    `bbox` is (min_lat, min_lon, max_lat, max_lon). Returns a list of shapely
    LineString objects in EPSG:4326.
    """
    import urllib.request, urllib.parse
    import json as _json
    from shapely.geometry import LineString

    min_lat, min_lon, max_lat, max_lon = bbox
    # Escape double-quotes in name for Overpass QL
    esc = name.replace('"', '\\"')
    q = (
        f"[out:json][timeout:{timeout}];"
        f'way["name"="{esc}"]({min_lat},{min_lon},{max_lat},{max_lon});'
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=data,
        headers={
            "User-Agent": "crowd-service/0.1",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = _json.loads(resp.read().decode())
    lines = []
    for el in body.get("elements", []):
        if el.get("type") != "way":
            continue
        coords = [(float(n["lon"]), float(n["lat"])) for n in el.get("geometry", [])]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def _overpass_fetch_nodes(node_ids: list[int], *, timeout: int = 60) -> dict[int, tuple[float, float]]:
    """Batch-fetch node coordinates from Overpass. Returns {node_id: (lat, lon)}.

    Chunks at 500 IDs per request to stay under Overpass query-length limits.
    """
    import urllib.request
    import urllib.parse
    import json as _json

    out: dict[int, tuple[float, float]] = {}
    CHUNK = 500
    for start in range(0, len(node_ids), CHUNK):
        chunk = node_ids[start:start + CHUNK]
        ids_csv = ",".join(str(i) for i in chunk)
        q = f"[out:json][timeout:{timeout}];node(id:{ids_csv});out;"
        data = urllib.parse.urlencode({"data": q}).encode()
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=data,
            headers={
                "User-Agent": "crowd-service/0.1",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = _json.loads(resp.read().decode())
        for el in body.get("elements", []):
            if el.get("type") == "node":
                out[int(el["id"])] = (float(el["lat"]), float(el["lon"]))
    return out


def load_user_polygons(
    json_path: Path | str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    cache_dir: Path | str = "outputs/.osm_cache",
    auto_close: bool = True,
    min_nodes: int = 3,
):
    """Load a user-supplied polygon definition file and return a GeoDataFrame.

    Each entry in the JSON may take one of three forms:

        "Wood Street": [node_id, node_id, ...]           # single-ring polygon
        "Central Square": [[...], [...]]                 # multi-ring (unioned)
        "Wood Street": {"from_osm_name": "Wood Street",  # auto-buffer a
                        "buffer_m": 4.0}                 # named OSM way

    The auto-buffer form fetches every OSM way with that `name` tag inside
    the scene `bbox` (LineStrings), unions them, then buffers by `buffer_m`
    metres in UTM to produce a polygon. Handy for streets OSM doesn't
    publish as a closed area.

    `bbox` (min_lat, min_lon, max_lat, max_lon) is required only when any
    entry uses the dict form; ignored otherwise.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    json_path = Path(json_path)
    if not json_path.exists():
        print(f"  user polygons: no file at {json_path} (ok, skipping)")
        return gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")

    data_raw = json.loads(json_path.read_text() or "{}")

    # Split entries into two buckets by form:
    #   data_nodes: {name: [[ids_of_sector_1], [ids_of_sector_2], ...]}
    #   data_buffer: {name: {"from_osm_name": str, "buffer_m": float}}
    data_nodes: dict[str, list[list[int]]] = {}
    data_buffer: dict[str, dict] = {}
    for k, v in data_raw.items():
        if not v:
            continue
        if isinstance(v, dict):
            data_buffer[k] = v
        elif isinstance(v, list):
            if isinstance(v[0], list):
                rings = [[int(x) for x in ring] for ring in v if ring]
            else:
                rings = [[int(x) for x in v]]
            if rings:
                data_nodes[k] = rings
    if not data_nodes and not data_buffer:
        print(f"  user polygons: file is empty at {json_path}")
        return gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Node-coord fetch only if any node-form entries
    coords: dict[int, tuple[float, float]] = {}
    if data_nodes:
        all_ids = sorted({nid for rings in data_nodes.values() for ring in rings for nid in ring})
        key = hashlib.sha1((",".join(str(i) for i in all_ids)).encode()).hexdigest()[:16]
        cache_path = cache_dir / f"nodes_{key}.json"
        if cache_path.exists():
            print(f"  user-polygon node-coord cache hit: {cache_path.name}")
            coords = {int(k): (v[0], v[1]) for k, v in json.loads(cache_path.read_text()).items()}
        else:
            print(f"  fetching {len(all_ids)} OSM node coordinates from Overpass …")
            coords = _overpass_fetch_nodes(all_ids)
            cache_path.write_text(json.dumps({str(k): [v[0], v[1]] for k, v in coords.items()}))
            print(f"  cached → {cache_path} ({len(coords)} nodes)")

    # Build polygons; multiple rings with the same name are merged via unary_union
    from shapely.ops import unary_union
    names, geoms = [], []
    missing: dict[str, list[int]] = {}
    for name, rings in data_nodes.items():
        ring_polys = []
        for ids in rings:
            pts: list[tuple[float, float]] = []
            miss: list[int] = []
            for nid in ids:
                if nid in coords:
                    lat, lon = coords[nid]
                    pts.append((lon, lat))  # shapely uses (x=lon, y=lat)
                else:
                    miss.append(nid)
            if miss:
                missing.setdefault(name, []).extend(miss)
            if len(pts) < min_nodes:
                continue
            if auto_close and pts[0] != pts[-1]:
                pts.append(pts[0])
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    ring_polys.append(poly)
            except Exception as e:
                print(f"  warning: could not build ring for {name!r}: {e}")
        if not ring_polys:
            continue
        combined = ring_polys[0] if len(ring_polys) == 1 else unary_union(ring_polys)
        if not combined.is_valid:
            combined = combined.buffer(0)
        names.append(name)
        geoms.append(combined)

    # ── Auto-buffer entries ({"from_osm_name": name, "buffer_m": width}) ──
    if data_buffer:
        import geopandas as gpd2
        from shapely.ops import unary_union
        if bbox is None:
            print("  warning: bbox not provided; skipping auto-buffer entries: "
                  f"{list(data_buffer.keys())}")
        else:
            ways_cache = cache_dir / "ways_by_name"
            ways_cache.mkdir(parents=True, exist_ok=True)
            for name, spec in data_buffer.items():
                osm_name = spec.get("from_osm_name", name)
                buffer_m = float(spec.get("buffer_m", 4.0))
                cache_path = ways_cache / f"{hashlib.sha1((osm_name + ':' + str(bbox)).encode()).hexdigest()[:16]}.json"
                if cache_path.exists():
                    print(f"  way-by-name cache hit: {osm_name!r}")
                    line_coords = json.loads(cache_path.read_text())
                    from shapely.geometry import LineString
                    lines = [LineString(c) for c in line_coords if len(c) >= 2]
                else:
                    print(f"  fetching LineStrings for name={osm_name!r} …")
                    lines = _overpass_fetch_ways_by_name(osm_name, bbox)
                    cache_path.write_text(json.dumps([list(l.coords) for l in lines]))
                    print(f"  cached → {cache_path.name} ({len(lines)} linestrings)")
                if not lines:
                    print(f"  warning: no OSM ways named {osm_name!r} in bbox; skipping {name!r}")
                    continue
                # Union lines, project to UTM 32630 for metric buffer, buffer, re-project to 4326
                merged = unary_union(lines)
                line_gdf = gpd2.GeoDataFrame(geometry=[merged], crs="EPSG:4326").to_crs(epsg=32630)
                buffered = line_gdf.buffer(buffer_m)
                out_poly = buffered.to_crs(epsg=4326).iloc[0]
                if not out_poly.is_valid:
                    out_poly = out_poly.buffer(0)
                if out_poly.is_empty:
                    continue
                names.append(name)
                geoms.append(out_poly)
                print(f"  auto-buffered {name!r} from {len(lines)} line(s) × {buffer_m} m")

    if missing:
        print(f"  warning: {sum(len(v) for v in missing.values())} node IDs were "
              f"not returned by Overpass:")
        for n, ids in missing.items():
            print(f"    {n}: {ids[:5]}{' …' if len(ids) > 5 else ''}")

    gdf = gpd.GeoDataFrame({"name": names, "geometry": geoms}, crs="EPSG:4326")
    if not gdf.empty:
        areas_m2 = gdf.to_crs(epsg=32630).geometry.area
        print(f"  built {len(gdf)} user polygon(s):")
        for name, a in zip(gdf["name"], areas_m2):
            print(f"    {name:<40s} {a:>8.0f} m²")
    return gdf


def fetch_named_pedestrian_polygons(
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Path | str = "outputs/.osm_cache",
    force_fetch: bool = False,
):
    """Fetch named pedestrian-area / plaza polygons from OSM as a GeoDataFrame.

    Pulls features tagged as `highway=pedestrian`, `place=square`,
    `leisure=plaza`, `amenity=marketplace`, or a landuse polygon with a
    `name` (e.g. relation 17630421 "Central Square"). Filters to geometries
    that are `Polygon` / `MultiPolygon` AND have a non-empty `name` tag.

    Returns a GeoDataFrame (projected to UTM) with columns `name`,
    `geometry`, or an empty GeoDataFrame if nothing is found.
    """
    if not HAS_OSMNX:
        raise RuntimeError("osmnx is not installed")
    import geopandas as gpd  # noqa: F401  — we return one

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(bbox, "named-polygons")
    cache_path = cache_dir / f"polys_{cache_key}.gpkg"

    if cache_path.exists() and not force_fetch:
        print(f"  named-polygon cache hit: {cache_path.name}")
        import geopandas as gpd
        gdf = gpd.read_file(cache_path)
    else:
        min_lat, min_lon, max_lat, max_lon = bbox
        tags = {
            "highway": "pedestrian",
            "place": ["square", "plaza"],
            "leisure": "plaza",
            "amenity": "marketplace",
        }
        print(f"  Fetching named pedestrian-area polygons …")
        try:
            gdf = ox.features.features_from_bbox(
                bbox=(min_lon, min_lat, max_lon, max_lat),
                tags=tags,
            )
        except Exception as e:
            print(f"  warning: polygon fetch failed ({e}); returning empty set")
            import geopandas as gpd
            return gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")

        # Filter to polygon/multipolygon with non-empty name
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        if "name" in gdf.columns:
            gdf = gdf[gdf["name"].notna() & (gdf["name"].astype(str).str.len() > 0)].copy()
            gdf = gdf[["name", "geometry"]].reset_index(drop=True)
        else:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")

        # Cache
        if not gdf.empty:
            gdf.to_file(cache_path, driver="GPKG")
            print(f"  Saved polygons → {cache_path} ({len(gdf)} named areas)")
        else:
            print("  No named pedestrian polygons in bbox")

    # Project to UTM (match typical graph CRS for metric containment later)
    if not gdf.empty:
        # Use the same projection as the road graph for consistency
        gdf = gdf.to_crs(epsg=32630) if gdf.crs.to_epsg() in (4326, None) else gdf
    return gdf


def snap_persons_polygon_only(
    persons: list[dict],
    polygons_gdf,
    *,
    name_overrides: Optional[dict] = None,
) -> list[tuple[Optional[str], float]]:
    """Per-person-row snap: each row is individually assigned to the SMALLEST
    polygon containing its (lat, lon). Returns a list of (name_or_None, 0.0)
    with one entry per input row (same length, same order).

    Unlike snap_tracks_polygon_only, this does NOT use per-track medians — so
    a track that crosses polygon boundaries mid-clip will stamp different
    street_index values across frames, as the user expected.
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    overrides = name_overrides or {}
    if not persons:
        return []
    if polygons_gdf is None or polygons_gdf.empty:
        return [(None, 0.0)] * len(persons)

    if polygons_gdf.crs is None or polygons_gdf.crs.to_epsg() != 4326:
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)

    areas_m2 = polygons_gdf.to_crs(epsg=32630).geometry.area.values
    polys = list(polygons_gdf.geometry.values)
    names = list(polygons_gdf["name"].astype(str).values)
    tree = STRtree(polys)

    out: list[tuple[Optional[str], float]] = []
    for p in persons:
        lat, lon = p.get("lat"), p.get("lon")
        if lat is None or lon is None:
            out.append((None, 0.0))
            continue
        pt = Point(float(lon), float(lat))
        cands = tree.query(pt)
        hits = []
        for c in cands:
            j = int(c) if isinstance(c, (int, np.integer)) else polys.index(c)
            if polys[j].contains(pt):
                hits.append(j)
        if not hits:
            out.append((None, 0.0))
            continue
        j = min(hits, key=lambda k: areas_m2[k])
        name = names[j]
        out.append((overrides.get(name, name), 0.0))
    return out


def snap_tracks_polygon_only(
    track_medians: dict[int, tuple[float, float]],
    polygons_gdf,
    *,
    name_overrides: Optional[dict] = None,
) -> dict[int, tuple[Optional[str], float]]:
    """Polygon-only snap (no edge fallback).

    Each track's median (lat, lon) is tested for containment against every
    polygon in `polygons_gdf` (expected CRS: EPSG:4326). If inside one,
    the track is assigned that polygon's name. If inside multiple polygons
    (overlap), the SMALLEST polygon wins (most specific). Everything else →
    (None, 0.0) → `__unassigned__`.

    Distance in the return tuple is always 0.0; kept for API compatibility.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    out: dict[int, tuple[Optional[str], float]] = {}
    overrides = name_overrides or {}

    if not track_medians:
        return out
    if polygons_gdf is None or polygons_gdf.empty:
        return {tid: (None, 0.0) for tid in track_medians}

    # Ensure EPSG:4326
    if polygons_gdf.crs is None or polygons_gdf.crs.to_epsg() != 4326:
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)

    # Pre-compute areas (metric) for tie-breaking
    areas_m2 = polygons_gdf.to_crs(epsg=32630).geometry.area.values
    polys = list(polygons_gdf.geometry.values)
    names = list(polygons_gdf["name"].astype(str).values)
    tree = STRtree(polys)

    for tid, (lat, lon) in track_medians.items():
        pt = Point(lon, lat)
        cands = tree.query(pt)
        hits: list[int] = []
        for c in cands:
            j = int(c) if isinstance(c, (int, np.integer)) else polys.index(c)
            if polys[j].contains(pt):
                hits.append(j)
        if not hits:
            out[tid] = (None, 0.0)
            continue
        # smallest area wins → most specific polygon
        j = min(hits, key=lambda k: areas_m2[k])
        name = names[j]
        out[tid] = (overrides.get(name, name), 0.0)
    return out


def compute_bbox(tracks, margin_deg: float = 0.0015) -> tuple[float, float, float, float]:
    """Return (min_lat, min_lon, max_lat, max_lon) with a lat/lon-degree margin."""
    lats, lons = [], []
    for t in tracks:
        lat, lon = t.get("lat"), t.get("lon")
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))
    if not lats:
        raise ValueError("No valid lat/lon values found in tracks.")
    return (
        min(lats) - margin_deg,
        min(lons) - margin_deg,
        max(lats) + margin_deg,
        max(lons) + margin_deg,
    )


def _cache_key(bbox, network_type) -> str:
    min_lat, min_lon, max_lat, max_lon = bbox
    payload = (
        f"{round(min_lat, 4)}_{round(min_lon, 4)}_"
        f"{round(max_lat, 4)}_{round(max_lon, 4)}_{network_type}"
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def build_street_index_map(
    track_street: dict[int, tuple[Optional[str], float]],
) -> dict[str, int]:
    """Assign stable integer indices to street names.

    Index 0 is reserved for __unassigned__. Real names are sorted
    alphabetically and numbered from 1 upwards so runs on the same scene
    produce the same mapping.
    """
    names = {sk for (sk, _) in track_street.values() if sk is not None}
    return {UNASSIGNED_KEY: 0, **{n: i + 1 for i, n in enumerate(sorted(names))}}


