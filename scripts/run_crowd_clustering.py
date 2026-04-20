#!/usr/bin/env python3
"""
Run the VGGT crowd-analysis subsystem on a tracks JSON that already contains
pos_ned / vel_ned values (e.g. geocage output from the VGGT service).

Pipeline:
  1. Load tracks JSON.
  2. (Optional) Snap each track to its nearest OSM street and partition
     person rows by street name. The graph is fetched once per run and
     cached as GraphML under outputs/.osm_cache/.
  3. Run HDBSCAN parameter optimisation over z-score normalised position
     and velocity features. In street-partition mode HDBSCAN runs
     independently on each street bucket and the resulting crowd_ids are
     offset by street_index * id_stride so they stay globally unique.
  4. Re-run clustering with the best params to stamp a stable namespaced
     crowd_id onto every person detection.
  5. Write an enriched state_estimation.json compatible with
     crowd_service/preprocess.py and scripts/visualise_vggt_output.py.

Usage:
    python scripts/run_crowd_clustering.py \
        -i inputs/cc_run_20260407_200138/intel/geocage_*.json \
        -o outputs/cc_run_20260407_200138_crowd/state_estimation.json \
        --n-iter 40 --max-seconds 900
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "crowd_service"))

from main_v2_state_est import (
    cluster_crowds_per_frame,
    compute_crowd_metrics,
    crowd_composite_cost,
)

try:
    from osm_streets import (
        UNASSIGNED_KEY,
        compute_bbox,
        fetch_named_pedestrian_polygons,
        load_user_polygons,
        snap_persons_polygon_only,
    )
    HAS_OSM = True
except ImportError as e:
    HAS_OSM = False
    _OSM_IMPORT_ERROR = e


PARAM_GRID = {
    "coherence_weight": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "min_cluster_size": [5, 8, 10, 15],
    "min_samples": [2, 3, 5],
    "cluster_selection_epsilon": [0.3, 0.5, 0.7, 1.0],
    "max_match_dist": [3, 5, 8, 10],
    "ema_alpha": [0.2, 0.3, 0.4, 0.5, 0.7],
    "memory_frames": [5, 10, 15, 20, 30],
}

ID_STRIDE = 10_000  # crowd_id = street_index * ID_STRIDE + local_id


def _group_by_street(tracks):
    """Return dict[street_index -> list[person_row_ref]]. Ignores non-person rows."""
    by_si = defaultdict(list)
    for t in tracks:
        if t.get("class_name") != "person":
            continue
        by_si[t.get("street_index", 0)].append(t)
    return by_si


def cluster_by_street(
    tracks,
    params,
    max_cluster_population,
    street_index_map,
    *,
    id_stride=ID_STRIDE,
    unassigned_relaxed=True,
    params_by_si: Optional[dict] = None,
):
    """Run cluster_crowds_per_frame once per street bucket, then offset ids.

    `params` is the default param dict. `params_by_si` (optional) is a
    per-street mapping {street_index: param_dict} that overrides `params`
    for specific streets.
    """
    for t in tracks:
        if t.get("class_name") == "person":
            t["crowd_id"] = None

    by_si = _group_by_street(tracks)
    unassigned_si = street_index_map.get(UNASSIGNED_KEY, 0)
    n_used = 0

    for si, rows in by_si.items():
        if not rows:
            continue

        p = dict((params_by_si or {}).get(si, params))
        if si == unassigned_si and unassigned_relaxed:
            p["min_cluster_size"] = max(3, int(p["min_cluster_size"]) // 2 + 1)
            p["cluster_selection_epsilon"] = max(float(p["cluster_selection_epsilon"]), 1.0)
        # HDBSCAN requires min_samples <= min_cluster_size (and <= available samples).
        # Clamp defensively to prevent errors on sparse per-frame buckets.
        p["min_samples"] = min(int(p["min_samples"]), int(p["min_cluster_size"]))

        cluster_crowds_per_frame(
            rows,
            min_cluster_size=int(p["min_cluster_size"]),
            min_samples=int(p["min_samples"]),
            coherence_weight=float(p["coherence_weight"]),
            cluster_selection_epsilon=float(p["cluster_selection_epsilon"]),
            max_match_dist=float(p["max_match_dist"]),
            ema_alpha=float(p["ema_alpha"]),
            memory_frames=int(p["memory_frames"]),
            max_cluster_population=max_cluster_population,
        )

        any_clustered = False
        for t in rows:
            cid = t.get("crowd_id")
            if cid is not None:
                t["crowd_id"] = int(si) * id_stride + int(cid)
                any_clustered = True
        if any_clustered:
            n_used += 1

    return n_used


def autotune_one_polygon(rows, n_iter, max_seconds, seed, fixed_params, label=""):
    """Random-search HDBSCAN params on a single polygon's person rows.

    Returns (best_cost, best_params, best_metrics) or None if there aren't
    enough tracks to run HDBSCAN sensibly.
    """
    n_unique = len({t["track_id"] for t in rows})
    if n_unique < 5:
        return None
    rng = np.random.RandomState(seed)
    keys = list(PARAM_GRID.keys())
    t0 = time.monotonic()
    best = (float("inf"), None, None)

    for i in range(n_iter):
        if time.monotonic() - t0 > max_seconds:
            break
        combo = {k: PARAM_GRID[k][rng.randint(len(PARAM_GRID[k]))] for k in keys}
        params = {**fixed_params, **combo}

        for t in rows:
            t["crowd_id"] = None

        cluster_crowds_per_frame(
            rows,
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
            coherence_weight=float(params["coherence_weight"]),
            cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
            max_match_dist=float(params["max_match_dist"]),
            ema_alpha=float(params["ema_alpha"]),
            memory_frames=int(params["memory_frames"]),
            max_cluster_population=fixed_params.get("max_cluster_population"),
        )
        metrics = compute_crowd_metrics(rows)
        cost = crowd_composite_cost(metrics, n_unique)
        if cost < best[0]:
            best = (cost, combo.copy(), metrics.copy())

    for t in rows:
        t["crowd_id"] = None

    if label:
        bcost, bp, bm = best
        if bp is not None:
            print(f"    [{label}] tracks={n_unique:>4} cost={bcost:>7.2f} "
                  f"sw={bm['switches']} ids={bm['unique_ids']} "
                  f"noise={bm['noise_ratio']:.2f}")
    return best


def _cost_street_aware(metrics, n_tracks, n_streets):
    """Composite cost with unique_ids normalised by the number of non-empty streets.

    Mirrors main_v2_state_est.crowd_composite_cost but divides the unique_ids
    term by max(n_streets, 1) so street partitioning (which multiplies
    unique_ids by ~N_streets) does not dominate the autotune signal.
    """
    return (
        (metrics["switches"] / max(n_tracks, 1)) * 1.0
        + metrics["noise_ratio"] * 50.0
        + (metrics["unique_ids"] / max(n_streets, 1)) * 0.5
        + metrics["count_std"] * 2.0
    )


def autotune(tracks, n_iter, max_seconds, seed, fixed_params,
             street_index_map=None):
    n_unique = len(set(
        t["track_id"] for t in tracks
        if t.get("class_name") == "person" and t.get("pos_ned") is not None
    ))
    rng = np.random.RandomState(seed)
    keys = list(PARAM_GRID.keys())
    t0 = time.monotonic()
    partition = street_index_map is not None

    best = (float("inf"), None, None)
    for i in range(n_iter):
        if time.monotonic() - t0 > max_seconds:
            print(f"  autotune: hit {max_seconds}s budget at iter {i}")
            break

        combo = {k: PARAM_GRID[k][rng.randint(len(PARAM_GRID[k]))] for k in keys}
        params = {**fixed_params, **combo}

        if partition:
            n_streets = cluster_by_street(
                tracks, params,
                max_cluster_population=fixed_params.get("max_cluster_population"),
                street_index_map=street_index_map,
            )
            metrics = compute_crowd_metrics(tracks)
            cost = _cost_street_aware(metrics, n_unique, n_streets)
        else:
            for t in tracks:
                if t.get("class_name") == "person":
                    t["crowd_id"] = None
            cluster_crowds_per_frame(
                tracks,
                min_cluster_size=int(params["min_cluster_size"]),
                min_samples=int(params["min_samples"]),
                coherence_weight=float(params["coherence_weight"]),
                cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
                max_match_dist=float(params["max_match_dist"]),
                ema_alpha=float(params["ema_alpha"]),
                memory_frames=int(params["memory_frames"]),
                max_cluster_population=fixed_params.get("max_cluster_population"),
            )
            metrics = compute_crowd_metrics(tracks)
            cost = crowd_composite_cost(metrics, n_unique)

        elapsed = time.monotonic() - t0
        tag = "  *" if cost < best[0] else "   "
        print(f"  [{i+1:>3}/{n_iter}] cost={cost:>7.2f}  "
              f"sw={metrics['switches']:>4}  noise={metrics['noise_ratio']:.3f}  "
              f"ids={metrics['unique_ids']:>3}  cstd={metrics['count_std']:.2f}  "
              f"({elapsed:.0f}s){tag}")

        if cost < best[0]:
            best = (cost, combo.copy(), metrics.copy())

    for t in tracks:
        if t.get("class_name") == "person":
            t["crowd_id"] = None

    return best


def build_metadata(tracks, best_params, best_cost, best_metrics, fps_hint,
                   street_index_map=None, bbox=None, osm_network_type=None,
                   lateral_tol_m=None, polygon_stats=None, params_by_si=None):
    uav = next((t for t in tracks if t.get("class_name") == "UAV"), None)
    home = None
    if uav is not None:
        home = {
            "latitude": uav.get("lat"),
            "longitude": uav.get("lon"),
            "altitude": uav.get("alt"),
        }

    frame_ids = [t.get("frame_id") for t in tracks if t.get("frame_id") is not None]
    total_frames = (max(frame_ids) + 1) if frame_ids else 0

    cost_val = float(best_cost) if best_cost is not None else None
    if cost_val is not None and not np.isfinite(cost_val):
        cost_val = None
    crowd_meta = {
        "method": "HDBSCAN_per_frame_hungarian",
        "z_score_normalisation": True,
        "auto_tuned": cost_val is not None,
        "tuning_best_cost": cost_val,
        "tuning_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in (best_metrics or {}).items()},
        "note": "Per-frame HDBSCAN with z-score normalised features (pos_N, pos_E, "
                "coherence_weight * vel_N, coherence_weight * vel_E), Hungarian "
                "centroid matching, EMA smoothing, and multi-frame memory for "
                "stable crowd IDs.",
    }
    if best_params:
        crowd_meta.update({k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                           for k, v in best_params.items()})

    if street_index_map is not None:
        crowd_meta["street_partitioning"] = {
            "enabled": True,
            "id_stride": ID_STRIDE,
            "network_type": osm_network_type,
            "lateral_tol_m": lateral_tol_m,
            "bbox": list(bbox) if bbox is not None else None,
            "unassigned_key": UNASSIGNED_KEY,
            "snap_method": "per_frame_polygon_only",
            **(polygon_stats or {}),
        }
        crowd_meta["street_index_map"] = {str(i): n for n, i in street_index_map.items()}
        if params_by_si:
            # Key per-polygon param overrides by street name for readability
            idx_to_name = {v: n for n, v in street_index_map.items()}
            crowd_meta["params_by_street"] = {
                idx_to_name.get(si, str(si)): {
                    k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                    for k, v in (params or {}).items()
                }
                for si, params in params_by_si.items()
            }

    return {
        "fps": fps_hint,
        "total_frames": total_frames,
        "home_position": home,
        "pipeline": "crowd-clustering-only",
        "crowd_clustering": crowd_meta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Input tracks JSON")
    parser.add_argument("-o", "--output", required=True, help="Output enriched JSON")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="Max autotune iterations (default: 40)")
    parser.add_argument("--max-seconds", type=float, default=900.0,
                        help="Max autotune wall-clock seconds (default: 900)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=None,
                        help="FPS hint to write into metadata (default: auto from timestamps)")
    parser.add_argument("--max-cluster-population", type=int, default=30,
                        help="Split clusters exceeding this size (default: 30)")
    parser.add_argument("--no-autotune", action="store_true",
                        help="Skip autotune; use default params (optionally overridden by --hdbscan-*)")
    # Manual HDBSCAN param overrides (used when --no-autotune is set)
    parser.add_argument("--coherence-weight", type=float, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=None)
    parser.add_argument("--max-match-dist", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--memory-frames", type=int, default=None)
    # Street partitioning (polygon-only)
    parser.add_argument("--no-street-partition", action="store_true",
                        help="Disable polygon-based street partitioning; cluster globally (legacy behaviour)")
    parser.add_argument("--user-polygons", default="crowd_service/user_polygons.json",
                        help="JSON file mapping {street_name: [osm_node_ids]} "
                             "(default: crowd_service/user_polygons.json)")
    parser.add_argument("--osm-cache-dir", default="outputs/.osm_cache",
                        help="GraphML / polygon cache directory (default: outputs/.osm_cache)")
    parser.add_argument("--street-name-overrides", default="crowd_service/street_overrides.json",
                        help="JSON file of {osm_name: friendly_name} rename map. "
                             "Default: crowd_service/street_overrides.json")
    # Per-polygon autotune
    parser.add_argument("--no-per-polygon-autotune", action="store_true",
                        help="Skip the per-polygon autotune step after global autotune")
    parser.add_argument("--min-tracks-per-polygon", type=int, default=30,
                        help="Polygons with fewer unique tracks than this use the global "
                             "fallback params (default: 30)")
    parser.add_argument("--per-polygon-n-iter", type=int, default=20,
                        help="Max random-search iters per polygon (default: 20)")
    parser.add_argument("--per-polygon-max-seconds", type=float, default=120.0,
                        help="Wall-clock budget per polygon (default: 120s)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    with open(in_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        tracks = data
    else:
        tracks = data.get("tracks", [])

    n_person = sum(1 for t in tracks if t.get("class_name") == "person")
    n_uav = sum(1 for t in tracks if t.get("class_name") == "UAV")
    n_unique_tids = len({t["track_id"] for t in tracks if t.get("class_name") == "person"})
    print(f"  {len(tracks)} total, {n_person} persons, {n_uav} UAV, "
          f"{n_unique_tids} unique person track_ids")

    # Auto FPS from UAV timestamps if not provided
    fps_hint = args.fps
    if fps_hint is None:
        uav_ts = [t for t in tracks if t.get("class_name") == "UAV"]
        if len(uav_ts) > 1 and uav_ts[0].get("timestamp") and uav_ts[-1].get("timestamp"):
            from datetime import datetime
            t0 = datetime.fromisoformat(uav_ts[0]["timestamp"].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(uav_ts[-1]["timestamp"].replace("Z", "+00:00"))
            dt = (t1 - t0).total_seconds()
            if dt > 0:
                fps_hint = (len(uav_ts) - 1) / dt
        if fps_hint is None:
            fps_hint = 24.0

    # ── Street partitioning pre-pass ─────────────────────────────────────
    street_index_map = None
    bbox = None
    if args.no_street_partition:
        print("\nStreet partitioning DISABLED (--no-street-partition)")
    elif not HAS_OSM:
        print(f"\nStreet partitioning unavailable: {_OSM_IMPORT_ERROR}\n"
              f"  Falling back to global clustering. "
              f"Install with: pip install -r crowd_service/requirements.txt")
    else:
        print("\nStreet partitioning (polygon-only, per-frame snap):")
        bbox = compute_bbox(tracks)

        # Auto-fetch OSM named pedestrian polygons (e.g. Central Square)
        osm_polys = fetch_named_pedestrian_polygons(bbox, cache_dir=args.osm_cache_dir)
        if osm_polys is not None and not osm_polys.empty:
            if osm_polys.crs is None or osm_polys.crs.to_epsg() != 4326:
                osm_polys = osm_polys.to_crs(epsg=4326)

        user_polys = load_user_polygons(args.user_polygons, bbox=bbox,
                                        cache_dir=args.osm_cache_dir)

        import geopandas as gpd
        import pandas as pd
        osm_names = set(osm_polys["name"].astype(str)) if not osm_polys.empty else set()
        user_names = set(user_polys["name"].astype(str)) if not user_polys.empty else set()
        collisions = osm_names & user_names
        if collisions:
            print(f"  name collision(s) — user entry wins: {sorted(collisions)}")
            osm_polys = osm_polys[~osm_polys["name"].astype(str).isin(collisions)]
        parts = [g for g in (osm_polys, user_polys) if g is not None and not g.empty]
        polys = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326") \
            if parts else gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")
        print(f"  merged polygons: {len(polys)} "
              f"(OSM={len(osm_polys)}, user={len(user_polys)})")

        overrides = {}
        if args.street_name_overrides:
            ov_path = Path(args.street_name_overrides)
            if ov_path.exists():
                with open(ov_path) as f:
                    overrides = json.load(f) or {}
                if overrides:
                    print(f"  Loaded {len(overrides)} name override(s) from {ov_path}")

        # ── Per-frame (per-row) snap ─────────────────────────────────────
        persons = [t for t in tracks if t.get("class_name") == "person"]
        print(f"  snapping {len(persons)} person-frames to polygons …")
        row_streets = snap_persons_polygon_only(persons, polys, name_overrides=overrides)
        seen_names = {s for (s, _) in row_streets if s is not None}
        street_index_map = {UNASSIGNED_KEY: 0}
        for i, name in enumerate(sorted(seen_names), start=1):
            street_index_map[name] = i

        unassigned_rows = 0
        for t, (s, _) in zip(persons, row_streets):
            key = s if s is not None else UNASSIGNED_KEY
            t["street_key"] = key
            t["street_index"] = street_index_map[key]
            t["street_dist_m"] = 0.0
            if s is None:
                unassigned_rows += 1

        # Per-track dominant street summary for reporting
        from collections import Counter
        per_tid_counter: dict[int, Counter] = defaultdict(Counter)
        for t in persons:
            per_tid_counter[t["track_id"]][t["street_key"]] += 1
        per_track_dominant = {tid: c.most_common(1)[0][0] for tid, c in per_tid_counter.items()}
        dom_counts = Counter(per_track_dominant.values())
        print(f"  snap: rows_in_polygons={len(persons) - unassigned_rows} / {len(persons)} "
              f"(unassigned rows={unassigned_rows})")
        print(f"  {len(street_index_map)} streets (incl. unassigned); top dominant per track:")
        for name, n in sorted(dom_counts.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {name:<40s}  {n:>4d} tracks (dominant)")
        if len(dom_counts) > 8:
            print(f"    … and {len(dom_counts) - 8} smaller buckets")

    partition_active = street_index_map is not None
    fixed = {"max_cluster_population": args.max_cluster_population}
    params_by_si: dict = {}  # populated when per-polygon autotune runs

    if args.no_autotune:
        print("\nAutotune disabled — using default parameters (overrides applied where given)")
        best_params = {
            "coherence_weight": 1.5,
            "min_cluster_size": 15,
            "min_samples": 3,
            "cluster_selection_epsilon": 0.5,
            "max_match_dist": 5.0,
            "ema_alpha": 0.7,
            "memory_frames": 30,
        }
        for k in list(best_params.keys()):
            v = getattr(args, k, None)
            if v is not None:
                best_params[k] = v
        print("  Params:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
        best_cost = None
        best_metrics = None
    else:
        print(f"\nAutotuning HDBSCAN (max {args.n_iter} iter / {args.max_seconds}s budget)")
        print("  Feature z-normalisation: per-frame (inside cluster_crowds_per_frame)")
        if partition_active:
            print(f"  Clustering runs per-street on {len(street_index_map)} buckets; "
                  f"cost uses unique_ids / n_streets")
        best_cost, best_params, best_metrics = autotune(
            tracks, args.n_iter, args.max_seconds, args.seed, fixed,
            street_index_map=street_index_map,
        )
        print(f"\nGlobal autotune best cost: {best_cost:.2f}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # ── Per-polygon autotune (with global fallback for thin polygons) ──
        if partition_active and not args.no_per_polygon_autotune:
            print(f"\nPer-polygon autotune (min_tracks={args.min_tracks_per_polygon}, "
                  f"n_iter={args.per_polygon_n_iter})")
            # Invalidate any clustering state left behind by autotune
            for t in tracks:
                if t.get("class_name") == "person":
                    t["crowd_id"] = None

            by_si = _group_by_street(tracks)
            for si, rows in by_si.items():
                street_name = next((k for k, v in street_index_map.items() if v == si),
                                   f"si={si}")
                n_unique = len({t["track_id"] for t in rows})
                if n_unique < args.min_tracks_per_polygon:
                    params_by_si[si] = best_params  # fallback
                    print(f"    [fallback] {street_name:<35s} tracks={n_unique:>4}  "
                          f"(using global params)")
                    continue
                res = autotune_one_polygon(
                    rows, args.per_polygon_n_iter,
                    max_seconds=args.per_polygon_max_seconds,
                    seed=args.seed + si,
                    fixed_params=fixed,
                    label=street_name,
                )
                if res is not None and res[1] is not None:
                    params_by_si[si] = res[1]
                else:
                    params_by_si[si] = best_params

    print("\nRunning final clustering with best params ...")
    if partition_active:
        n_streets_used = cluster_by_street(
            tracks, best_params,
            max_cluster_population=fixed.get("max_cluster_population"),
            street_index_map=street_index_map,
            params_by_si=params_by_si or None,
        )
        print(f"  Streets used: {n_streets_used}/{len(street_index_map)}")
    else:
        cluster_crowds_per_frame(
            tracks,
            min_cluster_size=int(best_params["min_cluster_size"]),
            min_samples=int(best_params["min_samples"]),
            coherence_weight=float(best_params["coherence_weight"]),
            cluster_selection_epsilon=float(best_params["cluster_selection_epsilon"]),
            max_match_dist=float(best_params["max_match_dist"]),
            ema_alpha=float(best_params["ema_alpha"]),
            memory_frames=int(best_params["memory_frames"]),
            max_cluster_population=fixed.get("max_cluster_population"),
        )

    final_metrics = compute_crowd_metrics(tracks)
    if partition_active:
        # Count non-empty street buckets for cost normalisation
        ne = len({t.get("street_index") for t in tracks
                  if t.get("class_name") == "person" and t.get("crowd_id") is not None})
        final_cost = _cost_street_aware(final_metrics, n_unique_tids, ne)
        cost_label = f"cost_street_aware={final_cost:.2f}"
    else:
        final_cost = crowd_composite_cost(final_metrics, n_unique_tids)
        cost_label = f"cost={final_cost:.2f}"
    print(f"  Final: switches={final_metrics['switches']}, "
          f"noise={final_metrics['noise_ratio']:.3f}, "
          f"unique_ids={final_metrics['unique_ids']}, "
          f"count_std={final_metrics['count_std']:.2f}, {cost_label}")

    polygon_stats = None
    if partition_active:
        # Recompute from stamped tracks
        si_to_dist = {}
        for t in tracks:
            if t.get("class_name") == "person":
                si = t.get("street_index")
                d = t.get("street_dist_m")
                if si is not None and si not in si_to_dist:
                    si_to_dist[si] = d
        # polygon hits have dist==0.0 by convention
        n_poly = sum(1 for si, d in si_to_dist.items() if d == 0.0)
        polygon_stats = {
            "polygon_street_count": int(n_poly),
            "total_street_count": len(street_index_map),
        }
    metadata = build_metadata(
        tracks, best_params, best_cost,
        best_metrics or final_metrics, fps_hint,
        street_index_map=street_index_map,
        bbox=bbox,
        osm_network_type=None,
        lateral_tol_m=None,
        polygon_stats=polygon_stats,
        params_by_si=params_by_si,
    )

    out = {"metadata": metadata, "tracks": tracks}
    print(f"\nWriting {out_path} ...")
    with open(out_path, "w") as f:
        json.dump(out, f)

    n_clustered = sum(1 for t in tracks
                      if t.get("class_name") == "person" and t.get("crowd_id") is not None)
    print(f"  {n_clustered}/{n_person} person detections clustered "
          f"({100 * n_clustered / max(n_person, 1):.1f}%)")


if __name__ == "__main__":
    main()
