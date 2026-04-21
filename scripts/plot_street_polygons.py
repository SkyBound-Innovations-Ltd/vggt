#!/usr/bin/env python3
"""
Diagnostic street-polygon map (polygon-only): overlay every polygon used by
the crowd-clustering snap (OSM-fetched named areas + user-supplied polygons),
and plot each track's median (lat, lon) coloured by its assigned street.

No edge buffers, no lateral tolerance — what you see is exactly what the
point-in-polygon assignment uses.

Usage:
    python scripts/plot_street_polygons.py \
        -i outputs/cc_run_20260407_200138_crowd/state_estimation.json \
        -o outputs/cc_run_20260407_200138_crowd/streets_debug.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "crowd_service"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import Point

from collections import Counter, defaultdict

from osm_streets import (
    UNASSIGNED_KEY,
    compute_bbox,
    fetch_named_pedestrian_polygons,
    load_user_polygons,
)


def palette(n):
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    if n <= len(base):
        return base[:n]
    extra = plt.get_cmap("hsv", n - len(base))
    return base + [extra(i) for i in range(n - len(base))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True,
                    help="state_estimation.json produced by run_crowd_clustering.py")
    ap.add_argument("-o", "--output", required=True, help="Output PNG path")
    ap.add_argument("--user-polygons", default="crowd_service/user_polygons.json")
    ap.add_argument("--osm-cache-dir", default="outputs/.osm_cache")
    ap.add_argument("--figsize", type=float, nargs=2, default=[14, 12])
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--point-size", type=float, default=18)
    args = ap.parse_args()

    print(f"Loading {args.input} …")
    with open(args.input) as f:
        data = json.load(f)
    tracks = data.get("tracks", [])
    meta = data.get("metadata") or {}
    cc_meta = meta.get("crowd_clustering") or {}
    street_index_map = cc_meta.get("street_index_map") or {}
    if not street_index_map:
        sys.exit("ERROR: state_estimation.json has no street_index_map")

    name_to_idx = {v: int(k) for k, v in street_index_map.items()} \
        if all(str(k).isdigit() for k in street_index_map) \
        else {k: int(v) for k, v in street_index_map.items()}
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    bbox = compute_bbox(tracks)

    # ── Pull polygons (OSM + user), same merge logic as the clustering script ─
    osm_polys = fetch_named_pedestrian_polygons(bbox, cache_dir=args.osm_cache_dir)
    if osm_polys is not None and not osm_polys.empty:
        if osm_polys.crs is None or osm_polys.crs.to_epsg() != 4326:
            osm_polys = osm_polys.to_crs(epsg=4326)
    user_polys = load_user_polygons(args.user_polygons, bbox=bbox,
                                    cache_dir=args.osm_cache_dir)

    osm_names = set(osm_polys["name"].astype(str)) if not osm_polys.empty else set()
    user_names = set(user_polys["name"].astype(str)) if not user_polys.empty else set()
    collisions = osm_names & user_names
    if collisions and not osm_polys.empty:
        osm_polys = osm_polys[~osm_polys["name"].astype(str).isin(collisions)]
    parts = [g for g in (osm_polys, user_polys) if g is not None and not g.empty]
    polys = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326") \
        if parts else gpd.GeoDataFrame({"name": [], "geometry": []}, crs="EPSG:4326")

    if polys.empty:
        sys.exit("ERROR: no polygons available (neither OSM-fetched nor user-supplied)")

    print(f"  {len(polys)} polygons on the map (OSM={len(osm_polys)}, user={len(user_polys)})")

    # ── Colours per street name (match idx ordering) ─────────────────────
    street_names = sorted(set(polys["name"].astype(str)))
    pal = palette(len(street_names))
    street_to_color = {name: pal[i] for i, name in enumerate(street_names)}
    street_to_color[UNASSIGNED_KEY] = (0.6, 0.6, 0.6)

    polys_wm = polys.to_crs(epsg=3857)

    DOMINANT_THRESH = 0.80  # residency cut-off for "cleanly in polygon"

    track_frames: dict[int, list[tuple]] = defaultdict(list)
    for t in tracks:
        if t.get("class_name") != "person":
            continue
        lat, lon = t.get("lat"), t.get("lon")
        si = t.get("street_index")
        if lat is None or lon is None or si is None:
            continue
        track_frames[t["track_id"]].append((float(lat), float(lon), int(si)))

    track_info: list[dict] = []
    for tid, frames_list in track_frames.items():
        n = len(frames_list)
        if n == 0:
            continue
        si_counter = Counter(si for _, _, si in frames_list)
        top = si_counter.most_common(2)
        dom_si, dom_n = top[0]
        frac_dom = dom_n / n
        if len(top) > 1:
            si_2nd, second_n = top[1]
            frac_2nd = second_n / n
        else:
            si_2nd, frac_2nd = None, 0.0

        in_dom = [(la, lo) for la, lo, si in frames_list if si == dom_si]
        if len(in_dom) >= 3:
            med_lat = float(np.median([la for la, _ in in_dom]))
            med_lon = float(np.median([lo for _, lo in in_dom]))
        else:
            med_lat = float(np.median([la for la, _, _ in frames_list]))
            med_lon = float(np.median([lo for _, lo, _ in frames_list]))

        track_info.append({
            "tid": tid,
            "dominant_si": int(dom_si),
            "frac_dom": frac_dom,
            "si_2nd": si_2nd,
            "frac_2nd": frac_2nd,
            "lat": med_lat,
            "lon": med_lon,
            "cross_segment": frac_dom < DOMINANT_THRESH,
        })

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    tids = [ti["tid"] for ti in track_info]
    lats = [ti["lat"] for ti in track_info]
    lons = [ti["lon"] for ti in track_info]
    idxs = [ti["dominant_si"] for ti in track_info]
    xs, ys = transformer.transform(lons, lats)

    # ── Residency QA report (stdout) ────────────────────────────────────
    n_tracks = len(track_info)
    clean = [ti for ti in track_info if not ti["cross_segment"]]
    cross = [ti for ti in track_info if ti["cross_segment"]]
    avg_max_res = float(np.mean([ti["frac_dom"] for ti in cross])) if cross else 0.0
    print("\n=== residency QA ===")
    print(f"  n_tracks                  = {n_tracks}")
    print(f"  cleanly_in_polygon (≥{int(DOMINANT_THRESH * 100)}%) = {len(clean)}")
    print(f"  cross_segment  (<{int(DOMINANT_THRESH * 100)}%)     = {len(cross)}")
    if cross:
        print(f"    avg_max_residency       = {avg_max_res:.2f}")
        print(f"    top mismatches (tid, dominant, 2nd, frac_dom, frac_2nd):")
        for ti in sorted(cross, key=lambda r: r["frac_dom"])[:15]:
            dom_name = idx_to_name.get(ti["dominant_si"], f"si={ti['dominant_si']}")
            snd_name = idx_to_name.get(ti["si_2nd"], "—") if ti["si_2nd"] is not None else "—"
            print(f"      tid={ti['tid']:<5} {dom_name[:22]:<22} "
                  f"({ti['frac_dom']:.2f}) → {snd_name[:22]:<22} ({ti['frac_2nd']:.2f})")

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Bounds from polygons OR tracks, whichever is wider, + padding
    all_x = np.concatenate([[b for b in polys_wm.total_bounds[[0, 2]]], xs])
    all_y = np.concatenate([[b for b in polys_wm.total_bounds[[1, 3]]], ys])
    minx, maxx = float(all_x.min()), float(all_x.max())
    miny, maxy = float(all_y.min()), float(all_y.max())
    pad = 0.05 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")

    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=17)
    except Exception as e:
        print(f"  warning: basemap fetch failed ({e}); drawing without basemap")
        ax.set_facecolor("#ececec")

    # Filled polygons + labels
    for _, row in polys_wm.iterrows():
        name = str(row["name"])
        geom = row.geometry
        color = street_to_color.get(name, (0.5, 0.5, 0.5))
        ps = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in ps:
            patch = mpatches.Polygon(
                np.array(p.exterior.coords),
                closed=True, facecolor=color, edgecolor="black",
                alpha=0.45, linewidth=1.2, zorder=3,
            )
            ax.add_patch(patch)
        c = geom.representative_point()
        ax.annotate(
            name, (c.x, c.y), fontsize=8.5, ha="center", va="center",
            color="black", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc=(1, 1, 1, 0.85), ec="black", lw=0.5),
        )

    point_colors = [street_to_color.get(idx_to_name.get(i, UNASSIGNED_KEY),
                                        (0.6, 0.6, 0.6)) for i in idxs]
    ax.scatter(xs, ys, c=point_colors, s=args.point_size,
               edgecolors="white", linewidths=0.6, zorder=7)

    # Legend
    counts = Counter(idxs)
    top = [(idx_to_name.get(i, UNASSIGNED_KEY), counts[i])
           for i, _ in counts.most_common(16)]
    handles = [
        mpatches.Patch(color=street_to_color.get(name, (0.5, 0.5, 0.5)),
                       alpha=0.9, label=f"{name}  ({n})")
        for name, n in top
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92,
                    title="Tracks by street (median position)")
    leg.get_frame().set_facecolor("white")

    ax.set_title("Polygon-only snap — OSM + user polygons, no edge buffers",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Web-Mercator X (m)")
    ax.set_ylabel("Web-Mercator Y (m)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    # Mismatch report
    polys_wm_idx = polys_wm.set_index("name")
    inside_any = 0
    inside_own = 0
    for x, y, idx in zip(xs, ys, idxs):
        pt = Point(x, y)
        assigned = idx_to_name.get(idx, UNASSIGNED_KEY)
        if assigned in polys_wm_idx.index:
            geom = polys_wm_idx.loc[assigned].geometry
            inside_own += int(geom.contains(pt))
        for _, row in polys_wm_idx.iterrows():
            if row.geometry.contains(pt):
                inside_any += 1
                break
    print(f"  tracks inside ANY polygon: {inside_any}/{len(xs)}")
    print(f"  tracks inside OWN polygon: {inside_own}/{len(xs)} "
          f"(= {100*inside_own/max(inside_any,1):.1f}% of those inside a polygon)")


if __name__ == "__main__":
    main()
