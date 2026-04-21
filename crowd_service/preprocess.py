#!/usr/bin/env python3
"""
Convert a state_estimation.json (VGGT output with crowd_id / crowd_density)
into a compact, frame-bucketed JSON for the browser viewer.

Input  : one track entry per (frame_id, track_id), all rows in a flat list.
Output : { metadata, frames: [ { frame_id, uav: {...}, persons: [...] } ] }
         indexed by frame_id for O(1) seeking in the viewer.

Also writes a gzipped variant so `python -m http.server` can serve it if you
manually rename it to frames.json and the browser decompresses transparently.

Usage:
    python crowd_service/preprocess.py \
        -i outputs/cc_run_20260407_200138_crowd/state_estimation.json \
        -o crowd_service/frames.json
"""

import argparse
import gzip
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def sanitize(obj):
    """Recursively replace NaN/Infinity with None so json.dump output
    passes JSON.parse in the browser."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def stabilise_subcluster_ids(
    tracks,
    *,
    max_match_dist_m: float = 40.0,
    memory_frames: int = 30,
    ema_alpha: float = 0.4,
):
    """Re-number crowd_id values so that spatially-stable sub-clusters keep
    the same id across frames.

    Per-frame k-means splits (inside cluster_crowds_per_frame's Phase 4)
    assign fresh sub-cluster ids each frame. The upstream Hungarian match
    only operates at the whole-crowd level, so sub-cluster identity gets
    reset every frame → a visible "stream" of id flips on long streets.

    This runs AFTER clustering, on the stamped state_estimation.json, and
    does a second Hungarian match — this time on SUB-cluster centroids
    within each street. Matched sub-clusters inherit the canonical id;
    unmatched ones keep their own and enter the active pool. Absent ones
    age out after `memory_frames` so re-emerging crowds re-grab their id.

    Args:
        max_match_dist_m: reject a Hungarian assignment whose centroid
            distance exceeds this (metres).
        memory_frames: frames to remember an absent canonical id.
        ema_alpha: EMA smoothing on the canonical centroid.

    Returns count of crowd_id values rewritten.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict as _dd

    # Group person rows by (street_index, frame_id, crowd_id)
    groups: dict = _dd(lambda: _dd(lambda: _dd(list)))  # si -> fid -> cid -> [row_idx]
    for idx, t in enumerate(tracks):
        if t.get("class_name") != "person":
            continue
        si = t.get("street_index")
        fid = t.get("frame_id")
        cid = t.get("crowd_id")
        if si is None or fid is None or cid is None:
            continue
        groups[si][fid][cid].append(idx)

    # Cardiff-latitude metres per degree (rough)
    M_PER_DEG_LAT = 111_111.0
    m_per_deg_lon = 111_111.0 * np.cos(np.deg2rad(51.476))

    changed = 0

    for si, frames_map in groups.items():
        frame_ids = sorted(frames_map.keys())
        # canonical_id -> {centroid: (lat, lon), age: int}
        canon: dict = {}

        for fid in frame_ids:
            cid_to_rows = frames_map[fid]
            # compute current-frame centroids per crowd_id
            cur: dict = {}  # cid -> (lat, lon)
            for cid, row_idxs in cid_to_rows.items():
                lats = [tracks[i]["lat"] for i in row_idxs]
                lons = [tracks[i]["lon"] for i in row_idxs]
                cur[cid] = (float(np.mean(lats)), float(np.mean(lons)))

            cur_cids = list(cur.keys())
            cand = [(ck, c["centroid"]) for ck, c in canon.items() if c["age"] < memory_frames]

            matched_cur: set = set()
            matched_cand: set = set()
            if cand and cur_cids:
                n_cand = len(cand)
                n_cur = len(cur_cids)
                cost = np.empty((n_cand, n_cur), dtype=np.float64)
                for i, (_, pc) in enumerate(cand):
                    for j, cc in enumerate(cur_cids):
                        lat1, lon1 = pc
                        lat2, lon2 = cur[cc]
                        dlat = (lat1 - lat2) * M_PER_DEG_LAT
                        dlon = (lon1 - lon2) * m_per_deg_lon
                        cost[i, j] = (dlat * dlat + dlon * dlon) ** 0.5
                row_ind, col_ind = linear_sum_assignment(cost)
                for ri, ci in zip(row_ind, col_ind):
                    if cost[ri, ci] > max_match_dist_m:
                        continue
                    canonical_id = cand[ri][0]
                    cur_id = cur_cids[ci]
                    matched_cand.add(ri)
                    matched_cur.add(ci)
                    if cur_id != canonical_id:
                        for row_idx in cid_to_rows[cur_id]:
                            tracks[row_idx]["crowd_id"] = canonical_id
                            changed += 1
                    # EMA update canonical centroid
                    old_c = canon[canonical_id]["centroid"]
                    new_c = cur[cur_id]
                    canon[canonical_id]["centroid"] = (
                        ema_alpha * new_c[0] + (1 - ema_alpha) * old_c[0],
                        ema_alpha * new_c[1] + (1 - ema_alpha) * old_c[1],
                    )
                    canon[canonical_id]["age"] = 0

            # Unmatched current-frame crowds join the canonical pool as-is
            for j, cid in enumerate(cur_cids):
                if j in matched_cur:
                    continue
                canon[cid] = {"centroid": cur[cid], "age": 0}

            # Age unmatched canonicals; prune stale
            for ck in list(canon.keys()):
                hit = any(canonical_id == ck for _, canonical_id in
                          ((ri, cand[ri][0]) for ri in matched_cand))
                if not hit:
                    canon[ck]["age"] += 1
                    if canon[ck]["age"] >= memory_frames:
                        del canon[ck]

    return changed


def smooth_crowd_ids(tracks, window):
    """Segment-aware per-track sliding-window mode filter on `crowd_id`.

    Splits each track's timeline into runs of constant `street_index`, then
    applies the mode filter INDEPENDENTLY within each run. Never smooths a
    crowd_id across a segment boundary, so a genuine A→B polygon crossing is
    preserved (crowd_id flips exactly at the boundary frame).

    Returns (n_changed, n_transitions_preserved):
      n_changed                = crowd_id values rewritten by in-segment smoothing
      n_transitions_preserved  = count of street_index changes across all tracks
                                 that were NOT masked by cross-run smoothing
                                 (trivially all of them now that the filter is
                                 segment-aware — reported for visibility).
    """
    if window <= 1:
        return 0, 0

    from itertools import groupby

    timelines: dict[int, list[int]] = defaultdict(list)
    for idx, t in enumerate(tracks):
        if t.get("class_name") == "person":
            timelines[t["track_id"]].append(idx)

    half = window // 2
    changed = 0
    transitions_preserved = 0

    for tid, indices in timelines.items():
        indices.sort(key=lambda i: tracks[i]["frame_id"])
        si_seq = [tracks[i].get("street_index") for i in indices]
        cids = [tracks[i].get("crowd_id") for i in indices]

        pos = 0
        runs: list[tuple[int, int]] = []
        for _, group in groupby(si_seq):
            run_len = sum(1 for _ in group)
            runs.append((pos, pos + run_len))
            pos += run_len
        transitions_preserved += max(0, len(runs) - 1)

        for run_s, run_e in runs:
            for j in range(run_s, run_e):
                lo = max(run_s, j - half)
                hi = min(run_e, j + half + 1)
                non_none = [c for c in cids[lo:hi] if c is not None]
                if not non_none:
                    new = None
                else:
                    counts = Counter(non_none).most_common()
                    top = counts[0][1]
                    tied = {c for c, k in counts if k == top}
                    new = cids[j] if cids[j] in tied else counts[0][0]
                if new != cids[j]:
                    changed += 1
                    tracks[indices[j]]["crowd_id"] = new

    return changed, transitions_preserved


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True,
                   help="state_estimation.json with pos_ned / vel_ned / crowd_id")
    p.add_argument("-o", "--output", default="crowd_service/frames.json",
                   help="output path (default: crowd_service/frames.json)")
    p.add_argument("--round", type=int, default=6,
                   help="lat/lon decimal digits (default: 6 ≈ 11 cm)")
    p.add_argument("--gzip", action="store_true",
                   help="also emit a .gz copy alongside the output")
    p.add_argument("--smooth-window", type=int, default=15,
                   help="Per-track sliding mode filter window on crowd_id "
                        "(default: 15 ≈ 0.65s at 23fps; 0 = disabled)")
    p.add_argument("--no-stabilise", action="store_true",
                   help="Skip the sub-cluster id stabiliser (restores raw "
                        "cluster_crowds_per_frame output)")
    p.add_argument("--stabilise-max-match-m", type=float, default=40.0,
                   help="Max centroid-distance (m) for a sub-cluster match "
                        "across frames (default: 40)")
    p.add_argument("--stabilise-memory-frames", type=int, default=30,
                   help="Frames to remember absent sub-clusters before pruning "
                        "(default: 30)")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ({in_path.stat().st_size/1e6:.1f} MB)…")
    with open(in_path) as f:
        data = json.load(f)

    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    tracks = data.get("tracks", data) if isinstance(data, dict) else data

    if not args.no_stabilise:
        print(f"Stabilising sub-cluster ids "
              f"(max_match={args.stabilise_max_match_m}m, "
              f"memory={args.stabilise_memory_frames}f)…")
        stab_changed = stabilise_subcluster_ids(
            tracks,
            max_match_dist_m=args.stabilise_max_match_m,
            memory_frames=args.stabilise_memory_frames,
        )
        print(f"  sub-cluster ids re-matched : {stab_changed} rewrites")

    if args.smooth_window and args.smooth_window > 1:
        print(f"Applying segment-aware per-track mode filter, window={args.smooth_window}…")
        changed, transitions_preserved = smooth_crowd_ids(tracks, args.smooth_window)
        total_person = sum(1 for t in tracks if t.get("class_name") == "person")
        pct = 100 * changed / max(total_person, 1)
        print(f"  crowd_ids changed (within-segment mode filter) : {changed}/{total_person} "
              f"({pct:.1f}%)")
        print(f"  segment transitions preserved                   : {transitions_preserved}")

    # Compact crowd_id remap: collect all distinct non-null ids in order of
    # first appearance (sorted by frame_id), renumber to 1..N. Makes the
    # viewer tooltip show "Crowd 7" instead of "Crowd 70001". Street name is
    # still available via the per-person `si` field.
    cid_order: list[int] = []
    seen_cids: set = set()
    ordered_rows = sorted(
        (t for t in tracks if t.get("class_name") == "person"),
        key=lambda t: (t.get("frame_id") or 0),
    )
    for t in ordered_rows:
        cid = t.get("crowd_id")
        if cid is not None and cid not in seen_cids:
            seen_cids.add(cid)
            cid_order.append(cid)
    cid_remap = {old: new + 1 for new, old in enumerate(cid_order)}
    for t in tracks:
        if t.get("class_name") == "person" and t.get("crowd_id") is not None:
            t["crowd_id"] = cid_remap[t["crowd_id"]]
    print(f"  crowd_id compacted : {len(cid_remap)} distinct ids → 1..{len(cid_remap)}")

    frames: dict[int, dict] = {}
    max_frame = -1

    def _round(v, nd):
        if v is None:
            return None
        try:
            return round(float(v), nd)
        except (TypeError, ValueError):
            return None

    for t in tracks:
        fid = t.get("frame_id")
        if fid is None:
            continue
        max_frame = max(max_frame, fid)
        bucket = frames.setdefault(fid, {"uav": None, "persons": []})

        if t.get("class_name") == "UAV":
            bucket["uav"] = {
                "lat": _round(t.get("lat"), args.round),
                "lon": _round(t.get("lon"), args.round),
                "alt": _round(t.get("alt"), 2),
                "heading": _round(t.get("heading_deg"), 1),
                "speed_mph": _round(t.get("speed_mph"), 1),
            }
        else:
            vel = t.get("vel_ned") or [0.0, 0.0, 0.0]
            person = {
                "tid": t.get("track_id"),
                "lat": _round(t.get("lat"), args.round),
                "lon": _round(t.get("lon"), args.round),
                "cid": t.get("crowd_id"),
                "cd": _round(t.get("crowd_density"), 4),
                "vn": _round(vel[0] if len(vel) > 0 else 0.0, 3),
                "ve": _round(vel[1] if len(vel) > 1 else 0.0, 3),
            }
            if t.get("street_index") is not None:
                person["si"] = int(t["street_index"])
            bucket["persons"].append(person)

    n_frames = max_frame + 1 if max_frame >= 0 else 0
    frame_list = [frames.get(i, {"uav": None, "persons": []}) for i in range(n_frames)]

    crowd_meta = metadata.get("crowd_clustering", {}) or {}
    out = {
        "metadata": {
            "fps": metadata.get("fps", 24),
            "total_frames": n_frames,
            "home": metadata.get("home_position"),
            "crowd_clustering": crowd_meta,
            "street_index_map": crowd_meta.get("street_index_map"),
            # crowd_id values are compacted 1..N in this output; the viewer
            # reads `si` directly for street name, no stride decode needed.
            "id_stride": None,
        },
        "frames": frame_list,
    }

    n_persons = sum(len(f["persons"]) for f in frame_list)
    n_unique_crowds = len({p["cid"] for f in frame_list for p in f["persons"] if p["cid"] is not None})
    print(f"  frames: {n_frames}  persons: {n_persons}  unique crowds: {n_unique_crowds}")

    print(f"Writing {out_path}…")
    with open(out_path, "w") as f:
        json.dump(sanitize(out), f, separators=(",", ":"), allow_nan=False)
    size = out_path.stat().st_size
    print(f"  JSON  : {size/1e6:.1f} MB")

    if args.gzip:
        gz_path = Path(str(out_path) + ".gz")
        with open(out_path, "rb") as fin, gzip.open(gz_path, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
        print(f"  GZIP  : {gz_path.stat().st_size/1e6:.1f} MB → {gz_path}")


if __name__ == "__main__":
    main()
