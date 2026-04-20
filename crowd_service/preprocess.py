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
        n = len(cids)

        # Split indices into contiguous runs of constant street_index
        runs: list[tuple[int, int]] = []   # [(start, end_exclusive), ...]
        run_start = 0
        for j in range(1, n):
            if si_seq[j] != si_seq[j - 1]:
                runs.append((run_start, j))
                transitions_preserved += 1
                run_start = j
        runs.append((run_start, n))

        # Mode filter WITHIN each run only
        for run_s, run_e in runs:
            run_len = run_e - run_s
            for j in range(run_s, run_e):
                lo = max(run_s, j - half)
                hi = min(run_e, j + half + 1)
                window_cids = cids[lo:hi]
                non_none = [c for c in window_cids if c is not None]
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
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ({in_path.stat().st_size/1e6:.1f} MB)…")
    with open(in_path) as f:
        data = json.load(f)

    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    tracks = data.get("tracks", data) if isinstance(data, dict) else data

    if args.smooth_window and args.smooth_window > 1:
        print(f"Applying segment-aware per-track mode filter, window={args.smooth_window}…")
        changed, transitions_preserved = smooth_crowd_ids(tracks, args.smooth_window)
        total_person = sum(1 for t in tracks if t.get("class_name") == "person")
        pct = 100 * changed / max(total_person, 1)
        print(f"  crowd_ids changed (within-segment mode filter) : {changed}/{total_person} "
              f"({pct:.1f}%)")
        print(f"  segment transitions preserved                   : {transitions_preserved}")

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
            "id_stride": (crowd_meta.get("street_partitioning") or {}).get("id_stride"),
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
