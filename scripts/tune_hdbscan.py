#!/usr/bin/env python3
"""
Automated HDBSCAN parameter tuning for crowd clustering.

Loads a state_estimation.json, re-runs only cluster_crowds_per_frame()
with different parameter combinations, and scores each by proxy metrics.

No ground-truth crowd labels needed — uses temporal stability proxies.

Usage:
    python scripts/tune_hdbscan.py \
        -i outputs/sim_260306_.../state_estimation.json \
        --method random --n-iter 200 --top 20
"""

import argparse
import json
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_v2_state_est import cluster_crowds_per_frame


# ── Parameter search space ────────────────────────────────────────────

PARAM_GRID = {
    "coherence_weight": [5, 10, 15, 20],
    "min_cluster_size": [5, 8, 10, 15],
    "min_samples": [2, 3, 5],
    "cluster_selection_epsilon": [1, 2, 3, 5],
    "max_match_dist": [15, 20, 30, 50],
    "ema_alpha": [0.2, 0.3, 0.4, 0.5, 0.7],
    "memory_frames": [5, 10, 15, 20, 30],
}


# ── Proxy metrics ─────────────────────────────────────────────────────

def compute_metrics(tracks):
    """Compute temporal stability proxy metrics from clustered tracks."""
    # Build per-track crowd_id history
    track_history = defaultdict(list)  # track_id -> [(frame_id, crowd_id)]
    person_count = 0

    for t in tracks:
        if t.get("class_name") != "person":
            continue
        person_count += 1
        tid = t["track_id"]
        fid = t["frame_id"]
        cid = t.get("crowd_id")
        track_history[tid].append((fid, cid))

    if person_count == 0:
        return {"switches": 0, "noise_ratio": 1.0, "unique_ids": 0, "count_std": 0.0}

    # Sort each track by frame_id
    for tid in track_history:
        track_history[tid].sort(key=lambda x: x[0])

    # Switches: total crowd_id changes across all tracks
    switches = 0
    for tid, history in track_history.items():
        prev_cid = None
        for _, cid in history:
            if cid is not None and prev_cid is not None and cid != prev_cid:
                switches += 1
            if cid is not None:
                prev_cid = cid

    # Noise ratio: fraction of person detections with crowd_id=None
    noise_count = sum(1 for t in tracks if t.get("class_name") == "person" and t.get("crowd_id") is None)
    noise_ratio = noise_count / person_count

    # Unique IDs
    unique_ids = len(set(
        t["crowd_id"] for t in tracks
        if t.get("class_name") == "person" and t.get("crowd_id") is not None
    ))

    # Count std: std of crowds-per-frame
    frame_crowds = defaultdict(set)
    for t in tracks:
        if t.get("class_name") == "person" and t.get("crowd_id") is not None:
            frame_crowds[t["frame_id"]].add(t["crowd_id"])
    if frame_crowds:
        counts = [len(v) for v in frame_crowds.values()]
        count_std = float(np.std(counts))
    else:
        count_std = 0.0

    return {
        "switches": switches,
        "noise_ratio": noise_ratio,
        "unique_ids": unique_ids,
        "count_std": count_std,
    }


def composite_cost(metrics, n_tracks):
    """Compute composite cost from proxy metrics."""
    return (
        (metrics["switches"] / max(n_tracks, 1)) * 1.0
        + metrics["noise_ratio"] * 50.0
        + metrics["unique_ids"] * 0.5
        + metrics["count_std"] * 2.0
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tune HDBSCAN crowd clustering parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Path to state_estimation.json")
    parser.add_argument("--method", choices=["random", "grid"], default="random",
                        help="Search method (default: random)")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="Number of random iterations (default: 200, ignored for grid)")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N results (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)

    tracks = data["tracks"]
    n_persons = sum(1 for t in tracks if t.get("class_name") == "person")
    n_unique_tracks = len(set(t["track_id"] for t in tracks if t.get("class_name") == "person"))
    print(f"  {len(tracks)} total detections, {n_persons} person detections, {n_unique_tracks} unique tracks")

    # Build parameter combinations
    keys = list(PARAM_GRID.keys())
    if args.method == "grid":
        combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
        print(f"Grid search: {len(combos)} combinations")
    else:
        rng = np.random.RandomState(args.seed)
        combos = []
        for _ in range(args.n_iter):
            combo = tuple(rng.choice(PARAM_GRID[k]) for k in keys)
            combos.append(combo)
        print(f"Random search: {args.n_iter} iterations (seed={args.seed})")

    # Compute baseline metrics
    baseline_metrics = compute_metrics(tracks)
    baseline_cost = composite_cost(baseline_metrics, n_unique_tracks)
    print(f"\nBaseline (current params):")
    print(f"  switches={baseline_metrics['switches']}, noise={baseline_metrics['noise_ratio']:.3f}, "
          f"unique_ids={baseline_metrics['unique_ids']}, count_std={baseline_metrics['count_std']:.2f}, "
          f"cost={baseline_cost:.2f}")

    # Run search
    results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # Reset crowd_id for all person tracks
        for t in tracks:
            if t.get("class_name") == "person":
                t["crowd_id"] = None

        # Run clustering
        cluster_crowds_per_frame(
            tracks,
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
            coherence_weight=float(params["coherence_weight"]),
            cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
            max_match_dist=float(params["max_match_dist"]),
            ema_alpha=float(params["ema_alpha"]),
            memory_frames=int(params["memory_frames"]),
        )

        metrics = compute_metrics(tracks)
        cost = composite_cost(metrics, n_unique_tracks)
        results.append((cost, params, metrics))

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(combos) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(combos)}] best_cost={min(r[0] for r in results):.2f}  "
                  f"({rate:.1f} iter/s, ETA {eta:.0f}s)")

    # Sort by cost
    results.sort(key=lambda x: x[0])

    elapsed = time.time() - t0
    print(f"\nSearch complete in {elapsed:.1f}s")

    # Print top results
    print(f"\n{'='*100}")
    print(f"Top {args.top} parameter combinations (lower cost = better):")
    print(f"{'='*100}")
    print(f"{'Rank':>4}  {'Cost':>8}  {'Sw':>5}  {'Noise':>6}  {'IDs':>4}  {'Std':>5}  | "
          f"{'cw':>3} {'mcs':>3} {'ms':>2} {'eps':>3} {'mmd':>3} {'ema':>4} {'mem':>3}")
    print(f"{'-'*100}")

    for rank, (cost, params, metrics) in enumerate(results[:args.top], 1):
        print(f"{rank:>4}  {cost:>8.2f}  {metrics['switches']:>5}  {metrics['noise_ratio']:>6.3f}  "
              f"{metrics['unique_ids']:>4}  {metrics['count_std']:>5.2f}  | "
              f"{params['coherence_weight']:>3.0f} {params['min_cluster_size']:>3.0f} "
              f"{params['min_samples']:>2.0f} {params['cluster_selection_epsilon']:>3.0f} "
              f"{params['max_match_dist']:>3.0f} {params['ema_alpha']:>4.1f} "
              f"{params['memory_frames']:>3.0f}")

    # Print best vs baseline
    best_cost, best_params, best_metrics = results[0]
    print(f"\n{'='*100}")
    print(f"Best vs Baseline:")
    print(f"  switches:   {baseline_metrics['switches']:>6} -> {best_metrics['switches']:>6}  "
          f"({best_metrics['switches'] - baseline_metrics['switches']:+d})")
    print(f"  noise:      {baseline_metrics['noise_ratio']:>6.3f} -> {best_metrics['noise_ratio']:>6.3f}  "
          f"({best_metrics['noise_ratio'] - baseline_metrics['noise_ratio']:+.3f})")
    print(f"  unique_ids: {baseline_metrics['unique_ids']:>6} -> {best_metrics['unique_ids']:>6}  "
          f"({best_metrics['unique_ids'] - baseline_metrics['unique_ids']:+d})")
    print(f"  count_std:  {baseline_metrics['count_std']:>6.2f} -> {best_metrics['count_std']:>6.2f}  "
          f"({best_metrics['count_std'] - baseline_metrics['count_std']:+.2f})")
    print(f"  cost:       {baseline_cost:>6.2f} -> {best_cost:>6.2f}  "
          f"({best_cost - baseline_cost:+.2f})")
    print(f"\nBest params:")
    for k, v in best_params.items():
        print(f"  --hdbscan-{k.replace('_', '-')}={v}")


if __name__ == "__main__":
    main()
