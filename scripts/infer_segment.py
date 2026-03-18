#!/usr/bin/env python3
"""Standalone single-segment VGGT inference — designed to run as a subprocess.

Loads the model, runs inference on one segment directory, saves .npy outputs,
then exits. Process termination fully reclaims all GPU memory (no fragmentation).

Inference is done in mini-batches (default 20 frames) to avoid OOM from VGGT's
global attention layer, which is O((S*P)^2) in memory.

Usage:
    python infer_segment.py --seg-dir /path/to/segments/seg00000
    python infer_segment.py --seg-dir /path/to/segments/seg00000 --batch-size 10
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def main():
    parser = argparse.ArgumentParser(description="VGGT single-segment inference (subprocess)")
    parser.add_argument("--seg-dir", type=str, required=True,
                        help="Absolute path to segment directory (e.g. .../segments/seg00000)")
    parser.add_argument("--model-name", type=str, default="facebook/VGGT-1B",
                        help="HuggingFace model ID (default: facebook/VGGT-1B)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Frames per inference batch (default: 20)")
    parser.add_argument("--save-world-points", action="store_true",
                        help="Also save world_points.npy")
    args = parser.parse_args()

    seg_dir = args.seg_dir
    seg_name = os.path.basename(seg_dir)
    frame_dir = os.path.join(seg_dir, "frames")

    # Discover frame PNGs
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if len(frame_paths) == 0:
        print(f"ERROR: No PNGs found in {frame_dir}", file=sys.stderr)
        sys.exit(1)

    num_frames = len(frame_paths)
    print(f"  [{seg_name}] {num_frames} frames, batch_size={args.batch_size}")

    # Device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    # Load model
    tic = time.time()
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()
    print(f"  [{seg_name}] Model loaded in {time.time() - tic:.1f}s")

    # Preprocess all frames (kept on CPU to save GPU memory)
    seg_images = load_and_preprocess_images(frame_paths)
    print(f"  [{seg_name}] Preprocessed: {seg_images.shape}")

    # Run inference in mini-batches (only batch slice moves to GPU)
    all_depths = []
    all_pose_encs = []
    all_world_points = []
    batch_size = args.batch_size
    num_batches = (num_frames + batch_size - 1) // batch_size

    tic = time.time()
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_frames)
        batch_images = seg_images[start:end].to(device)  # only batch on GPU

        print(f"    [{seg_name}] Batch {batch_idx + 1}/{num_batches} "
              f"(frames {start}-{end - 1})")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(batch_images)

        all_depths.append(preds["depth"].cpu())
        all_pose_encs.append(preds["pose_enc"].cpu())

        if args.save_world_points and "world_points" in preds:
            all_world_points.append(preds["world_points"].cpu())

        del preds
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"  [{seg_name}] Inference done in {time.time() - tic:.1f}s")

    # Concatenate batch results along frame dim (each is [1, B, ...])
    depth_seg = torch.cat(all_depths, dim=1).squeeze(0).numpy().squeeze(-1)  # [S,H,W]
    pose_enc_seg = torch.cat(all_pose_encs, dim=1).squeeze(0).numpy()        # [S,9]

    np.save(os.path.join(seg_dir, "depth.npy"), depth_seg)
    np.save(os.path.join(seg_dir, "pose_enc.npy"), pose_enc_seg)

    if all_world_points:
        wp = torch.cat(all_world_points, dim=1).squeeze(0).numpy()
        np.save(os.path.join(seg_dir, "world_points.npy"), wp)

    print(f"  [{seg_name}] Saved depth.npy {depth_seg.shape}, "
          f"pose_enc.npy {pose_enc_seg.shape}")


if __name__ == "__main__":
    main()
