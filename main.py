import torch
import cv2
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def extract_frames_from_video(video_path, fps=1):
    """Extract frames from video at specified fps and return list of frame arrays."""
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {video_fps}, Total frames: {total_frames}")
    print(f"Extracting 1 frame every {frame_interval} frames (target: {fps} fps)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def save_frames_and_load(frames, tmp_dir):
    """Save frames to temp directory and load with VGGT preprocessing."""
    os.makedirs(tmp_dir, exist_ok=True)
    frame_paths = []

    for i, frame in enumerate(frames):
        frame_path = os.path.join(tmp_dir, f"{i:06d}.png")
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)

    images = load_and_preprocess_images(frame_paths)
    return images


def save_depth_maps(depth_maps, output_dir):
    """Save depth maps as both .npy and visualization .png files."""
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    for i, depth in enumerate(depth_maps):
        depth_np = depth.cpu().numpy().squeeze()  # [H, W]

        # Save raw depth as numpy
        np.save(os.path.join(depth_dir, f"depth_{i:04d}.npy"), depth_np)

        # Save normalized visualization
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        depth_img = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_img).save(os.path.join(depth_dir, f"depth_{i:04d}.png"))

    print(f"Saved {len(depth_maps)} depth maps to {depth_dir}/")


def save_point_maps(point_maps, output_dir):
    """Save world point maps as .npy files."""
    points_dir = os.path.join(output_dir, "points")
    os.makedirs(points_dir, exist_ok=True)

    for i, points in enumerate(point_maps):
        points_np = points.cpu().numpy()  # [H, W, 3]
        np.save(os.path.join(points_dir, f"points_{i:04d}.npy"), points_np)

    print(f"Saved {len(point_maps)} point maps to {points_dir}/")


def save_cameras(predictions, output_dir):
    """Save camera intrinsics and extrinsics."""
    camera_dir = os.path.join(output_dir, "cameras")
    os.makedirs(camera_dir, exist_ok=True)

    if "extrinsic" in predictions:
        extrinsics = predictions["extrinsic"].cpu().numpy()
        np.save(os.path.join(camera_dir, "extrinsics.npy"), extrinsics)
        print(f"Saved extrinsics: {extrinsics.shape}")

    if "intrinsic" in predictions:
        intrinsics = predictions["intrinsic"].cpu().numpy()
        np.save(os.path.join(camera_dir, "intrinsics.npy"), intrinsics)
        print(f"Saved intrinsics: {intrinsics.shape}")


def create_depth_colormap(depth_np, colormap='turbo'):
    """Convert depth map to colored visualization using matplotlib colormap."""
    # Normalize depth to [0, 1]
    depth_min, depth_max = depth_np.min(), depth_np.max()
    depth_normalized = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha channel
    depth_colored = (depth_colored * 255).astype(np.uint8)

    return depth_colored, depth_min, depth_max


def create_colorbar(height, width=120, depth_min=0, depth_max=1, colormap='turbo'):
    """Create a vertical colorbar image with depth range labels."""
    # Create figure with colorbar and labels - wider figure for readable text
    fig_height = max(6, height / 60)
    fig = plt.figure(figsize=(2.5, fig_height))
    fig.patch.set_facecolor('black')

    # Create axes for colorbar with more space for labels
    ax = fig.add_axes([0.1, 0.08, 0.25, 0.84])  # [left, bottom, width, height]

    # Create colorbar
    norm = plt.Normalize(vmin=depth_min, vmax=depth_max)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=16, width=2, length=6)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(2)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=16)
    cbar.set_label('Depth', color='white', fontsize=18, labelpad=15)

    # Convert figure to image using buffer_rgba
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    colorbar_img = np.asarray(buf)[:, :, :3]  # Remove alpha channel
    plt.close(fig)

    # Resize to match target height
    colorbar_img = cv2.resize(colorbar_img, (width, height))

    return colorbar_img


def create_overlay_colorbar(height, width=40, depth_min=0, depth_max=1, colormap='turbo', margin=10):
    """Create a vertical colorbar image for overlay with transparent-friendly design."""
    # Create gradient
    gradient = np.linspace(1, 0, height - 2 * margin).reshape(-1, 1)
    gradient = np.repeat(gradient, width - 10, axis=1)

    # Apply colormap
    cmap = plt.colormaps.get_cmap(colormap)
    colorbar = cmap(gradient)[:, :, :3]
    colorbar = (colorbar * 255).astype(np.uint8)

    # Create background with padding
    bg = np.zeros((height, width + 60, 3), dtype=np.uint8)
    bg[:] = (0, 0, 0)  # Black background

    # Place colorbar
    bg[margin:margin + colorbar.shape[0], 5:5 + colorbar.shape[1]] = colorbar

    # Add border to colorbar
    cv2.rectangle(bg, (5, margin), (5 + colorbar.shape[1], margin + colorbar.shape[0]), (255, 255, 255), 1)

    # Add tick labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Top label (max depth)
    max_text = f"{depth_max:.1f}"
    cv2.putText(bg, max_text, (width - 5, margin + 15), font, font_scale, (255, 255, 255), thickness)

    # Middle label
    mid_val = (depth_min + depth_max) / 2
    mid_text = f"{mid_val:.1f}"
    cv2.putText(bg, mid_text, (width - 5, height // 2 + 5), font, font_scale, (255, 255, 255), thickness)

    # Bottom label (min depth)
    min_text = f"{depth_min:.1f}"
    cv2.putText(bg, min_text, (width - 5, height - margin), font, font_scale, (255, 255, 255), thickness)

    return bg


def create_side_by_side_frame(original_frame, depth_map, colormap='turbo', add_colorbar=True, overlay_colorbar=True):
    """Create a side-by-side visualization of original frame and depth map."""
    h, w = original_frame.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]

    # Resize depth to match original frame size
    if (depth_h, depth_w) != (h, w):
        depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_map

    # Create colored depth visualization
    depth_colored, depth_min, depth_max = create_depth_colormap(depth_resized, colormap)

    # Overlay colorbar on depth map
    if add_colorbar and overlay_colorbar:
        colorbar = create_overlay_colorbar(h, width=40, depth_min=depth_min, depth_max=depth_max, colormap=colormap)
        # Overlay on right side of depth image
        cb_h, cb_w = colorbar.shape[:2]
        x_offset = w - cb_w - 10  # 10px from right edge
        y_offset = 0

        # Blend colorbar onto depth (semi-transparent background)
        roi = depth_colored[y_offset:y_offset + cb_h, x_offset:x_offset + cb_w]
        mask = (colorbar > 0).any(axis=2)
        alpha = 0.8
        roi[mask] = (alpha * colorbar[mask] + (1 - alpha) * roi[mask]).astype(np.uint8)

    # Create side-by-side image
    combined = np.hstack([original_frame, depth_colored])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Depth', (w + 10, 30), font, 1, (255, 255, 255), 2)

    return combined


def encode_depth_video(frames, depth_maps, output_path, fps=10, colormap='turbo'):
    """Encode original frames and depth maps into a side-by-side video with colorbar."""
    if len(frames) == 0 or len(depth_maps) == 0:
        raise ValueError("No frames or depth maps to encode")

    # Get dimensions from first combined frame
    first_combined = create_side_by_side_frame(frames[0], depth_maps[0], colormap)
    h, w = first_combined.shape[:2]

    # Initialize video writer with H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Fallback to mp4v if H.264 not available
    if not out.isOpened():
        print("  H.264 codec not available, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Encoding video: {len(frames)} frames at {fps} fps")

    for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
        combined = create_side_by_side_frame(frame, depth, colormap)
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i + 1}/{len(frames)} frames")

    out.release()
    print(f"Video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VGGT Depth Estimation from Video")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1.0)")
    parser.add_argument("--save-points", action="store_true",
                        help="Also save world point maps")
    parser.add_argument("--save-cameras", action="store_true",
                        help="Also save camera parameters")
    parser.add_argument("--save-video", action="store_true", default=True,
                        help="Save side-by-side depth video (default: True)")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video output")
    parser.add_argument("--colormap", type=str, default="turbo",
                        help="Colormap for depth visualization (default: turbo)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of frames to process at once (default: 20, reduce if OOM)")
    args = parser.parse_args()

    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Extract frames from video
    print(f"\nExtracting frames from: {args.input}")
    frames = extract_frames_from_video(args.input, fps=args.fps)

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Save frames temporarily and load with preprocessing
    tmp_dir = os.path.join(args.output, "frames")
    print(f"\nPreprocessing {len(frames)} frames...")
    images = save_frames_and_load(frames, tmp_dir)
    images = images.to(device)
    print(f"Input tensor shape: {images.shape}")

    # Load model
    print("\nLoading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()

    # Run inference in batches to avoid OOM
    print(f"\nRunning inference (batch size: {args.batch_size})...")
    all_depths = []
    all_world_points = []
    all_extrinsics = []
    all_intrinsics = []

    num_frames = images.shape[0]
    num_batches = (num_frames + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_frames)
        batch_images = images[start_idx:end_idx]

        print(f"  Processing batch {batch_idx + 1}/{num_batches} (frames {start_idx}-{end_idx - 1})...")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                batch_predictions = model(batch_images)

        # Collect results
        if "depth" in batch_predictions:
            all_depths.append(batch_predictions["depth"].cpu())
        if "world_points" in batch_predictions:
            all_world_points.append(batch_predictions["world_points"].cpu())
        if "extrinsic" in batch_predictions:
            all_extrinsics.append(batch_predictions["extrinsic"].cpu())
        if "intrinsic" in batch_predictions:
            all_intrinsics.append(batch_predictions["intrinsic"].cpu())

        # Clear GPU memory
        del batch_predictions
        torch.cuda.empty_cache()

    # Combine batch results
    # Model outputs shape: [1, S, H, W, C] where S is sequence length
    # Concatenate along dim=1 (sequence), then squeeze dim=0
    predictions = {}
    if all_depths:
        predictions["depth"] = torch.cat(all_depths, dim=1).squeeze(0)  # [S, H, W, 1]
    if all_world_points:
        predictions["world_points"] = torch.cat(all_world_points, dim=1).squeeze(0)  # [S, H, W, 3]
    if all_extrinsics:
        predictions["extrinsic"] = torch.cat(all_extrinsics, dim=1).squeeze(0)  # [S, 4, 4]
    if all_intrinsics:
        predictions["intrinsic"] = torch.cat(all_intrinsics, dim=1).squeeze(0)  # [S, 3, 3]

    # Save outputs
    print("\nSaving outputs...")

    # Always save depth maps
    if "depth" in predictions:
        save_depth_maps(predictions["depth"], args.output)

    # Optionally save point maps
    if args.save_points and "world_points" in predictions:
        save_point_maps(predictions["world_points"], args.output)

    # Optionally save camera parameters
    if args.save_cameras:
        save_cameras(predictions, args.output)

    # Encode depth video with colorbar
    if args.save_video and not args.no_video:
        print("\nEncoding depth video...")
        video_output_path = os.path.join(args.output, "depth_video.mp4")
        depth_maps_np = [d.cpu().numpy().squeeze() for d in predictions["depth"]]
        encode_depth_video(frames, depth_maps_np, video_output_path,
                           fps=args.fps, colormap=args.colormap)

    print(f"\nDone! Outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
