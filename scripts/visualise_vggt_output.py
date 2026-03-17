#!/usr/bin/env python3
"""
Visualise VGGT state estimation output as MP4 map videos.

Produces two videos from a state_estimation.json:
  - *_map.mp4   : Standard 2D map with class-colored markers and UAV trajectory
  - *_crowd.mp4 : Crowd map with persons colored by crowd_id

Usage:
    python scripts/visualise_vggt_output.py \
        -i outputs/DJI_20260215_132741_V_1_30s/state_estimation.json
"""

import argparse
import colorsys
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import contextily as ctx
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate MP4 map videos from VGGT state_estimation.json')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to state_estimation.json')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Output video FPS (default: 10)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Figure DPI (default: 100)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 10],
                        help='Figure size in inches (default: 14 10)')
    parser.add_argument('--video', type=str, default=None,
                        help='Comma-separated list of videos to generate: '
                             'map, crowd, density (default: all three). '
                             'E.g. --video map,density')
    parser.add_argument('--no-arrows', action='store_true',
                        help='Hide velocity arrows on all video types')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(json_path):
    """Load state_estimation.json and return organised data structures.

    Returns:
        tracks_by_frame : dict[int, list[dict]]  — non-UAV tracks grouped by frame
        metadata        : dict
        uav_tracks      : list[dict]  — UAV entries sorted by frame_id
    """
    with open(json_path) as f:
        data = json.load(f)

    # Support both formats: dict with 'tracks' key, or plain list of tracks
    if isinstance(data, list):
        metadata = {}
        all_tracks = data
    else:
        metadata = data.get('metadata', {})
        all_tracks = data.get('tracks', [])

    uav_tracks = []
    tracks_by_frame = {}

    for t in all_tracks:
        frame_id = t.get('frame_id')
        if frame_id is None:
            continue
        if t.get('class_name') == 'UAV':
            uav_tracks.append(t)
        else:
            tracks_by_frame.setdefault(frame_id, []).append(t)

    uav_tracks.sort(key=lambda t: t['frame_id'])
    return tracks_by_frame, metadata, uav_tracks


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

def get_class_colors():
    """Return the standard class→hex-color mapping (with fallback)."""
    return {
        'person':     '#00FF00',
        'car':        '#FF0000',
        'vehicle':    '#0000FF',
        'cycle':      '#00FFFF',
        'bus':        '#FF00FF',
        'track_leaf': '#FFD700',
    }


_FALLBACK_COLOR = '#AAAAAA'


def generate_crowd_colors(crowd_ids):
    """Generate a distinct color for each crowd_id using golden-angle HSL rotation.

    Args:
        crowd_ids: iterable of integer crowd ids

    Returns:
        dict[int, str] mapping crowd_id → hex color string
    """
    sorted_ids = sorted(set(crowd_ids))
    golden_angle = 137.508  # degrees
    colors = {}
    for i, cid in enumerate(sorted_ids):
        hue = (i * golden_angle) % 360 / 360.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.85)
        colors[cid] = '#{:02X}{:02X}{:02X}'.format(
            int(r * 255), int(g * 255), int(b * 255))
    return colors


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

# Module-level home Mercator position (set once in main via set_home)
_HOME_MX = 0.0
_HOME_MY = 0.0
_HOME_COS_LAT = 1.0  # cos(home_lat) for Mercator→real-metre scaling


def latlon_to_mercator(lat, lon):
    """Convert lat/lon to Web Mercator (EPSG:3857) coordinates."""
    x = np.asarray(lon, dtype=np.float64) * 20037508.34 / 180.0
    lat_rad = np.deg2rad(np.clip(np.asarray(lat, dtype=np.float64), -85, 85))
    y = np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0)) * 20037508.34 / np.pi
    return x, y


def set_home(lat, lon):
    """Set the home lat/lon — used for relative-metre axis tick labels."""
    global _HOME_MX, _HOME_MY, _HOME_COS_LAT
    _HOME_MX, _HOME_MY = latlon_to_mercator(float(lat), float(lon))
    _HOME_COS_LAT = np.cos(np.deg2rad(float(lat)))


def compute_bounds(tracks_by_frame, uav_tracks):
    """Compute fixed map bounds from all positions, with padding and 1:1 aspect.

    Returns (x_min, x_max, y_min, y_max) in Web Mercator (EPSG:3857).
    """
    all_lats = []
    all_lons = []

    for t in uav_tracks:
        all_lats.append(t['lat'])
        all_lons.append(t['lon'])

    for frame_tracks in tracks_by_frame.values():
        for t in frame_tracks:
            lat, lon = t.get('lat'), t.get('lon')
            if lat is not None and lon is not None:
                all_lats.append(lat)
                all_lons.append(lon)

    all_x, all_y = latlon_to_mercator(np.array(all_lats), np.array(all_lons))

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    # 10% padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    # 1:1 aspect ratio fix
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range > y_range:
        y_center = (y_min + y_max) / 2
        y_min = y_center - x_range / 2
        y_max = y_center + x_range / 2
    else:
        x_center = (x_min + x_max) / 2
        x_min = x_center - y_range / 2
        x_max = x_center + y_range / 2

    return x_min, x_max, y_min, y_max


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _setup_basemap(bounds):
    """Test basemap providers and return the first working one (or None)."""
    x_min, x_max, y_min, y_max = bounds
    providers = [
        ctx.providers.CartoDB.Positron,
        ctx.providers.CartoDB.DarkMatter,
        ctx.providers.OpenStreetMap.Mapnik,
    ]
    for provider in providers:
        try:
            fig_test, ax_test = plt.subplots(figsize=(4, 3))
            ax_test.set_xlim(x_min, x_max)
            ax_test.set_ylim(y_min, y_max)
            ctx.add_basemap(ax_test, source=provider, zoom=16)
            plt.close(fig_test)
            name = provider.get('name', str(provider))
            print(f"  Using basemap provider: {name}")
            return provider
        except Exception:
            plt.close('all')
            continue
    print("  Warning: No basemap provider available, using plain background")
    return None


# Cached basemap provider (set once in main)
_BASEMAP_PROVIDER = None


def _metre_formatter_x(val, pos):
    """Format Mercator x-tick as relative East metres from home."""
    return f'{(val - _HOME_MX) * _HOME_COS_LAT:.0f}'


def _metre_formatter_y(val, pos):
    """Format Mercator y-tick as relative North metres from home."""
    return f'{(val - _HOME_MY) * _HOME_COS_LAT:.0f}'


def _render_base(ax, bounds, uav_x, uav_y, frame_idx, figsize):
    """Shared rendering: basemap, metre-scale axes, UAV trajectory."""
    x_min, x_max, y_min, y_max = bounds

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    # Basemap tiles
    if _BASEMAP_PROVIDER is not None:
        try:
            ctx.add_basemap(ax, source=_BASEMAP_PROVIDER, zoom=17)
        except Exception:
            ax.set_facecolor('#E0E0E0')
    else:
        ax.set_facecolor('#E0E0E0')

    # Relative-metre tick labels
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_metre_formatter_x))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_metre_formatter_y))
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')

    # Full UAV trajectory (faded)
    ax.plot(uav_x, uav_y, color='gray', linewidth=1, alpha=0.3, zorder=1)

    # Progressive UAV trajectory up to current frame
    if frame_idx > 0:
        idx = min(frame_idx, len(uav_x) - 1)
        ax.plot(uav_x[:idx + 1], uav_y[:idx + 1],
                color='white', linewidth=3, alpha=0.9, zorder=2)
        ax.plot(uav_x[:idx + 1], uav_y[:idx + 1],
                color='dodgerblue', linewidth=2, alpha=0.9, zorder=3)

    # Start marker (lime triangle)
    ax.scatter(uav_x[0], uav_y[0], c='lime', s=100, marker='^',
               zorder=6, edgecolors='white', linewidth=1.5)

    # Current UAV position (black circle)
    cur = min(frame_idx, len(uav_x) - 1)
    ax.scatter(uav_x[cur], uav_y[cur], c='black', s=150, marker='o',
               zorder=7, edgecolors='white', linewidth=2)

    return cur


def _draw_heading(ax, uav_x, uav_y, uav_tracks, frame_idx, bounds,
                   fov_deg=60.0):
    """Draw the drone's line-of-sight as a filled sector (FOV cone).

    The sector is centered on heading_deg with angular width fov_deg,
    filled with translucent red (alpha=0.3).
    """
    from matplotlib.patches import Wedge

    cur = min(frame_idx, len(uav_x) - 1)
    heading_deg = uav_tracks[cur].get('heading_deg')
    if heading_deg is None:
        return

    x_min, x_max = bounds[0], bounds[1]
    radius = (x_max - x_min) * 0.12

    # Wedge angles are measured counter-clockwise from the +x axis (East).
    # heading_deg: 0=North, 90=East  →  matplotlib angle = 90 - heading_deg
    center_angle = 90.0 - heading_deg
    theta1 = center_angle - fov_deg / 2.0
    theta2 = center_angle + fov_deg / 2.0

    wedge = Wedge((uav_x[cur], uav_y[cur]), radius,
                  theta1, theta2,
                  facecolor='red', edgecolor='red',
                  alpha=0.3, linewidth=1.0, zorder=8)
    ax.add_patch(wedge)


def _draw_velocity_arrows(ax, frame_tracks, bounds, class_colors,
                          crowd_colors=None):
    """Draw velocity vectors as arrows proportional to speed.

    vel_ned = [north, east, down] in m/s.
    Arrow length is proportional to horizontal speed; scaled so that
    ~2 m/s (typical walking speed) ≈ 5% of the map extent.

    If crowd_colors is provided, person arrows use their crowd_id colour
    (unclustered persons get neutral gray).  Non-person classes use
    class_colors as before.
    """
    x_span = bounds[1] - bounds[0]
    # Scale: 2 m/s → 5 % of map width (in mercator metres)
    arrow_scale = x_span * 0.025 / 2.0  # mercator-metres per m/s (half length)

    for track in frame_tracks:
        lat, lon = track.get('lat'), track.get('lon')
        vel = track.get('vel_ned')
        if lat is None or lon is None or vel is None:
            continue
        vn, ve, _ = vel
        speed = math.sqrt(vn ** 2 + ve ** 2)
        if speed < 0.05:  # skip near-stationary
            continue

        x, y = latlon_to_mercator(lat, lon)
        # NED → mercator: dx = east, dy = north
        dx = ve * arrow_scale
        dy = vn * arrow_scale

        cls = track.get('class_name', 'unknown')
        if cls == 'person' and crowd_colors:
            cid = track.get('crowd_id')
            if cid is not None and cid in crowd_colors:
                color = crowd_colors[cid]
            else:
                color = '#888888'
        else:
            color = class_colors.get(cls, _FALLBACK_COLOR)

        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, alpha=0.8),
                    zorder=6)


def _draw_crowd_hulls(ax, frame_tracks, crowd_colors):
    """Draw dotted convex-hull outlines around groups sharing the same crowd_id."""
    from scipy.spatial import ConvexHull

    # Group person positions by crowd_id
    crowd_points = {}  # crowd_id → list of (x, y)
    for track in frame_tracks:
        cid = track.get('crowd_id')
        if cid is None:
            continue
        lat, lon = track.get('lat'), track.get('lon')
        if lat is None or lon is None:
            continue
        x, y = latlon_to_mercator(lat, lon)
        crowd_points.setdefault(cid, []).append((x, y))

    for cid, pts in crowd_points.items():
        if len(pts) < 3:
            continue  # need ≥ 3 points for a hull
        pts_arr = np.array(pts)
        try:
            hull = ConvexHull(pts_arr)
        except Exception:
            continue
        color = crowd_colors.get(cid, _FALLBACK_COLOR)
        # Close the hull polygon
        hull_verts = pts_arr[hull.vertices]
        hull_verts = np.vstack([hull_verts, hull_verts[0]])
        ax.plot(hull_verts[:, 0], hull_verts[:, 1],
                linestyle=':', linewidth=2.0, color=color, alpha=0.8,
                zorder=4)


def render_frame_standard(ax, bounds, uav_x, uav_y,
                          uav_tracks, frame_tracks, frame_idx, class_colors,
                          class_list, figsize, crowd_colors=None,
                          show_arrows=True):
    """Render one frame of the standard (class-colored) map."""
    _render_base(ax, bounds, uav_x, uav_y, frame_idx, figsize)
    _draw_heading(ax, uav_x, uav_y, uav_tracks, frame_idx, bounds)

    # Crowd convex hulls (drawn first so markers sit on top)
    if crowd_colors:
        _draw_crowd_hulls(ax, frame_tracks, crowd_colors)

    # Object markers (class-colored)
    for track in frame_tracks:
        lat, lon = track.get('lat'), track.get('lon')
        if lat is None or lon is None:
            continue
        x, y = latlon_to_mercator(lat, lon)
        cls = track.get('class_name', 'unknown')
        color = class_colors.get(cls, '#FFFFFF')
        ax.scatter(x, y, c=color, s=100, alpha=0.9,
                   edgecolors='white', linewidth=2, zorder=5)

    # Velocity arrows
    if show_arrows:
        _draw_velocity_arrows(ax, frame_tracks, bounds, class_colors)

    # Frame info text
    ax.text(0.02, 0.98, f'Frame: {frame_idx}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Class legend
    handles = []
    for cls in class_list:
        h = plt.scatter([], [], c=class_colors.get(cls, _FALLBACK_COLOR), s=60,
                        edgecolors='white', linewidth=1, label=cls)
        handles.append(h)
    if handles:
        leg = ax.legend(handles=handles, loc='upper right',
                        framealpha=0.9, fontsize=9)
        leg.get_frame().set_facecolor('white')

    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.set_title('UAV Trajectory and Object Geolocalization',
                 fontsize=12, fontweight='bold')


def render_frame_crowd(ax, bounds, uav_x, uav_y,
                       uav_tracks, frame_tracks, frame_idx, class_colors,
                       crowd_colors, all_crowd_ids_sorted, class_list, figsize,
                       show_arrows=True):
    """Render one frame of the crowd-colored map."""
    _render_base(ax, bounds, uav_x, uav_y, frame_idx, figsize)
    _draw_heading(ax, uav_x, uav_y, uav_tracks, frame_idx, bounds)

    # Crowd convex hulls (drawn first so markers sit on top)
    _draw_crowd_hulls(ax, frame_tracks, crowd_colors)

    active_crowds = set()

    for track in frame_tracks:
        lat, lon = track.get('lat'), track.get('lon')
        if lat is None or lon is None:
            continue
        x, y = latlon_to_mercator(lat, lon)
        cls = track.get('class_name', 'unknown')

        if cls == 'person':
            cid = track.get('crowd_id')
            if cid is not None:
                color = crowd_colors[cid]
                alpha = 0.9
                size = 100
                active_crowds.add(cid)
            else:
                color = '#888888'
                alpha = 0.4
                size = 50
        else:
            color = class_colors.get(cls, '#FFFFFF')
            alpha = 0.9
            size = 100

        ax.scatter(x, y, c=color, s=size, alpha=alpha,
                   edgecolors='white', linewidth=2, zorder=5)

    # Velocity arrows
    if show_arrows:
        _draw_velocity_arrows(ax, frame_tracks, bounds, class_colors)

    # Frame info text with crowd count
    n_crowds = len(active_crowds)
    ax.text(0.02, 0.98, f'Frame: {frame_idx} | Crowds: {n_crowds} active',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend: crowd clusters + other classes
    handles = []
    for cid in all_crowd_ids_sorted:
        h = plt.scatter([], [], c=crowd_colors[cid], s=60,
                        edgecolors='white', linewidth=1,
                        label=f'Crowd {cid}')
        handles.append(h)
    # Noise marker
    handles.append(plt.scatter([], [], c='#888888', s=30, alpha=0.4,
                               edgecolors='white', linewidth=1, label='Noise'))
    # Non-person classes
    for cls in class_list:
        if cls != 'person':
            handles.append(plt.scatter([], [], c=class_colors.get(cls, _FALLBACK_COLOR), s=60,
                                       edgecolors='white', linewidth=1,
                                       label=cls))

    if handles:
        leg = ax.legend(handles=handles, loc='upper right',
                        framealpha=0.9, fontsize=8, ncol=1)
        leg.get_frame().set_facecolor('white')

    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.set_title('UAV Trajectory — Crowd Clustering',
                 fontsize=12, fontweight='bold')


def _density_to_color(density, vmin=0.0, vmax=5.0):
    """Map a crowd_density value to a light-gray → dark-blue hex color.

    0 → light gray (#CCCCCC), max → dark navy (#0A2463).
    Values are clamped to [vmin, vmax].
    """
    t = max(0.0, min(1.0, (density - vmin) / (vmax - vmin)))
    # Linear interpolation: light gray (0) → dark navy (1)
    r = int(204 * (1.0 - t) + 10 * t)
    g = int(204 * (1.0 - t) + 36 * t)
    b = int(204 * (1.0 - t) + 99 * t)
    return '#{:02X}{:02X}{:02X}'.format(r, g, b)


def render_frame_density(ax, bounds, uav_x, uav_y,
                         uav_tracks, frame_tracks, frame_idx, class_colors,
                         class_list, figsize, density_min=0.0, density_max=0.5,
                         crowd_colors=None, show_arrows=True):
    """Render one frame with dual-encoded markers per person.

    Fill colour encodes crowd_density (light gray → dark navy).
    Border colour encodes crowd_id (golden-angle palette).
    Persons with no crowd_id get a mid-gray border at reduced alpha.

    Default scale 0–0.5 p/m² suits typical outdoor UAV scenes.
    For packed indoor/stadium scenes, increase density_max (up to ~5).
    """
    _render_base(ax, bounds, uav_x, uav_y, frame_idx, figsize)
    _draw_heading(ax, uav_x, uav_y, uav_tracks, frame_idx, bounds)

    # No contour hulls — crowd_id is encoded in the marker border instead

    max_density_this_frame = 0.0

    for track in frame_tracks:
        lat, lon = track.get('lat'), track.get('lon')
        if lat is None or lon is None:
            continue
        x, y = latlon_to_mercator(lat, lon)
        cls = track.get('class_name', 'unknown')

        if cls == 'person':
            d = track.get('crowd_density')
            cid = track.get('crowd_id')
            if d is not None:
                fill_color = _density_to_color(d, density_min, density_max)
                alpha = 0.9
                size = 100
                max_density_this_frame = max(max_density_this_frame, d)
            else:
                fill_color = '#888888'
                alpha = 0.4
                size = 50

            # Border encodes crowd_id (semi-transparent so fill shows through)
            if cid is not None and crowd_colors:
                _hex = crowd_colors.get(cid, '#888888')
                _r, _g, _b = int(_hex[1:3], 16)/255, int(_hex[3:5], 16)/255, int(_hex[5:7], 16)/255
                edge_color = (_r, _g, _b, 0.45)
                edge_width = 2.5
            else:
                edge_color = (0.53, 0.53, 0.53, 0.35)
                edge_width = 1.5
                if d is not None:
                    alpha = 0.6  # has density but no crowd — slightly faded
        else:
            fill_color = class_colors.get(cls, '#FFFFFF')
            edge_color = 'white'
            edge_width = 2
            alpha = 0.9
            size = 100

        ax.scatter(x, y, c=fill_color, s=size, alpha=alpha,
                   edgecolors=edge_color, linewidth=edge_width, zorder=5)

    # Velocity arrows (coloured by crowd_id when available)
    if show_arrows:
        _draw_velocity_arrows(ax, frame_tracks, bounds, class_colors,
                              crowd_colors=crowd_colors)

    # Frame info
    ax.text(0.02, 0.98,
            f'Frame: {frame_idx} | Peak density: {max_density_this_frame:.2f} p/m\u00b2',
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend — density gradient bar + unclustered + non-person classes
    handles = []
    # Adaptive density ticks: nice round values spanning the full range
    # Include a ">" swatch to indicate values above the P99 cap
    _nice_steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    _step = min((s for s in _nice_steps if density_max / s <= 6), default=1.0)
    _ticks = []
    v = density_min
    while v <= density_max + 1e-9:
        _ticks.append(round(v, 4))
        v += _step
    # Always include density_max if not already close to last tick
    if abs(_ticks[-1] - density_max) > _step * 0.3:
        _ticks.append(round(density_max, 2))
    for val in _ticks:
        c = _density_to_color(val, density_min, density_max)
        handles.append(plt.scatter([], [], c=c, s=60, edgecolors='white',
                                   linewidth=1,
                                   label=f'{val:.2f} p/m\u00b2'))
    # "Above cap" swatch — diamond marker, larger so shape is visible
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker='D', color='w', markerfacecolor=_density_to_color(density_max, density_min, density_max),
                          markeredgecolor='white', markersize=10, linewidth=0,
                          label=f'>{density_max:.2f} p/m\u00b2'))
    # Unclustered
    handles.append(plt.scatter([], [], c='#888888', s=30, alpha=0.4,
                               edgecolors=(0.53, 0.53, 0.53, 0.35), linewidth=1.5,
                               label='Unclustered'))
    # Non-person classes
    for cls in class_list:
        if cls != 'person':
            handles.append(plt.scatter([], [], c=class_colors.get(cls, _FALLBACK_COLOR), s=60,
                                       edgecolors='white', linewidth=1,
                                       label=cls))

    if handles:
        leg = ax.legend(handles=handles, loc='upper right',
                        framealpha=0.9, fontsize=8, ncol=1)
        leg.get_frame().set_facecolor('white')

    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.set_title(f'UAV Trajectory \u2014 Crowd Density',
                 fontsize=12, fontweight='bold')


# ---------------------------------------------------------------------------
# Video compilation
# ---------------------------------------------------------------------------

def compile_video(frame_dir, output_path, fps):
    """Compile PNGs from *frame_dir* into an MP4 at *output_path*."""
    frame_paths = sorted(
        [os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
         if f.endswith('.png')])

    if not frame_paths:
        print("  No frames to compile!")
        return

    if HAS_CV2:
        first = cv2.imread(frame_paths[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for fp in frame_paths:
            writer.write(cv2.imread(fp))
        writer.release()
    else:
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frame_dir, 'frame_%05d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    print(f"  Video saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HAS_VIZ:
        print("Error: matplotlib and contextily are required.  "
              "pip install matplotlib contextily")
        sys.exit(1)

    args = parse_args()
    json_path = args.input
    fps = args.fps
    dpi = args.dpi
    figsize = tuple(args.figsize)
    show_arrows = not args.no_arrows

    # Parse --video selection
    ALL_VIDEOS = {'map', 'crowd', 'density'}
    if args.video:
        selected = {v.strip().lower() for v in args.video.split(',')}
        unknown = selected - ALL_VIDEOS
        if unknown:
            print(f"Error: unknown video type(s): {unknown}. Choose from: {ALL_VIDEOS}")
            sys.exit(1)
    else:
        selected = ALL_VIDEOS

    if not os.path.isfile(json_path):
        print(f"Error: file not found: {json_path}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(json_path))
    stem = os.path.splitext(os.path.basename(json_path))[0]
    map_video_path = os.path.join(out_dir, f'{stem}_map.mp4')
    crowd_video_path = os.path.join(out_dir, f'{stem}_crowd.mp4')
    density_video_path = os.path.join(out_dir, f'{stem}_density.mp4')

    print(f"Loading {json_path} ...")
    tracks_by_frame, metadata, uav_tracks = load_data(json_path)

    if not uav_tracks:
        print("Error: no UAV tracks found in data")
        sys.exit(1)

    # Frame range
    all_frame_ids = set(uav_tracks[i]['frame_id'] for i in range(len(uav_tracks)))
    for fid in tracks_by_frame:
        all_frame_ids.add(fid)
    max_frame = max(all_frame_ids)
    n_frames = max_frame + 1
    print(f"  {len(uav_tracks)} UAV entries, "
          f"{sum(len(v) for v in tracks_by_frame.values())} object tracks, "
          f"{n_frames} frames")

    # Set home position (first UAV point) as NED origin
    uav_by_frame = {t['frame_id']: t for t in uav_tracks}
    uav_lats = np.array([uav_by_frame[i]['lat'] for i in range(n_frames)])
    uav_lons = np.array([uav_by_frame[i]['lon'] for i in range(n_frames)])
    set_home(uav_lats[0], uav_lons[0])
    print(f"  Home (NED origin): lat={uav_lats[0]:.6f}, lon={uav_lons[0]:.6f}")

    # Precompute UAV local positions (indexed by frame_id)
    uav_x, uav_y = latlon_to_mercator(uav_lats, uav_lons)

    # Colors
    class_colors = get_class_colors()
    all_classes = set()
    all_crowd_ids = set()
    for frame_tracks in tracks_by_frame.values():
        for t in frame_tracks:
            cls = t.get('class_name', 'unknown')
            all_classes.add(cls)
            cid = t.get('crowd_id')
            if cid is not None:
                all_crowd_ids.add(cid)

    class_list = sorted(all_classes)
    all_crowd_ids_sorted = sorted(all_crowd_ids)
    crowd_colors = generate_crowd_colors(all_crowd_ids) if all_crowd_ids else {}

    # Detect crowd_density range for the density video colour scale
    all_densities = []
    for frame_tracks in tracks_by_frame.values():
        for t in frame_tracks:
            d = t.get('crowd_density')
            if d is not None:
                all_densities.append(d)
    if all_densities:
        density_max = float(np.percentile(all_densities, 99))
        density_max = max(density_max, 0.1)  # floor to avoid degenerate scale
        print(f"  Density range: 0 – {density_max:.4f} p/m² (P99 cap)")
    else:
        density_max = 0.5

    # Bounds
    bounds = compute_bounds(tracks_by_frame, uav_tracks)
    # Display bounds as relative metres
    e_min = (bounds[0] - _HOME_MX) * _HOME_COS_LAT
    e_max = (bounds[1] - _HOME_MX) * _HOME_COS_LAT
    n_min = (bounds[2] - _HOME_MY) * _HOME_COS_LAT
    n_max = (bounds[3] - _HOME_MY) * _HOME_COS_LAT
    print(f"  Map bounds (local m): East=[{e_min:.1f}, {e_max:.1f}], "
          f"North=[{n_min:.1f}, {n_max:.1f}]")

    # Basemap
    global _BASEMAP_PROVIDER
    print("  Setting up basemap ...")
    _BASEMAP_PROVIDER = _setup_basemap(bounds)

    # -----------------------------------------------------------------------
    # Video 1: Standard map
    # -----------------------------------------------------------------------
    saved_paths = []

    if 'map' in selected:
        print(f"\n--- Generating standard map video ({n_frames} frames) ---")
        temp_dir = tempfile.mkdtemp(prefix='vggt_viz_map_')
        try:
            for frame_idx in range(n_frames):
                fig, ax = plt.subplots(figsize=figsize)
                frame_tracks = tracks_by_frame.get(frame_idx, [])
                render_frame_standard(ax, bounds, uav_x, uav_y,
                                      uav_tracks, frame_tracks, frame_idx,
                                      class_colors, class_list, figsize,
                                      crowd_colors=crowd_colors,
                                      show_arrows=show_arrows)
                plt.tight_layout()
                plt.savefig(os.path.join(temp_dir, f'frame_{frame_idx:05d}.png'),
                            dpi=dpi, facecolor='white')
                plt.close(fig)
                if (frame_idx + 1) % 50 == 0 or frame_idx == max_frame:
                    print(f"    {frame_idx + 1}/{n_frames} frames")

            print("  Compiling video ...")
            compile_video(temp_dir, map_video_path, fps)
            saved_paths.append(map_video_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Video 2: Crowd map
    # -----------------------------------------------------------------------
    if 'crowd' in selected:
        print(f"\n--- Generating crowd map video ({n_frames} frames) ---")
        temp_dir = tempfile.mkdtemp(prefix='vggt_viz_crowd_')
        try:
            for frame_idx in range(n_frames):
                fig, ax = plt.subplots(figsize=figsize)
                frame_tracks = tracks_by_frame.get(frame_idx, [])
                render_frame_crowd(ax, bounds, uav_x, uav_y,
                                   uav_tracks, frame_tracks, frame_idx,
                                   class_colors, crowd_colors,
                                   all_crowd_ids_sorted, class_list, figsize,
                                   show_arrows=show_arrows)
                plt.tight_layout()
                plt.savefig(os.path.join(temp_dir, f'frame_{frame_idx:05d}.png'),
                            dpi=dpi, facecolor='white')
                plt.close(fig)
                if (frame_idx + 1) % 50 == 0 or frame_idx == max_frame:
                    print(f"    {frame_idx + 1}/{n_frames} frames")

            print("  Compiling video ...")
            compile_video(temp_dir, crowd_video_path, fps)
            saved_paths.append(crowd_video_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Video 3: Density map
    # -----------------------------------------------------------------------
    if 'density' in selected:
        print(f"\n--- Generating density map video ({n_frames} frames) ---")
        temp_dir = tempfile.mkdtemp(prefix='vggt_viz_density_')
        try:
            for frame_idx in range(n_frames):
                fig, ax = plt.subplots(figsize=figsize)
                frame_tracks = tracks_by_frame.get(frame_idx, [])
                render_frame_density(ax, bounds, uav_x, uav_y,
                                     uav_tracks, frame_tracks, frame_idx,
                                     class_colors, class_list, figsize,
                                     density_min=0.0, density_max=density_max,
                                     crowd_colors=crowd_colors,
                                     show_arrows=show_arrows)
                plt.tight_layout()
                plt.savefig(os.path.join(temp_dir, f'frame_{frame_idx:05d}.png'),
                            dpi=dpi, facecolor='white')
                plt.close(fig)
                if (frame_idx + 1) % 50 == 0 or frame_idx == max_frame:
                    print(f"    {frame_idx + 1}/{n_frames} frames")

            print("  Compiling video ...")
            compile_video(temp_dir, density_video_path, fps)
            saved_paths.append(density_video_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nDone! Videos saved to:")
    for p in saved_paths:
        print(f"  {p}")


if __name__ == '__main__':
    main()
