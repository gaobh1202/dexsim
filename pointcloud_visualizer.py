#!/usr/bin/env python3
"""
Visualize point clouds saved in per-demo NPZ files.

Each NPZ is expected to contain one array with shape [T, N, 3] / [T, N, 6]
or [N, 3] / [N, 6] (single frame). For [*, *, 6], the last 3 channels
are treated as RGB colors. Common keys are:
- point_cloud
- pc
- points
- xyz

Optional: overlay an axis-aligned workspace box in world coordinates via
--bbox-min and --bbox-max (three floats each: x y z).
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc

try:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "mpl_toolkits.mplot3d is required (usually bundled with matplotlib)."
    ) from exc


DEFAULT_PC_KEYS = ("point_cloud", "pc", "points", "xyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize point cloud frames from all NPZ files in a folder."
    )
    parser.add_argument(
        "npz_dir",
        type=Path,
        help="Directory that contains *.npz files (one demo per file).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS for frame animation (default: 10).",
    )
    parser.add_argument(
        "--point-key",
        type=str,
        default="",
        help="Force a specific key for point clouds. If empty, auto-detect.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Randomly sample at most this many points per frame (default: 5000).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle NPZ file order before playback.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat all files in a loop.",
    )
    parser.add_argument(
        "--bbox-min",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Workspace AABB minimum corner in world frame (use with --bbox-max).",
    )
    parser.add_argument(
        "--bbox-max",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Workspace AABB maximum corner in world frame (use with --bbox-min).",
    )
    parser.add_argument(
        "--bbox-color",
        type=str,
        default="red",
        help="Matplotlib color for the workspace wireframe (default: red).",
    )
    parser.add_argument(
        "--bbox-linewidth",
        type=float,
        default=1.5,
        help="Line width for the workspace wireframe (default: 1.5).",
    )
    parser.add_argument(
        "--no-bbox-expand-limits",
        action="store_true",
        help="Do not expand axis limits to include the bbox (may clip the box).",
    )
    return parser.parse_args()


def _parse_bbox(args: argparse.Namespace) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    lo = args.bbox_min
    hi = args.bbox_max
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        raise ValueError("Both --bbox-min and --bbox-max are required to draw a workspace box.")
    min_xyz = np.asarray(lo, dtype=np.float64)
    max_xyz = np.asarray(hi, dtype=np.float64)
    if np.any(max_xyz <= min_xyz):
        raise ValueError(
            f"--bbox-max must be strictly greater than --bbox-min per axis; "
            f"got min={min_xyz.tolist()} max={max_xyz.tolist()}"
        )
    return min_xyz.astype(np.float32), max_xyz.astype(np.float32)


def _merge_lim_with_bbox(
    lim_xyz: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    xlim, ylim, zlim = lim_xyz
    return (
        (min(xlim[0], float(bbox_min[0])), max(xlim[1], float(bbox_max[0]))),
        (min(ylim[0], float(bbox_min[1])), max(ylim[1], float(bbox_max[1]))),
        (min(zlim[0], float(bbox_min[2])), max(zlim[1], float(bbox_max[2]))),
    )


def _add_axis_aligned_bbox_wireframe(
    ax,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    color: str,
    linewidth: float,
) -> None:
    """Draw axis-aligned box edges in world coordinates."""
    mn = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    mx = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    # 12 edges of a box
    edge_idx = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    segments = np.array([[corners[i], corners[j]] for i, j in edge_idx], dtype=np.float64)
    lc = Line3DCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        linestyles="solid",
    )
    ax.add_collection3d(lc)


def _find_point_cloud_key(npz_obj: np.lib.npyio.NpzFile, forced_key: str) -> str:
    keys = list(npz_obj.keys())

    if forced_key:
        if forced_key not in npz_obj:
            raise KeyError(f"point key '{forced_key}' not found. Available keys: {keys}")
        return forced_key

    for key in DEFAULT_PC_KEYS:
        if key in npz_obj:
            return key

    # Fallback: first array whose last dim is 3 or 6 and rank is 2 or 3
    for key in keys:
        arr = np.asarray(npz_obj[key])
        if arr.ndim in (2, 3) and arr.shape[-1] in (3, 6):
            return key

    raise KeyError(
        f"Cannot find point cloud array key. Available keys: {keys}. "
        "Try --point-key <KEY>."
    )


def _load_demo_frames(npz_path: Path, forced_key: str) -> Tuple[np.ndarray, str]:
    with np.load(npz_path, allow_pickle=False) as data:
        pc_key = _find_point_cloud_key(data, forced_key)
        points = np.asarray(data[pc_key], dtype=np.float32)

    if points.ndim == 2 and points.shape[-1] in (3, 6):
        points = points[None, ...]  # [1, N, 3]
    if points.ndim != 3 or points.shape[-1] not in (3, 6):
        raise ValueError(
            f"{npz_path.name}: key '{pc_key}' has invalid shape {points.shape}, "
            "expected [T, N, 3]/[T, N, 6] or [N, 3]/[N, 6]."
        )

    return points, pc_key


def _sample_points(frame_points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or frame_points.shape[0] <= max_points:
        return frame_points
    idx = rng.choice(frame_points.shape[0], size=max_points, replace=False)
    return frame_points[idx]


def _compute_axis_limits(points_xyz: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    # Use robust percentiles to avoid outliers stretching the view too much.
    p_lo = np.percentile(points_xyz.reshape(-1, 3), 1, axis=0)
    p_hi = np.percentile(points_xyz.reshape(-1, 3), 99, axis=0)
    center = (p_lo + p_hi) * 0.5
    extent = float(np.max(p_hi - p_lo)) * 0.6
    extent = max(extent, 1e-3)
    return (
        (center[0] - extent, center[0] + extent),
        (center[1] - extent, center[1] + extent),
        (center[2] - extent, center[2] + extent),
    )


def _play_single_npz(
    npz_path: Path,
    fps: float,
    max_points: int,
    point_key: str,
    rng: np.random.Generator,
    bbox: Optional[Tuple[np.ndarray, np.ndarray]],
    bbox_color: str,
    bbox_linewidth: float,
    expand_limits_with_bbox: bool,
) -> bool:
    points, used_key = _load_demo_frames(npz_path, forced_key=point_key)
    n_frames = points.shape[0]
    interval = 1.0 / max(fps, 1e-6)

    points_xyz_all = points[..., :3]
    xlim, ylim, zlim = _compute_axis_limits(points_xyz_all)
    if bbox is not None and expand_limits_with_bbox:
        bbox_min, bbox_max = bbox
        xlim, ylim, zlim = _merge_lim_with_bbox((xlim, ylim, zlim), bbox_min, bbox_max)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    if bbox is not None:
        _add_axis_aligned_bbox_wireframe(
            ax,
            bbox[0],
            bbox[1],
            color=bbox_color,
            linewidth=bbox_linewidth,
        )

    plt.ion()
    fig.show()

    scatter = None
    for frame_idx in range(n_frames):
        if not plt.fignum_exists(fig.number):
            # User manually closed window: stop full playback.
            return False

        t0 = time.time()
        frame = _sample_points(points[frame_idx], max_points=max_points, rng=rng)
        xyz = frame[:, :3]

        if scatter is not None:
            scatter.remove()
        if frame.shape[1] >= 6:
            colors = frame[:, 3:6].astype(np.float32)
            # Auto-handle both [0, 255] and [0, 1] ranges.
            if np.nanmax(colors) > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
            scatter = ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=1.0,
                c=colors,
                alpha=0.9,
                linewidths=0,
            )
        else:
            scatter = ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=1.0,
                c=xyz[:, 2],
                cmap="viridis",
                alpha=0.85,
                linewidths=0,
            )
        ax.set_title(
            f"{npz_path.name} | key={used_key} | frame {frame_idx + 1}/{n_frames}\n"
            "Close window to quit"
        )

        fig.canvas.draw_idle()
        plt.pause(0.001)

        elapsed = time.time() - t0
        sleep_s = max(0.0, interval - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)

    plt.ioff()
    plt.show(block=True)
    return True


def main() -> None:
    args = parse_args()
    bbox = _parse_bbox(args)
    npz_dir = args.npz_dir.expanduser().resolve()
    if not npz_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {npz_dir}")

    files: List[Path] = sorted(npz_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    rng = np.random.default_rng(0)

    if args.shuffle:
        rng.shuffle(files)

    print(f"Found {len(files)} npz files in {npz_dir}")
    print("Playback starts... (close the figure window to stop)")

    while True:
        for npz_path in files:
            print(f"\n=== {npz_path.name} ===")
            keep_running = _play_single_npz(
                npz_path=npz_path,
                fps=args.fps,
                max_points=args.max_points,
                point_key=args.point_key.strip(),
                rng=rng,
                bbox=bbox,
                bbox_color=args.bbox_color,
                bbox_linewidth=float(args.bbox_linewidth),
                expand_limits_with_bbox=not args.no_bbox_expand_limits,
            )
            if not keep_running:
                print("Window closed. Exiting.")
                return
        if not args.repeat:
            break


if __name__ == "__main__":
    main()
