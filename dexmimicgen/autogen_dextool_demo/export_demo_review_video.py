#!/usr/bin/env python3
"""
Export a per-frame review video from an HDF5 demo (sim state playback).

Concatenates camera views horizontally (default: frontview | agentview).
One video frame per dataset step when --video-skip 1.

Example:
  cd /home/benhua/DexSim
  PYTHONPATH=robosuite:dexmimicgen MUJOCO_GL=osmesa \\
  python dexmimicgen/autogen_dextool_demo/export_demo_review_video.py \\
    --dataset dexmimicgen/datasets/generated/single_arm_hammer_cleanup_demo_4.hdf5 \\
    --demo demo_0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in (_REPO_ROOT / "robosuite", _REPO_ROOT / "dexmimicgen"):
    _p = str(_pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _configure_mujoco_gl(backend: str = "osmesa") -> None:
    os.environ.setdefault("MUJOCO_GL", backend)
    if backend in ("osmesa", "glfw", "glx"):
        try:
            import robosuite.macros as macros

            macros.MUJOCO_GPU_RENDERING = False
        except Exception:
            pass


def _normalize_frame(frame):
    import numpy as np

    if isinstance(frame, tuple):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _get_env_metadata(dataset_path: str) -> dict:
    import h5py

    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def _make_playback_env(env_meta: dict):
    import robosuite

    import dexmimicgen  # noqa: F401

    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env_kwargs.pop("env_lang", None)
    return robosuite.make(**env_kwargs)


def _reset_to_state(env, state_vec):
    env.sim.set_state_from_flattened(state_vec)
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def export_review_video(
    dataset_path: str,
    demo: str,
    video_path: str,
    camera_names: list[str],
    image_height: int,
    image_width: int,
    fps: float,
    video_skip: int,
    max_frames: int | None = None,
) -> int:
    """Replay demo states in sim and write a composite MP4."""
    import h5py
    import imageio
    import numpy as np

    _configure_mujoco_gl("osmesa")
    import robosuite  # noqa: F401
    import dexmimicgen  # noqa: F401

    env_meta = _get_env_metadata(dataset_path)
    env = _make_playback_env(env_meta)

    with h5py.File(dataset_path, "r") as f:
        states = f[f"data/{demo}/states"][()]

    num_steps = states.shape[0]
    if max_frames is not None:
        num_steps = min(num_steps, max_frames)

    os.makedirs(os.path.dirname(os.path.abspath(video_path)) or ".", exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps, format="FFMPEG", codec="libx264")
    frames_written = 0
    try:
        for i in range(num_steps):
            _reset_to_state(env, states[i])
            if i % video_skip != 0:
                continue
            tiles = []
            for cam in camera_names:
                raw = env.sim.render(
                    height=image_height,
                    width=image_width,
                    camera_name=cam,
                )
                tiles.append(_normalize_frame(raw)[::-1])
            writer.append_data(np.concatenate(tiles, axis=1))
            frames_written += 1
            if (i + 1) % 100 == 0 or i + 1 == num_steps:
                print(f"  frame {i + 1}/{num_steps} -> video frame {frames_written}")
    finally:
        writer.close()
        env.close()

    print(f"Wrote {video_path} ({frames_written} frames @ {fps} fps)")
    return frames_written


def _default_video_path(dataset_path: str, demo: str, output_dir: str) -> str:
    stem = Path(dataset_path).stem + f"_{demo}"
    return str(Path(output_dir) / f"{stem}_review.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(
            _REPO_ROOT
            / "dexmimicgen/datasets/generated/single_arm_hammer_cleanup_demo_4.hdf5"
        ),
    )
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
    parser.add_argument("--video", default=None, help="Output MP4 path (override default)")
    parser.add_argument(
        "--camera-names",
        nargs="+",
        default=["frontview", "agentview"],
    )
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=480)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    video_path = args.video or _default_video_path(dataset_path, args.demo, args.output_dir)
    video_path = os.path.abspath(video_path)

    fps = args.fps
    if fps is None:
        env_meta = _get_env_metadata(dataset_path)
        fps = float(env_meta.get("env_kwargs", {}).get("control_freq", 20))

    export_review_video(
        dataset_path=dataset_path,
        demo=args.demo,
        video_path=video_path,
        camera_names=args.camera_names,
        image_height=args.image_height,
        image_width=args.image_width,
        fps=fps,
        video_skip=args.video_skip,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
