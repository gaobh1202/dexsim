#!/usr/bin/env python3
"""
代码参考playback_datasets.py，添加点云观察，用于构造3d diffusion policy训练数据
Replay robomimic-style hdf5 demos and augment each episode with third-view RGB-D,
camera parameters, and point clouds.

Added datasets under each demo group (prefix comes from --camera_name):
    /data/<demo>/obs/<camera>_image      uint8   [T, H, W, 3]
    /data/<demo>/obs/<camera>_depth      float32 [T, H, W]
    /data/<demo>/obs/<camera>_cam_param  float32 [T, 2, 4, 4]
    /data/<demo>/obs/<camera>_pc         float32 [T, N, 6]  (XYZRGB)

Where <camera>_cam_param[t, 0] is intrinsic K expanded to 4x4,
and <camera>_cam_param[t, 1] is camera_to_world 4x4.

Point clouds are built from the configured camera (default pointview) RGB-D,
then filtered to an axis-aligned WORKSPACE in world coordinates before uniform
random subsampling to N points (default 2048).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import h5py
import imageio
import numpy as np

import robosuite

# IMPORTANT: import package to register environments.
import dexmimicgen  # noqa: F401

from robosuite.utils.camera_utils import (  # type: ignore[import-not-found]
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

try:
    import cv2
except Exception:
    cv2 = None


# World-frame axis-aligned workspace for tabletop-focused point sampling.
# Calibrated with pointcloud_visualizer.py bbox (covers desktop region).
WORKSPACE_MIN = np.array([-0.6, -0.6, 0.5], dtype=np.float32)
WORKSPACE_MAX = np.array([0.6, 0.6, 1.5], dtype=np.float32)


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Load environment metadata from dataset root attrs.
    """
    dataset_path = os.path.expanduser(str(dataset_path))
    with h5py.File(dataset_path, "r") as f:
        if ds_format == "robomimic":
            env_meta = json.loads(f["data"].attrs["env_args"])
        else:
            raise ValueError(f"Unsupported dataset format: {ds_format}")
    return env_meta


def reset_to(env, state):
    """
    Reset environment to specific xml/state payload.
    """
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

        env.reset()
        robosuite_minor = int(robosuite.__version__.split(".")[1])
        if robosuite_minor <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml  # type: ignore[import-not-found]

            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _render_rgbd(sim, camera_name, height, width):
    """
    Render rgb + depth and convert depth to metric units.
    """
    rgb, depth_norm = sim.render(
        height=height,
        width=width,
        camera_name=camera_name,
        depth=True,
    )
    # Keep consistent with robosuite playback convention.
    rgb = np.asarray(rgb)[::-1, :, :]
    depth_norm = np.asarray(depth_norm)[::-1, :]
    depth = get_real_depth_map(sim, depth_norm).astype(np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    return rgb, depth


def _render_rgb(sim, camera_name, height, width):
    """
    Render RGB only (used by optional video recording path).
    """
    rgb = sim.render(
        height=height,
        width=width,
        camera_name=camera_name,
        depth=False,
    )
    rgb = np.asarray(rgb)[::-1, :, :]
    return np.asarray(rgb, dtype=np.uint8)


def _normalize_video_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]
    elif frame.ndim != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _depth_to_vis(depth):
    """
    Convert metric depth map to uint8 RGB visualization for quick inspection.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    d_min = float(np.percentile(depth[valid], 1))
    d_max = float(np.percentile(depth[valid], 99))
    if d_max <= d_min + 1e-8:
        d_max = d_min + 1e-8

    depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)

    if cv2 is not None:
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    else:
        depth_color = np.repeat(depth_u8[..., None], 3, axis=2)

    return depth_color.astype(np.uint8)


def _build_pc_vis_frame(rgb, depth):
    """
    Compose side-by-side visualization: left RGB, right pseudo-colored depth.
    """
    rgb = _normalize_video_frame(rgb)
    depth_vis = _depth_to_vis(depth)
    if depth_vis.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"Depth visualization shape {depth_vis.shape[:2]} does not match RGB shape {rgb.shape[:2]}"
        )
    return np.concatenate([rgb, depth_vis], axis=1)


def _save_demo_pointcloud_npz(
    pc_npz_dir,
    demo_name,
    camera_name,
    point_clouds,
    image_height,
    image_width,
    num_points,
):
    """
    Save per-demo point clouds to npz for train-vs-deploy statistics.
    """
    os.makedirs(pc_npz_dir, exist_ok=True)
    npz_path = Path(pc_npz_dir) / f"{demo_name}_{camera_name}_pc.npz"
    np.savez_compressed(
        npz_path,
        point_cloud=point_clouds.astype(np.float32),  # [T, N, 6] (XYZRGB)
        demo_name=np.array([demo_name]),
        camera_name=np.array([camera_name]),
        image_height=np.array([int(image_height)], dtype=np.int32),
        image_width=np.array([int(image_width)], dtype=np.int32),
        num_points=np.array([int(num_points)], dtype=np.int32),
        num_frames=np.array([int(point_clouds.shape[0])], dtype=np.int32),
    )
    return npz_path


class _OpenCVVideoWriter:
    def __init__(self, video_path, fps=20):
        if cv2 is None:
            raise RuntimeError("opencv-python is not available for video fallback writer.")
        self.video_path = str(video_path)
        self.fps = float(fps)
        self._writer = None

    def append_data(self, frame):
        frame = _normalize_video_frame(frame)
        h, w = frame.shape[:2]
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open cv2 VideoWriter for {self.video_path}")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _parse_n_argument(value):
    """
    Parse --n argument. Accepts positive integer or "all".
    Returns None when all demos should be kept.
    """
    text = str(value).strip().lower()
    if text == "all":
        return None
    try:
        n = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--n must be a positive integer or 'all'") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError("--n must be > 0, or use 'all'")
    return n


def _make_video_writer(video_path, fps=20):
    attempts = [dict(format="FFMPEG", codec="libx264")]
    last_err = None
    for opts in attempts:
        try:
            return imageio.get_writer(str(video_path), fps=float(fps), **opts)
        except Exception as e:
            last_err = e
            print(f"warning: failed to create imageio writer with options {opts}: {e}")
    try:
        return _OpenCVVideoWriter(video_path, fps=fps)
    except Exception as e:
        last_err = e
        raise RuntimeError(
            "Failed to create video writer. Install one of: "
            "`pip install imageio[ffmpeg]` or `pip install opencv-python`."
        ) from last_err


def _depth_to_world_points_with_rgb(depth, rgb, intrinsic, cam_to_world):
    """
    Unproject depth map (meters) into world-frame point cloud with per-point RGB.
    """
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"rgb and depth shape mismatch: rgb={rgb.shape}, depth={depth.shape}")

    h, w = depth.shape
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.astype(np.float32)
    x = (xs - cx) * z / max(fx, 1e-8)
    y = (ys - cy) * z / max(fy, 1e-8)

    points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-6)
    points_cam = points_cam[valid]
    colors = rgb.reshape(-1, 3)[valid]

    if points_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    points_cam_h = np.concatenate([points_cam.astype(np.float32), ones], axis=1)
    points_world_h = points_cam_h @ cam_to_world.T.astype(np.float32)
    return points_world_h[:, :3].astype(np.float32), colors.astype(np.uint8)


def _filter_points_in_workspace(points_world, colors, workspace_min, workspace_max):
    """
    Keep points whose XYZ lies inside the axis-aligned box [min, max] per axis (world frame).
    """
    if points_world.shape[0] == 0:
        return points_world, colors
    mn = np.asarray(workspace_min, dtype=np.float32).reshape(1, 3)
    mx = np.asarray(workspace_max, dtype=np.float32).reshape(1, 3)
    inside = np.all((points_world >= mn) & (points_world <= mx), axis=1)
    return points_world[inside], colors[inside]


def _sample_points_with_rgb(points, colors, num_points, rng):
    """
    Random sample or repeat sample to fixed point count for XYZ + RGB pairs.
    """
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points and colors count mismatch: points={points.shape[0]}, colors={colors.shape[0]}"
        )

    n = points.shape[0]
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32), np.zeros((num_points, 3), dtype=np.uint8)
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return points[idx].astype(np.float32), colors[idx].astype(np.uint8)


def _copy_hdf5_recursive(src, dst):
    """
    Copy all groups/datasets and attributes from src file to dst file.
    """
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key in src.keys():
        src.copy(key, dst)


def _build_env(dataset_path, render, camera_name, render_gpu_device_id):
    env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = bool(render)
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    env_kwargs["render_gpu_device_id"] = int(render_gpu_device_id)
    if "env_lang" in env_kwargs:
        env_kwargs.pop("env_lang")
    env = robosuite.make(**env_kwargs)
    env.reset()
    if render and env.viewer is None:
        env.initialize_renderer()
    if render and env.viewer is not None:
        try:
            env.viewer.set_camera(camera_name=camera_name)
        except Exception:
            pass
    return env


def _configure_gl_backend(mujoco_gl):
    """
    Configure MUJOCO_GL in runtime, following replay_drillgrasp style.
    """
    backend = (mujoco_gl or "default").lower()
    if backend == "default":
        backend = os.environ.get("MUJOCO_GL", "glx").lower()

    os.environ["MUJOCO_GL"] = backend

    # robosuite may force egl when GPU rendering macro is enabled.
    # For glfw / glx / osmesa requests, disable that override explicitly.
    if backend in ("glfw", "glx", "osmesa"):
        try:
            import robosuite.macros as macros  # type: ignore[import-not-found]

            macros.MUJOCO_GPU_RENDERING = False
        except Exception:
            pass

    print(f"Using MUJOCO_GL={backend}")
    return backend


def _resolve_camera_name(sim, requested_name):
    """
    Resolve camera name with a simple alias fallback (e.g. third_view -> thirdview).
    """
    candidates = [requested_name, requested_name.replace("_", "")]
    for name in candidates:
        try:
            sim.model.camera_name2id(name)
            return name
        except Exception:
            continue
    raise ValueError(f"Camera '{requested_name}' not found in current environment model")


def _camera_obs_keys(camera_name):
    """
    Build hdf5 observation key names for selected camera.
    """
    return {
        "image": f"{camera_name}_image",
        "depth": f"{camera_name}_depth",
        "cam_param": f"{camera_name}_cam_param",
        "pc": f"{camera_name}_pc",
    }


def _replace_visual_obs_datasets(obs_grp, camera_name):
    """
    Remove existing third-view-like visual datasets in-place.

    Keeps robot eye-in-hand streams (e.g. robot0_eye_in_hand_image).
    Deletes:
      - legacy plain keys: image / depth / cam_param / pc
      - selected camera-prefixed keys for replacement (thirdview + current camera)
    """
    legacy_keys = {"image", "depth", "cam_param", "pc"}
    replace_camera_names = {
        "thirdview",
        "third_view",
        "pointview",
        camera_name,
        camera_name.replace("_", ""),
    }
    visual_suffixes = ("_image", "_depth", "_cam_param", "_pc")
    to_delete = []
    for key in list(obs_grp.keys()):
        if key in legacy_keys:
            to_delete.append(key)
            continue

        # Preserve eye-in-hand observations.
        if "eye_in_hand" in key:
            continue

        if any(key.startswith(f"{cam}_") for cam in replace_camera_names) and key.endswith(
            visual_suffixes
        ):
            to_delete.append(key)
    for key in to_delete:
        del obs_grp[key]


def _set_viewer_camera_by_name(env, camera_name):
    """
    Set mjviewer camera by camera name.
    """
    if env.viewer is None:
        return
    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
        env.viewer.set_camera(cam_id)
    except Exception:
        pass


def _get_viewer_resolution(env, fallback_h, fallback_w):
    """
    Read current mjviewer window size. Fallback to configured image size.
    """
    h = int(fallback_h)
    w = int(fallback_w)
    try:
        viewer_handle = getattr(env.viewer, "viewer", None)
        viewport = getattr(viewer_handle, "viewport", None)
        if viewport is not None:
            vp_h = int(getattr(viewport, "height", 0))
            vp_w = int(getattr(viewport, "width", 0))
            if vp_h > 0 and vp_w > 0:
                h, w = vp_h, vp_w
    except Exception:
        pass
    return h, w


def _augment_demo(
    env,
    demo_grp,
    demo_name,
    camera_name,
    img_h,
    img_w,
    num_points,
    render,
    rng,
    use_current_model=False,
    video_writer=None,
    video_skip=1,
    video_source="offscreen",
    video_height=None,
    video_width=None,
    replace_visual_obs=False,
    pc_vis_writer=None,
    pc_vis_skip=1,
    pc_npz_dir=None,
):
    states = demo_grp["states"][()]
    initial_state = {"states": states[0]}
    initial_state["model"] = demo_grp.attrs["model_file"]
    if use_current_model:
        initial_state["model"] = env.sim.model.get_xml()
    initial_state["ep_meta"] = demo_grp.attrs.get("ep_meta", None)
    reset_to(env, initial_state)
    keys = _camera_obs_keys(camera_name)

    images = []
    depths = []
    cam_params = []
    point_clouds = []

    for step_idx in range(states.shape[0]):
        tic = time.time()
        reset_to(env, {"states": states[step_idx]})

        rgb, depth = _render_rgbd(env.sim, camera_name=camera_name, height=img_h, width=img_w)
        if pc_vis_writer is not None and (step_idx % max(int(pc_vis_skip), 1) == 0):
            pc_vis_frame = _build_pc_vis_frame(rgb=rgb, depth=depth)
            pc_vis_writer.append_data(pc_vis_frame)
        if video_writer is not None and (step_idx % max(int(video_skip), 1) == 0):
            if video_source == "viewer":
                vh = int(video_height) if video_height is not None else int(img_h)
                vw = int(video_width) if video_width is not None else int(img_w)
                video_rgb = _render_rgb(env.sim, camera_name=camera_name, height=vh, width=vw)
                video_writer.append_data(video_rgb)
            else:
                video_writer.append_data(rgb)
        intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, img_h, img_w).astype(np.float32)
        cam_to_world = get_camera_extrinsic_matrix(env.sim, camera_name).astype(np.float32)
        points_world, point_colors = _depth_to_world_points_with_rgb(
            depth, rgb, intrinsic, cam_to_world
        )
        points_world, point_colors = _filter_points_in_workspace(
            points_world, point_colors, WORKSPACE_MIN, WORKSPACE_MAX
        )
        sampled_pc, sampled_pc_rgb = _sample_points_with_rgb(
            points_world, point_colors, num_points=num_points, rng=rng
        )
        sampled_pc_xyzrgb = np.concatenate(
            [sampled_pc, sampled_pc_rgb.astype(np.float32)], axis=-1
        )

        k_4x4 = np.eye(4, dtype=np.float32)
        k_4x4[:3, :3] = intrinsic
        cam_param = np.stack([k_4x4, cam_to_world], axis=0)

        images.append(rgb)
        depths.append(depth)
        cam_params.append(cam_param)
        point_clouds.append(sampled_pc_xyzrgb)

        if render and env.viewer is not None:
            # Keep the visible window camera aligned with collected camera frames.
            _set_viewer_camera_by_name(env, camera_name)
            env.viewer.update()
            delay = (1.0 / 60.0) - (time.time() - tic)
            if delay > 0:
                time.sleep(delay)

    images = np.asarray(images, dtype=np.uint8)
    depths = np.asarray(depths, dtype=np.float32)
    cam_params = np.asarray(cam_params, dtype=np.float32)
    point_clouds = np.asarray(point_clouds, dtype=np.float32)

    obs_grp = demo_grp.require_group("obs")
    if replace_visual_obs:
        _replace_visual_obs_datasets(obs_grp, camera_name=camera_name)
    else:
        for key in keys.values():
            if key in obs_grp:
                del obs_grp[key]
    obs_grp.create_dataset(keys["image"], data=images, compression="gzip")
    obs_grp.create_dataset(keys["depth"], data=depths, compression="gzip")
    obs_grp.create_dataset(keys["cam_param"], data=cam_params, compression="gzip")
    obs_grp.create_dataset(keys["pc"], data=point_clouds, compression="gzip")

    if pc_npz_dir:
        npz_path = _save_demo_pointcloud_npz(
            pc_npz_dir=pc_npz_dir,
            demo_name=demo_name,
            camera_name=camera_name,
            point_clouds=point_clouds,
            image_height=img_h,
            image_width=img_w,
            num_points=num_points,
        )
        print(f"saved point cloud npz for {demo_name} to: {npz_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="input hdf5 dataset path")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output hdf5 path; default <input>_with_pointview_pc.hdf5",
    )
    parser.add_argument("--camera_name", type=str, default="pointview", help="camera name to render")
    parser.add_argument(
        "--replace_visual_obs",
        action="store_true",
        help="delete existing visual obs datasets (*_image/*_depth/*_cam_param/*_pc and image/depth/cam_param/pc) before writing new camera data",
    )
    parser.add_argument(
        "--use_current_model",
        action="store_true",
        help="use current env model instead of dataset model_file (useful for newly added cameras in xml)",
    )
    parser.add_argument("--image_height", type=int, default=128, help="render image height")
    parser.add_argument("--image_width", type=int, default=128, help="render image width")
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="points per frame after WORKSPACE crop and random subsample (default: 2048)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for point sampling")
    parser.add_argument("--no_render", action="store_true", help="disable on-screen thirdview window")
    parser.add_argument(
        "--mujoco_gl",
        "--mujoco-gl",
        dest="mujoco_gl",
        type=str,
        default="egl",
        choices=["default", "glfw", "glx", "egl", "osmesa"],
        help="MUJOCO_GL backend. Default is egl.",
    )
    parser.add_argument(
        "--render_gpu_device_id",
        type=int,
        default=1,
        help="GPU index for offscreen rendering context (used by egl backend).",
    )
    parser.add_argument(
        "--n",
        type=_parse_n_argument,
        default=None,
        metavar="N|all",
        help="number of demos to keep/process, or 'all' for full dataset. Output hdf5 only contains selected demos.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="optional output video path (if empty, do not save video)",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="write one frame every n simulation steps when saving video",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=20.0,
        help="video fps when saving video",
    )
    parser.add_argument(
        "--video_source",
        type=str,
        default="offscreen",
        choices=["offscreen", "viewer"],
        help="video frame source: offscreen uses image_height/width; viewer uses current window resolution",
    )
    parser.add_argument(
        "--pc_vis_path",
        type=str,
        default="",
        help="optional output video path for RGB+Depth frames used to build point clouds",
    )
    parser.add_argument(
        "--pc_vis_skip",
        type=int,
        default=1,
        help="write one RGB+Depth visualization frame every n steps",
    )
    parser.add_argument(
        "--pc_vis_fps",
        type=float,
        default=20.0,
        help="fps for RGB+Depth visualization video",
    )
    parser.add_argument(
        "--pc_npz_dir",
        type=str,
        default="",
        help=(
            "optional output dir to save per-demo point clouds as npz files. "
            "Each processed demo writes one file: <demo_name>_<camera>_pc.npz"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _configure_gl_backend(args.mujoco_gl)
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if args.output.strip():
        output_path = Path(args.output).expanduser()
    else:
        output_path = dataset_path.with_name(f"{dataset_path.stem}_with_{args.camera_name}_pc.hdf5")

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    video_writer = None
    pc_vis_writer = None

    with h5py.File(dataset_path, "r") as fin, h5py.File(output_path, "w") as fout:
        _copy_hdf5_recursive(fin, fout)

        if "data" not in fout:
            raise RuntimeError("Invalid dataset: missing /data group")
        demos = list(fout["data"].keys())
        demos = sorted(demos, key=lambda x: (0, int(x[5:])) if x.startswith("demo_") else (1, str(x)))
        if args.n is not None:
            random.shuffle(demos)
            demos = demos[: args.n]
            selected_demo_names = set(demos)
            for demo_name in list(fout["data"].keys()):
                if demo_name not in selected_demo_names:
                    del fout["data"][demo_name]

        if args.video_path.strip():
            video_path = Path(args.video_path).expanduser()
        else:
            video_path = None
        if video_path is not None:
            video_writer = _make_video_writer(video_path=video_path, fps=float(args.video_fps))
            print(f"Saving playback video to: {video_path}")
        pc_vis_path = None
        if args.pc_vis_path.strip():
            pc_vis_path = Path(args.pc_vis_path).expanduser()
            pc_vis_writer = _make_video_writer(video_path=pc_vis_path, fps=float(args.pc_vis_fps))
            print(f"Saving point-cloud RGB+Depth visualization video to: {pc_vis_path}")

        env = _build_env(
            dataset_path=dataset_path,
            render=(not args.no_render),
            camera_name=args.camera_name,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        resolved_camera_name = _resolve_camera_name(env.sim, args.camera_name)
        if resolved_camera_name != args.camera_name:
            print(f"camera '{args.camera_name}' not found, fallback to '{resolved_camera_name}'")
        _set_viewer_camera_by_name(env, resolved_camera_name)

        video_height = int(args.image_height)
        video_width = int(args.image_width)
        if video_writer is not None and args.video_source == "viewer":
            if args.no_render:
                raise ValueError("--video_source viewer requires on-screen rendering (remove --no_render).")
            if env.viewer is None:
                env.initialize_renderer()
            env.viewer.update()
            _set_viewer_camera_by_name(env, resolved_camera_name)
            video_height, video_width = _get_viewer_resolution(
                env,
                fallback_h=int(args.image_height),
                fallback_w=int(args.image_width),
            )
            print(f"Viewer video resolution: {video_width}x{video_height}")
        try:
            for i, demo_name in enumerate(demos):
                print(f"[{i + 1}/{len(demos)}] augmenting {demo_name}")
                _augment_demo(
                    env=env,
                    demo_grp=fout[f"data/{demo_name}"],
                    demo_name=demo_name,
                    camera_name=resolved_camera_name,
                    img_h=int(args.image_height),
                    img_w=int(args.image_width),
                    num_points=int(args.num_points),
                    render=(not args.no_render),
                    rng=rng,
                    use_current_model=args.use_current_model,
                    video_writer=video_writer,
                    video_skip=max(1, int(args.video_skip)),
                    video_source=args.video_source,
                    video_height=video_height,
                    video_width=video_width,
                    replace_visual_obs=bool(args.replace_visual_obs),
                    pc_vis_writer=pc_vis_writer,
                    pc_vis_skip=max(1, int(args.pc_vis_skip)),
                    pc_npz_dir=args.pc_npz_dir.strip() if args.pc_npz_dir else None,
                )
        finally:
            try:
                env.close()
            except Exception:
                pass
            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception:
                    pass
            if pc_vis_writer is not None:
                try:
                    pc_vis_writer.close()
                except Exception:
                    pass

        fout.attrs["thirdview_augmentation_info"] = json.dumps(
            {
                "camera_name": args.camera_name,
                "workspace_min": WORKSPACE_MIN.astype(float).tolist(),
                "workspace_max": WORKSPACE_MAX.astype(float).tolist(),
                "image_height": int(args.image_height),
                "image_width": int(args.image_width),
                "num_points": int(args.num_points),
                "seed": int(args.seed),
                "render_window": bool(not args.no_render),
                "mujoco_gl": args.mujoco_gl,
                "render_gpu_device_id": int(args.render_gpu_device_id),
                "use_current_model": bool(args.use_current_model),
                "video_path": str(video_path) if video_path is not None else "",
                "video_skip": int(args.video_skip),
                "video_fps": float(args.video_fps),
                "video_source": args.video_source,
                "video_height": int(video_height),
                "video_width": int(video_width),
                "pc_vis_path": str(pc_vis_path) if pc_vis_path is not None else "",
                "pc_vis_skip": int(args.pc_vis_skip),
                "pc_vis_fps": float(args.pc_vis_fps),
                "pc_npz_dir": str(args.pc_npz_dir) if args.pc_npz_dir else "",
            },
            indent=2,
        )

    print(f"Saved augmented dataset to: {output_path}")


if __name__ == "__main__":
    main()