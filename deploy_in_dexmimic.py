"""
修改了部署时的point获取方式，和数据集中的一致
目测一致，下一步考虑修改state和action；
以及在数据集中查看点云形状
"""
import os
import pathlib
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace  # type: ignore[import-not-found]
from robosuite.utils.camera_utils import (  # type: ignore[import-not-found]
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_state(obs: Dict[str, np.ndarray]) -> np.ndarray:
    state_keys = [
        "robot0_joint_pos",
        "robot0_gripper_qpos",
        "robot1_joint_pos",
        "robot1_gripper_qpos",
    ]
    missing_keys = [k for k in state_keys if k not in obs]
    if missing_keys:
        raise KeyError(
            f"Missing keys in env obs for state construction: {missing_keys}. "
            "Expected dexmimic style keys."
        )
    state = np.concatenate([_to_numpy(obs[k]).reshape(-1) for k in state_keys], axis=0)
    return state.astype(np.float32)


def _depth_to_world_points(
    depth: np.ndarray, intrinsic: np.ndarray, cam_to_world: np.ndarray
) -> np.ndarray:
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
    if points_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    points_cam_h = np.concatenate([points_cam.astype(np.float32), ones], axis=1)
    points_world_h = points_cam_h @ cam_to_world.T.astype(np.float32)
    return points_world_h[:, :3].astype(np.float32)


def _sample_points(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if n >= num_points:
        idx = rng.choice(n, size=num_points, replace=False)
    else:
        idx = rng.choice(n, size=num_points, replace=True)
    return points[idx].astype(np.float32)


def _render_rgbd(
    sim,
    camera_name: str,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb, depth_norm = sim.render(
        height=image_height,
        width=image_width,
        camera_name=camera_name,
        depth=True,
    )
    rgb = np.asarray(rgb)[::-1, :, :3]
    depth_norm = np.asarray(depth_norm)[::-1, :]
    depth = get_real_depth_map(sim, depth_norm).astype(np.float32)
    return rgb, depth


def _extract_point_cloud_from_render(
    env,
    num_points: int,
    camera_name: str,
    image_height: int,
    image_width: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb, depth = _render_rgbd(
        sim=env.sim,
        camera_name=camera_name,
        image_height=image_height,
        image_width=image_width,
    )
    intrinsic = get_camera_intrinsic_matrix(
        env.sim, camera_name, image_height, image_width
    ).astype(np.float32)
    cam_to_world = get_camera_extrinsic_matrix(env.sim, camera_name).astype(np.float32)
    points_world = _depth_to_world_points(depth, intrinsic, cam_to_world)
    return _sample_points(points_world, num_points, rng), rgb, depth


def _depth_to_display(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 1e-6)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [5, 95])
        if high <= low:
            high = low + 1e-6
        depth_norm = np.clip((depth - low) / (high - low), 0.0, 1.0)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    return (depth_norm * 255).astype(np.uint8)


def _resolve_camera_name(sim, requested_name: str) -> str:
    candidates = [requested_name, requested_name.replace("_", "")]
    for name in candidates:
        try:
            sim.model.camera_name2id(name)
            return name
        except Exception:
            continue
    raise ValueError(f"Camera '{requested_name}' not found in current environment model")


def _configure_gl_backend(mujoco_gl: str) -> str:
    backend = (mujoco_gl or "default").lower()
    if backend == "default":
        backend = os.environ.get("MUJOCO_GL", "egl").lower()
    os.environ["MUJOCO_GL"] = backend

    if backend in ("glfw", "glx", "osmesa"):
        try:
            import robosuite.macros as macros  # type: ignore[import-not-found]

            macros.MUJOCO_GPU_RENDERING = False
        except Exception:
            pass
    return backend


def _extract_success(env, info: Dict) -> bool:
    for key in ("success", "task_success", "is_success"):
        if key in info:
            return bool(info[key])
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False


def _to_action_list(policy_out, action_dim: int) -> List[np.ndarray]:
    if isinstance(policy_out, dict):
        for k in ("action", "action_pred"):
            if k in policy_out:
                policy_out = policy_out[k]
                break
        else:
            raise KeyError(
                f"Unknown policy output keys: {list(policy_out.keys())}, "
                "expected one of ['action', 'action_pred']."
            )

    action_seq = _to_numpy(policy_out)
    if action_seq.ndim == 3:
        action_seq = action_seq[0]
    elif action_seq.ndim == 1:
        action_seq = action_seq[None]
    if action_seq.ndim != 2:
        raise ValueError(f"Unexpected action_seq shape: {action_seq.shape}")
    if action_seq.shape[-1] < action_dim:
        raise ValueError(f"Policy output dim {action_seq.shape[-1]} < expected {action_dim}.")

    action_list = [a[:action_dim].astype(np.float32) for a in action_seq]
    if not action_list:
        raise RuntimeError("Policy returned empty action sequence.")
    return action_list


def _save_episode_record(
    env,
    record_dir: str,
    episode_idx: int,
    roll_out_length: int,
    step_records: Optional[List[Dict[str, np.ndarray]]] = None,
) -> str:
    import h5py

    os.makedirs(record_dir, exist_ok=True)
    record_file_name = os.path.join(record_dir, f"demo_ep{episode_idx + 1:03d}.h5")

    color_array = np.array(getattr(env, "color_array", []))
    depth_array = np.array(getattr(env, "depth_array", []))
    cloud_array = np.array(getattr(env, "cloud_array", []))
    qpos_array = np.array(getattr(env, "qpos_array", []))

    with h5py.File(record_file_name, "w") as f:
        if color_array.size > 0:
            f.create_dataset("color", data=color_array)
        if depth_array.size > 0:
            f.create_dataset("depth", data=depth_array)
        if cloud_array.size > 0:
            f.create_dataset("cloud", data=cloud_array)
        if qpos_array.size > 0:
            f.create_dataset("qpos", data=qpos_array)

        # Fallback recording path when env does not expose *_array buffers.
        if step_records:
            f.create_dataset("state", data=np.asarray([r["state"] for r in step_records], dtype=np.float32))
            f.create_dataset(
                "point_cloud",
                data=np.asarray([r["point_cloud"] for r in step_records], dtype=np.float32),
            )
            f.create_dataset("action", data=np.asarray([r["action"] for r in step_records], dtype=np.float32))
            f.create_dataset("success", data=np.asarray([r["success"] for r in step_records], dtype=np.bool_))
        f.attrs["roll_out_length"] = int(roll_out_length)

    return record_file_name


def _save_policy_point_cloud_rollout(
    output_dir: str,
    episode_idx: int,
    point_cloud_records: List[np.ndarray],
    roll_out_length: int,
    camera_name: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"policy_point_cloud_ep{episode_idx + 1:03d}.npz")
    if point_cloud_records:
        point_cloud_seq = np.asarray(point_cloud_records, dtype=np.float32)
    else:
        point_cloud_seq = np.zeros((0, 0, 3), dtype=np.float32)
    np.savez_compressed(
        output_path,
        point_cloud=point_cloud_seq,
        roll_out_length=np.int32(roll_out_length),
        camera_name=np.asarray(camera_name),
    )
    return output_path


class DexMimicEnvInference:
    def __init__(
        self,
        env,
        obs_horizon: int,
        action_horizon: int,
        num_points: int,
        device: torch.device,
        camera_name: str,
        image_height: int,
        image_width: int,
        seed: int,
        show_reconstruction_images: bool = False,
        display_every_n_steps: int = 1,
    ):
        self.env = env
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_points = num_points
        self.device = device
        self.camera_name = camera_name
        self.image_height = image_height
        self.image_width = image_width
        self.rng = np.random.default_rng(seed)
        self.show_reconstruction_images = bool(show_reconstruction_images)
        self.display_every_n_steps = max(int(display_every_n_steps), 1)

        self.state_buffer = deque(maxlen=obs_horizon)
        self.point_cloud_buffer = deque(maxlen=obs_horizon)
        self.display_step_counter = 0
        self.display_window_name = "pointcloud_reconstruction_rgb_depth"
        self.cv2 = None

        if self.show_reconstruction_images:
            try:
                import cv2
                self.cv2 = cv2
            except Exception as exc:
                cprint(
                    f"show_reconstruction_images=True but OpenCV import failed: {exc}. "
                    "Disable image visualization.",
                    "yellow",
                )
                self.show_reconstruction_images = False

    def _build_obs_dict(self):
        state_seq = np.stack(list(self.state_buffer), axis=0)
        pc_seq = np.stack(list(self.point_cloud_buffer), axis=0)
        return {
            "agent_pos": torch.from_numpy(state_seq).unsqueeze(0).to(self.device),
            "point_cloud": torch.from_numpy(pc_seq).unsqueeze(0).to(self.device),
        }

    def _append_obs(self, raw_obs: Dict[str, np.ndarray]):
        state = _extract_state(raw_obs)
        point_cloud, rgb, depth = _extract_point_cloud_from_render(
            env=self.env,
            num_points=self.num_points,
            camera_name=self.camera_name,
            image_height=self.image_height,
            image_width=self.image_width,
            rng=self.rng,
        )
        self._maybe_show_reconstruction_images(rgb, depth)
        self.state_buffer.append(state)
        self.point_cloud_buffer.append(point_cloud)
        return state, point_cloud

    def _maybe_show_reconstruction_images(self, rgb: np.ndarray, depth: np.ndarray):
        if not self.show_reconstruction_images or self.cv2 is None:
            return
        self.display_step_counter += 1
        if self.display_step_counter % self.display_every_n_steps != 0:
            return
        try:
            rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb_bgr = self.cv2.cvtColor(rgb_u8, self.cv2.COLOR_RGB2BGR)
            depth_u8 = _depth_to_display(depth)
            depth_color = self.cv2.applyColorMap(depth_u8, self.cv2.COLORMAP_TURBO)
            panel = np.concatenate([rgb_bgr, depth_color], axis=1)
            self.cv2.putText(
                panel,
                "RGB (for point cloud)",
                (10, 24),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
            self.cv2.putText(
                panel,
                "Depth (for point cloud)",
                (rgb_bgr.shape[1] + 10, 24),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
            self.cv2.imshow(self.display_window_name, panel)
            self.cv2.waitKey(1)
        except Exception as exc:
            cprint(
                f"Failed to display reconstruction images ({exc}), disable visualization.",
                "yellow",
            )
            self.show_reconstruction_images = False

    def close(self):
        if self.cv2 is not None:
            self.cv2.destroyWindow(self.display_window_name)

    def reset(self):
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            raw_obs = reset_out[0]
        else:
            raw_obs = reset_out

        self.state_buffer.clear()
        self.point_cloud_buffer.clear()
        self._append_obs(raw_obs)
        while len(self.state_buffer) < self.obs_horizon:
            self.state_buffer.append(self.state_buffer[-1].copy())
            self.point_cloud_buffer.append(self.point_cloud_buffer[-1].copy())
        return self._build_obs_dict()

    def step(
        self,
        action_list: List[np.ndarray],
        step_records: Optional[List[Dict[str, np.ndarray]]] = None,
        point_cloud_records: Optional[List[np.ndarray]] = None,
    ):
        done = False
        success = False
        env_steps = 0
        for i in range(min(self.action_horizon, len(action_list))):
            action = np.asarray(action_list[i], dtype=np.float32).reshape(-1)
            step_out = self.env.step(action)
            if len(step_out) == 5:
                raw_obs, _, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                raw_obs, _, done, info = step_out
                done = bool(done)
            success = _extract_success(self.env, info)

            state, point_cloud = self._append_obs(raw_obs)
            if point_cloud_records is not None:
                point_cloud_records.append(point_cloud.copy())
            if step_records is not None:
                step_records.append(
                    {
                        "state": state.copy(),
                        "point_cloud": point_cloud.copy(),
                        "action": action.copy(),
                        "success": np.array(success, dtype=np.bool_),
                    }
                )
            env_steps += 1
            if done:
                break

        return self._build_obs_dict(), done, success, env_steps


def create_two_arm_drawer_cleanup_env(
    max_steps: int,
    camera_name: str = "pointview",
    mujoco_gl: str = "egl",
    render_gpu_device_id: int = 1,
    render_window: bool = True,
):
    has_renderer = bool(render_window)
    import dexmimicgen  # type: ignore[import-not-found]  # noqa: F401
    import robosuite as suite  # type: ignore[import-not-found]
    from robosuite import load_composite_controller_config  # type: ignore[import-not-found]
    requested_backend = str(mujoco_gl).lower()
    requested_device_id = int(render_gpu_device_id)
    backend_attempts: List[Tuple[str, int]] = [(requested_backend, requested_device_id)]
    if requested_backend == "egl":
        # Common on desktop machines without headless EGL support.
        backend_attempts.append(("glx", -1))

    last_exc = None
    for idx, (backend_name, device_id) in enumerate(backend_attempts):
        backend = _configure_gl_backend(backend_name)
        if backend == "egl":
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device_id)
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

        cprint(f"Using MUJOCO_GL={os.environ.get('MUJOCO_GL')}", "cyan")
        cprint(
            "Render config: "
            f"backend={backend}, "
            f"effective_render_gpu_device_id={device_id}, "
            f"render_window={has_renderer}, "
            f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '<unset>')}, "
            f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID', '<unset>')}",
            "cyan",
        )
        try:
            controller_configs = [
                load_composite_controller_config(robot="PandaDexRH"),
                load_composite_controller_config(robot="PandaDexLH"),
            ]
            env = suite.make(
                "TwoArmDrawerCleanup",
                robots=["PandaDexRH", "PandaDexLH"],
                controller_configs=controller_configs,
                env_configuration="default",
                use_camera_obs=False,
                use_object_obs=True,
                has_renderer=has_renderer,
                has_offscreen_renderer=True,
                render_camera=camera_name,
                control_freq=20,
                horizon=max_steps,
                ignore_done=False,
                hard_reset=True,
                render_gpu_device_id=device_id,
                renderer="mjviewer" if has_renderer else "mujoco",
            )
            if has_renderer and getattr(env, "viewer", None) is None:
                env.initialize_renderer()
            if has_renderer and getattr(env, "viewer", None) is not None:
                try:
                    cam_id = env.sim.model.camera_name2id(camera_name)
                    env.viewer.set_camera(cam_id)
                except Exception:
                    pass
            if idx > 0:
                cprint(
                    f"Fell back to backend '{backend}' after '{requested_backend}' failed.",
                    "yellow",
                )
            return env
        except Exception as exc:
            last_exc = exc
            if idx < len(backend_attempts) - 1:
                cprint(
                    f"Failed to create env with backend '{backend}'. "
                    f"Trying fallback backend '{backend_attempts[idx + 1][0]}'.",
                    "yellow",
                )
            continue

    raise RuntimeError(
        "Failed to create TwoArmDrawerCleanup env. "
        "Ensure dexmimicgen + robosuite are installed and the env is registered."
    ) from last_exc


def _ensure_policy_normalizer(policy, cfg: OmegaConf):
    required_keys = {"action", "agent_pos", "point_cloud"}
    normalizer_keys = set()
    if hasattr(policy, "normalizer") and hasattr(policy.normalizer, "params_dict"):
        normalizer_keys = set(policy.normalizer.params_dict.keys())

    if required_keys.issubset(normalizer_keys):
        return

    cprint(
        f"Policy normalizer keys {sorted(list(normalizer_keys))} incomplete, "
        "refitting from deployment dataset.",
        "yellow",
    )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config"))
)
def main(cfg: OmegaConf):
    deploy_default = OmegaConf.create(
        {
            "deploy": {
                "episodes": 5,
                "max_steps": 300,
                "device": "cuda",
                "camera_name": "pointview",
                "mujoco_gl": "egl",
                "render_gpu_device_id": 1,
                "render_window": True,
                "image_height": 128,
                "image_width": 128,
                "show_reconstruction_images": False,
                "display_every_n_steps": 1,
                "record_data": False,
                "record_dir": "./deploy_record",
                "save_policy_point_cloud": False,
                "policy_point_cloud_dir": "./deploy_policy_point_cloud",
            }
        }
    )
    cfg = OmegaConf.merge(deploy_default, cfg)

    seed = int(cfg.training.seed) if "training" in cfg and "seed" in cfg.training else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    if hasattr(workspace, "get_checkpoint_path"):
        ckpt_path = workspace.get_checkpoint_path(tag="latest")
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. "
                "Please verify hydra.run.dir and training outputs."
            )
    policy = workspace.get_model()
    _ensure_policy_normalizer(policy, cfg)

    if str(cfg.deploy.device) == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    policy.to(device)
    policy.eval()

    obs_horizon = policy.n_obs_steps
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    num_points = int(cfg.task.shape_meta.obs.point_cloud.shape[0])
    action_dim = int(np.prod(cfg.task.shape_meta.action.shape))

    env = create_two_arm_drawer_cleanup_env(
        max_steps=int(cfg.deploy.max_steps),
        camera_name=str(cfg.deploy.camera_name),
        mujoco_gl=str(cfg.deploy.mujoco_gl),
        render_gpu_device_id=int(cfg.deploy.render_gpu_device_id),
        render_window=bool(cfg.deploy.render_window),
    )
    resolved_camera_name = _resolve_camera_name(env.sim, str(cfg.deploy.camera_name))
    if resolved_camera_name != str(cfg.deploy.camera_name):
        cprint(
            f"camera '{cfg.deploy.camera_name}' not found, fallback to '{resolved_camera_name}'",
            "yellow",
        )

    env_action_dim = int(env.action_spec[0].shape[0])
    if env_action_dim != action_dim:
        raise ValueError(
            f"Action dim mismatch: cfg expects {action_dim}, env expects {env_action_dim}. "
            "Please check robot / controller_configs used at deployment."
        )
    infer_env = DexMimicEnvInference(
        env=env,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_points=num_points,
        device=device,
        camera_name=resolved_camera_name,
        image_height=int(cfg.deploy.image_height),
        image_width=int(cfg.deploy.image_width),
        seed=seed,
        show_reconstruction_images=bool(cfg.deploy.show_reconstruction_images),
        display_every_n_steps=int(cfg.deploy.display_every_n_steps),
    )

    cprint(f"Policy loaded, action_horizon={action_horizon}, action_dim={action_dim}", "cyan")
    cprint(f"Start deployment in TwoArmDrawerCleanup for {int(cfg.deploy.episodes)} episodes", "cyan")

    success_count = 0
    episode_returns = []
    episode_lengths = []

    for episode_idx in range(int(cfg.deploy.episodes)):
        obs_dict = infer_env.reset()
        done = False
        step_count = 0
        episode_return = 0.0
        success = False
        episode_step_records = [] if bool(cfg.deploy.record_data) else None
        episode_policy_point_cloud_records = (
            [] if bool(cfg.deploy.save_policy_point_cloud) else None
        )
        if episode_policy_point_cloud_records is not None:
            # The first observation used by policy comes from env reset.
            episode_policy_point_cloud_records.append(
                infer_env.point_cloud_buffer[-1].copy()
            )

        while not done and step_count < int(cfg.deploy.max_steps):
            with torch.no_grad():
                action_list = _to_action_list(policy(obs_dict), action_dim=action_dim)

            obs_dict, done, success, env_steps = infer_env.step(
                action_list,
                step_records=episode_step_records,
                point_cloud_records=episode_policy_point_cloud_records,
            )
            step_count += env_steps
            if success:
                done = True
                episode_return = 1.0

        success_count += int(success)
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        cprint(
            f"[Episode {episode_idx + 1}/{int(cfg.deploy.episodes)}] "
            f"success={success} steps={step_count} return={episode_return:.2f}",
            "yellow",
        )
        if bool(cfg.deploy.record_data):
            record_path = _save_episode_record(
                env=env,
                record_dir=str(cfg.deploy.record_dir),
                episode_idx=episode_idx,
                roll_out_length=step_count,
                step_records=episode_step_records,
            )
            cprint(f"saved rollout data ({step_count} steps) to {record_path}", "yellow")
        if episode_policy_point_cloud_records is not None:
            point_cloud_path = _save_policy_point_cloud_rollout(
                output_dir=str(cfg.deploy.policy_point_cloud_dir),
                episode_idx=episode_idx,
                point_cloud_records=episode_policy_point_cloud_records,
                roll_out_length=step_count,
                camera_name=resolved_camera_name,
            )
            cprint(
                f"saved policy point cloud rollout ({len(episode_policy_point_cloud_records)} frames) "
                f"to {point_cloud_path}",
                "yellow",
            )

    success_rate = success_count / max(int(cfg.deploy.episodes), 1)
    avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    cprint("==== Deployment Result (DexMimic TwoArmDrawerCleanup) ====", "green")
    cprint(f"success_rate: {success_rate:.3f}", "green")
    cprint(f"avg_return:   {avg_return:.3f}", "green")
    cprint(f"avg_length:   {avg_len:.1f}", "green")

    infer_env.close()
    env.close()


if __name__ == "__main__":
    main()
