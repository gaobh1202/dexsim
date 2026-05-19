"""
将 DexMimicGen hdf5 数据转换为 DP3 训练使用的 zarr 数据集。

hdf5 结构:
- 顶层 keys: ['data']
- data keys: ['demo_0', 'demo_1', ...]
- demo_i keys: ['action_dict', 'actions', 'datagen_info', 'obs', 'states']

字段映射:
- action:
  直接使用 /data/demo_i/actions
  常见含义:
    [
      right_gripper_rel_pos(3), right_gripper_rel_axis_angle(3), right_gripper(6),
      left_gripper_rel_pos(3),  left_gripper_rel_axis_angle(3),  left_gripper(6)
    ]
- img:
  使用 /obs/robot0_eye_in_hand_image 与 /obs/robot1_eye_in_hand_image
  按相机维拼接为 (T, 2, 84, 84, 3)
- point_cloud:
  使用 /obs/thirdview_pc
- state:
  与 policy 训练 low-dim observation 对齐，按顺序拼接:
    /obs/robot0_eef_pos + /obs/robot0_eef_quat + /obs/robot0_gripper_qpos +
    /obs/robot1_eef_pos + /obs/robot1_eef_quat + /obs/robot1_gripper_qpos
- traj_gen (可选):
  当 --add_traj 开启时，从 /trajectory_gen 提取。
  若为 group 且包含 robot0_eef_traj 与 robot1_eef_traj，
  则按机器人维拼接为 (T, 2, 4, 4)。
- phase_idx (可选):
  当 --add_phase_idx 开启时，从 /phase_idx 提取。
  支持以下两种格式:
  1) /phase_idx 为 dataset
  2) /phase_idx/value 为 dataset
  若为 (2, T) 会自动转为 (T, 2) 后保存。

输出 zarr:
- data/img
- data/point_cloud
- data/action
- data/state
- data/traj_gen (可选)
- data/phase_idx (可选)
- meta/episode_ends
"""

import argparse
import os
import re
import shutil

import h5py
import numpy as np
import tqdm
import zarr
from termcolor import cprint


def sorted_demo_keys(data_group):
    """按 demo_0, demo_1, ... 的数字顺序返回 key。"""
    demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]

    def key_fn(name):
        match = re.match(r"demo_(\d+)$", name)
        return int(match.group(1)) if match else name

    return sorted(demo_keys, key=key_fn)


def normalize_image_array(images, key_name):
    """
    将图像标准化为 (T, 84, 84, 3) uint8。
    """
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"{key_name} should be 4D, got shape {images.shape}")

    # 支持 (T, C, H, W) 或 (T, H, W, C)
    if images.shape[-1] in (3, 4):
        pass
    elif images.shape[1] in (3, 4):
        images = np.transpose(images, (0, 2, 3, 1))
    else:
        raise ValueError(f"{key_name} channel dim not found, got shape {images.shape}")

    # 只保留 RGB
    if images.shape[-1] == 4:
        images = images[..., :3]

    if images.shape[1:4] != (84, 84, 3):
        raise ValueError(
            f"{key_name} must be (T,84,84,3), got shape {images.shape}"
        )

    if np.issubdtype(images.dtype, np.floating):
        if images.max() <= 1.0:
            images = images * 255.0
        images = np.clip(images, 0.0, 255.0).astype(np.uint8)
    else:
        images = np.clip(images, 0, 255).astype(np.uint8)
    return images


def load_trajectory_array(demo_group, demo_key):
    """
    读取 demo 下的 trajectory_gen 并转换为统一数组格式。

    支持两种情况:
    1) trajectory_gen 是 dataset，直接读取；
    2) trajectory_gen 是 group，包含 robot0_eef_traj 和 robot1_eef_traj，
       输出为 (T, 2, 4, 4)。
    """
    if "trajectory_gen" not in demo_group:
        raise KeyError(f"{demo_key} does not contain 'trajectory_gen'")

    traj_obj = demo_group["trajectory_gen"]
    if isinstance(traj_obj, h5py.Dataset):
        traj = np.asarray(traj_obj, dtype=np.float32)
        return traj

    if isinstance(traj_obj, h5py.Group):
        if "robot0_eef_traj" in traj_obj and "robot1_eef_traj" in traj_obj:
            robot0_traj = np.asarray(traj_obj["robot0_eef_traj"], dtype=np.float32)
            robot1_traj = np.asarray(traj_obj["robot1_eef_traj"], dtype=np.float32)
            if robot0_traj.shape != robot1_traj.shape:
                raise ValueError(
                    f"{demo_key} trajectory_gen robot traj shape mismatch: "
                    f"robot0={robot0_traj.shape}, robot1={robot1_traj.shape}"
                )
            # (T, 2, 4, 4)
            return np.stack([robot0_traj, robot1_traj], axis=1)

        raise ValueError(
            f"{demo_key} trajectory_gen group format not supported. "
            f"Found keys: {list(traj_obj.keys())}"
        )

    raise TypeError(
        f"{demo_key} trajectory_gen has unsupported type: {type(traj_obj).__name__}"
    )


def load_phase_idx_array(demo_group, demo_key, step_count):
    """
    读取 demo 下的 phase_idx，并规范为 (T, 2) uint8。

    支持:
    1) phase_idx 是 dataset，形状 (T,2) 或 (2,T)
    2) phase_idx 是 group 且包含 dataset 'value'，形状同上
    """
    if "phase_idx" not in demo_group:
        raise KeyError(f"{demo_key} does not contain 'phase_idx'")

    phase_obj = demo_group["phase_idx"]
    if isinstance(phase_obj, h5py.Dataset):
        phase = np.asarray(phase_obj)
    elif isinstance(phase_obj, h5py.Group):
        if "value" not in phase_obj:
            raise ValueError(
                f"{demo_key} phase_idx group format not supported. "
                f"Found keys: {list(phase_obj.keys())}"
            )
        phase = np.asarray(phase_obj["value"])
    else:
        raise TypeError(
            f"{demo_key} phase_idx has unsupported type: {type(phase_obj).__name__}"
        )

    if phase.ndim != 2:
        raise ValueError(f"{demo_key} phase_idx must be 2D, got shape {phase.shape}")

    # Normalize to (T, 2)
    if phase.shape == (step_count, 2):
        pass
    elif phase.shape == (2, step_count):
        phase = phase.T
    else:
        raise ValueError(
            f"{demo_key} phase_idx shape not compatible with step_count={step_count}, "
            f"got shape {phase.shape}, expected (T,2) or (2,T)"
        )

    return phase.astype(np.uint8, copy=False)


def convert_hdf5_to_zarr(
    original_dataset_path,
    save_dataset_path,
    overwrite=False,
    add_traj=False,
    add_phase_idx=False,
):
    if os.path.exists(save_dataset_path):
        if overwrite:
            cprint(f"Overwriting existing directory: {save_dataset_path}", "red")
            shutil.rmtree(save_dataset_path)
        else:
            raise FileExistsError(
                f"Path exists: {save_dataset_path}. Use --overwrite to replace it."
            )
    os.makedirs(save_dataset_path, exist_ok=True)

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    traj_gen_arrays = [] if add_traj else None
    phase_idx_arrays = [] if add_phase_idx else None
    episode_ends_arrays = []

    with h5py.File(original_dataset_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Top-level key 'data' not found in {original_dataset_path}")

        demos = f["data"]
        demo_keys = sorted_demo_keys(demos)
        if not demo_keys:
            raise ValueError(f"No demo_* groups found under {original_dataset_path}/data")

        for demo_key in tqdm.tqdm(demo_keys, desc="Converting demos"):
            demo_group = demos[demo_key]
            obs_group = demo_group["obs"]

            action = np.asarray(demo_group["actions"], dtype=np.float32)
            img0 = normalize_image_array(
                obs_group["robot0_eye_in_hand_image"], "robot0_eye_in_hand_image"
            )
            img1 = normalize_image_array(
                obs_group["robot1_eye_in_hand_image"], "robot1_eye_in_hand_image"
            )
            point_cloud = np.asarray(obs_group["pointview_pc"], dtype=np.float32)

            state = np.concatenate(
                [
                    np.asarray(obs_group["robot0_eef_pos"], dtype=np.float32),
                    np.asarray(obs_group["robot0_eef_quat"], dtype=np.float32),
                    np.asarray(obs_group["robot0_gripper_qpos"], dtype=np.float32),
                    np.asarray(obs_group["robot1_eef_pos"], dtype=np.float32),
                    np.asarray(obs_group["robot1_eef_quat"], dtype=np.float32),
                    np.asarray(obs_group["robot1_gripper_qpos"], dtype=np.float32),
                ],
                axis=-1,
            )
            traj_gen = None
            if add_traj:
                traj_gen = load_trajectory_array(demo_group, demo_key)
            phase_idx = None
            if add_phase_idx:
                phase_idx = load_phase_idx_array(demo_group, demo_key, step_count=action.shape[0])

            step_count = action.shape[0]
            if (
                img0.shape[0] != step_count
                or img1.shape[0] != step_count
                or point_cloud.shape[0] != step_count
                or state.shape[0] != step_count
            ):
                raise ValueError(
                    f"{demo_key} has inconsistent step count: "
                    f"actions={step_count}, img0={img0.shape[0]}, img1={img1.shape[0]}, "
                    f"point_cloud={point_cloud.shape[0]}, state={state.shape[0]}"
                )
            if add_traj and traj_gen.shape[0] != step_count:
                raise ValueError(
                    f"{demo_key} traj_gen step count mismatch: "
                    f"actions={step_count}, traj_gen={traj_gen.shape[0]}"
                )
            if add_phase_idx and phase_idx.shape[0] != step_count:
                raise ValueError(
                    f"{demo_key} phase_idx step count mismatch: "
                    f"actions={step_count}, phase_idx={phase_idx.shape[0]}"
                )

            # (T, 2, 84, 84, 3)
            img = np.stack([img0, img1], axis=1)

            img_arrays.append(img)
            point_cloud_arrays.append(point_cloud)
            state_arrays.append(state)
            action_arrays.append(action)
            if add_traj:
                traj_gen_arrays.append(traj_gen)
            if add_phase_idx:
                phase_idx_arrays.append(phase_idx)

            total_count += step_count
            episode_ends_arrays.append(total_count)

    img_arrays = np.concatenate(img_arrays, axis=0)
    point_cloud_arrays = np.concatenate(point_cloud_arrays, axis=0)
    state_arrays = np.concatenate(state_arrays, axis=0)
    action_arrays = np.concatenate(action_arrays, axis=0)
    if add_traj:
        traj_gen_arrays = np.concatenate(traj_gen_arrays, axis=0)
    if add_phase_idx:
        phase_idx_arrays = np.concatenate(phase_idx_arrays, axis=0)
    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)

    zarr_root = zarr.group(save_dataset_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    chunk_t = min(500, img_arrays.shape[0])

    zarr_data.create_dataset(
        "img",
        data=img_arrays,
        chunks=(chunk_t, *img_arrays.shape[1:]),
        dtype="uint8",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=(chunk_t, *point_cloud_arrays.shape[1:]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=(chunk_t, *action_arrays.shape[1:]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(chunk_t, *state_arrays.shape[1:]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    if add_traj:
        zarr_data.create_dataset(
            "traj_gen",
            data=traj_gen_arrays,
            chunks=(chunk_t, *traj_gen_arrays.shape[1:]),
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
    if add_phase_idx:
        zarr_data.create_dataset(
            "phase_idx",
            data=phase_idx_arrays,
            chunks=(chunk_t, *phase_idx_arrays.shape[1:]),
            dtype="uint8",
            overwrite=True,
            compressor=compressor,
        )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        chunks=(min(500, episode_ends_arrays.shape[0]),),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    cprint(f"img shape: {img_arrays.shape}", "green")
    cprint(f"point_cloud shape: {point_cloud_arrays.shape}", "green")
    cprint(f"action shape: {action_arrays.shape}", "green")
    cprint(f"state shape: {state_arrays.shape}", "green")
    if add_traj:
        cprint(f"traj_gen shape: {traj_gen_arrays.shape}", "green")
    if add_phase_idx:
        cprint(f"phase_idx shape: {phase_idx_arrays.shape}", "green")
    cprint(f"episode_ends shape: {episode_ends_arrays.shape}", "green")
    cprint(f"total_count: {total_count}", "green")
    cprint(f"Saved zarr file to {save_dataset_path}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DexMimicGen hdf5 to DP3 zarr.")
    parser.add_argument(
        "--input",
        default="/home/benhua/Improved-3D-Diffusion-Policy/dexmimicgen_dataset/two_arm_drawer_cleanup_part1_w_pc.hdf5",
        help="Path to input hdf5 file",
    )
    parser.add_argument(
        "--output",
        default="/home/benhua/Improved-3D-Diffusion-Policy/dexmimicgen_twoarm_drawer_cleanup/twoarm_drawer_cleanup_1",
        help="Path to output zarr directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists",
    )
    parser.add_argument(
        "--add_traj",
        action="store_true",
        help="If set, extract 'trajectory_gen' from hdf5 and save to data/traj_gen",
    )
    parser.add_argument(
        "--add_phase_idx",
        action="store_true",
        help="If set, extract per-demo 'phase_idx' from hdf5 and save to data/phase_idx",
    )
    args = parser.parse_args()

    convert_hdf5_to_zarr(
        original_dataset_path=args.input,
        save_dataset_path=args.output,
        overwrite=args.overwrite,
        add_traj=args.add_traj,
        add_phase_idx=args.add_phase_idx,
    )
