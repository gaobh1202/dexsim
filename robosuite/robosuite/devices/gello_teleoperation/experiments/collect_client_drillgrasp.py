#!/usr/bin/env python3
"""
Interactive data collection client for DrillGrasp teleoperation (Gello + DexUMI).

This client keeps the original interactive workflow:
1) press Enter to start one rollout
2) collect fixed-length frames from teleop server
3) ask user whether to save
4) repeat or quit

Saved HDF5 schema follows collect_human_demonstration_in_drill_grasp.py style:
  data/
    attrs: env_args, total
    demo_0/
      attrs: model_file, num_samples
      states
      actions
      action_dict/*
      obs/*
      datagen_info/*
"""

import json
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import robosuite.utils.transform_utils as T


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5000
    out_dir: str = "/home/benhua/DexSim/robosuite/robosuite/demonstration_collection_client"
    hz: int = 25
    length: int = 700
    verbose: bool = True
    sock_timeout_s: float = 0.0  # <=0 disables recv timeout (block until server replies)
    collect_mode: str = "lite"  # "lite" for action/state only, "full" for rich obs
    include_images: bool = True  # only used in full mode


def _pose_from_pos_quat_xyzw(pos, quat_xyzw):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = T.quat2mat(np.asarray(quat_xyzw, dtype=np.float64))
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def _quat_wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def _axis_angle_from_quat_delta(prev_xyzw, cur_xyzw):
    # q_delta = q_prev^-1 * q_cur
    q_prev_inv = T.quat_inverse(np.asarray(prev_xyzw, dtype=np.float64))
    q_delta = T.quat_multiply(q_prev_inv, np.asarray(cur_xyzw, dtype=np.float64))
    return T.quat2axisangle(np.asarray(q_delta, dtype=np.float64))


def _rot6d_from_axis_angle(axis_angle):
    quat_xyzw = T.axisangle2quat(np.asarray(axis_angle, dtype=np.float64))
    rot = T.quat2mat(quat_xyzw).astype(np.float64)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0)


def _depth_to_pointcloud_world(depth, intrinsic, extrinsic, max_points=4096):
    """
    Convert depth map + camera intr/extr to world-frame point cloud.
    Returns:
      points: (max_points, 3) float32
      count: valid point count before padding / truncation
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        return np.zeros((max_points, 3), dtype=np.float32), 0

    H, W = d.shape
    K = np.asarray(intrinsic, dtype=np.float64)
    Tcw = np.asarray(extrinsic, dtype=np.float64)  # camera -> world
    if K.shape != (3, 3) or Tcw.shape != (4, 4):
        return np.zeros((max_points, 3), dtype=np.float32), 0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        return np.zeros((max_points, 3), dtype=np.float32), 0

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = d
    valid = np.isfinite(z) & (z > 1e-8)
    if not np.any(valid):
        return np.zeros((max_points, 3), dtype=np.float32), 0

    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)  # (N,3)

    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    pts_world = (R @ pts_cam.T).T + t[None, :]

    n = pts_world.shape[0]
    if n >= max_points:
        ids = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
        out = pts_world[ids].astype(np.float32)
        return out, int(n)

    out = np.zeros((max_points, 3), dtype=np.float32)
    out[:n] = pts_world.astype(np.float32)
    return out, int(n)


class DrillGraspCollectorClient:
    def __init__(self, args: Args):
        self.args = args
        self.addr = (args.host, args.port)
        self.dt = 1.0 / args.hz
        self.save_dir = Path(args.out_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.collect_mode = str(args.collect_mode).strip().lower()
        if self.collect_mode not in ("lite", "full"):
            raise ValueError(f"Invalid collect_mode={args.collect_mode}, expected 'lite' or 'full'")


    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.addr)
        sock.settimeout(1.0)
        # timeout_s = float(self.args.sock_timeout_s)
        # if timeout_s > 0:
        #     sock.settimeout(timeout_s)
        # else:
        #     sock.settimeout(None)
        return sock

    @staticmethod
    def _send(sock, obj):
        sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    @staticmethod
    def _recv(sock):
        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:                return None
            if not chunk:
                raise RuntimeError("server disconnected")
            buf += chunk
            if b"\n" in buf:
                line, _ = buf.split(b"\n", 1)
                return json.loads(line.decode("utf-8"))

    def run(self):
        sock = self._connect()
        try:
            demo_count = 0
            print("\n" + "=" * 60)
            print("DrillGrasp interactive collection ready")
            print("=" * 60)
            while True:
                print("\nCommands:")
                print("  [Enter] collect one demo")
                print("  q       quit")
                cmd = input("Input: ").strip().lower()
                if cmd == "q":
                    break
                if cmd != "":
                    print("Unknown command")
                    continue

                demo_count += 1
                try:
                    out = self.collect_one_demo(sock=sock, demo_number=demo_count)
                    if out is not None:
                        print(f"[collector] Saved demo #{demo_count}: {out}")
                    else:
                        demo_count -= 1
                        print("[collector] Demo discarded")
                except KeyboardInterrupt:
                    demo_count -= 1
                    print("[collector] Interrupted while collecting this demo")
                except Exception as e:
                    demo_count -= 1
                    print(f"[collector] Failed: {e}")
            print(f"\nCollected demos: {demo_count}")
        finally:
            sock.close()

    def collect_one_demo(self, sock, demo_number: int):
        if self.args.verbose:
            print(
                f"\n[collector] Start demo {demo_number}, frames={self.args.length}, "
                f"hz={self.args.hz}, mode={self.collect_mode}"
            )

        # Raw buffers (fallback-compatible)
        arm_qpos_list = []
        hand_qpos_list = []
        ee_pos_list = []
        ee_quat_xyzw_list = []
        gello_act_list = []
        umi_act_list = []
        force_list = []
        obs_buffer = {}
        camera_info_buffer = {}
        pointcloud_buffer = {}
        pointcloud_count_buffer = {}
        sim_state_list = []
        actions_list = []
        drill_pos_list = []
        drill_quat_xyzw_list = []
        signal_grasped_list = []
        signal_lifted_list = []

        prev_ee_pos = None
        prev_ee_quat_xyzw = None
        right_rel_pos_list = []
        right_rel_rot_axis_angle_list = []
        right_rel_rot_6d_list = []
        next_progress_print = 100

        lite_mode = self.collect_mode == "lite"
        include_images = (not lite_mode) and bool(self.args.include_images)
        for i in range(self.args.length):
            start_t = time.time()

            self._send(sock, {"type": "GET_STATE", "include_images": include_images, "lite": lite_mode})
            resp = self._recv(sock)
            if resp is None:
                if self.args.verbose and (i % 25 == 0):
                    print("[collector] recv timeout, skip this frame")
                continue
            if not resp.get("ok", False):
                continue

            obs_ext = resp.get("obs", {})
            camera_info = resp.get("camera_info", {})
            action_dict_ext = resp.get("action_dict", {})
            dgi_signals = resp.get("datagen_info", {}).get("subtask_term_signals", {})

            arm_qpos = np.asarray(
                obs_ext.get("robot0_joint_pos", resp.get("armqpos6", np.zeros(6))),
                dtype=np.float64,
            )
            hand_qpos = np.asarray(
                obs_ext.get("robot0_gripper_qpos", resp.get("handqpos6", np.zeros(6))),
                dtype=np.float64,
            )
            ee_pose = np.asarray(resp.get("armee7", np.zeros(7)), dtype=np.float64)  # [pos(3), quat_wxyz(4)]
            gello_act = np.asarray(resp.get("gelloaction6", np.zeros(6)), dtype=np.float64)
            umi_act = np.asarray(resp.get("umiaction6", np.zeros(6)), dtype=np.float64)
            force = np.asarray(resp.get("force6", np.zeros(6)), dtype=np.float64)
            sim_state = np.asarray(resp.get("sim_state", np.array([], dtype=np.float64)), dtype=np.float64)
            action_12 = np.asarray(resp.get("actions", np.array([], dtype=np.float64)), dtype=np.float64)

            ee_pos = np.asarray(obs_ext.get("robot0_eef_pos", ee_pose[:3]), dtype=np.float64)
            ee_quat_xyzw = np.asarray(obs_ext.get("robot0_eef_quat", _quat_wxyz_to_xyzw(ee_pose[3:7])), dtype=np.float64)
            drill_pos = np.asarray(obs_ext.get("drill_001_pos", np.zeros(3)), dtype=np.float64)
            drill_quat_xyzw = np.asarray(obs_ext.get("drill_001_quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=np.float64)

            if prev_ee_pos is None:
                rel_pos = np.zeros(3, dtype=np.float64)
                rel_rot_aa = np.zeros(3, dtype=np.float64)
            else:
                rel_pos = ee_pos - prev_ee_pos
                rel_rot_aa = _axis_angle_from_quat_delta(prev_ee_quat_xyzw, ee_quat_xyzw)
            rel_rot_6d = _rot6d_from_axis_angle(rel_rot_aa)

            prev_ee_pos = ee_pos.copy()
            prev_ee_quat_xyzw = ee_quat_xyzw.copy()

            # Prefer server-provided action_dict right fields if available
            if "right" in action_dict_ext:
                right_cmd = np.asarray(action_dict_ext["right"], dtype=np.float64).reshape(-1)
                if right_cmd.shape[0] >= 6:
                    rel_pos = right_cmd[:3]
                    rel_rot_aa = right_cmd[3:6]
                    rel_rot_6d = _rot6d_from_axis_angle(rel_rot_aa)

            if "right_gripper" in action_dict_ext:
                hand_qpos = np.asarray(action_dict_ext["right_gripper"], dtype=np.float64).reshape(-1)

            arm_qpos_list.append(arm_qpos)
            hand_qpos_list.append(hand_qpos)
            ee_pos_list.append(ee_pos)
            ee_quat_xyzw_list.append(ee_quat_xyzw)
            gello_act_list.append(gello_act)
            umi_act_list.append(umi_act)
            force_list.append(force)
            sim_state_list.append(sim_state)
            actions_list.append(action_12)
            drill_pos_list.append(drill_pos)
            drill_quat_xyzw_list.append(drill_quat_xyzw)
            signal_grasped_list.append(int(dgi_signals.get("drill_grasped", 0)))
            signal_lifted_list.append(int(dgi_signals.get("drill_lifted", 0)))
            right_rel_pos_list.append(rel_pos)
            right_rel_rot_axis_angle_list.append(rel_rot_aa)
            right_rel_rot_6d_list.append(rel_rot_6d)

            captured = len(arm_qpos_list)
            if captured >= next_progress_print:
                print(
                    f"[collector] progress: captured={captured} valid frames "
                    f"(loop_step={i + 1}/{self.args.length})"
                )
                next_progress_print += 100

            if not lite_mode:
                # Collect simulator obs directly, same style as collect_human_demonstration_in_drill_grasp.py
                for k, v in obs_ext.items():
                    arr = np.asarray(v)
                    if arr.ndim == 0 or k.endswith("-state"):
                        continue
                    if k not in obs_buffer:
                        obs_buffer[k] = []
                    obs_buffer[k].append(arr)

                # Per-step camera intrinsics / extrinsics for later pointcloud synthesis
                for cam_name, cinfo in camera_info.items():
                    K = np.asarray(cinfo.get("intrinsic", np.eye(3)), dtype=np.float64)
                    E = np.asarray(cinfo.get("extrinsic", np.eye(4)), dtype=np.float64)
                    if cam_name not in camera_info_buffer:
                        camera_info_buffer[cam_name] = {"intrinsic": [], "extrinsic": []}
                    camera_info_buffer[cam_name]["intrinsic"].append(K)
                    camera_info_buffer[cam_name]["extrinsic"].append(E)

            if self.args.verbose and (i % 25 == 0):
                print(f"[collector] step {i:04d}/{self.args.length}")

            dt = self.dt - (time.time() - start_t)
            if dt > 0:
                time.sleep(dt)

        if len(arm_qpos_list) == 0:
            return None

        save = input("Save this demo? (y/n): ").strip().lower()
        if save != "y":
            return None

        out_path = self._next_hdf5_path()
        demo = self._build_demo_payload(
            arm_qpos=np.asarray(arm_qpos_list, dtype=np.float64),
            hand_qpos=np.asarray(hand_qpos_list, dtype=np.float64),
            ee_pos=np.asarray(ee_pos_list, dtype=np.float64),
            ee_quat_xyzw=np.asarray(ee_quat_xyzw_list, dtype=np.float64),
            gello_action=np.asarray(gello_act_list, dtype=np.float64),
            umi_action=np.asarray(umi_act_list, dtype=np.float64),
            force=np.asarray(force_list, dtype=np.float64),
            sim_state=np.asarray(sim_state_list, dtype=np.float64),
            action_12=np.asarray(actions_list, dtype=np.float64),
            drill_pos=np.asarray(drill_pos_list, dtype=np.float64),
            drill_quat_xyzw=np.asarray(drill_quat_xyzw_list, dtype=np.float64),
            signal_grasped=np.asarray(signal_grasped_list, dtype=np.int64),
            signal_lifted=np.asarray(signal_lifted_list, dtype=np.int64),
            right_rel_pos=np.asarray(right_rel_pos_list, dtype=np.float64),
            right_rel_rot_axis_angle=np.asarray(right_rel_rot_axis_angle_list, dtype=np.float64),
            right_rel_rot_6d=np.asarray(right_rel_rot_6d_list, dtype=np.float64),
            obs_buffer=obs_buffer,
            camera_info_buffer=camera_info_buffer,
            pointcloud_buffer=pointcloud_buffer,
            pointcloud_count_buffer=pointcloud_count_buffer,
            lite_mode=lite_mode,
        )
        self._write_demo_hdf5(out_path, demo)
        return str(out_path)

    def _build_demo_payload(
        self,
        arm_qpos,
        hand_qpos,
        ee_pos,
        ee_quat_xyzw,
        gello_action,
        umi_action,
        force,
        sim_state,
        action_12,
        drill_pos,
        drill_quat_xyzw,
        signal_grasped,
        signal_lifted,
        right_rel_pos,
        right_rel_rot_axis_angle,
        right_rel_rot_6d,
        obs_buffer,
        camera_info_buffer,
        pointcloud_buffer,
        pointcloud_count_buffer,
        lite_mode: bool = False,
    ):
        t = arm_qpos.shape[0]

        # Prefer server-provided sim_state/actions when available
        if sim_state.size > 0 and sim_state.ndim == 2 and sim_state.shape[0] == t:
            states = sim_state.astype(np.float64)
        else:
            states = np.concatenate(
                [arm_qpos, hand_qpos, ee_pos, ee_quat_xyzw, gello_action, umi_action, force],
                axis=1,
            ).astype(np.float64)

        if action_12.size > 0 and action_12.ndim == 2 and action_12.shape[0] == t:
            actions = action_12.astype(np.float64)
        else:
            actions = np.concatenate([gello_action, hand_qpos], axis=1).astype(np.float64)

        obs = {}
        if not lite_mode:
            # Simulator observations (camera and proprioception) from GET_STATE obs
            obs = {k: np.asarray(v) for k, v in obs_buffer.items()}
            # Ensure required keys exist even if server stream misses them
            obs.setdefault("robot0_joint_pos", arm_qpos.astype(np.float64))
            obs.setdefault("robot0_gripper_qpos", hand_qpos.astype(np.float64))
            obs.setdefault("robot0_eef_pos", ee_pos.astype(np.float64))
            obs.setdefault("robot0_eef_quat", ee_quat_xyzw.astype(np.float64))

            # Build point clouds from saved RGB-D + camera intr/extr
            for k in list(obs.keys()):
                if not k.endswith("_depth"):
                    continue
                cam_name = k[: -len("_depth")]
                if cam_name not in camera_info_buffer:
                    continue
                depth_seq = np.asarray(obs[k])
                intr_seq = np.asarray(camera_info_buffer[cam_name]["intrinsic"], dtype=np.float64)
                ext_seq = np.asarray(camera_info_buffer[cam_name]["extrinsic"], dtype=np.float64)
                if depth_seq.shape[0] != t or intr_seq.shape[0] != t or ext_seq.shape[0] != t:
                    continue

                pcs = []
                counts = []
                for i in range(t):
                    pts, cnt = _depth_to_pointcloud_world(
                        depth=depth_seq[i],
                        intrinsic=intr_seq[i],
                        extrinsic=ext_seq[i],
                        max_points=4096,
                    )
                    pcs.append(pts)
                    counts.append(cnt)
                pointcloud_buffer[cam_name] = np.asarray(pcs, dtype=np.float32)
                pointcloud_count_buffer[cam_name] = np.asarray(counts, dtype=np.int32)

        eef_pose = np.stack([_pose_from_pos_quat_xyzw(ee_pos[i], ee_quat_xyzw[i]) for i in range(t)], axis=0)
        if drill_pos.size > 0 and drill_quat_xyzw.size > 0 and drill_pos.shape[0] == t:
            object_pose = np.stack(
                [_pose_from_pos_quat_xyzw(drill_pos[i], drill_quat_xyzw[i]) for i in range(t)],
                axis=0,
            )
        else:
            object_pose = np.tile(np.eye(4, dtype=np.float64)[None, :, :], (t, 1, 1))

        demo = {
            "states": states,
            "actions": actions,
            "action_dict": {
                "right_gripper": hand_qpos.astype(np.float64),
                "right_rel_pos": right_rel_pos.astype(np.float64),
                "right_rel_rot_axis_angle": right_rel_rot_axis_angle.astype(np.float64),
                "right_rel_rot_6d": right_rel_rot_6d.astype(np.float64),
            },
            "obs": obs,
            "datagen_info": {
                "eef_pose": eef_pose[:, None, :, :],  # (T,1,4,4)
                "target_pose": eef_pose[:, None, :, :],  # fallback
                "gripper_action": hand_qpos[:, None, :],  # (T,1,6)
                "object_poses": {"drill_001": object_pose},
                "camera_info": (
                    {
                        cam: {
                            "intrinsic": np.asarray(v["intrinsic"], dtype=np.float64),
                            "extrinsic": np.asarray(v["extrinsic"], dtype=np.float64),
                        }
                        for cam, v in camera_info_buffer.items()
                        if len(v["intrinsic"]) == t and len(v["extrinsic"]) == t
                    }
                    if (not lite_mode)
                    else {}
                ),
                "pointclouds": (
                    {
                        cam: {
                            "points": np.asarray(pointcloud_buffer[cam], dtype=np.float32),
                            "counts": np.asarray(pointcloud_count_buffer[cam], dtype=np.int32),
                        }
                        for cam in pointcloud_buffer.keys()
                    }
                    if (not lite_mode)
                    else {}
                ),
                "subtask_term_signals": {
                    "drill_grasped": signal_grasped.astype(np.int64) if signal_grasped.shape[0] == t else np.zeros((t,), dtype=np.int64),
                    "drill_lifted": signal_lifted.astype(np.int64) if signal_lifted.shape[0] == t else np.zeros((t,), dtype=np.int64),
                },
            },
            "num_samples": int(t),
        }
        return demo

    def _next_hdf5_path(self):
        files = [f for f in os.listdir(self.save_dir) if f.startswith("demo_") and f.endswith(".hdf5")]
        max_idx = -1
        for f in files:
            m = re.match(r"demo_(\d+)\.hdf5", f)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return self.save_dir / f"demo_{max_idx + 1:06d}.hdf5"

    def _write_group_recursive(self, group, key, value):
        if isinstance(value, dict):
            sub = group.create_group(key)
            for k, v in value.items():
                self._write_group_recursive(sub, k, v)
        else:
            group.create_dataset(key, data=value)

    def _write_demo_hdf5(self, filepath: Path, demo: Dict):
        camera_names = [k.replace("_image", "") for k in demo["obs"].keys() if k.endswith("_image")]
        env_args = {
            "env_name": "DrillGrasp",
            "collector": "collect_client_drillgrasp",
            "collector_mode": self.collect_mode,
            "teleop_source": f"tcp://{self.addr[0]}:{self.addr[1]}",
            "note": "Client-side collection from teleop server streams; lite mode stores action/state only for later replay.",
            "env_kwargs": {
                "robots": "UR5eDex",
                "camera_names": camera_names if len(camera_names) > 0 else ["thirdview", "robot0_eye_in_hand"],
                "camera_heights": 84,
                "camera_widths": 84,
                "camera_depths": [True, True],
                "use_camera_obs": any(k.endswith("_image") for k in demo["obs"].keys()),
                "control_freq": int(self.args.hz),
            },
        }
        with h5py.File(filepath, "w") as f:
            data_grp = f.create_group("data")
            data_grp.attrs["env_args"] = json.dumps(env_args, indent=4)
            data_grp.attrs["total"] = int(demo["num_samples"])
            demo_grp = data_grp.create_group("demo_0")
            demo_grp.attrs["model_file"] = ""
            demo_grp.attrs["num_samples"] = int(demo["num_samples"])
            demo_grp.create_dataset("states", data=demo["states"])
            demo_grp.create_dataset("actions", data=demo["actions"])
            self._write_group_recursive(demo_grp, "action_dict", demo["action_dict"])
            self._write_group_recursive(demo_grp, "obs", demo["obs"])
            self._write_group_recursive(demo_grp, "datagen_info", demo["datagen_info"])


def main(args: Args):
    DrillGraspCollectorClient(args).run()


if __name__ == "__main__":
    try:
        import tyro

        main(tyro.cli(Args))
    except Exception:
        main(Args())

