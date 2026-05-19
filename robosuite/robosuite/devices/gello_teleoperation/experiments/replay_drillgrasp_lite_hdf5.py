#!/usr/bin/env python3
"""
Replay DrillGrasp demos collected by collect_client_drillgrasp.py and regenerate observations.

Default behavior:
1) load states / actions from input hdf5 (data/demo_0)
2) replay with actions in simulator
3) export regenerated observations to output hdf5 under data/demo_0/obs_replay
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

# 6D hand command groups in actuator order:
# [pinky, ring, middle, index, thumb_bend, thumb_proximal_1]
INSPIRE_GROUPS_ACTUATOR = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9, 10),
    (11,),
)


def _build_absolute_joint_controller_config():
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_POSITION",
                "input_max": [6.28] * 6,
                "input_min": [-6.28] * 6,
                "output_max": [0.5] * 6,
                "output_min": [-0.5] * 6,
                "kp": [150] * 6,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "qpos_limits": None,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "input_type": "absolute",
                "gripper": {
                    "type": "JOINT_POSITION",
                    "use_action_scaling": False,
                },
            }
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-hdf5", type=str, required=True, help="Input demo hdf5 path")
    parser.add_argument("--output-hdf5", type=str, default="", help="Output hdf5 path; default: <input>_replay_obs.hdf5")
    parser.add_argument("--mode", type=str, default="actions", choices=["actions", "states"])
    parser.add_argument("--no-render", action="store_true", help="Disable on-screen window rendering")
    parser.add_argument(
        "--camera-obs",
        action="store_true",
        help="Regenerate image/depth observations (requires offscreen OpenGL context)",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default="egl",
        choices=["", "egl", "osmesa", "glfw"],
        help="Optional MUJOCO_GL backend override for camera-obs mode (recommended: osmesa on laptops)",
    )
    parser.add_argument("--replace-obs", action="store_true", help="Replace data/demo_0/obs with replayed obs")
    parser.add_argument("--camera-names", nargs="+", default=["thirdview", "robot0_eye_in_hand"])
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument(
        "--rebuild-right-action",
        action="store_true",
        help="Rebuild right-arm action as [rel_pos(3), rel_axis_angle(3), right_gripper(6)] from replayed observations.",
    )
    parser.add_argument(
        "--replace-actions",
        action="store_true",
        help="When rebuilding actions, replace demo_0/actions and action_dict instead of writing *_replay datasets.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        choices=["mjviewer", "mujoco"],
        help="On-screen renderer backend when rendering is enabled.",
    )
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=1,
        help="GPU index for robosuite render_gpu_device_id (try 0 or 1)",
    )
    return parser.parse_args()


def _joint_limits_from_ids(sim, joint_ids):
    lows = np.zeros(len(joint_ids), dtype=float)
    highs = np.zeros(len(joint_ids), dtype=float)
    for i, j_id in enumerate(joint_ids):
        low, high = sim.model.jnt_range[j_id]
        lows[i] = low
        highs[i] = high
    return lows, highs


def _build_hand_qpos_groups(hand_joint_names):
    name_to_idx = {name: i for i, name in enumerate(hand_joint_names)}

    def _idx_by_suffix(raw_name):
        if raw_name in name_to_idx:
            return name_to_idx[raw_name]
        matches = [idx for name, idx in name_to_idx.items() if name.endswith(raw_name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise KeyError(f"Joint suffix not found: {raw_name}. Available: {list(name_to_idx.keys())}")
        raise KeyError(f"Joint suffix ambiguous: {raw_name}. Matches: {matches}")

    return (
        (_idx_by_suffix("joint_r_pinky_distal"), _idx_by_suffix("joint_r_pinky_proximal")),
        (_idx_by_suffix("joint_r_ring_distal"), _idx_by_suffix("joint_r_ring_proximal")),
        (_idx_by_suffix("joint_r_middle_distal"), _idx_by_suffix("joint_r_middle_proximal")),
        (_idx_by_suffix("joint_r_index_distal"), _idx_by_suffix("joint_r_index_proximal")),
        (
            _idx_by_suffix("joint_r_thumb_distal"),
            _idx_by_suffix("joint_r_thumb_middle"),
            _idx_by_suffix("joint_r_thumb_proximal_2"),
        ),
        (_idx_by_suffix("joint_r_thumb_proximal_1"),),
    )


def _hand_norm6_to_ctrl12(hand_norm6, ctrl_low12, ctrl_high12):
    """
    Convert normalized 6D hand command to 12D actuator targets.
    Convention: 0 = closed, 1 = open.
    """
    hand_norm6 = np.clip(np.asarray(hand_norm6, dtype=float), 0.0, 1.0)
    if hand_norm6.shape[0] != 6:
        raise ValueError(f"Expected 6D normalized hand command, got shape {hand_norm6.shape}.")
    ctrl12 = np.zeros(12, dtype=float)
    for g, act_indices in enumerate(INSPIRE_GROUPS_ACTUATOR):
        for i in act_indices:
            ctrl12[i] = ctrl_low12[i] + (1.0 - hand_norm6[g]) * (ctrl_high12[i] - ctrl_low12[i])
    return ctrl12


def _hand_qpos12_to_norm6(hand_qpos12, hand_low12, hand_high12, qpos_groups):
    q12 = np.asarray(hand_qpos12, dtype=float)
    norm6 = np.zeros(6, dtype=float)
    for g, joint_indices in enumerate(qpos_groups):
        vals = []
        for j in joint_indices:
            span = max(hand_high12[j] - hand_low12[j], 1e-8)
            vals.append((hand_high12[j] - np.clip(q12[j], hand_low12[j], hand_high12[j])) / span)
        norm6[g] = float(np.mean(vals))
    return np.clip(norm6, 0.0, 1.0)


def _infer_hand_norm6_from_action(hand_action, hand_low12, hand_high12, hand_qpos_groups):
    """
    Infer a normalized 6D hand command from dataset action payload.
    Supports compact 6D or expanded 12D hand vectors.
    """
    hand_action = np.asarray(hand_action, dtype=float).reshape(-1)
    if hand_action.shape[0] == 6:
        # Preferred dataset format from teleop server: normalized [0, 1], 0=closed, 1=open.
        if np.all(np.isfinite(hand_action)) and np.min(hand_action) >= -1e-6 and np.max(hand_action) <= 1.0 + 1e-6:
            return np.clip(hand_action, 0.0, 1.0)

        # Fallback: treat as compact qpos-like 6D command and normalize by grouped joint limits.
        norm6 = np.zeros(6, dtype=float)
        for g, idxs in enumerate(hand_qpos_groups):
            low = float(np.mean(hand_low12[list(idxs)]))
            high = float(np.mean(hand_high12[list(idxs)]))
            span = max(high - low, 1e-8)
            q = np.clip(hand_action[g], low, high)
            norm6[g] = (high - q) / span
        return np.clip(norm6, 0.0, 1.0)

    if hand_action.shape[0] == 12:
        # If already expanded but still normalized, collapse by command groups.
        if np.all(np.isfinite(hand_action)) and np.min(hand_action) >= -1e-6 and np.max(hand_action) <= 1.0 + 1e-6:
            norm6 = np.zeros(6, dtype=float)
            for g, idxs in enumerate(INSPIRE_GROUPS_ACTUATOR):
                norm6[g] = float(np.mean(hand_action[list(idxs)]))
            return np.clip(norm6, 0.0, 1.0)
        # Otherwise interpret as 12D qpos-like values.
        return _hand_qpos12_to_norm6(hand_action, hand_low12, hand_high12, hand_qpos_groups)

    raise RuntimeError(f"Unsupported hand action dimension {hand_action.shape[0]} (expected 6 or 12).")


def _build_direct_replay_context(env):
    robot = env.robots[0]
    arm_controller = robot.part_controllers["right"]
    hand_controller = robot.part_controllers["right_gripper"]

    arm_actuator_ids = np.array(robot._ref_actuators_indexes_dict["right"], dtype=int)
    hand_actuator_ids = np.array(robot._ref_actuators_indexes_dict["right_gripper"], dtype=int)
    hand_joint_ids = np.array(hand_controller.joint_index, dtype=int)

    hand_ctrl_low12 = np.array(env.sim.model.actuator_ctrlrange[hand_actuator_ids, 0], dtype=float)
    hand_ctrl_high12 = np.array(env.sim.model.actuator_ctrlrange[hand_actuator_ids, 1], dtype=float)
    hand_low12, hand_high12 = _joint_limits_from_ids(env.sim, hand_joint_ids)
    hand_qpos_groups = _build_hand_qpos_groups(hand_controller.joint_names)

    return {
        "robot": robot,
        "arm_controller": arm_controller,
        "arm_dim": len(arm_controller.qpos_index),
        "arm_actuator_ids": arm_actuator_ids,
        "hand_actuator_ids": hand_actuator_ids,
        "hand_ctrl_low12": hand_ctrl_low12,
        "hand_ctrl_high12": hand_ctrl_high12,
        "hand_low12": hand_low12,
        "hand_high12": hand_high12,
        "hand_qpos_groups": hand_qpos_groups,
    }


def _apply_action_direct(env, replay_ctx, action):
    """
    Replay action with absolute arm target + direct hand ctrl target.
    This matches the teleop server path and avoids gripper controller mismatches.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    arm_dim = replay_ctx["arm_dim"]
    if action.shape[0] < arm_dim + 6:
        raise RuntimeError(f"Action dim too small: got {action.shape[0]}, expected at least {arm_dim + 6}.")

    arm_target = action[:arm_dim]
    hand_payload = action[arm_dim:]
    hand_norm6 = _infer_hand_norm6_from_action(
        hand_payload,
        replay_ctx["hand_low12"],
        replay_ctx["hand_high12"],
        replay_ctx["hand_qpos_groups"],
    )
    hand_ctrl_target12 = _hand_norm6_to_ctrl12(
        hand_norm6,
        replay_ctx["hand_ctrl_low12"],
        replay_ctx["hand_ctrl_high12"],
    )

    robot = replay_ctx["robot"]
    robot.composite_controller.update_state()
    replay_ctx["arm_controller"].set_goal(arm_target)
    applied = robot.composite_controller.run_controller(robot._enabled_parts)
    for part_name, applied_action in applied.items():
        if part_name == "right_gripper":
            continue
        actuator_ids = robot._ref_actuators_indexes_dict[part_name]
        low = robot.sim.model.actuator_ctrlrange[actuator_ids, 0]
        high = robot.sim.model.actuator_ctrlrange[actuator_ids, 1]
        robot.sim.data.ctrl[actuator_ids] = np.clip(applied_action, low, high)

    env.sim.data.ctrl[replay_ctx["hand_actuator_ids"]] = np.clip(
        hand_ctrl_target12,
        replay_ctx["hand_ctrl_low12"],
        replay_ctx["hand_ctrl_high12"],
    )


def _configure_gl_backend(args):
    if not args.camera_obs:
        return
    backend = args.mujoco_gl if args.mujoco_gl else "osmesa"
    os.environ["MUJOCO_GL"] = backend
    print(f"[replay] MUJOCO_GL={backend}")
    if args.no_render:
        print("[replay] camera-obs enabled in headless mode (no on-screen window)")
    else:
        print("[replay] camera-obs enabled (offscreen + on-screen window)")


def _make_env(
    control_freq: int,
    render_window: bool,
    renderer: str,
    collect_camera_obs: bool,
    camera_names,
    camera_height: int,
    camera_width: int,
    render_gpu_device_id: int,
):
    import robosuite

    return robosuite.make(
        env_name="DrillGrasp",
        robots="UR5eDex",
        controller_configs=_build_absolute_joint_controller_config(),
        has_renderer=bool(render_window),
        renderer=str(renderer),
        has_offscreen_renderer=bool(collect_camera_obs),
        use_camera_obs=bool(collect_camera_obs),
        camera_names=list(camera_names),
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        camera_depths=[True] * len(camera_names),
        control_freq=int(control_freq),
        ignore_done=True,
        render_camera=None,
        initialization_noise=None,
        render_gpu_device_id=int(render_gpu_device_id),
    )


def _append_obs(obs_buffer, obs_dict):
    for k, v in obs_dict.items():
        if not isinstance(v, np.ndarray) or k.endswith("-state"):
            continue
        if k not in obs_buffer:
            obs_buffer[k] = []
        obs_buffer[k].append(np.asarray(v))


def _stack_obs(obs_buffer):
    return {k: np.asarray(v) for k, v in obs_buffer.items()}


def _write_group_recursive(group, key, value):
    if isinstance(value, dict):
        sub = group.create_group(key)
        for k, v in value.items():
            _write_group_recursive(sub, k, v)
    else:
        group.create_dataset(key, data=value)


def _quat_inverse_xyzw(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_multiply_xyzw(q1_xyzw, q2_xyzw):
    x1, y1, z1, w1 = [float(v) for v in q1_xyzw]
    x2, y2, z2, w2 = [float(v) for v in q2_xyzw]
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_delta_to_axis_angle_xyzw(prev_xyzw, cur_xyzw):
    q_prev_inv = _quat_inverse_xyzw(prev_xyzw)
    q_delta = _quat_multiply_xyzw(q_prev_inv, cur_xyzw)
    q_delta = q_delta / max(np.linalg.norm(q_delta), 1e-12)
    # Keep the shortest rotation representation.
    if q_delta[3] < 0:
        q_delta = -q_delta
    xyz = q_delta[:3]
    w = float(np.clip(q_delta[3], -1.0, 1.0))
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(xyz_norm, w)
    axis = xyz / xyz_norm
    return axis * angle


def _rot6d_from_axis_angle(axis_angle):
    aa = np.asarray(axis_angle, dtype=np.float64)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        rot = np.eye(3, dtype=np.float64)
    else:
        k = aa / theta
        kx, ky, kz = k
        K = np.array(
            [
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0],
            ],
            dtype=np.float64,
        )
        rot = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0)


def _rebuild_right_action_from_obs(obs_replay, hand_qpos_groups):
    if "robot0_eef_pos" not in obs_replay or "robot0_eef_quat" not in obs_replay:
        raise RuntimeError("Cannot rebuild actions: missing robot0_eef_pos / robot0_eef_quat in replay observations.")
    if "robot0_gripper_qpos" not in obs_replay:
        raise RuntimeError("Cannot rebuild actions: missing robot0_gripper_qpos in replay observations.")

    eef_pos = np.asarray(obs_replay["robot0_eef_pos"], dtype=np.float64)
    eef_quat = np.asarray(obs_replay["robot0_eef_quat"], dtype=np.float64)
    gripper_qpos = np.asarray(obs_replay["robot0_gripper_qpos"], dtype=np.float64)
    t = int(eef_pos.shape[0])
    if eef_quat.shape[0] != t or gripper_qpos.shape[0] != t:
        raise RuntimeError("Cannot rebuild actions: inconsistent sequence lengths among eef / gripper observations.")

    rel_pos = np.zeros((t, 3), dtype=np.float64)
    rel_rot_aa = np.zeros((t, 3), dtype=np.float64)
    for i in range(1, t):
        rel_pos[i] = eef_pos[i] - eef_pos[i - 1]
        rel_rot_aa[i] = _quat_delta_to_axis_angle_xyzw(eef_quat[i - 1], eef_quat[i])
    rel_rot_6d = np.asarray([_rot6d_from_axis_angle(v) for v in rel_rot_aa], dtype=np.float64)

    if gripper_qpos.ndim != 2:
        raise RuntimeError(f"Expected robot0_gripper_qpos shape (T,D), got {gripper_qpos.shape}.")
    if gripper_qpos.shape[1] == 6:
        right_gripper = gripper_qpos.copy()
    elif gripper_qpos.shape[1] == 12:
        right_gripper = np.zeros((t, 6), dtype=np.float64)
        for g, idxs in enumerate(hand_qpos_groups):
            right_gripper[:, g] = np.mean(gripper_qpos[:, list(idxs)], axis=1)
    else:
        raise RuntimeError(
            f"Unsupported robot0_gripper_qpos dim={gripper_qpos.shape[1]} (expected 6 or 12 for DrillGrasp)."
        )

    action12 = np.concatenate([rel_pos, rel_rot_aa, right_gripper], axis=1).astype(np.float64)
    action_dict = {
        "right_gripper": right_gripper.astype(np.float64),
        "right_rel_pos": rel_pos.astype(np.float64),
        "right_rel_rot_axis_angle": rel_rot_aa.astype(np.float64),
        "right_rel_rot_6d": rel_rot_6d.astype(np.float64),
    }
    return action12, action_dict


def replay_and_collect_obs(env, states, actions, mode: str, render: bool):
    obs_buffer = {}
    divergence = []
    replay_ctx = _build_direct_replay_context(env)
    n_substeps = max(1, int(round(float(env.control_timestep) / float(env.model_timestep))))

    if mode == "actions":
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        for i, action in enumerate(actions):
            start = time.time()
            _apply_action_direct(env, replay_ctx, np.asarray(action))
            for _ in range(n_substeps):
                env.sim.step()
            env._update_observables()
            if render:
                if env.viewer is None:
                    env.initialize_renderer()
                env.viewer.update()
                diff = (1.0 / 60.0) - (time.time() - start)
                if diff > 0:
                    time.sleep(diff)
            obs = env._get_observations(force_update=False)
            _append_obs(obs_buffer, obs)
            if i < (len(states) - 1):
                state_playback = env.sim.get_state().flatten()
                err = float(np.linalg.norm(np.asarray(states[i + 1]) - np.asarray(state_playback)))
                divergence.append(err)
    else:
        for state in states:
            start = time.time()
            env.sim.set_state_from_flattened(np.asarray(state))
            env.sim.forward()
            if render:
                if env.viewer is None:
                    env.initialize_renderer()
                env.viewer.update()
                diff = (1.0 / 60.0) - (time.time() - start)
                if diff > 0:
                    time.sleep(diff)
            obs = env._get_observations(force_update=False)
            _append_obs(obs_buffer, obs)

    return _stack_obs(obs_buffer), divergence, replay_ctx


def main():
    args = _parse_args()
    _configure_gl_backend(args)

    input_path = Path(args.input_hdf5).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = (
        Path(args.output_hdf5).expanduser()
        if args.output_hdf5.strip()
        else input_path.with_name(f"{input_path.stem}_replay_obs.hdf5")
    )

    with h5py.File(input_path, "r") as fin:
        states = np.asarray(fin["data/demo_0/states"], dtype=np.float64)
        actions = np.asarray(fin["data/demo_0/actions"], dtype=np.float64)
        print(f"actions shape: {actions.shape}")
        if states.ndim != 2:
            raise RuntimeError(f"Expected 2D states, got shape={states.shape}")
        if args.mode == "actions" and actions.ndim != 2:
            raise RuntimeError(f"Expected 2D actions, got shape={actions.shape}")
        if args.mode == "actions" and actions.shape[0] > states.shape[0]:
            raise RuntimeError(f"actions longer than states: {actions.shape[0]} > {states.shape[0]}")

        env = _make_env(
            control_freq=args.control_freq,
            render_window=(not args.no_render),
            renderer=args.renderer,
            collect_camera_obs=bool(args.camera_obs),
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        try:
            env.reset()
            if (not args.no_render) and env.viewer is None:
                env.initialize_renderer()
            obs_replay, divergence, replay_ctx = replay_and_collect_obs(
                env=env,
                states=states,
                actions=actions,
                mode=args.mode,
                render=(not args.no_render),
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

        with h5py.File(output_path, "w") as fout:
            fin.copy("data", fout)
            demo_grp = fout["data/demo_0"]

            if args.replace_obs and "obs" in demo_grp:
                del demo_grp["obs"]
            if args.replace_obs:
                _write_group_recursive(demo_grp, "obs", obs_replay)
            else:
                if "obs_replay" in demo_grp:
                    del demo_grp["obs_replay"]
                _write_group_recursive(demo_grp, "obs_replay", obs_replay)

            if args.rebuild_right_action:
                rebuilt_actions, rebuilt_action_dict = _rebuild_right_action_from_obs(
                    obs_replay,
                    replay_ctx["hand_qpos_groups"],
                )
                if args.replace_actions:
                    if "actions" in demo_grp:
                        del demo_grp["actions"]
                    if "action_dict" in demo_grp:
                        del demo_grp["action_dict"]
                    demo_grp.create_dataset("actions", data=rebuilt_actions)
                    _write_group_recursive(demo_grp, "action_dict", rebuilt_action_dict)
                else:
                    if "actions_replay" in demo_grp:
                        del demo_grp["actions_replay"]
                    if "action_dict_replay" in demo_grp:
                        del demo_grp["action_dict_replay"]
                    demo_grp.create_dataset("actions_replay", data=rebuilt_actions)
                    _write_group_recursive(demo_grp, "action_dict_replay", rebuilt_action_dict)

            replay_meta = {
                "mode": args.mode,
                "camera_names": list(args.camera_names),
                "camera_height": int(args.camera_height),
                "camera_width": int(args.camera_width),
                "control_freq": int(args.control_freq),
                "render_window": bool(not args.no_render),
                "use_camera_obs": bool(args.camera_obs),
                "mujoco_gl": args.mujoco_gl if args.mujoco_gl else "default",
                "render_gpu_device_id": int(args.render_gpu_device_id),
                "replace_obs": bool(args.replace_obs),
                "rebuild_right_action": bool(args.rebuild_right_action),
                "replace_actions": bool(args.replace_actions),
                "mean_state_divergence_l2": float(np.mean(divergence)) if len(divergence) > 0 else 0.0,
                "max_state_divergence_l2": float(np.max(divergence)) if len(divergence) > 0 else 0.0,
            }
            demo_grp.attrs["replay_info"] = json.dumps(replay_meta, indent=2)

    print(f"[replay] wrote: {output_path}")


if __name__ == "__main__":
    main()

