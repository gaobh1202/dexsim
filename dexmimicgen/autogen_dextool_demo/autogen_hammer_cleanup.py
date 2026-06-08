#!/usr/bin/env python3
"""
Autogenerate training-ready HammerCleanup demos from ONE fixed source teleop + frame labels.

All output demos (demo_0 .. demo_{n-1}) are derived from the same --source-hdf5
(single episode inside that file); only the hammer initial pose differs per generated demo.

Writes consistent states / actions / datagen_info / obs. Adds datagen_info/stage_label
(per-frame 0=arm motion, 1=hand motion).

Stage policy (sequential rollout; see demo_generation/trajectory_gen_pipeline.md):
  - motion 1 (hand): replay source actions; states from physics (no object state copy).
  - motion 0 + drawer (pre-hammer / close-drawer): drawer-relative EE + OSC re-sim.
  - motion 0 + hammer: EE transformed by T_delta + OSC re-sim.
  - motion 0 + drawer (post-hammer, stages 7/9): segment-local EE delta re-anchored at segment start.

Example:
  cd /home/benhua/DexSim
  PYTHONPATH=robosuite:dexmimicgen MUJOCO_GL=osmesa \\
  python dexmimicgen/autogen_dextool_demo/autogen_hammer_cleanup.py \\
    --source-hdf5 dexmimicgen/datasets/generated/single_arm_hammer_cleanup_demo_4.hdf5 \\
    --labels dexmimicgen/autogen_dextool_demo/outputs/single_arm_hammer_cleanup_demo_4_demo_0_review_labels.json \\
    --num-demos 5 --seed 0

  # single demo with explicit hammer offset:
  python .../autogen_hammer_cleanup.py -n 1 --hammer-dx 0.05 --hammer-dy -0.05 --hammer-dyaw-deg 10

Output layout matches robomimic merged datasets (e.g. datasets/generated/single_arm_hammer_cleanup.hdf5):
  /<output>.hdf5
    data/attrs: env_args, total
    data/demo_0 .. data/demo_{n-1}: states, actions, obs, datagen_info/stage_label, ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_HDF5 = (
    _REPO_ROOT / "dexmimicgen/datasets/generated/single_arm_hammer_cleanup_autogen.hdf5"
)
for _pkg in (
    _REPO_ROOT / "robosuite",
    _REPO_ROOT / "dexmimicgen",
    _REPO_ROOT / "dexmimicgen" / "demo_generation",
):
    _p = str(_pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco as _mujoco  # noqa: E402 – needed for IK Jacobian
from scipy.spatial.transform import Rotation as _Rot, Slerp as _Slerp

from generate_demo_from_source import (  # noqa: E402
    build_env_action,
    compose_flattened_state,
    get_env_from_dataset,
    get_eef_site_name_and_id,
    get_gripper_joint_and_dof_indices,
    mat_to_quat_wxyz,
    mat_to_quat_xyzw,
    quat_xyzw_to_mat,
    rotvec_from_matrix,
    split_flattened_state,
)
from robosuite.utils.control_utils import orientation_error as _ori_err

import dexmimicgen  # noqa: F401,E402

ARM_REF_SEQUENCE = ["drawer", "drawer", "hammer", "drawer", "drawer", "drawer"]

# stage_id -> metadata (matches review_labels.txt)
STAGE_CATALOG = [
    {"stage_id": 1, "motion_label": 0, "ref": "drawer", "name": "move_to_drawer"},
    {"stage_id": 2, "motion_label": 1, "ref": "hand", "name": "grasp_handle"},
    {"stage_id": 3, "motion_label": 0, "ref": "drawer", "name": "open_drawer"},
    {"stage_id": 4, "motion_label": 1, "ref": "hand", "name": "open_hand"},
    {"stage_id": 5, "motion_label": 0, "ref": "hammer", "name": "move_to_hammer"},
    {"stage_id": 6, "motion_label": 1, "ref": "hand", "name": "grasp_hammer"},
    {"stage_id": 7, "motion_label": 0, "ref": "drawer", "name": "move_to_drawer"},
    {"stage_id": 8, "motion_label": 1, "ref": "hand", "name": "open_hand"},
    {"stage_id": 9, "motion_label": 0, "ref": "drawer", "name": "move_to_handle"},
    {"stage_id": 10, "motion_label": 1, "ref": "hand", "name": "grasp_handle"},
    {"stage_id": 11, "motion_label": 0, "ref": "drawer", "name": "close_and_home"},
]

@dataclass
class Segment:
    start: int
    end: int
    mode: str
    ref: str
    stage_id: int
    motion_label: int


def _eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _pose_from_body(env, body_id: int) -> np.ndarray:
    T = _eye4()
    T[:3, :3] = env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
    T[:3, 3] = env.sim.data.body_xpos[body_id].copy()
    return T


def _transform_pose(T_delta: np.ndarray, T: np.ndarray) -> np.ndarray:
    return T_delta @ T


def _inv_pose(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    out = _eye4()
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def _pose_from_site(env, site_id: int) -> np.ndarray:
    T = _eye4()
    T[:3, :3] = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    T[:3, 3] = env.sim.data.site_xpos[site_id].copy()
    return T


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _interp_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """SE(3) interpolation: linear translation + Slerp rotation."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    T = _eye4()
    T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    rots = _Rot.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    T[:3, :3] = _Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]
    return T




def _delta_from_pose_change(T_src: np.ndarray, T_new: np.ndarray) -> np.ndarray:
    R_src, p_src = T_src[:3, :3], T_src[:3, 3]
    R_new, p_new = T_new[:3, :3], T_new[:3, 3]
    dR = R_new @ R_src.T
    dp = p_new - dR @ p_src
    T_delta = _eye4()
    T_delta[:3, :3] = dR
    T_delta[:3, 3] = dp
    return T_delta


def _new_hammer_pose(T_src_ref: np.ndarray, dx: float, dy: float, dyaw_deg: float) -> np.ndarray:
    th = np.deg2rad(dyaw_deg)
    c, s = np.cos(th), np.sin(th)
    R_delta = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T_new = _eye4()
    T_new[:3, :3] = R_delta @ T_src_ref[:3, :3]
    T_new[:3, 3] = T_src_ref[:3, 3] + np.array([dx, dy, 0.0], dtype=np.float64)
    return T_new


def load_labels(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        return np.asarray(json.load(f)["labels"], dtype=np.int64)


def segments_from_labels(labels: np.ndarray) -> list[Segment]:
    T = len(labels)
    segments: list[Segment] = []
    arm_ref_i = 0
    stage_i = 0
    i = 0
    while i < T:
        if labels[i] == 1:
            s = i
            while i < T and labels[i] == 1:
                i += 1
            meta = STAGE_CATALOG[stage_i]
            stage_i += 1
            segments.append(
                Segment(s, i - 1, "replay_action", "hand", meta["stage_id"], meta["motion_label"])
            )
        else:
            s = i
            while i < T and labels[i] == 0:
                i += 1
            ref = ARM_REF_SEQUENCE[min(arm_ref_i, len(ARM_REF_SEQUENCE) - 1)]
            arm_ref_i += 1
            meta = STAGE_CATALOG[stage_i]
            stage_i += 1
            if ref == "hammer":
                mode = "transform_hammer"
            elif meta["stage_id"] in (7, 9):
                mode = "transform_drawer_anchor"
            else:
                mode = "transform_drawer"
            segments.append(
                Segment(s, i - 1, mode, ref, meta["stage_id"], meta["motion_label"])
            )
    if stage_i != len(STAGE_CATALOG):
        raise ValueError(
            f"Expected {len(STAGE_CATALOG)} stages, got {stage_i} from labels. "
            "Check labels.json matches the 11-stage plan."
        )
    return segments


def _first_grasp_hammer_frame(segments: list[Segment]) -> int | None:
    seen_hammer_arm = False
    for seg in segments:
        if seg.mode == "transform_hammer":
            seen_hammer_arm = True
        elif seen_hammer_arm and seg.ref == "hand":
            return seg.start
    return None


def _first_carry_start_frame(
    segments: list[Segment], grasp_start: int | None
) -> int | None:
    """First frame of Stage 7 (carry to drawer) — hammer pinned to arm from here."""
    if grasp_start is None:
        return None
    after_grasp = False
    for seg in segments:
        if seg.start >= grasp_start:
            after_grasp = True
        if after_grasp and seg.mode == "transform_drawer_anchor":
            return seg.start
    return None


def _first_open_start_frame(
    segments: list[Segment], carry_start: int | None
) -> int | None:
    """First frame of Stage 8 (open_hand) — hammer released from arm here."""
    if carry_start is None:
        return None
    for seg in segments:
        if seg.start > carry_start and seg.mode == "replay_action":
            return seg.start
    return None


def _find_hammer_free_joint(env) -> tuple[int, int]:
    jid = env.sim.model.joint_name2id("hammer_joint0")
    return int(env.sim.model.jnt_qposadr[jid]), int(env.sim.model.jnt_dofadr[jid])


def _set_hammer_qpos_in_parts(parts, hammer_qadr: int, T_h: np.ndarray) -> None:
    parts["qpos"][hammer_qadr : hammer_qadr + 3] = T_h[:3, 3]
    parts["qpos"][hammer_qadr + 3 : hammer_qadr + 7] = mat_to_quat_wxyz(T_h[:3, :3])


def _patch_hammer_qpos_in_state(
    state_vec: np.ndarray,
    hammer_qadr: int,
    T_h: np.ndarray,
    nq: int,
    nv: int,
    na: int,
    has_time: bool,
) -> np.ndarray:
    parts = split_flattened_state(state_vec, nq, nv, na)
    _set_hammer_qpos_in_parts(parts, hammer_qadr, T_h)
    return compose_flattened_state(parts, has_time=has_time)


def _rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    return rotvec_from_matrix(R)


def _update_action_dict_from_eef(
    action_dict_grp,
    eef_T: np.ndarray,
    transform_mask: np.ndarray,
    gripper_src: dict[str, np.ndarray],
) -> None:
    """Update right_rel_* for frames where EE trajectory changed."""
    T = eef_T.shape[0]
    rel_pos = np.array(action_dict_grp["right_rel_pos"])
    rel_rot_aa = np.array(action_dict_grp["right_rel_rot_axis_angle"])
    gripper = np.array(action_dict_grp["right_gripper"])

    for t in range(1, T):
        if not transform_mask[t]:
            continue
        R_prev, R_cur = eef_T[t - 1, :3, :3], eef_T[t, :3, :3]
        p_prev, p_cur = eef_T[t - 1, :3, 3], eef_T[t, :3, 3]
        R_delta = R_prev.T @ R_cur
        rel_pos[t] = R_prev.T @ (p_cur - p_prev)
        rel_rot_aa[t] = _rot_to_axis_angle(R_delta)
        gripper[t] = gripper_src["right_gripper"][t]

    action_dict_grp["right_rel_pos"][...] = rel_pos
    action_dict_grp["right_rel_rot_axis_angle"][...] = rel_rot_aa
    action_dict_grp["right_gripper"][...] = gripper
    if "right_rel_rot_6d" in action_dict_grp:
        # Keep source 6d where unchanged; training often uses axis_angle path.
        pass


def _ur5e_arm_qpos_indices(model, ridx: int = 0) -> np.ndarray:
    names = [
        f"robot{ridx}_shoulder_pan_joint",
        f"robot{ridx}_shoulder_lift_joint",
        f"robot{ridx}_elbow_joint",
        f"robot{ridx}_wrist_1_joint",
        f"robot{ridx}_wrist_2_joint",
        f"robot{ridx}_wrist_3_joint",
    ]
    return np.array([int(model.jnt_qposadr[model.joint_name2id(n)]) for n in names], dtype=np.int64)


def _ur5e_arm_dof_indices(model, ridx: int = 0) -> np.ndarray:
    names = [
        f"robot{ridx}_shoulder_pan_joint",
        f"robot{ridx}_shoulder_lift_joint",
        f"robot{ridx}_elbow_joint",
        f"robot{ridx}_wrist_1_joint",
        f"robot{ridx}_wrist_2_joint",
        f"robot{ridx}_wrist_3_joint",
    ]
    return np.array([int(model.jnt_dofadr[model.joint_name2id(n)]) for n in names], dtype=np.int64)


def _sync_datagen_and_obs_from_states(
    env,
    demo_out,
    states: np.ndarray,
    hammer_bid: int,
    ridx: int,
    site_id: int,
    nq: int,
    nv: int,
    na: int,
    has_time: bool,
    gripper_src: np.ndarray,
) -> np.ndarray:
    T = states.shape[0]
    eef_pose = np.zeros((T, 1, 4, 4), dtype=np.float64)

    for t in range(T):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        T_eef = _eye4()
        T_eef[:3, :3] = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
        T_eef[:3, 3] = env.sim.data.site_xpos[site_id].copy()
        eef_pose[t, 0] = T_eef
        demo_out["datagen_info/object_poses/hammer_1"][t] = _pose_from_body(env, hammer_bid)

    demo_out["datagen_info/eef_pose"][...] = eef_pose
    demo_out["datagen_info/target_pose"][...] = eef_pose
    demo_out["datagen_info/gripper_action"][...] = gripper_src

    if "obs" in demo_out:
        obs = demo_out["obs"]
        if f"robot{ridx}_eef_pos" in obs:
            obs[f"robot{ridx}_eef_pos"][...] = eef_pose[:, 0, :3, 3]
        quat_key = (
            f"robot{ridx}_eef_quat_site"
            if f"robot{ridx}_eef_quat_site" in obs
            else f"robot{ridx}_eef_quat"
        )
        if quat_key in obs:
            quat = np.zeros((T, 4), dtype=np.float64)
            for t in range(T):
                quat[t] = mat_to_quat_xyzw(eef_pose[t, 0, :3, :3])
            obs[quat_key][...] = quat
        if "hammer_pos" in obs:
            obs["hammer_pos"][...] = demo_out["datagen_info/object_poses/hammer_1"][:, :3, 3]
        if "hammer_quat" in obs:
            hq = np.zeros((T, 4), dtype=np.float64)
            for t in range(T):
                hq[t] = mat_to_quat_xyzw(demo_out["datagen_info/object_poses/hammer_1"][t, :3, :3])
            obs["hammer_quat"][...] = hq
        _sync_joint_obs_from_states(obs, env, states, ridx, nq, nv, na, has_time)

    return eef_pose[:, 0]


def _sync_joint_obs_from_states(obs_grp, env, states, ridx, nq, nv, na, has_time):
    T = states.shape[0]
    if f"robot{ridx}_joint_pos" not in obs_grp:
        return
    model = env.sim.model
    arm_qids = _ur5e_arm_qpos_indices(model, ridx)
    arm_dids = _ur5e_arm_dof_indices(model, ridx)
    grip_qids, grip_dids = get_gripper_joint_and_dof_indices(env)
    joint_keys = [
        f"robot{ridx}_joint_pos",
        f"robot{ridx}_joint_vel",
        f"robot{ridx}_joint_pos_cos",
        f"robot{ridx}_joint_pos_sin",
        f"robot{ridx}_gripper_qpos",
        f"robot{ridx}_gripper_qvel",
    ]
    buf = {k: np.zeros((T, obs_grp[k].shape[1]), dtype=np.float64) for k in joint_keys if k in obs_grp}
    for t in range(T):
        parts = split_flattened_state(states[t], nq, nv, na)
        arm_q = parts["qpos"][arm_qids]
        arm_qv = parts["qvel"][arm_dids]
        if f"robot{ridx}_joint_pos" in buf:
            buf[f"robot{ridx}_joint_pos"][t] = arm_q
            buf[f"robot{ridx}_joint_pos_cos"][t] = np.cos(arm_q)
            buf[f"robot{ridx}_joint_pos_sin"][t] = np.sin(arm_q)
            buf[f"robot{ridx}_joint_vel"][t] = arm_qv
        if f"robot{ridx}_gripper_qpos" in buf:
            buf[f"robot{ridx}_gripper_qpos"][t] = parts["qpos"][grip_qids]
            buf[f"robot{ridx}_gripper_qvel"][t] = parts["qvel"][grip_dids]
    for k, arr in buf.items():
        obs_grp[k][...] = arr


def _frame_segment_map(segments: list[Segment], T: int) -> list[Segment]:
    seg_at: list[Segment | None] = [None] * T
    for seg in segments:
        for t in range(seg.start, seg.end + 1):
            seg_at[t] = seg
    if any(s is None for s in seg_at):
        raise ValueError("segments do not cover all frames")
    return seg_at  # type: ignore[return-value]


def _pin_hammer_free(
    env,
    hammer_qadr: int,
    hammer_dadr: int,
    T_h: np.ndarray,
) -> None:
    env.sim.data.qpos[hammer_qadr : hammer_qadr + 3] = T_h[:3, 3]
    env.sim.data.qpos[hammer_qadr + 3 : hammer_qadr + 7] = mat_to_quat_wxyz(T_h[:3, :3])
    if hammer_dadr + 6 <= env.sim.data.qvel.shape[0]:
        env.sim.data.qvel[hammer_dadr : hammer_dadr + 6] = 0.0


def _pin_gripper_from_src(
    env,
    src_parts: dict,
    gripper_qpos_ids: np.ndarray,
    gripper_dof_ids: np.ndarray,
) -> None:
    env.sim.data.qpos[gripper_qpos_ids] = src_parts["qpos"][gripper_qpos_ids]
    env.sim.data.qvel[gripper_dof_ids] = src_parts["qvel"][gripper_dof_ids]


def _ik_arm_to_eef(
    env,
    site_id: int,
    arm_qpos_ids: np.ndarray,
    arm_dof_ids: np.ndarray,
    target_T: np.ndarray,
    max_iters: int = 15,
    tol: float = 5e-5,
    damping: float = 0.04,
) -> np.ndarray:
    """Damped-least-squares IK.

    max_iters is intentionally small (default 15) to cap total joint displacement
    at 1.5 rad (15 × 0.1 rad/iter), preventing IK branch switches (wrist flips
    require ~2.2 rad norm). With chain warm-starting, frame-to-frame targets are
    tiny so 15 iterations is more than sufficient for accurate tracking.
    """
    nv = env.sim.model.nv
    for _ in range(max_iters):
        env.sim.forward()
        p_cur = env.sim.data.site_xpos[site_id].copy()
        R_cur = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = target_T[:3, 3] - p_cur
        ori_err = _ori_err(target_T[:3, :3], R_cur)
        err = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(err) < tol:
            break

        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        _mujoco.mj_jacSite(
            env.sim.model._model, env.sim.data._data, jacp, jacr, site_id
        )
        J = np.vstack([jacp[:, arm_dof_ids], jacr[:, arm_dof_ids]])  # 6 × n_arm
        A = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)
        # Clamp step so joints move at most 0.1 rad per iter.
        step = np.linalg.norm(dq)
        if step > 0.1:
            dq = dq * (0.1 / step)
        env.sim.data.qpos[arm_qpos_ids] += dq

    return env.sim.data.qpos[arm_qpos_ids].copy()


def _compute_arm_eef_target(
    t: int,
    seg: Segment,
    env,
    site_id: int,
    eef_src: np.ndarray,
    segment_anchors: dict[int, np.ndarray],
) -> np.ndarray:
    """Two-endpoint SE(3) trajectory deformation (guideline Section 5).

    T_new[t] = interp_SE3(Delta_a, Delta_b, smoothstep(u)) @ T_src[t]

    Delta_a and Delta_b are pre-computed at each segment's entry frame:
      transform_hammer:         stored by the outer handler (Delta_b = T_delta).
      transform_drawer:         stored by the outer handler (Delta_b = I; drawer fixed).
      transform_drawer_anchor:  computed here on the first call (Delta_b = I; drawer fixed).

    Result: start has no jump, end converges to the correct object-relative goal,
    and any drift accumulated during hammer stages fades out by segment end.
    """
    if seg.mode == "transform_drawer_anchor" and t == seg.start:
        # Entry frame: capture actual sim EEF; endpoint converges back to source.
        T_a_new = _pose_from_site(env, site_id)
        segment_anchors[f"Delta_a_{seg.start}"] = T_a_new @ _inv_pose(eef_src[seg.start])
        segment_anchors[f"Delta_b_{seg.start}"] = _eye4()
        return T_a_new  # IK target = current EEF (no movement at first frame)

    Da = segment_anchors[f"Delta_a_{seg.start}"]
    Db = segment_anchors[f"Delta_b_{seg.start}"]
    u = (t - seg.start) / max(seg.end - seg.start, 1)
    alpha = _smoothstep(u)
    return _interp_se3(Da, Db, alpha) @ eef_src[t]


def _build_states_and_actions(
    env,
    states_src: np.ndarray,
    actions_src: np.ndarray,
    segments: list[Segment],
    eef_src: np.ndarray,
    hammer_world_src: np.ndarray,
    drawer_world_src: np.ndarray,
    T_delta: np.ndarray,
    hammer_world_new: np.ndarray,
    grasp_start: int | None,
    carry_start: int | None,
    open_start: int | None,
    hammer_qadr: int,
    hammer_dadr: int,
    drawer_bid: int,
    gripper_qpos_ids: np.ndarray,
    gripper_dof_ids: np.ndarray,
    nq: int,
    nv: int,
    na: int,
    has_time: bool,
    ridx: int,
    arm_name: str,
    site_id: int,
    control_substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    T = states_src.shape[0]
    states_new = np.zeros_like(states_src)
    actions_new = np.zeros_like(actions_src)
    seg_at = _frame_segment_map(segments, T)
    segment_anchors: dict[int, np.ndarray] = {}

    # Stage 7 carry: pin hammer at fixed pose relative to EEF (kinematic attachment).
    # grip_rel = inv(EEF) @ hammer at carry_start.  Regardless of T_delta:
    #   grip_rel ≈ inv(EEF_src(carry_start)) @ H_src(carry_start) = source_grip_rel
    # because Stage 6 IK tracks T_delta @ EEF_src and hammer is at T_delta @ H_src,
    # so the T_delta cancels.  The arm returns to source trajectory by Stage 7 end
    # (Delta_b = I), so hammer_carry_pos converges near the drawer for any offset.
    hammer_grip_rel: np.ndarray | None = None   # set once at carry_start
    hammer_carry_pos: np.ndarray | None = None  # EEF(t) @ hammer_grip_rel each carry frame

    # Offset into the flat action vector where the gripper action begins.
    # build_env_action zeros the gripper; we overwrite it with source signals
    # so the hand controller receives the correct open/close commands.
    arm_ctrl_dim = env.robots[ridx].composite_controller.part_controllers[arm_name].control_dim

    # Arm joint indices (qpos / dof addresses) for the UR5e.
    arm_qpos_ids = _ur5e_arm_qpos_indices(env.sim.model, ridx)  # e.g. [0,1,2,3,4,5]
    arm_dof_ids  = _ur5e_arm_dof_indices(env.sim.model, ridx)
    arm_ndof = len(arm_dof_ids)

    init_state = states_src[0].copy()
    if grasp_start is None or 0 < grasp_start:
        init_state = _patch_hammer_qpos_in_state(
            init_state, hammer_qadr, hammer_world_new[0], nq, nv, na, has_time
        )

    init_parts = split_flattened_state(init_state, nq, nv, na)
    init_parts["qvel"][arm_dof_ids] = 0.0
    init_state = compose_flattened_state(init_parts, has_time=has_time)

    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()

    # current_arm_qpos tracks the arm configuration we *want* at each frame.
    # For pre-hammer stages it is copied directly from source; for hammer /
    # carry / close-drawer stages it is the IK solution for the target EEF pose.
    # We pin this configuration during substeps so finger-force coupling cannot
    # drift the wrist, and we store it in states_new so the demo is self-consistent.
    current_arm_qpos: np.ndarray = init_parts["qpos"][arm_qpos_ids].copy()

    # arm_follows_source: True until the first transform_hammer segment.
    arm_follows_source = True

    for t in range(T):
        seg = seg_at[t]
        src_parts = split_flattened_state(states_src[t], nq, nv, na)
        # Hammer pinning phases:
        #   Stages 1-6: pin to hammer_world_new[t]  = T_delta @ H_src[t]
        #   Stage 7:    EEF-relative kinematic carry (hammer_carry_pos = EEF(t) @ grip_rel)
        #   Stage 8+:   free (hammer placed in drawer, arm opens)
        pin_hammer = open_start is not None and t < open_start
        in_carry = (carry_start is not None and t >= carry_start
                    and open_start is not None and t < open_start)

        # ── 1. Update current_arm_qpos ────────────────────────────────────
        if arm_follows_source:
            if seg.mode == "transform_hammer":
                arm_follows_source = False  # fall through to IK branch below
            else:
                # Pre-hammer: arm follows source exactly.
                current_arm_qpos = src_parts["qpos"][arm_qpos_ids].copy()

        if not arm_follows_source:
            # Set env to current_arm_qpos so (a) anchor captures are correct and
            # (b) the IK warm-start is consistent with the previous frame.
            env.sim.data.qpos[arm_qpos_ids] = current_arm_qpos
            # Before grasp_start (hand open, no object contact): safe to teleport
            # fingers to source positions for consistency.
            # After grasp_start: keep physics gripper state so the hand maintains
            # contact with the hammer without ejecting it via non-physical overlaps.
            if grasp_start is None or t < grasp_start:
                env.sim.data.qpos[gripper_qpos_ids] = src_parts["qpos"][gripper_qpos_ids]
                env.sim.data.qvel[gripper_dof_ids] = src_parts["qvel"][gripper_dof_ids]
            env.sim.data.qvel[arm_dof_ids] = 0.0
            if pin_hammer and not in_carry:
                _pin_hammer_free(env, hammer_qadr, hammer_dadr, hammer_world_new[t])
            elif not pin_hammer and open_start is not None and t == open_start:
                # Zero out residual carry velocity so the hammer drops cleanly.
                last_carry = hammer_carry_pos if hammer_carry_pos is not None else hammer_world_new[t - 1]
                _pin_hammer_free(env, hammer_qadr, hammer_dadr, last_carry)
            env.sim.forward()

            if seg.mode == "replay_action":
                if t == seg.start:
                    segment_anchors[seg.start] = _pose_from_site(env, site_id)
                # During the grasp phase (Stage 6), the source arm actively moves
                # into the hammer to complete the grasp.  The hammer moves in
                # source as the fingers close, so we cannot use T_delta @ eef_src.
                # Instead map the source EEF-hammer relative pose onto the gen
                # hammer: T_target = hammer_world_new[t] @ inv(src_hm[t]) @ eef_src[t].
                # This gives the same EEF-hammer relationship as source regardless
                # of how much the hammer drifted in the source physics.
                if (grasp_start is not None and t >= grasp_start
                        and carry_start is not None and t < carry_start):
                    T_target = hammer_world_new[t] @ _inv_pose(hammer_world_src[t]) @ eef_src[t]
                    current_arm_qpos = _ik_arm_to_eef(
                        env, site_id, arm_qpos_ids, arm_dof_ids, T_target
                    )
                    env.sim.data.qpos[arm_qpos_ids] = current_arm_qpos
                    env.sim.forward()
                # else: current_arm_qpos unchanged (arm holds at anchor)
            elif seg.mode == "transform_hammer" and t == seg.start:
                # Entry frame: pre-compute two-endpoint deltas for Stage 5.
                # Delta_a corrects the ~1 cm discrepancy between sim and source at entry.
                # Delta_b = T_delta so the endpoint aligns to the new hammer-relative goal.
                T_a_new = _pose_from_site(env, site_id)
                segment_anchors["hammer_entry"] = T_a_new
                segment_anchors[f"Delta_a_{seg.start}"] = T_a_new @ _inv_pose(eef_src[seg.start])
                segment_anchors[f"Delta_b_{seg.start}"] = T_delta
                # current_arm_qpos unchanged (arm holds at entry frame)
            elif seg.mode == "transform_drawer" and t == seg.start:
                # Entry frame: pre-compute two-endpoint deltas.
                # Delta_b = I because the drawer is not randomized: the endpoint
                # target is the same as source, so drift gradually fades to zero.
                T_a_new = _pose_from_site(env, site_id)
                segment_anchors[f"Delta_a_{seg.start}"] = T_a_new @ _inv_pose(eef_src[seg.start])
                segment_anchors[f"Delta_b_{seg.start}"] = _eye4()
                # current_arm_qpos unchanged (arm holds at entry frame)
            else:
                T_target = _compute_arm_eef_target(
                    t, seg, env, site_id, eef_src, segment_anchors,
                )
                current_arm_qpos = _ik_arm_to_eef(
                    env, site_id, arm_qpos_ids, arm_dof_ids, T_target
                )
                env.sim.data.qpos[arm_qpos_ids] = current_arm_qpos
                env.sim.forward()

                # Stage 7 EEF-relative carry: compute hammer pose from arm EEF.
                # grip_rel is set once at carry_start ≈ inv(EEF_src) @ H_src (source grip),
                # so hammer_carry_pos = EEF(t) @ grip_rel converges to the drawer as the
                # arm blends back to the source trajectory (Delta_b = I for Stage 7).
                if in_carry:
                    eef_now = _pose_from_site(env, site_id)
                    if hammer_grip_rel is None:
                        hammer_grip_rel = _inv_pose(eef_now) @ hammer_world_new[carry_start]
                    hammer_carry_pos = eef_now @ hammer_grip_rel
                    _pin_hammer_free(env, hammer_qadr, hammer_dadr, hammer_carry_pos)
                    env.sim.forward()

        # ── 1b. Determine hammer pin target for step 4 (state storage) ───
        if pin_hammer:
            if in_carry:
                hammer_pin_t: np.ndarray | None = hammer_carry_pos
            else:
                hammer_pin_t = hammer_world_new[t]
        elif open_start is not None and t == open_start:
            # Store the last carry position so states_new[open_start] is clean.
            hammer_pin_t = hammer_carry_pos if hammer_carry_pos is not None else hammer_world_new[t - 1]
        else:
            hammer_pin_t = None

        # ── 2. For pre-hammer stages: set env arm joints from source ──────
        if arm_follows_source:
            env.sim.data.qpos[arm_qpos_ids] = current_arm_qpos
            env.sim.data.qvel[arm_dof_ids] = src_parts["qvel"][arm_dof_ids]
            env.sim.data.qpos[gripper_qpos_ids] = src_parts["qpos"][gripper_qpos_ids]
            env.sim.data.qvel[gripper_dof_ids] = src_parts["qvel"][gripper_dof_ids]
            if pin_hammer:
                # arm_follows_source is only True before Stage 5 (carry hasn't started yet)
                _pin_hammer_free(env, hammer_qadr, hammer_dadr, hammer_world_new[t])
            env.sim.forward()

            if seg.mode == "replay_action":
                if t == seg.start:
                    segment_anchors[seg.start] = _pose_from_site(env, site_id)

        # ── 3. Compute OSC action (absolute target stored in actions_new) ─
        if seg.mode == "replay_action":
            # During Stage 6 (grasp phase) the arm tracks the transformed EEF;
            # for all other replay_action stages it holds at the anchor.
            if (grasp_start is not None and t >= grasp_start
                    and carry_start is not None and t < carry_start):
                cur_eef = _pose_from_site(env, site_id)
                arm_action = np.concatenate(
                    [cur_eef[:3, 3], rotvec_from_matrix(cur_eef[:3, :3])], axis=0
                )
            else:
                T_eef_hold = segment_anchors[seg.start]
                arm_action = np.concatenate(
                    [T_eef_hold[:3, 3], rotvec_from_matrix(T_eef_hold[:3, :3])], axis=0
                )
        else:
            cur_eef = _pose_from_site(env, site_id)
            arm_action = np.concatenate(
                [cur_eef[:3, 3], rotvec_from_matrix(cur_eef[:3, :3])], axis=0
            )

        env_action = build_env_action(env, {(ridx, arm_name): arm_action})
        # Restore source gripper action; build_env_action zeros the gripper which
        # prevents the hand from closing and makes grasping impossible.
        env_action[arm_ctrl_dim:] = actions_src[t, arm_ctrl_dim:]
        actions_new[t] = env_action

        # ── 4. Store state BEFORE substeps ────────────────────────────────
        # states[t] = sim state at the start of frame t (before action t).
        out_parts = split_flattened_state(env.sim.get_state().flatten(), nq, nv, na)
        out_parts["qpos"][arm_qpos_ids]    = current_arm_qpos
        out_parts["qvel"][arm_dof_ids]     = (
            src_parts["qvel"][arm_dof_ids] if arm_follows_source else np.zeros(arm_ndof)
        )
        # After grasp_start keep the actual physics gripper state in stored demos
        # so the carry/release phases reflect real finger-object contact.
        if grasp_start is None or t < grasp_start:
            out_parts["qpos"][gripper_qpos_ids] = src_parts["qpos"][gripper_qpos_ids]
            out_parts["qvel"][gripper_dof_ids]  = src_parts["qvel"][gripper_dof_ids]
        # Explicitly write the desired hammer pose into the stored state so that
        # physics drift from the last substep's env.step() never leaks into
        # datagen_info/object_poses.  Covers Stages 1-7 (approach, grasp, carry).
        if hammer_pin_t is not None:
            _set_hammer_qpos_in_parts(out_parts, hammer_qadr, hammer_pin_t)
        states_new[t] = compose_flattened_state(out_parts, has_time=has_time)

        # ── 5. Substeps (pin arm + gripper + hammer) ──────────────────────
        for _ in range(control_substeps):
            env.done = False
            env.sim.data.qpos[arm_qpos_ids] = current_arm_qpos
            env.sim.data.qvel[arm_dof_ids] = (
                src_parts["qvel"][arm_dof_ids] if arm_follows_source else np.zeros(arm_ndof)
            )
            if pin_hammer:
                hammer_pin_substep = (
                    hammer_carry_pos if in_carry else hammer_world_new[t]
                )
                _pin_hammer_free(env, hammer_qadr, hammer_dadr, hammer_pin_substep)
            # Before grasp_start fingers are open and safe to teleport for
            # consistency; after grasp_start let physics maintain the grip.
            if grasp_start is None or t < grasp_start:
                _pin_gripper_from_src(env, src_parts, gripper_qpos_ids, gripper_dof_ids)
            env.sim.forward()
            env.step(env_action)

        if (t + 1) % 100 == 0:
            print(f"  rollout {t + 1}/{T} ({seg.mode})")

    return states_new, actions_new


@dataclass
class _AutogenContext:
    source_hdf5: str
    labels_path: str
    source_demo: str
    stage_label: np.ndarray
    segments: list[Segment]
    states_src: np.ndarray
    actions_src: np.ndarray
    eef_src: np.ndarray
    target_src: np.ndarray
    gripper_src: np.ndarray
    hammer_world_src: np.ndarray
    drawer_world_src: np.ndarray
    hammer_arm_seg: Segment
    grasp_start: int | None
    carry_start: int | None
    open_start: int | None
    T_hammer_ref: np.ndarray
    env: object
    hammer_bid: int
    drawer_bid: int
    hammer_qadr: int
    hammer_dadr: int
    nq: int
    nv: int
    na: int
    has_time: bool
    gripper_qpos_ids: np.ndarray
    gripper_dof_ids: np.ndarray
    ridx: int
    arm_name: str
    site_id: int
    T: int


def _sample_hammer_offsets(
    rng: np.random.Generator,
    num_demos: int,
    dx_range: tuple[float, float],
    dy_range: tuple[float, float],
    yaw_range: tuple[float, float],
    fixed: tuple[float, float, float] | None,
) -> list[tuple[float, float, float]]:
    if fixed is not None:
        return [fixed] * num_demos
    offsets = []
    for _ in range(num_demos):
        offsets.append(
            (
                float(rng.uniform(dx_range[0], dx_range[1])),
                float(rng.uniform(dy_range[0], dy_range[1])),
                float(rng.uniform(yaw_range[0], yaw_range[1])),
            )
        )
    return offsets


def _dataset_demo_keys(data: h5py.Group) -> list[str]:
    return sorted(
        k for k in data.keys() if k.startswith("demo_") and isinstance(data[k], h5py.Group)
    )


def _resolve_source_demo(data: h5py.Group, source_hdf5: str) -> str:
    keys = _dataset_demo_keys(data)
    if not keys:
        raise ValueError(f"{source_hdf5}: no demo_* groups under data/")
    if len(keys) > 1:
        raise ValueError(
            f"{source_hdf5} has {len(keys)} episodes ({', '.join(keys)}); "
            "use an HDF5 with exactly one source demo"
        )
    return keys[0]


def _build_autogen_context(
    source_hdf5: str,
    labels_path: str,
) -> _AutogenContext:
    labels = load_labels(labels_path)
    segments = segments_from_labels(labels)

    with h5py.File(source_hdf5, "r") as src:
        source_demo = _resolve_source_demo(src["data"], source_hdf5)
        demo_src = src[f"data/{source_demo}"]
        states_src = demo_src["states"][()]
        T = states_src.shape[0]
        if len(labels) != T:
            raise ValueError(f"labels length {len(labels)} != trajectory {T}")

        env = get_env_from_dataset(src)
        env.ignore_done = True
        env.done = False
        hammer_bid = env.obj_body_id["hammer"]
        drawer_bid = env.obj_body_id["drawer"]
        hammer_qadr, hammer_dadr = _find_hammer_free_joint(env)
        nq, nv, na = env.sim.model.nq, env.sim.model.nv, env.sim.model.na
        has_time = states_src.shape[1] == (nq + nv + na + 1)
        gripper_qpos_ids, gripper_dof_ids = get_gripper_joint_and_dof_indices(env)
        ridx, arm_name = 0, env.robots[0].arms[0]
        _, site_id = get_eef_site_name_and_id(env, ridx, arm_name)

        hammer_arm_seg = next(s for s in segments if s.mode == "transform_hammer")
        env.sim.set_state_from_flattened(states_src[hammer_arm_seg.start])
        env.sim.forward()
        T_hammer_ref = _pose_from_body(env, hammer_bid)

        hammer_world_src = np.zeros((T, 4, 4), dtype=np.float64)
        drawer_world_src = np.zeros((T, 4, 4), dtype=np.float64)
        # Re-derive eef_src by replaying source states through the sim so that
        # the site convention matches the grip_site used by the IK.  Reading
        # datagen_info/eef_pose from the source HDF5 uses a different rotation
        # frame (120° offset) that would cause the IK to target the wrong orientation.
        eef_src_sim = np.zeros((T, 1, 4, 4), dtype=np.float64)
        for t in range(T):
            env.sim.set_state_from_flattened(states_src[t])
            env.sim.forward()
            hammer_world_src[t] = _pose_from_body(env, hammer_bid)
            drawer_world_src[t] = _pose_from_body(env, drawer_bid)
            eef_src_sim[t, 0] = _pose_from_site(env, site_id)

        return _AutogenContext(
            source_hdf5=os.path.abspath(source_hdf5),
            labels_path=os.path.abspath(labels_path),
            source_demo=source_demo,
            stage_label=np.asarray(labels, dtype=np.int64),
            segments=segments,
            states_src=states_src,
            actions_src=demo_src["actions"][()],
            eef_src=eef_src_sim,
            target_src=demo_src["datagen_info/target_pose"][()],
            gripper_src=demo_src["datagen_info/gripper_action"][()],
            hammer_world_src=hammer_world_src,
            drawer_world_src=drawer_world_src,
            hammer_arm_seg=hammer_arm_seg,
            grasp_start=_first_grasp_hammer_frame(segments),
            carry_start=_first_carry_start_frame(
                segments, _first_grasp_hammer_frame(segments)
            ),
            open_start=_first_open_start_frame(
                segments,
                _first_carry_start_frame(
                    segments, _first_grasp_hammer_frame(segments)
                ),
            ),
            T_hammer_ref=T_hammer_ref,
            env=env,
            hammer_bid=hammer_bid,
            drawer_bid=drawer_bid,
            hammer_qadr=hammer_qadr,
            hammer_dadr=hammer_dadr,
            nq=nq,
            nv=nv,
            na=na,
            has_time=has_time,
            gripper_qpos_ids=gripper_qpos_ids,
            gripper_dof_ids=gripper_dof_ids,
            ridx=ridx,
            arm_name=arm_name,
            site_id=site_id,
            T=T,
        )


def _generate_variant(
    ctx: _AutogenContext,
    hammer_dx: float,
    hammer_dy: float,
    hammer_dyaw_deg: float,
    control_substeps: int,
    skip_sim: bool,
) -> dict:
    T_delta = _delta_from_pose_change(
        ctx.T_hammer_ref,
        _new_hammer_pose(ctx.T_hammer_ref, hammer_dx, hammer_dy, hammer_dyaw_deg),
    )

    hammer_world_new = ctx.hammer_world_src.copy()
    eef_src = ctx.eef_src[:, 0]
    eef_target = eef_src.copy()
    transform_mask = np.zeros(ctx.T, dtype=bool)

    for seg in ctx.segments:
        for t in range(seg.start, seg.end + 1):
            if seg.mode == "transform_hammer":
                eef_target[t] = _transform_pose(T_delta, eef_src[t])
                transform_mask[t] = True
            elif (seg.mode == "replay_action"
                    and ctx.grasp_start is not None and t >= ctx.grasp_start
                    and ctx.carry_start is not None and t < ctx.carry_start):
                # Stage 6 (grasp): arm tracks T_delta-transformed source EEF.
                eef_target[t] = _transform_pose(T_delta, eef_src[t])
                transform_mask[t] = True
            elif seg.mode in ("transform_drawer", "transform_drawer_anchor"):
                # Drawer not randomized; source poses are the ideal targets in skip_sim mode.
                eef_target[t] = eef_src[t]
                transform_mask[t] = True
            # Stages 1-7: transform source hammer trajectory by T_delta (left multiply).
            #   For T_delta=I: hammer stays on source trajectory.
            #   For nonzero offset: hammer follows the shifted world trajectory.
            # Stage 8+: free (leave as source init; not used during Stage 8+).
            #
            # Extending through Stage 7 ensures the hammer follows the source
            # carry path (tilting into the drawer) rather than a constant grip
            # offset, which would put it at the wrong place for Stage 8.
            if ctx.open_start is None or t < ctx.open_start:
                hammer_world_new[t] = _transform_pose(T_delta, hammer_world_new[t])

    if skip_sim:
        states_new = ctx.states_src.copy()
        actions_new = ctx.actions_src.copy()
        for t in range(ctx.T):
            if ctx.grasp_start is not None and t < ctx.grasp_start:
                states_new[t] = _patch_hammer_qpos_in_state(
                    ctx.states_src[t],
                    ctx.hammer_qadr,
                    hammer_world_new[t],
                    ctx.nq,
                    ctx.nv,
                    ctx.na,
                    ctx.has_time,
                )
        eef_pose_out = ctx.eef_src.copy()
        eef_pose_out[:, 0] = eef_target
        target_out = ctx.target_src.copy()
        target_out[:, 0] = eef_target
    else:
        ctx.env.done = False
        states_new, actions_new = _build_states_and_actions(
            env=ctx.env,
            states_src=ctx.states_src,
            actions_src=ctx.actions_src,
            segments=ctx.segments,
            eef_src=eef_src,
            hammer_world_src=ctx.hammer_world_src,
            drawer_world_src=ctx.drawer_world_src,
            T_delta=T_delta,
            hammer_world_new=hammer_world_new,
            grasp_start=ctx.grasp_start,
            carry_start=ctx.carry_start,
            open_start=ctx.open_start,
            hammer_qadr=ctx.hammer_qadr,
            hammer_dadr=ctx.hammer_dadr,
            drawer_bid=ctx.drawer_bid,
            gripper_qpos_ids=ctx.gripper_qpos_ids,
            gripper_dof_ids=ctx.gripper_dof_ids,
            nq=ctx.nq,
            nv=ctx.nv,
            na=ctx.na,
            has_time=ctx.has_time,
            ridx=ctx.ridx,
            arm_name=ctx.arm_name,
            site_id=ctx.site_id,
            control_substeps=control_substeps,
        )
        eef_pose_out = None
        target_out = None

    transform_mask = transform_mask | np.array(
        [not np.allclose(states_new[t], ctx.states_src[t]) for t in range(ctx.T)]
    )

    return {
        "states": states_new,
        "actions": actions_new,
        "eef_pose_out": eef_pose_out,
        "target_out": target_out,
        "hammer_world_new": hammer_world_new,
        "transform_mask": transform_mask,
        "T_delta": T_delta,
        "hammer_offset": (hammer_dx, hammer_dy, hammer_dyaw_deg),
        "skip_sim": skip_sim,
    }


def _write_demo_group(demo_out, ctx: _AutogenContext, variant: dict, demo_index: int) -> None:
    demo_out["states"][...] = variant["states"]
    demo_out["actions"][...] = variant["actions"]

    dg = demo_out["datagen_info"]
    for _legacy in ("stage_label", "stage_motion_label", "stage_ref"):
        if _legacy in dg:
            del dg[_legacy]
    for _attr in ("stage_catalog", "stage_ref_codes"):
        if _attr in dg.attrs:
            del dg.attrs[_attr]
    if "stage_label" not in dg:
        dg.create_dataset("stage_label", shape=(ctx.T,), dtype=np.int64)
    dg["stage_label"][...] = ctx.stage_label
    dg.attrs["label_schema"] = json.dumps(
        {"0": "arm_motion", "1": "hand_motion"}, ensure_ascii=False
    )

    if variant["skip_sim"]:
        demo_out["datagen_info/eef_pose"][...] = variant["eef_pose_out"]
        demo_out["datagen_info/target_pose"][...] = variant["target_out"]
        demo_out["datagen_info/gripper_action"][...] = ctx.gripper_src
    else:
        _sync_datagen_and_obs_from_states(
            ctx.env,
            demo_out,
            variant["states"],
            ctx.hammer_bid,
            ctx.ridx,
            ctx.site_id,
            ctx.nq,
            ctx.nv,
            ctx.na,
            ctx.has_time,
            ctx.gripper_src,
        )

    if "action_dict" in demo_out:
        ad = {k: demo_out["action_dict"][k][()] for k in demo_out["action_dict"].keys()}
        _update_action_dict_from_eef(
            demo_out["action_dict"],
            demo_out["datagen_info/eef_pose"][:, 0],
            variant["transform_mask"],
            ad,
        )

    dx, dy, dyaw = variant["hammer_offset"]
    gen_meta = {
        "source_hdf5": ctx.source_hdf5,
        "source_demo": ctx.source_demo,
        "labels_path": ctx.labels_path,
        "autogen_index": demo_index,
        "hammer_delta_xyz": [dx, dy, 0.0],
        "hammer_dyaw_deg": dyaw,
        "hammer_T_delta": variant["T_delta"].tolist(),
        "skip_sim": variant["skip_sim"],
    }
    demo_out.attrs["autogen"] = json.dumps(gen_meta)
    demo_out.attrs["num_samples"] = int(ctx.T)
    if "model_file" not in demo_out.attrs:
        demo_out.attrs["model_file"] = ""


def _write_dataset_shell(
    src: h5py.File,
    out: h5py.File,
    source_demo: str,
    num_demos: int,
) -> h5py.Group:
    """Create data/ group in robomimic merge style (env_args + demo_0..demo_{n-1} shells)."""
    if "data" in out:
        del out["data"]
    data_out = out.create_group("data")
    for k, v in src["data"].attrs.items():
        if k != "total":
            data_out.attrs[k] = v
    src.copy(f"data/{source_demo}", data_out, "demo_0")
    for i in range(1, num_demos):
        data_out.copy("demo_0", data_out, f"demo_{i}")
    return data_out


def _debug_report(ctx: _AutogenContext, output_hdf5: str) -> None:
    """Compare EEF-hammer relative poses between source demo and generated demo."""
    _stage_name = {s["stage_id"]: s["name"] for s in STAGE_CATALOG}

    # Source EEF from HDF5 (correctly recorded); source hammer from sim replay
    # because datagen_info/object_poses/hammer_1 in source HDF5 may be all-zero
    # if the original recording did not write body poses correctly.
    src_eef = ctx.eef_src[:, 0]      # (T,4,4) – from source HDF5 datagen_info/eef_pose
    src_hm  = ctx.hammer_world_src   # (T,4,4) – re-computed from source states

    with h5py.File(output_hdf5, "r") as f:
        gd = f["data/demo_0"]
        gen_eef = gd["datagen_info/eef_pose"][:, 0].copy()           # from _sync_datagen_and_obs
        gen_hm  = gd["datagen_info/object_poses/hammer_1"][()].copy() # from _sync_datagen_and_obs

    sep = "=" * 80
    print(f"\n{sep}")
    print("DEBUG: hammer 初始位姿 (t=0)  [src from sim replay; gen from _sync_datagen]")
    print(f"  src: {np.round(src_hm[0, :3, 3] * 100, 2)} cm")
    print(f"  gen: {np.round(gen_hm[0, :3, 3] * 100, 2)} cm")
    print(f"  Δ  : {np.round((gen_hm[0, :3, 3] - src_hm[0, :3, 3]) * 100, 2)} cm")

    print(f"\nDEBUG: EEF-hammer 相对位姿  T_rel = inv(T_hammer) @ T_eef")
    hdr = f"{'t':>5}  {'stage / event':<38}  {'src pos (mm)':>22}  {'gen pos (mm)':>22}  {'Δpos':>7}  {'Δori':>7}"
    print(hdr)
    print("-" * len(hdr))

    def _rel_compare(t: int, label: str, flag: str = "") -> None:
        src_rel = _inv_pose(src_hm[t]) @ src_eef[t]
        gen_rel = _inv_pose(gen_hm[t]) @ gen_eef[t]
        dpos = float(np.linalg.norm(src_rel[:3, 3] - gen_rel[:3, 3]) * 1000)
        R_err = src_rel[:3, :3].T @ gen_rel[:3, :3]
        dori = float(np.degrees(np.linalg.norm(_Rot.from_matrix(R_err).as_rotvec())))
        ps = np.round(src_rel[:3, 3] * 1000, 1)
        pg = np.round(gen_rel[:3, 3] * 1000, 1)
        print(f"{t:>5d}  {label:<38s}  {str(ps):>22s}  {str(pg):>22s}  {dpos:>7.2f}  {dori:>7.2f}{flag}")

    for seg in ctx.segments:
        sname = _stage_name[seg.stage_id]
        _rel_compare(seg.start, f"Stage{seg.stage_id:>2d} {sname:<20s} start")
        _rel_compare(seg.end,   f"Stage{seg.stage_id:>2d} {sname:<20s} end  ")

    if ctx.grasp_start is not None:
        t = ctx.grasp_start
        print()
        _rel_compare(t, f">>> grasp_start (Stage6 start) <<<", flag="  ← KEY")
    if ctx.carry_start is not None:
        t = ctx.carry_start
        _rel_compare(t, f">>> carry_start (Stage7 start) <<<", flag="  ← KEY")
    if ctx.open_start is not None:
        t = ctx.open_start
        _rel_compare(t, f">>> open_start  (Stage8 start) <<<", flag="  ← KEY")

    print(sep)
    print("NOTE: EEF pose = grip_site world frame;  hammer pose = hammer body world frame")
    print(sep)


def autogen_hammer_cleanup(
    source_hdf5: str,
    labels_path: str,
    output_hdf5: str,
    num_demos: int = 1,
    seed: int = 0,
    random: bool = False,
    hammer_dx: float = 0.0,
    hammer_dy: float = 0.0,
    hammer_dyaw_deg: float = 0.0,
    hammer_dx_range: tuple[float, float] = (-0.08, 0.08),
    hammer_dy_range: tuple[float, float] = (-0.12, 0.12),
    hammer_yaw_range_deg: tuple[float, float] = (-25.0, 25.0),
    control_substeps: int = 8,
    skip_sim: bool = False,
    debug: bool = False,
) -> Path:
    source_hdf5 = os.path.abspath(source_hdf5)
    output_hdf5 = os.path.abspath(output_hdf5)
    if debug:
        num_demos = 1
    if num_demos < 1:
        raise ValueError("--num-demos must be >= 1")

    os.makedirs(os.path.dirname(output_hdf5) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    fixed_offset = None if random else (hammer_dx, hammer_dy, hammer_dyaw_deg)
    offsets = _sample_hammer_offsets(
        rng,
        num_demos,
        hammer_dx_range,
        hammer_dy_range,
        hammer_yaw_range_deg,
        fixed_offset,
    )

    ctx = _build_autogen_context(source_hdf5, labels_path)
    print(f"Source (shared by all {num_demos} output demo(s)):")
    print(f"  {ctx.source_hdf5}  [{ctx.source_demo}]  ({ctx.T} steps)")
    print(
        f"Hammer stage {ctx.hammer_arm_seg.stage_id}: frames "
        f"{ctx.hammer_arm_seg.start + 1}-{ctx.hammer_arm_seg.end + 1}"
    )
    print(f"  src hammer @ segment start: {ctx.T_hammer_ref[:3, 3]}")

    if os.path.exists(output_hdf5):
        os.remove(output_hdf5)

    hammer_offsets_log = []

    with h5py.File(source_hdf5, "r") as src, h5py.File(output_hdf5, "w") as out:
        data_out = _write_dataset_shell(src, out, ctx.source_demo, num_demos)

        total_samples = 0
        for i in range(num_demos):
            dx, dy, dyaw = offsets[i]
            print(
                f"\n[{i + 1}/{num_demos}] demo_{i}: "
                f"hammer offset dx={dx:.4f} dy={dy:.4f} yaw={dyaw:.2f} deg"
            )
            T_h_new = _new_hammer_pose(ctx.T_hammer_ref, dx, dy, dyaw)
            print(f"  new hammer @ segment start: {T_h_new[:3, 3]}")

            variant = _generate_variant(ctx, dx, dy, dyaw, control_substeps, skip_sim)
            _write_demo_group(data_out[f"demo_{i}"], ctx, variant, i)
            total_samples += ctx.T
            hammer_offsets_log.append(
                {"demo": f"demo_{i}", "dx": dx, "dy": dy, "dyaw_deg": dyaw}
            )

        # Same convention as merge_hammer_cleanup_demos.py / single_arm_hammer_cleanup.hdf5
        data_out.attrs["total"] = int(total_samples)
        data_out.attrs["autogen_info"] = json.dumps(
            {
                "source_hdf5": ctx.source_hdf5,
                "source_demo": ctx.source_demo,
                "labels_path": ctx.labels_path,
                "num_demos": num_demos,
                "seed": seed,
                "hammer_offsets": hammer_offsets_log,
                "skip_sim": skip_sim,
            },
            ensure_ascii=False,
        )

    ctx.env.close()

    print(f"\nWrote {num_demos} demo(s) -> {output_hdf5}")
    print(f"  layout: data/demo_0 .. data/demo_{num_demos - 1}, total={total_samples}")
    n_hand = int(np.sum(ctx.stage_label == 1))
    print(f"  stage_label per demo: 0(arm)={len(ctx.stage_label) - n_hand}, 1(hand)={n_hand}")

    if debug:
        _debug_report(ctx, output_hdf5)

    return Path(output_hdf5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-hdf5",
        default=str(_REPO_ROOT / "dexmimicgen/datasets/generated/single_arm_hammer_cleanup_demo_4.hdf5"),
    )
    parser.add_argument(
        "--labels",
        default=str(
            _REPO_ROOT
            / "dexmimicgen/autogen_dextool_demo/outputs/single_arm_hammer_cleanup_demo_4_demo_0_review_labels.json"
        ),
    )
    parser.add_argument(
        "--output-hdf5",
        "--output",
        dest="output_hdf5",
        default=str(DEFAULT_OUTPUT_HDF5),
        help=(
            "Merged robomimic HDF5 (default: "
            "dexmimicgen/datasets/generated/single_arm_hammer_cleanup_autogen.hdf5)"
        ),
    )
    parser.add_argument(
        "-n",
        "--num-demos",
        type=int,
        default=1,
        help="Number of autogenerated demos in the output HDF5 (demo_0 .. demo_{n-1})",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for hammer placement")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample hammer offsets from --hammer-*-range; otherwise use fixed --hammer-dx/dy/dyaw-deg",
    )
    parser.add_argument(
        "--hammer-dx",
        type=float,
        default=0.02,
        help="Hammer offset +x (m); used for all demos when -n 1, else sampled in range",
    )
    parser.add_argument("--hammer-dy", type=float, default=-0.02)
    parser.add_argument("--hammer-dyaw-deg", type=float, default=5.0)
    parser.add_argument(
        "--hammer-dx-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-0.05, 0.05),
        help="Uniform sample range for x offset when --random (m)",
    )
    parser.add_argument("--hammer-dy-range", type=float, nargs=2, default=(-0.05, 0.05))
    parser.add_argument(
        "--hammer-yaw-range-deg",
        type=float,
        nargs=2,
        default=(-10.0, 10.0),
    )
    parser.add_argument("--control-substeps", type=int, default=8)
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Debug: only transform poses, do not re-sim states/actions",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate exactly 1 demo then print EEF-hammer relative pose comparison",
    )
    args = parser.parse_args()

    autogen_hammer_cleanup(
        source_hdf5=args.source_hdf5,
        labels_path=args.labels,
        output_hdf5=args.output_hdf5,
        num_demos=args.num_demos,
        seed=args.seed,
        random=args.random,
        hammer_dx=args.hammer_dx,
        hammer_dy=args.hammer_dy,
        hammer_dyaw_deg=args.hammer_dyaw_deg,
        hammer_dx_range=tuple(args.hammer_dx_range),
        hammer_dy_range=tuple(args.hammer_dy_range),
        hammer_yaw_range_deg=tuple(args.hammer_yaw_range_deg),
        control_substeps=args.control_substeps,
        skip_sim=args.skip_sim,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
