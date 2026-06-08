#!/usr/bin/env python3
"""
测试：UR5eInspireDexRH 弧形末端轨迹 + 结束位姿相对变换。

流程
----
1. 给定起始/结束 grip_site 位姿 T_start, T_end（世界系 SE(3)）。
2. 用 100 步生成弧形参考轨迹（位置二次 Bezier + 姿态 SLERP）。
3. 结束位姿：立方体正上方；grip_site **z → 立方体 -z**，**x → 立方体 -y**（随立方体旋转）。
4. Phase 2 立方体变换后，用渐进 object transform 重定向轨迹（起点不变、终点对齐新物体）：
   ``Delta_t = interp_SE3(I, T_cube_new @ inv(T_cube), smoothstep(u))``，
   ``T_new[t] = Delta_t @ T_src[t]``。
5. OSC 绝对位姿控制跟踪两条轨迹。

用法（conda activate dexsim）::

    cd /home/benhua/DexSim
    PYTHONPATH=robosuite:dexmimicgen:dexmimicgen/demo_generation MUJOCO_GL=osmesa \\
        python dexmimicgen/scripts/test_eef_arc_trajectory_delta.py

    PYTHONPATH=robosuite:dexmimicgen:dexmimicgen/demo_generation MUJOCO_GL=glfw \\
        python dexmimicgen/scripts/test_eef_arc_trajectory_delta.py --interactive

交互模式默认显示 grip_site 与立方体坐标轴（RGB=XYZ）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

_REPO = Path(__file__).resolve().parents[2]
for _p in (
    _REPO / "robosuite",
    _REPO / "dexmimicgen",
    _REPO / "dexmimicgen" / "demo_generation",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import robosuite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from generate_demo_from_source import (  # noqa: E402
    build_env_action,
    mat_to_quat_wxyz,
    rotvec_from_matrix,
)

try:
    import mujoco

    MUJOCO_VIEWER_AVAILABLE = True
except ImportError:
    MUJOCO_VIEWER_AVAILABLE = False

ROBOT = "UR5eInspireDexRH"
DEFAULT_STEPS = 100
DEFAULT_CUBE_XY = (-0.05, -0.10)
DEFAULT_EEF_CLEARANCE = 0.22
JOINT_ORIGINAL = np.array(
    [np.pi, -np.pi / 2, -3 * np.pi / 4, np.pi / 4, np.pi / 2, np.pi],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def eye4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = eye4()
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def inv_pose(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    out = eye4()
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def apply_world_xy_yaw(T: np.ndarray, dx: float, dy: float, dyaw_deg: float) -> np.ndarray:
    """世界系 xy 平移 + 绕世界 Z 轴旋转。"""
    th = np.deg2rad(dyaw_deg)
    c, s = np.cos(th), np.sin(th)
    R_w = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T_out = eye4()
    T_out[:3, :3] = R_w @ T[:3, :3]
    T_out[:3, 3] = T[:3, 3] + np.array([dx, dy, 0.0], dtype=np.float64)
    return T_out


def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def pose_from_body(env, body_id: int) -> np.ndarray:
    T = eye4()
    T[:3, :3] = env.sim.data.body_xmat[body_id].reshape(3, 3).copy()
    T[:3, 3] = env.sim.data.body_xpos[body_id].copy()
    return T


def pose_from_site(env, site_id: int) -> np.ndarray:
    T = eye4()
    T[:3, :3] = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    T[:3, 3] = env.sim.data.site_xpos[site_id].copy()
    return T


# ---------------------------------------------------------------------------
# Cube + end pose
# ---------------------------------------------------------------------------

def get_cube_joint_addrs(env) -> tuple[int, int]:
    model = env.sim.model
    jid = model.joint_name2id("cube_joint0")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def cube_half_height(env) -> float:
    model = env.sim.model
    body_id = env.cube_body_id
    half_z = [
        model.geom_size[i][2]
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == body_id
    ]
    return float(max(half_z)) if half_z else 0.021


def set_cube_pose(env, T_cube: np.ndarray) -> None:
    qadr, dadr = get_cube_joint_addrs(env)
    env.sim.data.qpos[qadr : qadr + 3] = T_cube[:3, 3]
    env.sim.data.qpos[qadr + 3 : qadr + 7] = mat_to_quat_wxyz(T_cube[:3, :3])
    if dadr + 6 <= env.sim.data.qvel.shape[0]:
        env.sim.data.qvel[dadr : dadr + 6] = 0.0
    env.sim.forward()


def place_cube_on_table(env, xy: tuple[float, float], yaw_deg: float = 0.0) -> np.ndarray:
    table_z = float(env.table_offset[2])
    half_h = cube_half_height(env)
    z = table_z + 0.01 + half_h
    R = Rot.from_euler("z", np.deg2rad(yaw_deg)).as_matrix()
    T_cube = make_pose(np.array([xy[0], xy[1], z], dtype=np.float64), R)
    set_cube_pose(env, T_cube)
    return T_cube


def grip_rotation_wrt_cube(yaw_deg: float = 0.0) -> np.ndarray:
    """
    grip 在立方体坐标系下的朝向：
      grip +x ∥ cube -y，grip +z ∥ cube -z（右手系 y = z × x）。
    yaw_deg：可选，绕立方体 +z 额外旋转。
    """
    x = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    y = np.cross(z, x)
    R_rel = np.column_stack([x, y, z])
    if abs(yaw_deg) > 1e-12:
        R_rel = R_rel @ Rot.from_euler("z", np.deg2rad(yaw_deg)).as_matrix()
    return R_rel


def grip_rotation_align_to_cube(T_cube: np.ndarray, yaw_deg: float = 0.0) -> np.ndarray:
    """世界系 grip 旋转：R_grip = R_cube @ R_rel。"""
    return T_cube[:3, :3] @ grip_rotation_wrt_cube(yaw_deg)


def grip_pose_above_cube(
    T_cube: np.ndarray,
    half_h: float,
    clearance: float,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    p = T_cube[:3, 3].copy()
    p[2] += half_h + clearance
    return make_pose(p, grip_rotation_align_to_cube(T_cube, yaw_deg))


def start_end_poses_from_cube(
    env,
    T_cube: np.ndarray,
    *,
    eef_clearance: float,
    end_yaw_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    grip_site_id = int(robot.eef_site_id[robot.arms[0]])
    T_start = pose_from_site(env, grip_site_id)
    half_h = cube_half_height(env)
    T_end = grip_pose_above_cube(T_cube, half_h, eef_clearance, end_yaw_deg)
    return T_start, T_end


def generate_arc_trajectory(
    T_start: np.ndarray,
    T_end: np.ndarray,
    n_steps: int,
    arc_height: float = 0.05,
) -> np.ndarray:
    """n_steps 个位姿：位置二次 Bezier 弧形 + 姿态 SLERP。"""
    p0, p1 = T_start[:3, 3], T_end[:3, 3]
    p_ctrl = 0.5 * (p0 + p1) + np.array([0.0, 0.0, arc_height])
    slerp = Slerp(
        [0.0, 1.0],
        Rot.from_matrix(np.stack([T_start[:3, :3], T_end[:3, :3]])),
    )
    traj = np.zeros((n_steps, 4, 4), dtype=np.float64)
    for i in range(n_steps):
        u = smoothstep(i / max(n_steps - 1, 1))
        p = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p_ctrl + u**2 * p1
        traj[i] = make_pose(p, slerp([u]).as_matrix()[0])
    traj[0] = T_start.copy()
    traj[-1] = T_end.copy()
    return traj


def interp_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """SE(3) 插值：平移线性 + 旋转 Slerp。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    T = eye4()
    T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    rots = Rot.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
    T[:3, :3] = Slerp([0.0, 1.0], rots)([alpha]).as_matrix()[0]
    return T


def deform_keep_start_follow_object_goal(
    T_src_seq: np.ndarray,
    T_o_src: np.ndarray,
    T_o_new: np.ndarray,
) -> np.ndarray:
    """
    起点不变、终点随物体相对目标变化，保留 source 轨迹中间趋势。

    Delta_obj = T_o_new @ inv(T_o_src)
    Delta_t   = interp_SE3(I, Delta_obj, smoothstep(u))
    T_new[t]  = Delta_t @ T_src[t]
    """
    n = len(T_src_seq)
    I = eye4()
    delta_obj = T_o_new @ inv_pose(T_o_src)
    out = np.zeros((n, 4, 4), dtype=np.float64)
    for i, T_src_i in enumerate(T_src_seq):
        u = i / max(n - 1, 1)
        alpha = smoothstep(u)
        delta_t = interp_se3(I, delta_obj, alpha)
        out[i] = delta_t @ T_src_i
    return out


def object_relative_grip_end(
    T_cube_src: np.ndarray,
    T_cube_new: np.ndarray,
    T_end_src: np.ndarray,
) -> np.ndarray:
    """T_end_new = T_cube_new @ inv(T_cube_src) @ T_end_src。"""
    return T_cube_new @ inv_pose(T_cube_src) @ T_end_src


# ---------------------------------------------------------------------------
# Frame visualization (grip_site + cube)
# ---------------------------------------------------------------------------

def _draw_axis(scene, pos, rot_mat, axis_dir, color, length, radius):
    world_dir = rot_mat @ axis_dir
    world_dir /= np.linalg.norm(world_dir) + 1e-12
    end_pos = pos + world_dir * length
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.size[0] = radius
    geom.size[1] = radius
    geom.size[2] = length * 0.5
    geom.pos[:] = (pos + end_pos) / 2
    z_axis = world_dir
    if abs(z_axis[2]) < 0.9:
        x_axis = np.array([0.0, 0.0, 1.0])
    else:
        x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    geom.mat[:] = np.column_stack([x_axis, y_axis, z_axis])
    geom.rgba[:] = color
    scene.ngeom += 1


def draw_coordinate_frame(scene, pos, rot_mat, axis_length=0.08, axis_radius=0.004):
    """RGB = XYZ。"""
    if not MUJOCO_VIEWER_AVAILABLE or scene is None:
        return
    for axis_dir, color in (
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 1.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
    ):
        _draw_axis(scene, pos, rot_mat, axis_dir, color, axis_length, axis_radius)


def _viewer_is_active(env) -> bool:
    viewer_handle = getattr(getattr(env, "viewer", None), "viewer", None)
    if viewer_handle is None:
        return False
    if hasattr(viewer_handle, "is_running"):
        return bool(viewer_handle.is_running())
    return True


def _get_viewer_scene(env):
    if not _viewer_is_active(env):
        return None
    viewer_handle = env.viewer.viewer
    if not hasattr(viewer_handle, "user_scn") or viewer_handle.user_scn is None:
        viewer_handle.user_scn = mujoco.MjvScene(env.sim.model._model, maxgeom=10000)
    scene = viewer_handle.user_scn
    scene.ngeom = 0
    return scene


def visualize_grip_and_cube_frames(env) -> None:
    """绘制 grip_site（较长）与立方体（较短）坐标轴。"""
    scene = _get_viewer_scene(env)
    if scene is None:
        return
    robot = env.robots[0]
    grip_site_id = int(robot.eef_site_id[robot.arms[0]])
    T_grip = pose_from_site(env, grip_site_id)
    T_cube = pose_from_body(env, env.cube_body_id)
    draw_coordinate_frame(scene, T_grip[:3, 3], T_grip[:3, :3], axis_length=0.10, axis_radius=0.004)
    draw_coordinate_frame(scene, T_cube[:3, 3], T_cube[:3, :3], axis_length=0.06, axis_radius=0.003)


def _release_viewer_user_scn(env) -> None:
    try:
        viewer_handle = getattr(getattr(env, "viewer", None), "viewer", None)
        if viewer_handle is not None and hasattr(viewer_handle, "user_scn"):
            if viewer_handle.user_scn is not None:
                viewer_handle.user_scn.ngeom = 0
            viewer_handle.user_scn = None
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _force_osc_absolute_world(controller_configs: dict) -> None:
    def _patch(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "OSC_POSE":
                obj["input_type"] = "absolute"
                obj["input_ref_frame"] = "world"
            for v in obj.values():
                _patch(v)
        elif isinstance(obj, list):
            for v in obj:
                _patch(v)

    _patch(controller_configs)


def make_test_env(interactive: bool) -> robosuite.environments.base.MujocoEnv:
    controller_configs = load_composite_controller_config(robot=ROBOT)
    _force_osc_absolute_world(controller_configs)
    env = robosuite.make(
        env_name="Lift",
        robots=[ROBOT],
        controller_configs=controller_configs,
        has_renderer=interactive,
        has_offscreen_renderer=not interactive,
        renderer="mjviewer" if interactive else "mujoco",
        render_camera="frontview",
        use_camera_obs=False,
        use_object_obs=False,
        ignore_done=True,
        control_freq=20,
        horizon=5000,
    )
    for robot in env.robots:
        for arm_name in robot.arms:
            ctrl = robot.composite_controller.part_controllers.get(arm_name)
            if ctrl is not None and getattr(ctrl, "input_type", None) is not None:
                ctrl.input_type = "absolute"
                ctrl.input_ref_frame = "world"
    return env


def set_robot_initial_joints(env) -> None:
    env.robots[0].set_robot_joint_positions(JOINT_ORIGINAL)
    robot = env.robots[0]
    if hasattr(robot, "composite_controller"):
        for part in robot.composite_controller.part_controllers.values():
            if hasattr(part, "update"):
                part.update(force=True)
    env.sim.forward()


def prepare_phase(env, T_cube: np.ndarray) -> None:
    """重置关节与立方体（不 env.reset，避免 mjviewer 崩溃）。"""
    set_robot_initial_joints(env)
    set_cube_pose(env, T_cube)


def execute_pose_trajectory(
    env,
    traj: np.ndarray,
    T_cube: np.ndarray,
    *,
    control_substeps: int = 20,
    settle_steps: int = 50,
    label: str = "",
    render: bool = False,
    show_frames: bool = False,
) -> tuple[np.ndarray, float]:
    robot = env.robots[0]
    arm_name = robot.arms[0]
    grip_site_id = int(robot.eef_site_id[arm_name])

    set_cube_pose(env, T_cube)

    def _maybe_draw():
        if render and _viewer_is_active(env):
            env.render()
            if show_frames:
                visualize_grip_and_cube_frames(env)

    for _ in range(settle_steps):
        set_cube_pose(env, T_cube)
        rotvec = rotvec_from_matrix(traj[0][:3, :3])
        action = build_env_action(
            env, {(0, arm_name): np.concatenate([traj[0][:3, 3], rotvec])}
        )
        env.step(action)
        _maybe_draw()

    for t in range(len(traj)):
        rotvec = rotvec_from_matrix(traj[t][:3, :3])
        arm_action = np.concatenate([traj[t][:3, 3], rotvec])
        env_action = build_env_action(env, {(0, arm_name): arm_action})
        for _ in range(control_substeps):
            set_cube_pose(env, T_cube)
            env.step(env_action)
        if t % max(len(traj) // 25, 1) == 0 or t == len(traj) - 1:
            _maybe_draw()

    T_final = pose_from_site(env, grip_site_id)
    T_target = traj[-1]
    pos_err = np.linalg.norm(T_final[:3, 3] - T_target[:3, 3])
    R_err = T_target[:3, :3].T @ T_final[:3, :3]
    ori_err = np.linalg.norm(Rot.from_matrix(R_err).as_rotvec())

    if label:
        print(f"\n--- {label} ---")
        print(f"  立方体位置     : {np.round(T_cube[:3, 3], 4)}")
        print(f"  目标 grip 位置 : {np.round(T_target[:3, 3], 4)}")
        print(f"  实际 grip 位置 : {np.round(T_final[:3, 3], 4)}")
        print(f"  位置误差 (mm)  : {pos_err * 1000:.2f}")
        print(f"  姿态误差 (rad) : {ori_err:.4f}")
        R_cube = env.sim.data.body_xmat[env.cube_body_id].reshape(3, 3)
        R_grip = env.sim.data.site_xmat[grip_site_id].reshape(3, 3)
        print(f"  grip x · (-cube y): {np.dot(R_grip[:, 0], -R_cube[:, 1]):.4f}")
        print(f"  grip z · (-cube z): {np.dot(R_grip[:, 2], -R_cube[:, 2]):.4f}")

    return T_final, pos_err


def main() -> None:
    parser = argparse.ArgumentParser(description="UR5e 弧形轨迹 + 结束位姿相对变换测试")
    parser.add_argument("--interactive", action="store_true", help="打开 MuJoCo 交互窗口")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--arc-height", type=float, default=0.05)
    parser.add_argument("--dx", type=float, default=0.08)
    parser.add_argument("--dy", type=float, default=-0.06)
    parser.add_argument("--dyaw-deg", type=float, default=25.0)
    parser.add_argument("--cube-x", type=float, default=DEFAULT_CUBE_XY[0])
    parser.add_argument("--cube-y", type=float, default=DEFAULT_CUBE_XY[1])
    parser.add_argument("--eef-clearance", type=float, default=DEFAULT_EEF_CLEARANCE)
    parser.add_argument(
        "--end-yaw-deg",
        type=float,
        default=0.0,
        help="绕立方体 +z 的额外 yaw (deg)，默认 0 保持 x→-y、z→-z",
    )
    parser.add_argument("--control-substeps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument(
        "--show-frames",
        action="store_true",
        default=None,
        help="显示 grip_site 与立方体坐标轴（交互模式默认开启）",
    )
    parser.add_argument(
        "--no-show-frames",
        action="store_true",
        help="关闭坐标轴显示",
    )
    args = parser.parse_args()

    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "glfw" if args.interactive else "osmesa"

    if args.no_show_frames:
        show_frames = False
    elif args.show_frames is not None:
        show_frames = args.show_frames
    else:
        show_frames = args.interactive

    env = make_test_env(args.interactive)
    env.reset()
    prepare_phase(env, place_cube_on_table(env, (args.cube_x, args.cube_y)))

    T_cube = pose_from_body(env, env.cube_body_id)
    T_start, T_end = start_end_poses_from_cube(
        env, T_cube, eef_clearance=args.eef_clearance, end_yaw_deg=args.end_yaw_deg
    )
    traj_src = generate_arc_trajectory(T_start, T_end, args.steps, arc_height=args.arc_height)

    T_cube_new = apply_world_xy_yaw(T_cube, args.dx, args.dy, args.dyaw_deg)
    T_end_new = object_relative_grip_end(T_cube, T_cube_new, T_end)
    traj_new = deform_keep_start_follow_object_goal(traj_src, T_cube, T_cube_new)

    delta_obj = T_cube_new @ inv_pose(T_cube)
    R_cube0 = T_cube[:3, :3]
    R_grip_end = T_end[:3, :3]
    R_grip_end_new = T_end_new[:3, :3]
    start_pos_diff = np.linalg.norm(traj_new[0][:3, 3] - traj_src[0][:3, 3])
    print("========== 立方体参考 & 轨迹变形 ==========")
    print(f"  T_cube         : {np.round(T_cube[:3, 3], 4)}")
    print(f"  T_cube_new     : {np.round(T_cube_new[:3, 3], 4)}")
    print(f"  T_grip_end     : {np.round(T_end[:3, 3], 4)}")
    print(f"  T_grip_end_new : {np.round(T_end_new[:3, 3], 4)}")
    print(f"  终点 grip x · (-cube y): {np.dot(R_grip_end[:, 0], -R_cube0[:, 1]):.4f}")
    print(f"  终点 grip z · (-cube z): {np.dot(R_grip_end[:, 2], -R_cube0[:, 2]):.4f}")
    print(f"  新终点 grip x · (-cube y): {np.dot(R_grip_end_new[:, 0], -T_cube_new[:3, :3][:, 1]):.4f}")
    print(f"  新终点 grip z · (-cube z): {np.dot(R_grip_end_new[:, 2], -T_cube_new[:3, :3][:, 2]):.4f}")
    print(f"  Delta_obj trans: {np.round(delta_obj[:3, 3], 4)}")
    print(f"  Phase2 起点偏移 (mm): {start_pos_diff * 1000:.4f}  (应 ≈ 0)")
    print(f"  轨迹变形       : T_new[t] = interp_SE3(I, Delta_obj, α) @ T_src[t]")
    if show_frames:
        print("  坐标轴         : 较长=grip_site，较短=立方体（RGB=XYZ）")

    render = args.interactive
    prepare_phase(env, T_cube)
    _, err_ref = execute_pose_trajectory(
        env, traj_src, T_cube,
        control_substeps=args.control_substeps,
        settle_steps=args.settle_steps,
        label="Phase 1 — 参考弧形轨迹",
        render=render,
        show_frames=show_frames,
    )

    if args.interactive:
        input("参考轨迹完成，按 Enter 继续 Phase 2…")

    prepare_phase(env, T_cube_new)
    _, err_new = execute_pose_trajectory(
        env, traj_new, T_cube_new,
        control_substeps=args.control_substeps,
        settle_steps=args.settle_steps,
        label="Phase 2 — 变换后轨迹",
        render=render,
        show_frames=show_frames,
    )

    print("\n========== 总结 ==========")
    print(f"  Phase 1 终点误差: {err_ref * 1000:.2f} mm")
    print(f"  Phase 2 终点误差: {err_new * 1000:.2f} mm")

    if args.interactive:
        input("\n按 Enter 关闭 viewer…")
    _release_viewer_user_scn(env)
    env.close()


if __name__ == "__main__":
    main()
