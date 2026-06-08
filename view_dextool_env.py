"""
查看自定义灵巧工具操作（dexterous tool manipulation）仿真环境的初始化结果。

HammerCleanup; 
DrillShelfNail (drill + red/yellow nails); 
DrillGrasp;
HammerShelfNail (hammer + red/yellow nails);

本脚本用于快速检查 DexSim / dexmimicgen 中单臂灵巧手场景：物体摆放、
桌子高度、相机视角等。通过 --env 指定环境名称，无需为每个任务单独写查看脚本。

用法（请先 conda activate dexsim）:
    python view_dextool_env.py --env HammerCleanup
    python view_dextool_env.py --env DrillNail
    python view_dextool_env.py --env DrillNail --interactive   # mjviewer + frontview
    python view_dextool_env.py --env DrillGrasp --interactive
    python view_dextool_env.py --env HammerCleanup --scale 0.15 --interactive
    python view_dextool_env.py --list

离屏模式会保存 PNG：{env_name}_view.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

# MuJoCo GL 后端：离屏用 OSMesa；交互窗口用 GLFW（避免 mujoco 渲染器强开 EGL 离屏）。
if "MUJOCO_GL" not in os.environ:
    if "--interactive" in sys.argv:
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = "osmesa"

from robosuite.environments.base import REGISTERED_ENVS, make

import dexmimicgen  # noqa: F401 — 加载 dexmimicgen 包
from dexmimicgen.environments.single_arm_drill_nail import DrillNail  # noqa: F401
from dexmimicgen.environments.single_arm_hammer_cleanup import HammerCleanup  # noqa: F401
from dexmimicgen.environments.single_arm_shelf_nail import ShelfNailScene  # noqa: F401
from dexmimicgen.environments.single_arm_drill_shelf_nail import DrillShelfNail  # noqa: F401
from dexmimicgen.environments.single_arm_hammer_shelf_nail import HammerShelfNail  # noqa: F401
from robosuite.environments.manipulation.drill_grasp import DrillGrasp  # noqa: F401

# 当前支持的 dexterous tool 单臂场景（可按需扩展）。
DEX_TOOL_ENV_NAMES = (
    "HammerCleanup",
    "DrillNail",
    "DrillGrasp",
    "ShelfNailScene",
    "DrillShelfNail",
    "HammerShelfNail",
)

DEFAULT_ROBOT = "UR5eInspireDexRH"


def _load_controller_config(robot: str) -> dict:
    from robosuite.controllers.composite.composite_controller_factory import (
        load_composite_controller_config,
    )

    return load_composite_controller_config(robot=robot)


def _env_specific_kwargs(env_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """各环境特有的构造参数。"""
    if env_name == "HammerCleanup":
        return {"hammer_scale": args.scale}
    if env_name in ("DrillNail", "ShelfNailScene", "DrillShelfNail", "HammerShelfNail"):
        kwargs = {"nail_scale": args.nail_scale}
        if args.nail_mjcf_path:
            kwargs["nail_mjcf_path"] = args.nail_mjcf_path
        return kwargs
    return {}


def make_env(env_name: str, args: argparse.Namespace):
    if env_name not in DEX_TOOL_ENV_NAMES:
        raise ValueError(
            f"Unknown env {env_name!r}. Choose from: {list(DEX_TOOL_ENV_NAMES)}"
        )
    if env_name not in REGISTERED_ENVS:
        raise ValueError(
            f"{env_name} is not registered. Registered: "
            f"{sorted(REGISTERED_ENVS.keys())}"
        )

    robot = args.robot or DEFAULT_ROBOT
    controller_config = _load_controller_config(robot)

    # renderer="mujoco" + has_renderer=True 会在 base.MujocoEnv 里强制 has_offscreen_renderer=True，
    # 从而走 EGL 离屏并在此机器上报错；交互查看请用原生 mjviewer。
    env_kwargs = {
        "robots": [robot],
        "controller_configs": controller_config,
        "has_renderer": args.interactive,
        "has_offscreen_renderer": not args.interactive,
        "renderer": "mjviewer" if args.interactive else "mujoco",
        "render_camera": "frontview" if args.interactive else "agentview",
        "use_camera_obs": False,
        "use_object_obs": True,
        "horizon": 500,
        "ignore_done": True,
        "hard_reset": True,
        **_env_specific_kwargs(env_name, args),
    }

    return make(env_name, **env_kwargs)


def _body_bbox(sim, body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom_ids = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == body_id]
    if not geom_ids:
        z = np.zeros(3)
        return z, z, z
    positions = np.array([sim.data.geom_xpos[i] for i in geom_ids])
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    return lo, hi, hi - lo


def print_scene_info(env_name: str, env) -> None:
    sim = env.sim
    print("\n========== Scene Info ==========")
    print(f"  Environment      : {env_name}")
    print(f"  Robot(s)         : {[r.name for r in env.robots]}")
    if hasattr(env, "table_offset"):
        print(f"  Table offset     : {env.table_offset}")

    if hasattr(env, "obj_body_id") and env.obj_body_id:
        for label, body_id in env.obj_body_id.items():
            pos = sim.data.body_xpos[body_id]
            lo, hi, size = _body_bbox(sim, body_id)
            print(f"  [{label}] body pos : {np.round(pos, 4)}")
            print(f"  [{label}] bbox size: {np.round(size, 4)}  (lo={np.round(lo, 4)}, hi={np.round(hi, 4)})")

    if env_name == "HammerCleanup" and hasattr(env, "drawer_qpos_addr"):
        print(f"  Drawer qpos      : {sim.data.qpos[env.drawer_qpos_addr]:.4f}  (0=closed)")

    if env_name == "DrillNail" and hasattr(env, "nails"):
        for nail in env.nails:
            print(f"  Nail object      : {nail.name} ({type(nail).__name__})")

    print("================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View initialization of custom dexterous-tool manipulation environments.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HammerCleanup",
        choices=DEX_TOOL_ENV_NAMES,
        help=f"Environment name (default: HammerCleanup). Options: {', '.join(DEX_TOOL_ENV_NAMES)}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported environment names and exit",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help=f"Robot model (default: {DEFAULT_ROBOT})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.28,
        help="HammerCleanup only: hammer mesh scale (default 0.28)",
    )
    parser.add_argument(
        "--nail-mjcf-path",
        type=str,
        default=None,
        help="DrillNail only: override nail MJCF path",
    )
    parser.add_argument(
        "--nail-scale",
        type=float,
        default=0.15,
        help="DrillNail only: nail mesh scale (default 0.15 ≈ 3× 原 5cm 钉)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a live MuJoCo viewer (robot holds zero action)",
    )
    args = parser.parse_args()

    if args.list:
        print("Dexterous tool manipulation environments:")
        for name in DEX_TOOL_ENV_NAMES:
            print(f"  - {name}")
        return

    env_name = args.env
    print(f"Building {env_name}  robot={args.robot or DEFAULT_ROBOT}")
    env = make_env(env_name, args)
    env.reset()
    print_scene_info(env_name, env)

    if args.interactive:
        if env.viewer is None:
            env.initialize_renderer()
        print("MuJoCo viewer (mjviewer, frontview) — zero action. Close window to exit.")
        while True:
            _, _, done, _ = env.step(np.zeros(env.action_dim))
            env.render()
            if done:
                env.reset()
                if env.viewer is None:
                    env.initialize_renderer()
    else:
        for _ in range(5):
            env.step(np.zeros(env.action_dim))

        frame = env.sim.render(width=640, height=480, camera_name="agentview")
        frame = frame[::-1]

        out_path = f"{env_name}_view.png"
        try:
            import imageio

            imageio.imwrite(out_path, frame)
            print(f"Saved  {out_path}")
        except ImportError:
            print("imageio not installed; use --interactive or pip install imageio.")

    env.close()


if __name__ == "__main__":
    main()
