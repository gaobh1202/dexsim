"""
Quick viewer for HammerCleanup environment.
Renders the scene so you can inspect hammer size relative to the drawer.

Usage:
    python view_hammer_cleanup.py                  # offscreen, saves hammer_cleanup_view.png
    python view_hammer_cleanup.py --interactive    # opens a live MuJoCo viewer window
    python view_hammer_cleanup.py --scale 0.15     # try a different hammer scale
"""

import argparse
import os
import sys

import numpy as np

# For headless runs, prefer CPU offscreen rendering unless user explicitly
# requests interactive viewing or has already selected a backend.
if "--interactive" not in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import robosuite
import dexmimicgen  # noqa: F401  — triggers env/robot registration
from dexmimicgen.environments.single_arm_hammer_cleanup import HammerCleanup


def _load_controller_config(robot: str) -> dict:
    from robosuite.controllers.composite.composite_controller_factory import load_composite_controller_config

    return load_composite_controller_config(robot=robot)


def make_env(scale: float, interactive: bool) -> HammerCleanup:
    controller_config = _load_controller_config("UR5eInspireDexRH")

    env = HammerCleanup(
        robots=["UR5eInspireDexRH"],
        controller_configs=controller_config,
        hammer_scale=scale,
        has_renderer=interactive,
        has_offscreen_renderer=not interactive,
        render_camera="agentview",
        use_camera_obs=False,
        use_object_obs=True,
        horizon=500,
        hard_reset=True,
    )

    return env


def print_scene_info(env: HammerCleanup) -> None:
    sim = env.sim

    hammer_id = env.obj_body_id["hammer"]
    drawer_id  = env.obj_body_id["drawer"]

    hammer_pos  = sim.data.body_xpos[hammer_id]
    drawer_pos  = sim.data.body_xpos[drawer_id]
    drawer_qpos = sim.data.qpos[env.drawer_qpos_addr]

    # Estimate hammer world-space bounding box from its geom positions
    geom_ids = [
        i for i in range(sim.model.ngeom)
        if sim.model.geom_bodyid[i] == hammer_id
    ]
    if geom_ids:
        positions = np.array([sim.data.geom_xpos[i] for i in geom_ids])
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        size = hi - lo
    else:
        lo = hi = size = np.zeros(3)

    print("\n========== Scene Info ==========")
    print(f"  Hammer body pos  : {np.round(hammer_pos, 4)}")
    print(f"  Hammer bbox lo   : {np.round(lo, 4)}")
    print(f"  Hammer bbox hi   : {np.round(hi, 4)}")
    print(f"  Hammer size (m)  : {np.round(size, 4)}")
    print(f"  Drawer body pos  : {np.round(drawer_pos, 4)}")
    print(f"  Drawer qpos      : {drawer_qpos:.4f}  (0=closed, ~-0.135=open)")
    print(f"  Table offset     : {env.table_offset}")
    print("================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale", type=float, default=0.28,
        help="Hammer mesh scale (default 0.28)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Open a live MuJoCo viewer window instead of saving an image"
    )
    args = parser.parse_args()

    print(f"Building HammerCleanup  robot=UR5eInspireDexRH  hammer_scale={args.scale}")
    env = make_env(scale=args.scale, interactive=args.interactive)
    env.reset()
    print_scene_info(env)

    if args.interactive:
        print("Viewer open — robot holds still (zero action). Close window to exit.")
        while True:
            action = np.zeros(env.action_dim)
            _, _, done, _ = env.step(action)
            env.render()
            if done:
                env.reset()
    else:
        # Offscreen: step a few frames then save one PNG
        for _ in range(5):
            env.step(np.zeros(env.action_dim))

        frame = env.sim.render(width=640, height=480, camera_name="agentview")
        frame = frame[::-1]  # MuJoCo renders upside-down

        try:
            import imageio
            imageio.imwrite("hammer_cleanup_view.png", frame)
            print("Saved  hammer_cleanup_view.png  — open it to inspect hammer size.")
        except ImportError:
            print("imageio not available; run with --interactive to see the scene live.")

    env.close()


if __name__ == "__main__":
    main()
