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
from pathlib import Path

import h5py
import numpy as np
import robosuite

from robosuite.devices.gello_teleoperation.gello.robots.sim_robot import DrillGraspRobotServer


def _make_env(control_freq: int, render: bool, camera_names, camera_height: int, camera_width: int):
    return robosuite.make(
        env_name="DrillGrasp",
        robots="UR5eDex",
        controller_configs=DrillGraspRobotServer._build_absolute_joint_controller_config(),
        has_renderer=bool(render),
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(camera_names),
        camera_heights=int(camera_height),
        camera_widths=int(camera_width),
        camera_depths=[True] * len(camera_names),
        control_freq=int(control_freq),
        ignore_done=True,
        render_camera=None,
        initialization_noise=None,
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


def replay_and_collect_obs(env, states, actions, mode: str, render: bool):
    obs_buffer = {}
    divergence = []

    if mode == "actions":
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        for i, action in enumerate(actions):
            env.step(np.asarray(action))
            if render:
                env.render()
            obs = env._get_observations(force_update=False)
            _append_obs(obs_buffer, obs)
            if i < (len(states) - 1):
                state_playback = env.sim.get_state().flatten()
                err = float(np.linalg.norm(np.asarray(states[i + 1]) - np.asarray(state_playback)))
                divergence.append(err)
    else:
        for state in states:
            env.sim.set_state_from_flattened(np.asarray(state))
            env.sim.forward()
            if render:
                env.render()
            obs = env._get_observations(force_update=False)
            _append_obs(obs_buffer, obs)

    return _stack_obs(obs_buffer), divergence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-hdf5", type=str, required=True, help="Input demo hdf5 path")
    parser.add_argument("--output-hdf5", type=str, default="", help="Output hdf5 path; default: <input>_replay_obs.hdf5")
    parser.add_argument("--mode", type=str, default="actions", choices=["actions", "states"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--replace-obs", action="store_true", help="Replace data/demo_0/obs with replayed obs")
    parser.add_argument("--camera-names", nargs="+", default=["thirdview", "robot0_eye_in_hand"])
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--control-freq", type=int, default=20)
    args = parser.parse_args()

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
        if states.ndim != 2:
            raise RuntimeError(f"Expected 2D states, got shape={states.shape}")
        if args.mode == "actions" and actions.ndim != 2:
            raise RuntimeError(f"Expected 2D actions, got shape={actions.shape}")
        if args.mode == "actions" and actions.shape[0] > states.shape[0]:
            raise RuntimeError(f"actions longer than states: {actions.shape[0]} > {states.shape[0]}")

        env = _make_env(
            control_freq=args.control_freq,
            render=args.render,
            camera_names=args.camera_names,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )
        try:
            env.reset()
            obs_replay, divergence = replay_and_collect_obs(
                env=env,
                states=states,
                actions=actions,
                mode=args.mode,
                render=bool(args.render),
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

            replay_meta = {
                "mode": args.mode,
                "camera_names": list(args.camera_names),
                "camera_height": int(args.camera_height),
                "camera_width": int(args.camera_width),
                "control_freq": int(args.control_freq),
                "replace_obs": bool(args.replace_obs),
                "mean_state_divergence_l2": float(np.mean(divergence)) if len(divergence) > 0 else 0.0,
                "max_state_divergence_l2": float(np.max(divergence)) if len(divergence) > 0 else 0.0,
            }
            demo_grp.attrs["replay_info"] = json.dumps(replay_meta, indent=2)

    print(f"[replay] wrote: {output_path}")


if __name__ == "__main__":
    main()

