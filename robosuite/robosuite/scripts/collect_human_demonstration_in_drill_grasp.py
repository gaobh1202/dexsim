"""
Collect human demonstrations for DrillGrasp and save in a dexmimic-like HDF5 schema.

Output structure (per demo):
    data/demo_i/
        attrs:
            model_file
            num_samples
        datasets:
            states
            actions
            action_dict/right_gripper
            action_dict/right_rel_pos
            action_dict/right_rel_rot_6d
            action_dict/right_rel_rot_axis_angle
            obs/<all raw obs keys except aggregated *-state>
            datagen_info/eef_pose
            datagen_info/target_pose
            datagen_info/gripper_action
            datagen_info/object_poses/drill_001
            datagen_info/subtask_term_signals/drill_grasped
            datagen_info/subtask_term_signals/drill_lifted

This script targets schema compatibility with dexmimic datasets rather than strict
task-field identity with TwoArmDrawerCleanup.
"""

import argparse
import json
import os
import time
from copy import deepcopy
from threading import Lock

import h5py
import numpy as np
from pynput.keyboard import Key, Listener

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper


def _rot6d_from_axis_angle(axis_angle):
    quat_xyzw = T.axisangle2quat(np.asarray(axis_angle, dtype=np.float64))
    rot = T.quat2mat(quat_xyzw).astype(np.float64)
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0)


def _pose_from_pos_quat_xyzw(pos, quat_xyzw):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = T.quat2mat(np.asarray(quat_xyzw, dtype=np.float64))
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def _get_model_xml(env):
    # Prefer simulator model xml if available
    if hasattr(env, "sim") and hasattr(env.sim, "model") and hasattr(env.sim.model, "get_xml"):
        return env.sim.model.get_xml()
    if hasattr(env, "model") and hasattr(env.model, "get_xml"):
        return env.model.get_xml()
    return ""


def _create_env_args(args, controller_config):
    env_kwargs = {
        "robots": args.robots,
        "controller_configs": controller_config,
        "env_configuration": args.config if "TwoArm" in args.environment else None,
        "reward_shaping": True,
        "camera_names": args.camera,
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "camera_depths": False,
        "render_gpu_device_id": args.render_gpu_device_id,
    }
    # Keep style similar to dexmimic datasets
    meta = {
        "env_name": args.environment,
        "env_version": suite.__version__,
        "type": 1,
        "env_kwargs": env_kwargs,
    }
    return json.dumps(meta, indent=4)


def _init_prev_gripper_actions(env):
    return [
        {
            f"{robot_arm}_gripper": np.repeat([0.0], robot.gripper[robot_arm].dof).astype(np.float64)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]


def _empty_demo_buffer():
    return {
        "states": [],
        "actions": [],
        "obs": {},
        "action_dict": {
            "right_gripper": [],
            "right_rel_pos": [],
            "right_rel_rot_axis_angle": [],
            "right_rel_rot_6d": [],
        },
        "datagen": {
            "eef_pose": [],
            "target_pose": [],
            "gripper_action": [],
            "object_pose_drill_001": [],
            "signal_drill_grasped": [],
            "signal_drill_lifted": [],
        },
    }


def _buffer_has_data(buf):
    return len(buf["actions"]) > 0


def _finalize_demo_buffer(buf):
    if not _buffer_has_data(buf):
        return None
    states = np.asarray(buf["states"], dtype=np.float64)
    actions = np.asarray(buf["actions"], dtype=np.float64)
    if states.shape[0] == actions.shape[0] + 1:
        states = states[:-1]
    if states.shape[0] != actions.shape[0]:
        raise RuntimeError(f"states/actions mismatch: {states.shape} vs {actions.shape}")
    return {
        "states": states,
        "actions": actions,
        "obs": {k: np.asarray(v) for k, v in buf["obs"].items()},
        "action_dict": {k: np.asarray(v) for k, v in buf["action_dict"].items()},
        "datagen_info": {
            "eef_pose": np.asarray(buf["datagen"]["eef_pose"], dtype=np.float64)[:, None, :, :],
            "target_pose": np.asarray(buf["datagen"]["target_pose"], dtype=np.float64)[:, None, :, :],
            "gripper_action": np.asarray(buf["datagen"]["gripper_action"], dtype=np.float64),
            "object_poses": {
                "drill_001": np.asarray(buf["datagen"]["object_pose_drill_001"], dtype=np.float64),
            },
            "subtask_term_signals": {
                "drill_grasped": np.asarray(buf["datagen"]["signal_drill_grasped"], dtype=np.int64),
                "drill_lifted": np.asarray(buf["datagen"]["signal_drill_lifted"], dtype=np.int64),
            },
        },
        "num_samples": int(actions.shape[0]),
        "successful": bool(np.any(np.asarray(buf["datagen"]["signal_drill_lifted"], dtype=np.int64) > 0)),
    }


def _compute_env_action(env, device, input_ac_dict, prev_gripper_actions):
    active_robot = env.robots[device.active_robot]
    step_action_dict = deepcopy(input_ac_dict)

    for arm in active_robot.arms:
        if isinstance(active_robot.composite_controller, WholeBody):
            controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
        else:
            controller_input_type = active_robot.part_controllers[arm].input_type
        if controller_input_type == "delta":
            step_action_dict[arm] = input_ac_dict[f"{arm}_delta"]
        elif controller_input_type == "absolute":
            step_action_dict[arm] = input_ac_dict[f"{arm}_abs"]
        else:
            raise ValueError(f"Unsupported controller input type: {controller_input_type}")

    env_action = [robot.create_action_vector(prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
    env_action[device.active_robot] = active_robot.create_action_vector(step_action_dict)
    env_action = np.concatenate(env_action).astype(np.float64)

    for gripper_key in prev_gripper_actions[device.active_robot]:
        prev_gripper_actions[device.active_robot][gripper_key] = np.asarray(step_action_dict[gripper_key], dtype=np.float64)
    return env_action, step_action_dict


def _append_step_to_buffer(buf, env, obs, env_action, step_action_dict):
    buf["states"].append(np.array(env.sim.get_state().flatten(), dtype=np.float64))
    buf["actions"].append(np.asarray(env_action, dtype=np.float64))

    for k, v in obs.items():
        if not isinstance(v, np.ndarray) or k.endswith("-state"):
            continue
        if k not in buf["obs"]:
            buf["obs"][k] = []
        buf["obs"][k].append(np.array(v))

    right_action = np.asarray(step_action_dict.get("right", np.zeros(6)), dtype=np.float64).reshape(-1)
    if right_action.shape[0] < 6:
        padded = np.zeros(6, dtype=np.float64)
        padded[: right_action.shape[0]] = right_action
        right_action = padded
    right_gripper = np.asarray(step_action_dict.get("right_gripper", np.zeros(1)), dtype=np.float64).reshape(-1)
    buf["action_dict"]["right_gripper"].append(right_gripper)
    buf["action_dict"]["right_rel_pos"].append(right_action[:3])
    buf["action_dict"]["right_rel_rot_axis_angle"].append(right_action[3:6])
    buf["action_dict"]["right_rel_rot_6d"].append(_rot6d_from_axis_angle(right_action[3:6]))

    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0]))
    eef_pose = _pose_from_pos_quat_xyzw(eef_pos, eef_quat)
    buf["datagen"]["eef_pose"].append(eef_pose)
    buf["datagen"]["target_pose"].append(eef_pose.copy())
    buf["datagen"]["gripper_action"].append(right_gripper[None, :])

    obj_pos = obs.get("drill_001_pos", np.zeros(3))
    obj_quat = obs.get("drill_001_quat", np.array([0.0, 0.0, 0.0, 1.0]))
    buf["datagen"]["object_pose_drill_001"].append(_pose_from_pos_quat_xyzw(obj_pos, obj_quat))

    is_grasped = int(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.objects[0]))
    is_lifted = int(env._check_success())
    buf["datagen"]["signal_drill_grasped"].append(is_grasped)
    buf["datagen"]["signal_drill_lifted"].append(is_lifted)


class RecordingHotkeys:
    """Global hotkeys for recording workflow."""

    def __init__(self):
        self.lock = Lock()
        self.events = []
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            if key == Key.space:
                self.events.append("start")
                return
            try:
                ch = key.char
            except AttributeError:
                return
            if ch == "z":
                self.events.append("stop")
            elif ch == "x":
                self.events.append("save")
            elif ch == "c":
                self.events.append("discard")

    def pop_events(self):
        with self.lock:
            out = self.events
            self.events = []
        return out

    def close(self):
        self.listener.stop()


def _write_dataset_recursive(group, key, value):
    if isinstance(value, dict):
        sub = group.create_group(key)
        for k, v in value.items():
            _write_dataset_recursive(sub, k, v)
    else:
        group.create_dataset(key, data=value)


def write_single_demo_hdf5(hdf5_path, demo, env_args, model_xml):
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = env_args
        data_grp.attrs["total"] = int(demo["num_samples"])

        demo_grp = data_grp.create_group("demo_0")
        demo_grp.attrs["model_file"] = model_xml
        demo_grp.attrs["num_samples"] = int(demo["num_samples"])

        demo_grp.create_dataset("states", data=demo["states"])
        demo_grp.create_dataset("actions", data=demo["actions"])
        _write_dataset_recursive(demo_grp, "action_dict", demo["action_dict"])
        _write_dataset_recursive(demo_grp, "obs", demo["obs"])
        _write_dataset_recursive(demo_grp, "datagen_info", demo["datagen_info"])


def _build_output_dir_and_stem(output_arg):
    """
    Convert --output into (output_dir, file_stem).
    - /path/to/dir               -> (/path/to/dir, "demo")
    - /path/to/name.hdf5         -> (/path/to, "name")
    """
    if output_arg.endswith(".hdf5"):
        output_dir = os.path.dirname(output_arg) or "."
        file_stem = os.path.splitext(os.path.basename(output_arg))[0]
    else:
        output_dir = output_arg
        file_stem = "demo"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, file_stem


def _get_next_demo_index(output_dir, file_stem):
    prefix = f"{file_stem}_"
    suffix = ".hdf5"
    max_idx = -1
    for name in os.listdir(output_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        mid = name[len(prefix) : -len(suffix)]
        if mid.isdigit():
            max_idx = max(max_idx, int(mid))
    return max_idx + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="Path to output hdf5 file.")
    parser.add_argument("--environment", type=str, default="DrillGrasp")
    parser.add_argument("--robots", nargs="+", type=str, default=["UR5eDex"], help="Robot(s) in env.")
    parser.add_argument("--config", type=str, default="default", help="Env configuration when needed.")
    parser.add_argument("--camera", nargs="*", type=str, default=["thirdview", "robot0_eye_in_hand"])
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse", "dualsense", "mjgui"])
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--max_fr", type=int, default=20)
    parser.add_argument("--goal_update_mode", type=str, default="target", choices=["target", "achieved"])
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--reverse_xy", type=bool, default=False)
    parser.add_argument("--render_gpu_device_id", type=int, default=0)
    args = parser.parse_args()

    output_dir, output_stem = _build_output_dir_and_stem(args.output)

    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0])
    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK  # noqa: F401
    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # mjviewer only supports one render camera; keep multi-camera in camera_names for observations
    render_camera = args.camera[0] if args.renderer == "mjviewer" and len(args.camera) > 0 else args.camera

    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        render_camera=render_camera,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        camera_names=args.camera,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        camera_depths=False,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    env = VisualizationWrapper(env)

    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    else:
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)

    model_xml = _get_model_xml(env)
    env_args = _create_env_args(args, controller_config)

    # Initialize runtime once; environment and device keep running continuously
    env.reset()
    env.render()
    device.start_control()
    for robot in env.robots:
        robot.print_action_info_dict()

    print("=" * 80)
    print(f"Continuous collection mode -> {output_dir}")
    print(f"Each saved demo is one file: {output_stem}_XXXXXX.hdf5")
    print("Hotkeys:")
    print("  space : start recording current demo")
    print("  z     : stop recording")
    print("  x     : save stopped demo")
    print("  c     : discard current / stopped demo")
    print("Press Ctrl+C in terminal to quit.")
    print("=" * 80)

    recorder = RecordingHotkeys()
    prev_gripper_actions = _init_prev_gripper_actions(env)
    buf = _empty_demo_buffer()
    is_recording = False
    saved = 0
    next_demo_idx = _get_next_demo_index(output_dir, output_stem)

    try:
        while True:
            start_t = time.time()

            # Handle recorder events
            for ev in recorder.pop_events():
                if ev == "start":
                    if is_recording:
                        print("Already recording.")
                    elif _buffer_has_data(buf):
                        print("Unsaved demo exists. Press x to save or c to discard before starting a new one.")
                    else:
                        is_recording = True
                        print("Recording started.")
                elif ev == "stop":
                    if not is_recording:
                        print("Not recording.")
                    else:
                        is_recording = False
                        print(f"Recording stopped. Captured {_buffer_has_data(buf) and len(buf['actions']) or 0} steps.")
                elif ev == "save":
                    if is_recording:
                        print("Stop recording first (press z), then save (x).")
                    else:
                        demo = _finalize_demo_buffer(buf)
                        if demo is None:
                            print("No demo to save.")
                        else:
                            hdf5_path = os.path.join(output_dir, f"{output_stem}_{next_demo_idx:06d}.hdf5")
                            write_single_demo_hdf5(
                                hdf5_path=hdf5_path,
                                demo=demo,
                                env_args=env_args,
                                model_xml=model_xml,
                            )
                            idx = next_demo_idx
                            next_demo_idx += 1
                            saved += 1
                            print(
                                f"Saved {os.path.basename(hdf5_path)}: num_samples={demo['num_samples']}, "
                                f"successful={demo['successful']} (saved_total={saved})"
                            )
                            buf = _empty_demo_buffer()
                elif ev == "discard":
                    if is_recording:
                        is_recording = False
                    if _buffer_has_data(buf):
                        steps = len(buf["actions"])
                        buf = _empty_demo_buffer()
                        print(f"Discarded current demo buffer ({steps} steps).")
                    else:
                        print("No demo to discard.")

            input_ac_dict = device.input2action(goal_update_mode=args.goal_update_mode)
            # Keyboard device uses Ctrl+q to reset (returns None). Keep runtime alive.
            if input_ac_dict is None:
                env.reset()
                env.render()
                device.start_control()
                prev_gripper_actions = _init_prev_gripper_actions(env)
                if is_recording:
                    is_recording = False
                    buf = _empty_demo_buffer()
                    print("Environment reset while recording; current demo discarded.")
                else:
                    print("Environment reset.")
                continue

            env_action, step_action_dict = _compute_env_action(
                env=env,
                device=device,
                input_ac_dict=input_ac_dict,
                prev_gripper_actions=prev_gripper_actions,
            )
            obs, _, _, _ = env.step(env_action)
            env.render()

            if is_recording:
                _append_step_to_buffer(
                    buf=buf,
                    env=env,
                    obs=obs,
                    env_action=env_action,
                    step_action_dict=step_action_dict,
                )

            if args.max_fr is not None:
                elapsed = time.time() - start_t
                remain = 1.0 / args.max_fr - elapsed
                if remain > 0:
                    time.sleep(remain)
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")
    finally:
        recorder.close()
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()

