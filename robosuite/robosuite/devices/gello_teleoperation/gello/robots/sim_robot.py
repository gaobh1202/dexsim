import pickle
import threading
import time
from typing import Any, Dict, Optional, Tuple, Sequence

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf

from gello.robots.robot import Robot

assert mujoco.viewer is mujoco.viewer

import robosuite

INSPIRE_GROUPS_ACTUATOR: Tuple[Tuple[int, ...], ...] = (
    (0, 1),      # pinky_distal, pinky_proximal
    (2, 3),      # ring_distal, ring_proximal
    (4, 5),      # middle_distal, middle_proximal
    (6, 7),      # index_distal, index_proximal
    (8, 9, 10),  # thumb_distal, thumb_middle, thumb_proximal_2
    (11,),       # thumb_proximal_1
)

INIT_ARM_QPOS = np.array(
    [
        np.pi,
        -1.570838,
        -1.570813,
        0.000021,
        1.570826,
        1.570804,
    ],
    dtype=float,
)

def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
) -> None:
    """Attaches a hand to an arm.

    The arm must have a site named "attachment_site".

    Taken from https://github.com/deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm

    Args:
      arm_mjcf: The mjcf.RootElement of the arm.
      hand_mjcf: The mjcf.RootElement of the hand.

    Raises:
      ValueError: If the arm does not have a site named "attachment_site".
    """
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


def build_scene(robot_xml_path: str, gripper_xml_path: Optional[str] = None):
    # assert robot_xml_path.endswith(".xml")

    arena = mjcf.RootElement()
    arm_simulate = mjcf.from_path(robot_xml_path)
    # arm_copy = mjcf.from_path(xml_path)

    if gripper_xml_path is not None:
        # attach gripper to the robot at "attachment_site"
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        attach_hand_to_arm(arm_simulate, gripper_simulate)

    arena.worldbody.attach(arm_simulate)
    # arena.worldbody.attach(arm_copy)

    return arena


class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()


class ZMQRobotServer:
    """A class representing a ZMQ server for a robot."""

    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the robot state and commands over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                print("Timeout in ZMQLeaderServer serve")
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()


class MujocoRobotServer:
    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
    ):
        self._has_gripper = gripper_xml_path is not None
        arena = build_scene(xml_path, gripper_xml_path)

        assets: Dict[str, str] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        xml_string = arena.to_xml_string()
        # save xml_string to file
        with open("arena.xml", "w") as f:
            f.write(xml_string)

        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu

        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

        self._print_joints = print_joints

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, (
            f"Expected joint state of length {self._num_joints}, "
            f"got {len(joint_state)}."
        )
        if self._has_gripper:
            _joint_state = joint_state.copy()
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        else:
            self._joint_cmd = joint_state.copy()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]
        ee_site = "attachment_site"
        try:
            ee_pos = self._data.site_xpos.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_mat = self._data.site_xmat.copy()[
                mujoco.mj_name2id(self._model, 6, ee_site)
            ]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
            ee_quat[0] = 1
        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
        }

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                self._data.ctrl[:] = self._joint_cmd
                # self._data.qpos[:] = self._joint_cmd
                mujoco.mj_step(self._model, self._data)
                self._joint_state = self._data.qpos.copy()[: self._num_joints]

                if self._print_joints:
                    print(self._joint_state)

                # Example modification of a viewer option: toggle contact points every two seconds.
                with viewer.lock():
                    # TODO remove?
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
                        self._data.time % 2
                    )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self._model.opt.timestep - (
                    time.time() - step_start
                )
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()


class DrillGraspRobotServer:
    """
    Robosuite DrillGrasp server with UR5eDex.

    ZMQ API is compatible with MujocoRobotServer:
      - num_dofs
      - get_joint_state
      - command_joint_state
      - get_observations
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
        control_freq: int = 20,
    ):
        self._env = robosuite.make(
            env_name="DrillGrasp",
            robots="UR5eDex",
            controller_configs=self._build_absolute_joint_controller_config(),
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["thirdview", "robot0_eye_in_hand"],
            camera_heights=84,
            camera_widths=84,
            camera_depths=[True, True],
            control_freq=control_freq,
            ignore_done=True,
            render_camera=None,
            initialization_noise=None,
        )
        self._env.reset()
        if self._env.has_renderer and self._env.viewer is None:
            self._env.initialize_renderer()

        self._robot = self._env.robots[0]
        self._arm_controller = self._robot.part_controllers["right"]
        self._hand_controller = self._robot.part_controllers["right_gripper"]

        self._arm_qpos_idx = np.array(self._arm_controller.qpos_index, dtype=int)
        self._hand_qpos_idx = np.array(self._hand_controller.qpos_index, dtype=int)
        self._arm_joint_ids = np.array(self._arm_controller.joint_index, dtype=int)
        self._hand_joint_ids = np.array(self._hand_controller.joint_index, dtype=int)

        # Match show_dex_drill.py initial arm posture.
        arm_init = self._clip_with_joint_limits(self._env.sim, self._arm_joint_ids, INIT_ARM_QPOS.copy())
        self._env.sim.data.qpos[self._arm_qpos_idx] = arm_init
        self._env.sim.data.qvel[self._arm_controller.qvel_index] = 0.0
        self._env.sim.forward()

        self._arm_dim = len(self._arm_qpos_idx)
        self._hand_cmd_dim = int(self._robot.gripper["right"].dof)  # Inspire command DoF (6)
        self._camera_names = list(self._env.camera_names)
        self._camera_height = int(self._env.camera_heights[0]) if isinstance(self._env.camera_heights, list) else int(self._env.camera_heights)
        self._camera_width = int(self._env.camera_widths[0]) if isinstance(self._env.camera_widths, list) else int(self._env.camera_widths)
        self._hand_actuator_ids = np.array(
            self._robot._ref_actuators_indexes_dict["right_gripper"], dtype=int
        )
        self._hand_ctrl_low12 = np.array(
            self._env.sim.model.actuator_ctrlrange[self._hand_actuator_ids, 0], dtype=float
        )
        self._hand_ctrl_high12 = np.array(
            self._env.sim.model.actuator_ctrlrange[self._hand_actuator_ids, 1], dtype=float
        )
        self._hand_low12, self._hand_high12 = self._joint_limits_from_ids(
            self._env.sim, self._hand_joint_ids
        )
        self._hand_qpos_groups = self._build_hand_qpos_groups(self._hand_controller.joint_names)

        # Keep ZMQ interface arm-centric by default to stay compatible with real UR teleop clients.
        self._num_joints = self._arm_dim
        self._joint_state = np.array(self._env.sim.data.qpos[self._arm_qpos_idx], dtype=float)
        self._arm_cmd = self._joint_state.copy()
        hand_qpos12 = np.array(self._env.sim.data.qpos[self._hand_qpos_idx], dtype=float)
        self._hand_cmd_norm6 = self._hand_qpos12_to_norm6(
            hand_qpos12, self._hand_low12, self._hand_high12, self._hand_qpos_groups
        )

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        self._print_joints = print_joints
        self._running = False

    @staticmethod
    def _build_absolute_joint_controller_config() -> Dict[str, Any]:
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

    @staticmethod
    def _clip_with_joint_limits(sim: Any, joint_ids: np.ndarray, qpos_target: np.ndarray) -> np.ndarray:
        clipped = qpos_target.copy()
        for i, j_id in enumerate(joint_ids):
            if bool(sim.model.jnt_limited[j_id]):
                low, high = sim.model.jnt_range[j_id]
                clipped[i] = np.clip(clipped[i], low, high)
        return clipped

    @staticmethod
    def _joint_limits_from_ids(sim: Any, joint_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lows = np.zeros(len(joint_ids), dtype=float)
        highs = np.zeros(len(joint_ids), dtype=float)
        for i, j_id in enumerate(joint_ids):
            low, high = sim.model.jnt_range[j_id]
            lows[i] = low
            highs[i] = high
        return lows, highs

    @staticmethod
    def _build_hand_qpos_groups(hand_joint_names: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
        name_to_idx = {name: i for i, name in enumerate(hand_joint_names)}

        def _idx_by_suffix(raw_name: str) -> int:
            if raw_name in name_to_idx:
                return name_to_idx[raw_name]
            matches = [idx for name, idx in name_to_idx.items() if name.endswith(raw_name)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) == 0:
                raise KeyError(
                    f"Joint suffix not found: {raw_name}. Available: {list(name_to_idx.keys())}"
                )
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

    @staticmethod
    def _hand_qpos12_to_norm6(
        hand_qpos12: np.ndarray,
        hand_low12: np.ndarray,
        hand_high12: np.ndarray,
        qpos_groups: Tuple[Tuple[int, ...], ...],
    ) -> np.ndarray:
        q12 = np.asarray(hand_qpos12, dtype=float)
        norm6 = np.zeros(6, dtype=float)
        for g, joint_indices in enumerate(qpos_groups):
            vals = []
            for j in joint_indices:
                span = max(hand_high12[j] - hand_low12[j], 1e-8)
                vals.append(
                    (hand_high12[j] - np.clip(q12[j], hand_low12[j], hand_high12[j])) / span
                )
            norm6[g] = float(np.mean(vals))
        return np.clip(norm6, 0.0, 1.0)

    @staticmethod
    def _hand_norm6_to_ctrl12(
        hand_norm6: np.ndarray,
        ctrl_low12: np.ndarray,
        ctrl_high12: np.ndarray,
    ) -> np.ndarray:
        hand_norm6 = np.clip(np.asarray(hand_norm6, dtype=float), 0.0, 1.0)
        if hand_norm6.shape[0] != 6:
            raise ValueError(f"Expected 6D hand command, got shape {hand_norm6.shape}.")
        ctrl12 = np.zeros(12, dtype=float)
        for g, act_indices in enumerate(INSPIRE_GROUPS_ACTUATOR):
            for i in act_indices:
                ctrl12[i] = ctrl_low12[i] + (1.0 - hand_norm6[g]) * (
                    ctrl_high12[i] - ctrl_low12[i]
                )
        return ctrl12

    def _apply_joint_targets(self, arm_target: np.ndarray, hand_ctrl_target12: np.ndarray) -> None:
        self._robot.composite_controller.update_state()
        self._arm_controller.set_goal(arm_target)

        applied = self._robot.composite_controller.run_controller(self._robot._enabled_parts)
        for part_name, applied_action in applied.items():
            if part_name == "right_gripper":
                continue
            actuator_ids = self._robot._ref_actuators_indexes_dict[part_name]
            ctrl_low = self._robot.sim.model.actuator_ctrlrange[actuator_ids, 0]
            ctrl_high = self._robot.sim.model.actuator_ctrlrange[actuator_ids, 1]
            self._robot.sim.data.ctrl[actuator_ids] = np.clip(applied_action, ctrl_low, ctrl_high)
        self._robot.sim.data.ctrl[self._hand_actuator_ids] = np.clip(
            hand_ctrl_target12, self._hand_ctrl_low12, self._hand_ctrl_high12
        )

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        joint_state = np.asarray(joint_state, dtype=float)
        n = len(joint_state)
        if n == self._arm_dim:
            self._arm_cmd = joint_state.copy()
        elif n == self._arm_dim + self._hand_cmd_dim:
            self._arm_cmd = joint_state[: self._arm_dim].copy()
            self._hand_cmd_norm6 = np.clip(joint_state[self._arm_dim :].copy(), 0.0, 1.0)
        else:
            raise AssertionError(
                f"Expected command length {self._arm_dim} (arm only), "
                f"{self._arm_dim + self._hand_cmd_dim} (arm + hand6 norm), got {n}."
            )

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs_dict = self._env._get_observations(force_update=False)
        # Keep only per-key obs entries; skip aggregated "*-state" tensors.
        obs_dict = {
            k: np.array(v)
            for k, v in obs_dict.items()
            if isinstance(v, np.ndarray) and (not k.endswith("-state"))
        }

        arm_pos = np.array(self._env.sim.data.qpos[self._arm_qpos_idx], dtype=float)
        arm_vel = np.array(self._env.sim.data.qvel[self._arm_controller.qvel_index], dtype=float)
        hand_pos = np.array(self._env.sim.data.qpos[self._hand_qpos_idx], dtype=float)
        hand_vel = np.array(self._env.sim.data.qvel[self._hand_controller.qvel_index], dtype=float)
        ee_pos = np.array(self._arm_controller.ref_pos, dtype=float)
        ee_mat = np.array(self._arm_controller.ref_ori_mat, dtype=float)
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, ee_mat.reshape(-1))
        # robosuite-style observations and extra fields used by collection scripts
        # Use env observable naming when available
        obj_pos = np.array(obs_dict.get("drill_001_pos", np.zeros(3)), dtype=float)
        obj_quat_xyzw = np.array(obs_dict.get("drill_001_quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=float)
        ee_pos_obs = np.array(obs_dict.get("robot0_eef_pos", ee_pos), dtype=float)
        ee_quat_xyzw_obs = np.array(obs_dict.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0])), dtype=float)
        joint_pos_obs = np.array(obs_dict.get("robot0_joint_pos", arm_pos), dtype=float)
        joint_vel_obs = np.array(obs_dict.get("robot0_joint_vel", arm_vel), dtype=float)
        grip_qpos_obs = np.array(obs_dict.get("robot0_gripper_qpos", hand_pos), dtype=float)
        grip_qvel_obs = np.array(obs_dict.get("robot0_gripper_qvel", hand_vel), dtype=float)

        is_grasped = int(self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=self._env.objects[0]))
        is_lifted = int(self._env._check_success())

        action_12 = np.concatenate([self._arm_cmd, self._hand_cmd_norm6], axis=0).astype(float)
        sim_state = np.array(self._env.sim.get_state().flatten(), dtype=float)

        camera_info = {}
        for cam_name in self._camera_names:
            try:
                cam_id = self._env.sim.model.camera_name2id(cam_name)
                fovy_deg = float(self._env.sim.model.cam_fovy[cam_id])
                h = float(self._camera_height)
                w = float(self._camera_width)
                fy = 0.5 * h / np.tan(np.deg2rad(fovy_deg) / 2.0)
                fx = fy
                cx = (w - 1.0) * 0.5
                cy = (h - 1.0) * 0.5
                intrinsic = np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )

                cam_pos = np.array(self._env.sim.data.cam_xpos[cam_id], dtype=float)
                cam_rot = np.array(self._env.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
                extrinsic = np.eye(4, dtype=float)
                extrinsic[:3, :3] = cam_rot
                extrinsic[:3, 3] = cam_pos
                camera_info[cam_name] = {
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic,
                    "width": np.array([self._camera_width], dtype=np.int32),
                    "height": np.array([self._camera_height], dtype=np.int32),
                    "depth_scale": np.array([1.0], dtype=float),
                }
            except Exception:
                continue

        return {
            # Backward-compatible fields
            "joint_positions": arm_pos,
            "joint_velocities": arm_vel,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "hand_joint_positions": hand_pos,
            "hand_joint_velocities": hand_vel,
            # Extended fields for dexmimic-style collection
            "sim_state": sim_state,
            "actions": action_12,
            "action_dict_right": np.array(self._arm_cmd, dtype=float),
            "action_dict_right_gripper": np.array(self._hand_cmd_norm6, dtype=float),
            "obs_robot0_joint_pos": joint_pos_obs,
            "obs_robot0_joint_vel": joint_vel_obs,
            "obs_robot0_gripper_qpos": grip_qpos_obs,
            "obs_robot0_gripper_qvel": grip_qvel_obs,
            "obs_robot0_eef_pos": ee_pos_obs,
            "obs_robot0_eef_quat": ee_quat_xyzw_obs,
            "obs_drill_001_pos": obj_pos,
            "obs_drill_001_quat": obj_quat_xyzw,
            "obs_dict": obs_dict,
            "camera_info": camera_info,
            "signal_drill_grasped": np.array([is_grasped], dtype=np.int64),
            "signal_drill_lifted": np.array([is_lifted], dtype=np.int64),
        }

    def serve(self) -> None:
        self._zmq_server_thread.start()
        self._running = True
        while self._running:

            step_start = time.time()

            arm_target = self._clip_with_joint_limits(self._env.sim, self._arm_joint_ids, self._arm_cmd)
            hand_norm6 = np.clip(self._hand_cmd_norm6, 0.0, 1.0)
            hand_ctrl_target12 = self._hand_norm6_to_ctrl12(
                hand_norm6, self._hand_ctrl_low12, self._hand_ctrl_high12
            )
            self._apply_joint_targets(arm_target, hand_ctrl_target12)

            self._env.sim.step()
            self._env._update_observables()
            if self._env.viewer is not None:
                try:
                    self._env.viewer.update()
                except Exception:
                    break

            self._joint_state = np.array(self._env.sim.data.qpos[self._arm_qpos_idx], dtype=float)
            if self._print_joints:
                print(self._joint_state)

            dt = self._env.model_timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)

    def stop(self) -> None:
        self._running = False
        self._zmq_server.stop()
        if self._zmq_server_thread.is_alive():
            self._zmq_server_thread.join()
        self._env.close()

    def __del__(self) -> None:
        self.stop()