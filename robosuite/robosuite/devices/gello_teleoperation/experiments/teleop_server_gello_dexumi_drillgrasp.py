import glob
import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from gello.agents.agent import DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot

from dexumi.encoder.encoder import InspireEncoder

# ===== DexUMI input =====
DEXUMI_UART_PORT = "/dev/ttyACM0"
DEXUMI_FREQ_HZ = 15.0

# DexUMI calibrated ranges (raw angle domain).
DEXUMI_RANGES = {
    "thumb_root": (200, 232),
    "thumb_mid": (136, 175),
    "index": (190, 240),
    "middle": (178, 240),
    "ring": (195, 233),
    "pinky": (195, 238),
}

# DrillGrasp hand command order:
# [pinky, ring, middle, index, thumb_bend, thumb_proximal_1]
# For Inspire hand in this sim setup, qpos=1 means closed and qpos=0 means open.
DEFAULT_HAND6_QPOS_CLOSED = np.ones(6, dtype=float)
DEFAULT_HAND6_QPOS_OPEN = np.zeros(6, dtype=float)

# Mapping from DrillGrasp expanded 12D hand qpos -> compact 6D command groups.
HAND12_TO_HAND6_GROUPS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9, 10),
    (11,),
)


def map_value(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    clamped = max(in_min, min(value, in_max))
    return out_min + (clamped - in_min) * (out_max - out_min) / (in_max - in_min)


def dex_angles_to_hand_norm6(angles: np.ndarray) -> np.ndarray:
    """
    Convert DexUMI 6D raw angles to DrillGrasp normalized hand command.

    Output convention:
      - 0.0 = closed
      - 1.0 = open
    Output order:
      [pinky, ring, middle, index, thumb_bend, thumb_proximal_1]
    """
    thumb_root = map_value(angles[0], *DEXUMI_RANGES["thumb_root"], 0.0, 1.0)
    thumb_mid = map_value(angles[1], *DEXUMI_RANGES["thumb_mid"], 0.0, 1.0)
    index = map_value(angles[5], *DEXUMI_RANGES["index"], 0.0, 1.0)
    middle = map_value(angles[3], *DEXUMI_RANGES["middle"], 0.0, 1.0)
    ring = map_value(angles[4], *DEXUMI_RANGES["ring"], 0.0, 1.0)
    pinky = map_value(angles[2], *DEXUMI_RANGES["pinky"], 0.0, 1.0)
    # return np.clip(
    #     np.array([pinky, ring, middle, index, thumb_mid, thumb_root], dtype=float),
    #     0.0,
    #     1.0,
    # )
    return np.clip(
        np.array([index, middle, ring, pinky, thumb_mid, thumb_root], dtype=float),
        0.0,
        1.0,
    )


def hand12_to_hand6(hand12: np.ndarray) -> np.ndarray:
    hand12 = np.asarray(hand12, dtype=float)
    hand6 = np.zeros(6, dtype=float)
    for i, idx_group in enumerate(HAND12_TO_HAND6_GROUPS):
        hand6[i] = float(np.mean(hand12[list(idx_group)]))
    return hand6


@dataclass
class Args:
    agent: str = "gello"
    robot_port: int = 6001
    socket_port: int = 5000
    fps: int = 25
    hostname: str = "127.0.0.1"
    robot_type: Optional[str] = None
    start_joints: Optional[Tuple[float, ...]] = None
    gello_port: Optional[str] = None
    mock: bool = False
    verbose: bool = False


class TeleopServerDrillGrasp:
    def __init__(self, args: Args):
        if args.mock:
            self.robot_client = PrintRobot(12, dont_print=True)
        else:
            self.robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"[INFO] Using gello port: {gello_port}")
                else:
                    raise ValueError("No gello port found, please specify --gello-port")
            self.agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            self.agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            self.agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent in ("dummy", "none"):
            self.agent = DummyAgent(num_dofs=6)
        else:
            raise ValueError(f"Invalid agent name: {args.agent}")

        self.encoder = InspireEncoder("inspire", verbose=False, uart_port=DEXUMI_UART_PORT)
        self.encoder.start_streaming()
        print("[INFO] InspireEncoder started.")

        self.hand_qpos_closed = DEFAULT_HAND6_QPOS_CLOSED.copy()
        self.hand_qpos_open = DEFAULT_HAND6_QPOS_OPEN.copy()

        self.fps = args.fps
        self.latest_arm_qpos = np.zeros(6)
        self.latest_hand_qpos = np.zeros(6)
        self.latest_gello_action = np.zeros(6)
        self.latest_umi_action = np.zeros(6)
        self.latest_ee_pose = np.zeros(7)
        self.latest_force = np.zeros(6)
        self.current_hand_cmd6 = np.zeros(6)

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.socket_port = args.socket_port
        self.override_ur_target = None
        self.override_hand_target = None
        self.initialization_complete = threading.Event()

        self.th_arm = None
        self.th_hand = None
        self.th_tcp = None

    def _get_obs(self) -> dict:
        return self.robot_client.get_observations()

    def _send_arm_hand_command(self, arm_target6: np.ndarray, hand_target6: np.ndarray) -> None:
        cmd = np.concatenate([np.asarray(arm_target6, dtype=float), np.asarray(hand_target6, dtype=float)])
        self.robot_client.command_joint_state(cmd)

    def _hand_norm_to_qpos(self, hand_norm6: np.ndarray) -> np.ndarray:
        hand_norm6 = np.clip(np.asarray(hand_norm6, dtype=float), 0.0, 1.0)
        return self.hand_qpos_closed + hand_norm6 * (self.hand_qpos_open - self.hand_qpos_closed)

    def go_to_start_position(self):
        print("[INFO] Moving to start position")
        obs = self._get_obs()
        start_pos = np.asarray(self.agent.act(obs), dtype=float)[:6]
        arm_joints = np.asarray(obs["joint_positions"], dtype=float)[:6]

        abs_deltas = np.abs(start_pos - arm_joints)
        if np.any(abs_deltas > 1.8):
            print("[WARN] Start delta too large, skip alignment for safety.")
            return

        max_delta = 0.05
        for _ in range(25):
            obs = self._get_obs()
            command_arm = np.asarray(self.agent.act(obs), dtype=float)[:6]
            current_arm = np.asarray(obs["joint_positions"], dtype=float)[:6]

            delta = command_arm - current_arm
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            new_arm = current_arm + delta

            with self.lock:
                hand_cmd6 = self.current_hand_cmd6.copy()
            self._send_arm_hand_command(new_arm, hand_cmd6)
            time.sleep(0.002)

    def arm_loop(self):
        step_time = 1.0 / self.fps
        print("[INFO] Arm control thread started.")
        while not self.stop_event.is_set():
            start = time.time()
            try:
                obs = self._get_obs()
                ur_current = np.asarray(obs.get("joint_positions", np.zeros(6)), dtype=float)[:6]
                ee_pose = np.asarray(obs.get("ee_pos_quat", np.zeros(7)), dtype=float)
                ur_target = np.asarray(self.agent.act(obs), dtype=float)[:6]

                with self.lock:
                    if self.override_ur_target is not None:
                        ur_target = np.asarray(self.override_ur_target, dtype=float)
                        self.override_ur_target = None
                        self.initialization_complete.set()
                    hand_cmd6 = self.current_hand_cmd6.copy()

                self._send_arm_hand_command(ur_target, hand_cmd6)

                hand12_obs = obs.get("hand_joint_positions", None)
                hand6_obs = hand12_to_hand6(hand12_obs) if hand12_obs is not None else hand_cmd6
                with self.lock:
                    self.latest_arm_qpos = ur_current
                    self.latest_hand_qpos = np.asarray(hand6_obs, dtype=float)
                    self.latest_gello_action = ur_target
                    self.latest_ee_pose = ee_pose
            except Exception as e:
                print(f"[WARN] arm_loop failed: {e}")

            sleep_time = step_time - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def hand_loop(self):
        step_time = 1.0 / DEXUMI_FREQ_HZ
        print("[INFO] DexUMI hand thread started.")
        while not self.stop_event.is_set():
            start = time.time()
            numeric = self.encoder.get_numeric_frame()
            if numeric is None:
                time.sleep(0.01)
                continue

            angles = getattr(numeric, "joint_angles", None)
            if angles is None or len(angles) < 6:
                continue

            angles = np.asarray(angles, dtype=float)
            hand_norm6 = dex_angles_to_hand_norm6(angles)
            hand_cmd6 = self._hand_norm_to_qpos(hand_norm6)

            with self.lock:
                if self.override_hand_target is not None:
                    hand_cmd6 = np.asarray(self.override_hand_target, dtype=float)
                    self.override_hand_target = None
                    self.initialization_complete.set()
                self.current_hand_cmd6 = hand_cmd6
                self.latest_hand_qpos = hand_cmd6
                self.latest_umi_action = angles

            sleep_time = step_time - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_request(self, req: dict) -> dict:
        def _to_serializable(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, dict):
                return {k: _to_serializable(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_to_serializable(v) for v in x]
            return x

        typ = req.get("type")
        if typ == "PING":
            return {"ok": True, "type": "PONG"}

        if typ == "GET_STATE":
            include_images = bool(req.get("include_images", False))
            with self.lock:
                obs = self._get_obs()
                obs_dict = obs.get("obs_dict", {})
                camera_info = obs.get("camera_info", {})
                if not include_images:
                    obs_dict = {k: v for k, v in obs_dict.items() if (not k.endswith("_image")) and (not k.endswith("_depth"))}
                return {
                    "ok": True,
                    "armqpos6": np.asarray(self.latest_arm_qpos).tolist(),
                    "handqpos6": np.asarray(self.latest_hand_qpos).tolist(),
                    "armee7": np.asarray(self.latest_ee_pose).tolist(),
                    "gelloaction6": np.asarray(self.latest_gello_action).tolist(),
                    "umiaction6": np.asarray(self.latest_umi_action).tolist(),
                    "force6": np.asarray(self.latest_force).tolist(),
                    # Extended payload for dexmimic-style collectors
                    "sim_state": _to_serializable(obs.get("sim_state", np.array([], dtype=float))),
                    "actions": _to_serializable(obs.get("actions", np.array([], dtype=float))),
                    "action_dict": {
                        "right": _to_serializable(obs.get("action_dict_right", np.zeros(6, dtype=float))),
                        "right_gripper": _to_serializable(
                            obs.get("action_dict_right_gripper", np.zeros(6, dtype=float))
                        ),
                    },
                    "obs": {
                        **_to_serializable(obs_dict),
                        "robot0_joint_pos": _to_serializable(obs.get("obs_robot0_joint_pos", np.zeros(6, dtype=float))),
                        "robot0_joint_vel": _to_serializable(obs.get("obs_robot0_joint_vel", np.zeros(6, dtype=float))),
                        "robot0_gripper_qpos": _to_serializable(obs.get("obs_robot0_gripper_qpos", np.zeros(12, dtype=float))),
                        "robot0_gripper_qvel": _to_serializable(obs.get("obs_robot0_gripper_qvel", np.zeros(12, dtype=float))),
                        "robot0_eef_pos": _to_serializable(obs.get("obs_robot0_eef_pos", np.zeros(3, dtype=float))),
                        "robot0_eef_quat": _to_serializable(obs.get("obs_robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0], dtype=float))),
                        "drill_001_pos": _to_serializable(obs.get("obs_drill_001_pos", np.zeros(3, dtype=float))),
                        "drill_001_quat": _to_serializable(obs.get("obs_drill_001_quat", np.array([0.0, 0.0, 0.0, 1.0], dtype=float))),
                    },
                    "datagen_info": {
                        "subtask_term_signals": {
                            "drill_grasped": int(np.asarray(obs.get("signal_drill_grasped", [0]))[0]),
                            "drill_lifted": int(np.asarray(obs.get("signal_drill_lifted", [0]))[0]),
                        }
                    },
                    "camera_info": _to_serializable(camera_info),
                }

        if typ == "SET_INITIAL":
            ur = req.get("ur")
            hand = req.get("hand")
            with self.lock:
                if ur is not None:
                    self.override_ur_target = ur
                if hand is not None:
                    self.override_hand_target = hand
            return {"ok": True}

        return {"ok": False, "error": "unknown type"}

    def tcp_server(self):
        print(f"[INFO] TCP server listening on 0.0.0.0:{self.socket_port}")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", self.socket_port))
        server.listen(1)
        server.settimeout(1.0)
        conn = None
        buf = b""
        try:
            while not self.stop_event.is_set():
                if conn is None:
                    try:
                        conn, _addr = server.accept()
                        conn.settimeout(0.1)
                        buf = b""
                        print("[INFO] TCP client connected")
                    except socket.timeout:
                        continue
                    except Exception as e:
                        print(f"[WARN] accept failed: {e}")
                        continue

                try:
                    data = conn.recv(4096)
                    if not data:
                        conn.close()
                        conn = None
                        print("[INFO] TCP client disconnected")
                        continue

                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line:
                            continue
                        try:
                            req = json.loads(line.decode("utf-8"))
                            resp = self._handle_request(req)
                        except Exception as e:
                            resp = {"ok": False, "error": str(e)}
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except socket.timeout:
                    pass
                except Exception as e:
                    print(f"[WARN] tcp_server client error: {e}")
                    try:
                        if conn:
                            conn.close()
                    finally:
                        conn = None
        finally:
            try:
                if conn:
                    conn.close()
            finally:
                server.close()

    def start(self):
        self.stop_event.clear()
        self.initialization_complete.clear()
        try:
            self.go_to_start_position()
        except Exception as e:
            print(f"[WARN] go_to_start_position skipped: {e}")

        self.th_arm = threading.Thread(target=self.arm_loop, daemon=True)
        self.th_hand = threading.Thread(target=self.hand_loop, daemon=True)
        self.th_tcp = threading.Thread(target=self.tcp_server, daemon=True)
        self.th_arm.start()
        self.th_hand.start()
        self.th_tcp.start()
        print("[INFO] TeleopServerDrillGrasp started.")

    def stop(self):
        self.stop_event.set()
        try:
            self.encoder.stop_streaming()
        except Exception:
            pass
        print("[INFO] TeleopServerDrillGrasp stopped.")


def main(args: Args):
    server = TeleopServerDrillGrasp(args)
    server.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    try:
        import tyro

        main(tyro.cli(Args))
    except Exception:
        main(Args())