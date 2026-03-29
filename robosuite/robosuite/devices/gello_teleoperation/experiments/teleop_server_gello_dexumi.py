import threading
import time
import numpy as np
import json
import socket
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# ===== Gello相关 =====
from gello.robots import ur
from gello.agents.agent import BimanualAgent, DummyAgent
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot

# ===== Inspire+DexUMI相关 =====
import inspire_hand # 注意串口的设置
from dexumi.encoder.encoder import InspireEncoder

# =========== Inspire Hand参数 =============
DEXUMI_UART_PORT = "/dev/ttyACM0"
DEXUMI_FREQ_HZ = 15.0

# 角度映射参数
DEXUMI_RANGES = {
    'thumb_root': (200, 232),
    'thumb_mid': (136, 175),
    'index': (190, 240),
    'middle': (178, 240),
    'ring': (195, 233),
    'pinky': (195, 238)
}
INSPIRE_RANGE = (1000, 0)


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)

def map_value(value, in_min, in_max, out_min, out_max):
    clamped = max(in_min, min(value, in_max))
    return out_min + (clamped - in_min) * (out_max - out_min) / (in_max - in_min)

def map_angles_to_inspire(angles):
    thumb_root = map_value(angles[0], *DEXUMI_RANGES['thumb_root'], *INSPIRE_RANGE)
    thumb_mid  = map_value(angles[1], *DEXUMI_RANGES['thumb_mid'],  *INSPIRE_RANGE)
    index      = map_value(angles[5], *DEXUMI_RANGES['index'],      *INSPIRE_RANGE)
    middle     = map_value(angles[3], *DEXUMI_RANGES['middle'],     *INSPIRE_RANGE)
    ring       = map_value(angles[4], *DEXUMI_RANGES['ring'],       *INSPIRE_RANGE)
    pinky      = map_value(angles[2], *DEXUMI_RANGES['pinky'],      *INSPIRE_RANGE)
    return (
        int(index),
        int(middle),
        int(ring),
        int(pinky),
        int(thumb_mid),
        int(thumb_root),
    )


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    socket_port: int = 5000
    fps: int = 25
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: Optional[str] = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "./demos_test"
    bimanual: bool = False
    verbose: bool = False


class TeleopServer:
    def __init__(self, args):
        # Gello/机械臂初始化
        if args.mock:
            self.robot_client = PrintRobot(8, dont_print=True)
            self.camera_clients = {}
        else:
            self.camera_clients = {
                # you can optionally add camera nodes here for imitation learning purposes
                # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
                # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
            }
            self.robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        self.env = RobotEnv(self.robot_client, control_rate_hz=args.hz, camera_dict=self.camera_clients)
        
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            print(args.start_joints)
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    # 根据启动流程文档，使用正确的初始关节角度
                    [90, -90, -90, 0, 90, 90]  # [1.57, -1.57, -1.57, 0, 1.57, 1.57] in radians
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
            self.agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
            curr_joints = self.env.get_obs()["joint_positions"][:6]
            print('reset_joints', reset_joints, 'curr joints', curr_joints)

            reset_joints = np.asarray(reset_joints)
            curr_joints = np.asarray(curr_joints)
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    # 根据机器人自由度动态构建动作
                    total_dofs = self.robot_client.num_dofs()
                    if total_dofs > 6:
                        cmd_full = np.concatenate([jnt, np.ones(total_dofs - 6)])
                    else:
                        cmd_full = jnt
                    self.env.step(cmd_full)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent
            self.agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent
            self.agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            self.agent = DummyAgent(num_dofs=self.robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")
        
        # Inspire Hand + DexUMI初始化
        self.encoder = InspireEncoder("inspire", verbose=False, uart_port=DEXUMI_UART_PORT)
        self.encoder.start_streaming()
        print("[INFO] InspireEncoder started.")
        
        # 验证 Inspire Hand 连接
        try:
            import inspire_hand
            # 测试连接
            test_angles = inspire_hand.get_actangle()
            print(f"[INFO] Inspire Hand connected successfully. Current angles: {test_angles}")
        except Exception as e:
            print(f"[ERROR] Inspire Hand connection failed: {e}")


        # 状态维护
        self.fps = args.fps
        self.latest_arm_qpos = np.zeros(6)  # 机械臂6 + 手6
        self.latest_hand_qpos = np.zeros(6)
        self.latest_gello_action = np.zeros(6)
        self.latest_umi_action = np.zeros(6)
        self.latest_ee_pose = np.zeros(7)
        self.latest_force = np.zeros(6)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.socket_port = args.socket_port

        # 允许外部通过SET_INITIAL设置初始位姿
        self.override_ur_target = None
        self.override_hand_target = None
        self.initialization_complete = threading.Event()

    def go_to_start_position(self):
        # self.env._robot.use_joint_control = True
        print("Going to start position")
        obs = self.env.get_obs()
        # 仅使用机械臂6维进行对齐（hand 由 hand_loop 独立控制）
        start_pos = np.asarray(self.agent.act(obs))  # (6,)
        print('start position (arm only)', start_pos)
        joints_all = np.asarray(obs["joint_positions"])  # 可能为(12,) 或 (7,)
        arm_joints = joints_all[:6]
        print('current arm joints', arm_joints)

        abs_deltas = np.abs(start_pos - arm_joints)
        id_max_joint_delta = np.argmax(abs_deltas)
        max_joint_delta_allowed = 1.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta_allowed:
            id_mask = abs_deltas > max_joint_delta_allowed
            print()
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                arm_joints[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return

        print(f"Start pos dim: {len(start_pos)}", f"Arm joints dim: {len(arm_joints)}")

        max_delta = 0.05
        # 进行若干步限幅推进，避免瞬时大动作
        for _ in range(25):
            obs = self.env.get_obs()
            command_arm = np.asarray(self.agent.act(obs))  # (6,)
            current_arm = np.asarray(obs["joint_positions"])[:6]

            delta = command_arm - current_arm
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta

            new_arm = current_arm + delta
            # hand 通道此处不做对齐，使用固定值占位（与 arm_loop 一致）
            cmd_full = np.concatenate([new_arm, np.ones(6)])

            self.env.step(cmd_full)

        # 事后安全检查
        obs = self.env.get_obs()
        joints_arm = np.asarray(obs["joint_positions"])[:6]
        action_arm = np.asarray(self.agent.act(obs))
        if (action_arm - joints_arm > 1.8).any():
            print("Action is too big")
            joint_index = np.where(action_arm - joints_arm > 0.8)
            for j in joint_index[0]:
                print(
                    f"Joint [{j}], leader: {action_arm[j]}, follower: {joints_arm[j]}, diff: {action_arm[j] - joints_arm[j]}"
                )

    def arm_loop(self):
        # self.env._robot.use_joint_control = False
        step_time = 1.0 / self.fps
        print("[INFO] Gello UR5e thread started.")
        while not self.stop_event.is_set():
            start = time.time()
            with self.lock:
                obs = self.env.get_obs()
                ur_current = obs['joint_positions'][:6]
                ee_pose = obs['ee_pos_quat']
                ur_target = self.agent.act(obs)
                # 覆盖外部指定的起始位姿（一次性）
                if self.override_ur_target is not None:
                    ur_target = np.array(self.override_ur_target, dtype=float)
                    self.override_ur_target = None
                    self.initialization_complete.set()
            # 控制机械臂
            try:
                # 根据机器人自由度动态构建动作（超过6维时补固定占位）
                total_dofs = self.robot_client.num_dofs()
                if total_dofs > 6:
                    action = np.concatenate([ur_target, np.ones(total_dofs - 6)])
                else:
                    action = ur_target
                print("arm loop action", action)
                self.env.step(action)
            except Exception as e:
                print(f"[ERROR] Env step failed: {e}")
            with self.lock:
                self.latest_arm_qpos = ur_current
                self.latest_gello_action = ur_target
                self.latest_ee_pose = ee_pose
            duration = time.time() - start
            sleep_time = step_time - duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def hand_loop(self):
        step_time = 1.0 / DEXUMI_FREQ_HZ
        print("[INFO] Inspire Hand thread started.")
        while not self.stop_event.is_set():
            start = time.time()
            numeric = self.encoder.get_numeric_frame()
            if numeric is None:
                print("[WARN] No data from Dexumi encoder")
                time.sleep(0.01)
                continue
            angles = numeric.joint_angles  # 6维
            # print('dexumi angles', angles)
            if angles is None or len(angles) < 6:
                print("[WARN] Invalid angle data from Dexumi")
                continue
            inspire_angles = None
            try:
                inspire_angles = map_angles_to_inspire(angles)
                # print('inspire angles', inspire_angles)
                # 外部覆盖一次性初始化
                with self.lock:
                    if self.override_hand_target is not None:
                        inspire_angles = tuple(self.override_hand_target)
                        self.override_hand_target = None
                        self.initialization_complete.set()
                inspire_hand.setangle(*inspire_angles)
                # print(f"[DEBUG] Sent angles to Inspire Hand: {inspire_angles}")
            except Exception as e:
                print(f"[WARN] setangle failed: {e}")
            # 记录手指角度
            if inspire_angles is not None:
                with self.lock:
                    self.latest_hand_qpos = np.array(inspire_angles)
                    self.latest_umi_action = np.array(angles)
            # 力觉
            try:
                forces_raw = inspire_hand.get_actforce()
                if forces_raw and len(forces_raw) >= 6:
                    forces_signed = [v-65536 if v>32767 else v for v in forces_raw[:6]]
                    with self.lock:
                        self.latest_force = np.array(forces_signed)
            except Exception as e:
                print(f"[WARN] get_actforce failed: {e}")
            duration = time.time() - start
            sleep_time = step_time - duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_request(self, req: dict) -> dict:
        # 通过'type'字段判断请求类型，分成三种情况：
        # 1. PING：返回PONG
        # 2. GET_STATE：返回当前状态
        # 3. SET_INITIAL：设置初始位姿
        typ = req.get('type')
        if typ == 'PING':
            return {'ok': True, 'type': 'PONG'}
        if typ == 'GET_STATE':
            with self.lock:
                def to_list(x):
                    try:
                        return np.asarray(x).tolist()
                    except Exception:
                        # 最差情况下返回原值，避免阻塞
                        return x
                return {
                    'ok': True,
                    'armqpos6': to_list(self.latest_arm_qpos),
                    'handqpos6': to_list(self.latest_hand_qpos),
                    'armee7': to_list(self.latest_ee_pose),
                    'gelloaction6': to_list(self.latest_gello_action),
                    'umiaction6': to_list(self.latest_umi_action),
                    'force6': to_list(self.latest_force),
                }
        if typ == 'SET_INITIAL':
            ur = req.get('ur')
            hand = req.get('hand')
            with self.lock:
                if ur is not None:
                    self.override_ur_target = ur
                if hand is not None:
                    self.override_hand_target = hand
            return {'ok': True}
        return {'ok': False, 'error': 'unknown type'}

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
                            req = json.loads(line.decode('utf-8'))
                            resp = self._handle_request(req)
                        except Exception as e:
                            resp = {'ok': False, 'error': str(e)}
                        msg = (json.dumps(resp) + "\n").encode('utf-8')
                        conn.sendall(msg)
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

        # 在启动控制线程之前，做一次安全的起始位姿对齐，避免大幅度动作
        try:
            self.go_to_start_position()
            print("go to start position")
        except Exception as e:
            print(f"[WARN] go_to_start_position skipped: {e}")
        self.th_arm = threading.Thread(target=self.arm_loop, daemon=True)
        self.th_hand = threading.Thread(target=self.hand_loop, daemon=True)
        self.th_tcp = threading.Thread(target=self.tcp_server, daemon=True)
        self.th_arm.start()
        self.th_hand.start()
        self.th_tcp.start()
        print("[INFO] TeleopServer started.")
        print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    def stop(self):
        self.stop_event.set()
        try:
            self.encoder.stop_streaming()
        except Exception:
            pass
        print("[INFO] TeleopServer stopped.")


def main(args: Args):
    server = TeleopServer(args)
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
        # 允许在交互式/IDE下直接运行
        main(Args())