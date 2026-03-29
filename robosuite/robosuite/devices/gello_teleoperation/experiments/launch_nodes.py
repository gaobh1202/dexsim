from dataclasses import dataclass
from pathlib import Path

import tyro

from gello.robots.robot import BimanualRobot, PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot
from typing import Optional, Tuple


@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.12.22"
    # 新增阻抗控制参数
    use_impedance: bool = True
    k_trans: float = 1000.0
    d_trans: float = 60.0
    k_rot: float = 32.0
    d_rot: float = 0.5
    f_max: float = 100.0
    tau_max: float = 10.0
    payload_mass: float = -2.0   # 末端负载质量 (kg)
    payload_cog: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 负载质心 (m)
    # 新增关节控制选项
    use_joint_control: bool = False  # 是否使用关节控制而不是阻抗控制
    sim_print_joints: bool = False
    sim_control_freq: int = 20


def launch_robot_server(args: Args):
    port = args.robot_port
    MENAGERIE_ROOT: Path = (
        Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
    )
    if args.robot == "sim_ur":
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=str(xml), gripper_xml_path=str(gripper_xml), port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_panda":
        from gello.robots.sim_robot import MujocoRobotServer
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=str(xml), gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_xarm":
        from gello.robots.sim_robot import MujocoRobotServer
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=str(xml), gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        server.serve()
    elif args.robot == "sim_drillgrasp":
        from gello.robots.sim_robot import DrillGraspRobotServer

        server = DrillGraspRobotServer(
            host=args.hostname,
            port=port,
            print_joints=args.sim_print_joints,
            control_freq=args.sim_control_freq,
        )
        server.serve()

    else:
        if args.robot == "xarm":
            from gello.robots.xarm_robot import XArmRobot

            robot = XArmRobot(ip=args.robot_ip)
        elif args.robot == "ur" or args.robot == "ur_impedance":
            if args.use_impedance:
                from gello.robots.ur_impedance import URRobotImpedance, ImpedanceGains
                gains = ImpedanceGains(
                    k_trans=args.k_trans,
                    d_trans=args.d_trans,
                    k_rot=args.k_rot,
                    d_rot=args.d_rot,
                    f_max=args.f_max,
                    tau_max=args.tau_max
                )
                robot = URRobotImpedance(
                    robot_ip=args.robot_ip,
                    no_gripper=True,
                    gains=gains,
                    control_rate_hz=125,
                    use_joint_control=args.use_joint_control  # 使用命令行参数
                )
                # 设置负载
                if args.payload_mass > 0:
                    robot.set_payload(args.payload_mass, list(args.payload_cog))

            else:
                from gello.robots.ur import URRobot

                robot = URRobot(robot_ip=args.robot_ip)
        elif args.robot == "ur_inspire":
            from gello.agents.ur_inspire_agent import URInspireRobot
            robot = URInspireRobot(robot_ip=args.robot_ip)
        elif args.robot == "panda":
            from gello.robots.panda import PandaRobot

            robot = PandaRobot(robot_ip=args.robot_ip)
        elif args.robot == "bimanual_ur":
            from gello.robots.ur import URRobot

            # IP for the bimanual robot setup is hardcoded
            _robot_l = URRobot(robot_ip="192.168.2.10")
            _robot_r = URRobot(robot_ip="192.168.1.10")
            robot = BimanualRobot(_robot_l, _robot_r)
        elif args.robot == "none" or args.robot == "print":
            robot = PrintRobot(8)

        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: sim_ur, sim_panda, sim_xarm, sim_drillgrasp, xarm, ur, bimanual_ur, none"
            )
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
        server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))