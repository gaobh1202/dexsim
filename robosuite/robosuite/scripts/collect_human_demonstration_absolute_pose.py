"""
使用VisionPro绝对位姿控制收集人类演示数据的脚本

与collect_human_demonstration_in_dexmimic.py的区别：
- 使用VisionProAbsolute设备，实现绝对位姿控制
- 在VisionPro坐标系中选择虚拟base，将手相对于虚拟base的位姿映射到机器人末端相对于机器人base的位姿
- 返回绝对位姿action而不是增量action

使用方法：
1. 运行脚本后，首先进行手部校准（保持初始姿态）
2. 然后进行虚拟base校准（将手放在期望的初始位置）
3. 开始收集演示数据
"""

import argparse
import datetime
import json
import os
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.mjmod import CameraModder

# IMPORTANT: import dexmimicgen to register the environments
import dexmimicgen


def collect_human_trajectory(env, device, arm, max_fr, goal_update_mode):
    """
    使用VisionPro绝对位姿控制收集演示数据
    
    Args:
        env: 环境
        device: VisionProAbsolute设备
        arm: 手臂名称（在绝对位姿模式下不使用）
        max_fr: 最大帧率
        goal_update_mode: 目标更新模式（在绝对位姿模式下不使用）
    """
    env.reset()

    env.render()
    
    # 设置机器人的固定关节位置（用于确定初始位姿）
    ur5e_joint_positions = np.array([np.pi, - np.pi / 2, - 3 *np.pi / 4, np.pi / 4, np.pi / 2, np.pi])
    panda_joint_positions = np.array([0, 0, 0, -3 * np.pi / 4.0, 0, np.pi * 1.1, np.pi / 3 - 0.2])
    
    for i, robot in enumerate(env.robots):
        robot_name = robot.name
        if "UR5e" in robot_name:
            if hasattr(robot, 'set_robot_joint_positions'):
                robot.set_robot_joint_positions(ur5e_joint_positions)
                if hasattr(robot, 'composite_controller'):
                    for part_name, part_controller in robot.composite_controller.part_controllers.items():
                        if hasattr(part_controller, 'update'):
                            part_controller.update(force=True)
        elif "Panda" in robot_name:
            if hasattr(robot, 'set_robot_joint_positions'):
                robot.set_robot_joint_positions(panda_joint_positions)
                if hasattr(robot, 'composite_controller'):
                    for part_name, part_controller in robot.composite_controller.part_controllers.items():
                        if hasattr(part_controller, 'update'):
                            part_controller.update(force=True)

    task_completion_hold_count = -1
    device.start_control()  # 这会触发手部校准和虚拟base校准

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # 创建文件用于保存env_action
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    action_log_file = f"env_action_log_absolute_{timestamp}.txt"
    timestep_count = 0

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Get the newest action (返回绝对位姿action)
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)
        print(f"input_ac_dict: {input_ac_dict}")

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break

        from copy import deepcopy

        # 为每个机器人构建action_dict
        robot_action_dicts = []
        
        # 调试：打印input_ac_dict的键名（仅第一次）
        if timestep_count == 0:
            print(f"Debug: input_ac_dict keys: {list(input_ac_dict.keys())}")
            for robot_idx, robot in enumerate(env.robots):
                print(f"Debug: Robot {robot_idx} ({robot.name}) arms: {robot.arms}")
                for arm in robot.arms:
                    print(f"  - arm '{arm}' gripper dof: {robot.gripper[arm].dof}")
                    # 检查控制器类型
                    if isinstance(robot.composite_controller, WholeBody):
                        print(f"  - controller type: WholeBody")
                    else:
                        controller = robot.part_controllers[arm]
                        print(f"  - controller type: {controller.name}, input_type: {controller.input_type}")
        
        # VisionProAbsolute返回的action key格式：left_abs, right_abs, left_gripper, right_gripper
        # 需要映射到机器人的实际arm名称
        visionpro_arm_mapping = ['left', 'right']
        
        for robot_idx, robot in enumerate(env.robots):
            robot_action_dict = deepcopy(all_prev_gripper_actions[robot_idx])
            
            # 根据机器人索引确定VisionPro返回的arm key
            visionpro_arm_key = visionpro_arm_mapping[robot_idx] if robot_idx < len(visionpro_arm_mapping) else robot.arms[0]
            
            # 检查input_ac_dict中是否有这个机器人的arm的action
            for arm in robot.arms:
                visionpro_arm_abs_key = f"{visionpro_arm_key}_abs"
                visionpro_gripper_key = f"{visionpro_arm_key}_gripper"
                
                # 检查是否有绝对位姿action
                if visionpro_arm_abs_key in input_ac_dict:
                    # 获取控制器类型
                    if isinstance(robot.composite_controller, WholeBody):
                        controller_input_type = robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = robot.part_controllers[arm].input_type
                    
                    # 对于绝对位姿控制，需要确保控制器支持absolute input_type
                    if controller_input_type == "absolute":
                        robot_action_dict[arm] = input_ac_dict[visionpro_arm_abs_key]
                    else:
                        # 如果控制器不支持absolute，打印警告并使用delta（虽然VisionProAbsolute不提供delta）
                        if timestep_count == 0:
                            print(f"Warning: Robot {robot_idx} ({robot.name}), arm '{arm}' "
                                  f"controller input_type is '{controller_input_type}', not 'absolute'. "
                                  f"Please configure controller to use 'absolute' input_type for absolute pose control.")
                        # 尝试使用delta（如果存在）
                        visionpro_arm_delta_key = f"{visionpro_arm_key}_delta"
                        if visionpro_arm_delta_key in input_ac_dict:
                            robot_action_dict[arm] = input_ac_dict[visionpro_arm_delta_key]
                        else:
                            # 如果没有delta，使用零动作
                            robot_action_dict[arm] = np.zeros(6)
                
                # 更新gripper action
                if visionpro_gripper_key in input_ac_dict:
                    gripper_action = input_ac_dict[visionpro_gripper_key]
                    expected_dof = robot.gripper[arm].dof
                    actual_dof = len(gripper_action) if hasattr(gripper_action, '__len__') else 1
                    
                    if actual_dof == expected_dof:
                        gripper_key = f"{arm}_gripper"
                        robot_action_dict[gripper_key] = gripper_action
                    else:
                        if timestep_count == 0:
                            print(f"Warning: Gripper action dimension mismatch for robot {robot_idx} ({robot.name}), "
                                  f"visionpro_key '{visionpro_gripper_key}', arm '{arm}'. "
                                  f"Expected {expected_dof}, got {actual_dof}.")
                        gripper_key = f"{arm}_gripper"
                        robot_action_dict[gripper_key] = np.zeros(expected_dof)
            
            robot_action_dicts.append(robot_action_dict)
        
        # 为每个机器人创建action vector
        env_action = [robot.create_action_vector(robot_action_dicts[i]) for i, robot in enumerate(env.robots)]
        env_action = np.concatenate(env_action)

        # 打印并保存env_action到文件
        timestep_count += 1
        print(f"Timestep {timestep_count} - env_action: {env_action}")
        print(f"  Shape: {env_action.shape}, Min: {np.min(env_action):.6f}, Max: {np.max(env_action):.6f}, Mean: {np.mean(env_action):.6f}")
        
        # 保存到文件
        with open(action_log_file, "a") as f:
            f.write(f"Timestep {timestep_count}:\n")
            f.write(f"  env_action: {env_action}\n")
            f.write(f"  Shape: {env_action.shape}, Min: {np.min(env_action):.6f}, Max: {np.max(env_action):.6f}, Mean: {np.mean(env_action):.6f}\n")
            f.write("\n")

        # 更新所有机器人的gripper actions
        for robot_idx, robot in enumerate(env.robots):
            for arm in robot.arms:
                gripper_key = f"{arm}_gripper"
                if gripper_key in robot_action_dicts[robot_idx]:
                    all_prev_gripper_actions[robot_idx][gripper_key] = robot_action_dicts[robot_idx][gripper_key]

        env.step(env_action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    收集演示数据并保存为hdf5文件（与原始脚本相同）
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        if success:
            print("Demonstration is successful and has been saved")
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}".format(now.hour, now.minute)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="TwoArmBoxCleanup")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["UR5eInspireDexRH", "Panda"],
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (not used in absolute pose mode)",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Must support 'absolute' input_type for absolute pose control.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="visionpro_absolute",
        choices=["visionpro_absolute"],
        help="Device to use: visionpro_absolute",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="Position sensitivity (not used in absolute pose mode)",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="Rotation sensitivity (not used in absolute pose mode)",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simulation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Not used in absolute pose mode, but kept for compatibility",
    )
    parser.add_argument(
        "--visionpro-pos-axis-map",
        type=str,
        default=None,
        help="VisionPro position axis mapping. Format: 'i,j,k' where i,j,k are indices 0-2.",
    )
    parser.add_argument(
        "--visionpro-pos-axis-signs",
        type=str,
        default=None,
        help="VisionPro position axis signs. Format: 'sx,sy,sz' where each is 1 or -1.",
    )
    parser.add_argument(
        "--visionpro-enable-coord-debug",
        action="store_true",
        help="Enable coordinate transformation debugging output for VisionPro device.",
    )
    parser.add_argument(
        "--virtual-base-calibration-frames",
        type=int,
        default=50,
        help="Number of frames to collect for virtual base calibration",
    )
    args = parser.parse_args()

    # Convert robots to list if it's a single string
    if isinstance(args.robots, str):
        args.robots = [args.robots]

    # Get controller config(s)
    if len(args.robots) > 1 and args.robots[0] != args.robots[1]:
        controller_configs = [
            load_composite_controller_config(
                controller=args.controller,
                robot=args.robots[0],
            ),
            load_composite_controller_config(
                controller=args.controller,
                robot=args.robots[1],
            ),
        ]
    else:
        controller_configs = load_composite_controller_config(
            controller=args.controller,
            robot=args.robots[0],
        )

    # 检查控制器配置是否支持absolute input_type
    def check_controller_config(config):
        if isinstance(config, dict):
            if config.get("type") == "BASIC":
                body_parts = config.get("body_parts", {})
                arms = body_parts.get("arms", {})
                for arm_name, arm_config in arms.items():
                    if isinstance(arm_config, dict):
                        input_type = arm_config.get("input_type", "delta")
                        if input_type != "absolute":
                            print(f"Warning: Controller for arm '{arm_name}' has input_type='{input_type}', "
                                  f"but absolute pose control requires input_type='absolute'.")
                            print(f"  Consider modifying the controller config to use 'absolute' input_type.")
            elif config.get("type") in ["WHOLE_BODY_MINK_IK", "WHOLE_BODY_IK"]:
                print("Warning: Whole body controllers may not support absolute pose control directly.")
        elif isinstance(config, list):
            for c in config:
                check_controller_config(c)
    
    check_controller_config(controller_configs)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_configs,
    }

    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)

    env_info = json.dumps(config)

    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Initialize VisionProAbsolute device
    if args.device == "visionpro_absolute":
        try:
            from robosuite.devices.visionpro_absolute import VisionProAbsolute
            from avp_stream import VisionProStreamer
        except ImportError as e:
            raise ImportError(
                f"VisionProAbsolute device requires avp_stream module. Error: {e}\n"
                "Please install avp_stream or ensure it is available."
            )
        
        print("=" * 60)
        print("初始化Vision Pro绝对位姿控制设备")
        print("=" * 60)
        
        import os
        room_code = os.getenv("VISIONPRO_ROOM_CODE", "ABC-1234")
        if room_code == "ABC-1234":
            print("Warning: 使用默认room code 'ABC-1234'，请通过环境变量 VISIONPRO_ROOM_CODE 设置正确的room code")
        
        print(f"使用room code: {room_code}")
        print("=" * 60)
        
        room_code = "SHQB-7053"  # 可以根据需要修改
        streamer = VisionProStreamer(ip=room_code)
        streamer.start_webrtc()
        
        swap_hands = os.getenv("VISIONPRO_SWAP_HANDS", "False").lower() == "true"
        if swap_hands:
            print("Warning: 已启用左右手数据交换（swap_hands=True）")
        
        pos_axis_map = None
        if args.visionpro_pos_axis_map is not None:
            try:
                axis_indices = [int(x.strip()) for x in args.visionpro_pos_axis_map.split(',')]
                if len(axis_indices) != 3 or not all(i in [0, 1, 2] for i in axis_indices):
                    raise ValueError("Must have 3 indices, each in [0, 1, 2]")
                pos_axis_map = tuple(axis_indices)
                print(f"使用位置轴映射: {pos_axis_map}")
            except Exception as e:
                print(f"Warning: 无法解析 --visionpro-pos-axis-map 参数: {e}")
        
        pos_axis_signs = None
        if args.visionpro_pos_axis_signs is not None:
            try:
                signs = [int(x.strip()) for x in args.visionpro_pos_axis_signs.split(',')]
                if len(signs) != 3 or not all(s in [1, -1] for s in signs):
                    raise ValueError("Must have 3 signs, each in [1, -1]")
                pos_axis_signs = tuple(signs)
                print(f"使用位置轴符号: {pos_axis_signs}")
            except Exception as e:
                print(f"Warning: 无法解析 --visionpro-pos-axis-signs 参数: {e}")
        
        device = VisionProAbsolute(
            env=env,
            streamer=streamer,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            calibration_frames=100,
            virtual_base_calibration_frames=args.virtual_base_calibration_frames,
            swap_hands=swap_hands,
            pos_axis_map=pos_axis_map,
            pos_axis_signs=pos_axis_signs,
            enable_coordinate_debug=args.visionpro_enable_coord_debug,
        )
    else:
        raise Exception("Invalid device choice: choose 'visionpro_absolute'")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
