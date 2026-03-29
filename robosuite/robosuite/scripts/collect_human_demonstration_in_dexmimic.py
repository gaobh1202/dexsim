"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
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
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        max_fr (int): if specified, pause the simulation whenever simulation runs faster than max_fr
    """

    env.reset()

    # # set camera position
    # camera_modder = CameraModder(env.sim)
    # camera_pos = np.array([0.5, 0, 1.35 + 2])
    # camera_modder.set_pos('agentview', camera_pos)

    env.render()
    # env.reset()将机械臂移动到xml中定义的初始位置，但不一定满足遥操作初始位置的需求
    # 人为定义一个_move_to_initial_pos() 
    
    # ============================================================
    # 设置机器人的固定关节位置（用于确定初始位姿）
    # 参考 demo_assigned_action.py 中的实现方式
    # ============================================================
    # UR5e的关节顺序：shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    ur5e_joint_positions = np.array([np.pi, - 2, - 1.75, 0.7, np.pi / 2, np.pi])
    # ur5e_joint_positions = np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])  # UR5e默认初始位置
    
    # Panda的关节位置（7个关节）
    panda_joint_positions = np.array([0, 0.4, 0, -1.8, 0, 3.75, 0.8])
    # panda_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 零位置
    
    # 根据机器人类型设置关节位置
    for i, robot in enumerate(env.robots):
        # 使用robot.name获取机器人类型名称（如"UR5eInspireDexRH", "Panda", 等）
        robot_name = robot.name
        # 检查机器人类型并设置对应的关节位置
        if "UR5e" in robot_name:
            # UR5e机器人（包括UR5eInspireDexRH等变体）
            if hasattr(robot, 'set_robot_joint_positions'):
                robot.set_robot_joint_positions(ur5e_joint_positions)
                # 更新控制器状态，使其知道当前的末端执行器位置
                if hasattr(robot, 'composite_controller'):
                    for part_name, part_controller in robot.composite_controller.part_controllers.items():
                        if hasattr(part_controller, 'update'):
                            part_controller.update(force=True)
        elif "Panda" in robot_name:
            # Panda机器人（包括PandaDexRH等变体）
            if hasattr(robot, 'set_robot_joint_positions'):
                robot.set_robot_joint_positions(panda_joint_positions)
                # 更新控制器状态
                if hasattr(robot, 'composite_controller'):
                    for part_name, part_controller in robot.composite_controller.part_controllers.items():
                        if hasattr(part_controller, 'update'):
                            part_controller.update(force=True)
        # 可以根据需要添加其他机器人类型的配置 

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    for robot in env.robots:
        robot.print_action_info_dict()

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
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
    action_log_file = f"env_action_log_{timestamp}.txt"
    timestep_count = 0

    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        # active_robot = env.robots[device.active_robot]

        # Get the newest action
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)
        # print(f"input_ac_dict: {input_ac_dict}")

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            break
        # if timestep_count >= 30:
        #     break

        from copy import deepcopy

        # 为每个机器人构建action_dict
        # 这样可以同时处理多个机器人的action（例如VisionPro同时控制两个机器人）
        robot_action_dicts = []
        
        # 调试：打印input_ac_dict的键名（仅第一次）
        if timestep_count == 0:
            print(f"Debug: input_ac_dict keys: {list(input_ac_dict.keys())}")
            for robot_idx, robot in enumerate(env.robots):
                print(f"Debug: Robot {robot_idx} ({robot.name}) arms: {robot.arms}")  # robot.arms = 'right'
                for arm in robot.arms:
                    print(f"  - arm '{arm}' gripper dof: {robot.gripper[arm].dof}")
        
        # 如果VisionPro返回了多个机器人的action，需要按机器人索引匹配
        # VisionPro在多机器人情况下，第一个机器人用左手，第二个机器人用右手
        # 所以我们可以根据机器人索引来匹配action
        # 映射关系：robot_idx=0 -> 'left', robot_idx=1 -> 'right'
        visionpro_arm_mapping = ['right', 'left']  # 根据机器人索引映射到VisionPro的arm key
        
        for robot_idx, robot in enumerate(env.robots):
            robot_action_dict = deepcopy(all_prev_gripper_actions[robot_idx])
            
            # 根据机器人索引确定VisionPro返回的arm key（left或right）
            visionpro_arm_key = visionpro_arm_mapping[robot_idx] if robot_idx < len(visionpro_arm_mapping) else robot.arms[0]
            
            # 检查input_ac_dict中是否有这个机器人的arm的action
            for arm in robot.arms:
                # 使用VisionPro的arm key来查找action（left或right）
                visionpro_arm_delta_key = f"{visionpro_arm_key}_delta"
                visionpro_arm_abs_key = f"{visionpro_arm_key}_abs"  # ee absolute pose?
                visionpro_gripper_key = f"{visionpro_arm_key}_gripper"
                
                # 检查是否有这个arm的action（可能是delta或absolute）
                if visionpro_arm_delta_key in input_ac_dict or visionpro_arm_abs_key in input_ac_dict:
                    if isinstance(robot.composite_controller, WholeBody):
                        controller_input_type = robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = robot.part_controllers[arm].input_type

                    if controller_input_type == "delta":
                        if visionpro_arm_delta_key in input_ac_dict:
                            robot_action_dict[arm] = input_ac_dict[visionpro_arm_delta_key]
                    elif controller_input_type == "absolute":
                        if visionpro_arm_abs_key in input_ac_dict:
                            robot_action_dict[arm] = input_ac_dict[visionpro_arm_abs_key]
                
                # 更新gripper action（需要验证维度是否匹配）
                if visionpro_gripper_key in input_ac_dict:
                    gripper_action = input_ac_dict[visionpro_gripper_key]
                    expected_dof = robot.gripper[arm].dof
                    actual_dof = len(gripper_action) if hasattr(gripper_action, '__len__') else 1
                    
                    # 验证维度是否匹配
                    if actual_dof == expected_dof:
                        gripper_key = f"{arm}_gripper"  # 使用robot的实际arm名称作为key
                        robot_action_dict[gripper_key] = gripper_action
                    else:
                        # 维度不匹配
                        if timestep_count == 0:
                            print(f"Warning: Gripper action dimension mismatch for robot {robot_idx} ({robot.name}), "
                                  f"visionpro_key '{visionpro_gripper_key}', arm '{arm}'. "
                                  f"Expected {expected_dof}, got {actual_dof}.")
                        
                        # 如果维度不匹配，使用零动作
                        gripper_key = f"{arm}_gripper"
                        robot_action_dict[gripper_key] = np.zeros(expected_dof)
            
            robot_action_dicts.append(robot_action_dict)
        
        # 为每个机器人创建action vector
        env_action = [robot.create_action_vector(robot_action_dicts[i]) for i, robot in enumerate(env.robots)]
        env_action = np.concatenate(env_action)

        # 打印并保存env_action到文件
        timestep_count += 1
        # print(f"Timestep {timestep_count} - env_action: {env_action}")
        # print(f"  Shape: {env_action.shape}, Min: {np.min(env_action):.6f}, Max: {np.max(env_action):.6f}, Mean: {np.mean(env_action):.6f}")
        

        # 更新所有机器人的gripper actions（用于保持状态）
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
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

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
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

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

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
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
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="birdview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse", "dualsense", "mjgui", "visionpro", "vr"],
        help="Device to use for teleoperation: keyboard, spacemouse, dualsense, mjgui, visionpro, or vr",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=5.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale rotation user inputs",
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
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    parser.add_argument(
        "--goal_update_mode",
        type=str,
        default="target",
        choices=["target", "achieved"],
        help="Used by the device to get the arm's actions. The mode to update the goal in. Can be 'target' or 'achieved'. If 'target', the goal is updated based on the current target pose. "
        "If 'achieved', the goal is updated based on the current achieved state. "
        "We recommend using 'achieved' (and input_ref_frame='base') if collecting demonstrations with a mobile base robot.",
    )
    parser.add_argument(
        "--visionpro-pos-axis-map",
        type=str,
        default=None,
        help="VisionPro position axis mapping. Format: 'i,j,k' where i,j,k are indices 0-2. "
        "Example: '1,0,2' maps Vision Pro [y,x,z] to robot [x,y,z]. "
        "Default: '0,1,2' (no remapping).",
    )
    parser.add_argument(
        "--visionpro-pos-axis-signs",
        type=str,
        default=None,
        help="VisionPro position axis signs. Format: 'sx,sy,sz' where each is 1 or -1. "
        "Example: '1,1,-1' flips the z-axis. "
        "Default: '1,1,1' (no sign changes).",
    )
    parser.add_argument(
        "--visionpro-enable-coord-debug",
        action="store_true",
        help="Enable coordinate transformation debugging output for VisionPro device.",
    )
    args = parser.parse_args()

    # Convert robots to list if it's a single string
    if isinstance(args.robots, str):
        args.robots = [args.robots]

    # Get controller config(s) - handle different robots with different controllers
    if len(args.robots) > 1 and args.robots[0] != args.robots[1]:
        # If robots are different, load controller config for each robot
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
        # Check for WHOLE_BODY_MINK_IK or WHOLE_BODY_IK in any config
        for controller_config in controller_configs:
            if controller_config["type"] == "WHOLE_BODY_MINK_IK":
                # mink-specific import. requires installing mink
                from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
            if controller_config["type"] == "WHOLE_BODY_IK":
                assert len(args.robots) == 1, "Whole Body IK only supports one robot"
    else:
        # If robots are the same, use single controller config
        controller_configs = load_composite_controller_config(
            controller=args.controller,
            robot=args.robots[0],
        )
        if isinstance(controller_configs, dict) and controller_configs["type"] == "WHOLE_BODY_MINK_IK":
            # mink-specific import. requires installing mink
            from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK
        if isinstance(controller_configs, dict) and controller_configs["type"] == "WHOLE_BODY_IK":
            assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_configs,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
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

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "dualsense":
        from robosuite.devices import DualSense

        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        assert args.renderer == "mjviewer", "Mocap is only supported with the mjviewer renderer"
        from robosuite.devices.mjgui import MJGUI

        device = MJGUI(env=env)
    elif args.device == "visionpro":
        try:
            from robosuite.devices.visionpro import VisionPro
            from avp_stream import VisionProStreamer
        except ImportError as e:
            raise ImportError(
                f"VisionPro device requires avp_stream module. Error: {e}\n"
                "Please install avp_stream or ensure it is available."
            )
        
        # 初始化Vision Pro Streamer
        # 注意：需要根据实际情况设置room code
        print("=" * 60)
        print("初始化Vision Pro设备")
        print("=" * 60)
        print("请确保Vision Pro已连接并显示room code")
        
        # 从环境变量或参数获取room code，这里使用默认值
        import os
        room_code = os.getenv("VISIONPRO_ROOM_CODE", "ABC-1234")
        if room_code == "ABC-1234":
            print("Warning: 使用默认room code 'ABC-1234'，请通过环境变量 VISIONPRO_ROOM_CODE 设置正确的room code")
        
        print(f"使用room code: {room_code}")
        print("=" * 60)
        
        # room_code = "10.12.15.204"
        room_code = "SHQB-7053"
        streamer = VisionProStreamer(ip=room_code)
        # 配置视频（如果需要）
        # streamer.configure_video(device="/dev/video0", format="v4l2", size="1280x720", fps=30)
        streamer.start_webrtc()
        
        device = VisionPro(
            env=env,
            streamer=streamer,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            calibration_frames=50,  # 可以根据需要调整
        )
    elif args.device == "vr":
        # 未实现
        print("VR device not implemented")
    else:
        raise Exception("Invalid device choice: choose 'keyboard', 'spacemouse', 'dualsense', 'mjgui', 'visionpro', or 'vr'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.max_fr, args.goal_update_mode)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
