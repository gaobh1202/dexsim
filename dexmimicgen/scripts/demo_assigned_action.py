# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Run random actions in dexmimicgen environments.

Args:
    --env (str): Name of the environment to run (default: "TwoArmThreading").
    --render (bool): Whether to render the environment.

Example usage:
    python script.py --env TwoArmPouring --render
"""

import argparse

import numpy as np
import robosuite
from robosuite import load_composite_controller_config
from robosuite.utils.mjmod import CameraModder
import robosuite.utils.transform_utils as T

# IMPORTANT: you need to import the package to register the environments
import dexmimicgen

# 导入MuJoCo viewer相关模块用于绘制坐标系
try:
    from mujoco import viewer
    import mujoco
    MUJOCO_VIEWER_AVAILABLE = True
except ImportError:
    MUJOCO_VIEWER_AVAILABLE = False
    print("警告: MuJoCo viewer不可用，无法绘制坐标系")

ENV_ROBOTS = {
    "TwoArmThreading": ["Panda", "Panda"],  # 修改：将第二个机器人改为 UR5e; UR5eDexRH
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["UR5eInspireDexRH", "Panda"], # ["PandaDexRH", "PandaDexLH"]
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
}

# 定义每个环境下每个机器人的动作空间维度（arm和gripper）
# 格式: {环境名: [{"arm": arm维度, "gripper": gripper维度}, ...]}
ENV_ROBOTS_ACTION_SPACE = {
    "TwoArmThreading": [
        {"arm": 7, "gripper": 1},  # 第一个机器人
        {"arm": 7, "gripper": 1},  # 第二个机器人
    ],
    "TwoArmThreePieceAssembly": [
        {"arm": 7, "gripper": 1},
        {"arm": 7, "gripper": 1},
    ],
    "TwoArmTransport": [
        {"arm": 7, "gripper": 1},
        {"arm": 7, "gripper": 1},
    ],
    "TwoArmLiftTray": [
        {"arm": 7, "gripper": 6},
        {"arm": 7, "gripper": 6},
    ],
    "TwoArmBoxCleanup": [
        {"arm": 6, "gripper": 6},
        {"arm": 6, "gripper": 1},
    ],
    "TwoArmDrawerCleanup": [
        {"arm": 7, "gripper": 6},
        {"arm": 7, "gripper": 6},
    ],

}


def draw_coordinate_frame(scene, pos, rot_mat, axis_length=1):
    """
    在MuJoCo scene中绘制坐标系（三个轴：X红色，Y绿色，Z蓝色）
    使用mjv_addGeoms函数来添加geoms到scene中
    
    Args:
        scene: MuJoCo scene对象
        pos: 坐标系原点位置 (3,)
        rot_mat: 旋转矩阵 (3, 3)，表示坐标系的方向
        axis_length: 坐标轴长度（米）
    """
    if not MUJOCO_VIEWER_AVAILABLE or scene is None:
        return
    
    # 定义三个轴的方向和颜色（在局部坐标系中）
    # X轴：红色，Y轴：绿色，Z轴：蓝色
    axes = [
        (np.array([1, 0, 0]), np.array([1, 0, 0, 1])),  # X轴：红色
        (np.array([0, 1, 0]), np.array([0, 1, 0, 1])),  # Y轴：绿色
        (np.array([0, 0, 1]), np.array([0, 0, 1, 1])),  # Z轴：蓝色
    ]
    
    # 将局部坐标系的轴转换到世界坐标系并绘制
    for axis_dir, color in axes:
        # 在世界坐标系中的方向
        world_dir = rot_mat @ axis_dir
        world_dir = world_dir / np.linalg.norm(world_dir)  # 归一化
        # 轴的另一端点
        end_pos = pos + world_dir * axis_length
        
        # 使用MuJoCo的mjvGeom来绘制线条
        try:
            if hasattr(scene, 'ngeom') and scene.ngeom < scene.maxgeom:
                ngeom = scene.ngeom
                geom = scene.geoms[ngeom]
                
                # 使用cylinder类型来绘制坐标轴（更可见）
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                
                # 设置圆柱体的尺寸（更粗一些以便观察）
                radius = max(axis_length * 0.02, 0.01)  # 半径，至少0.01m
                geom.size[0] = radius  # 半径
                geom.size[1] = radius  # 半径（圆柱体需要两个半径）
                geom.size[2] = axis_length * 0.5   # 高度的一半
                
                # 设置位置（线段的中点）
                geom.pos[:] = (pos + end_pos) / 2
                
                # 设置旋转矩阵，使圆柱体指向world_dir方向
                # 创建一个旋转矩阵，使z轴指向world_dir
                z_axis = world_dir
                # 选择一个垂直于z_axis的x_axis
                if abs(z_axis[2]) < 0.9:
                    x_axis = np.array([0, 0, 1])
                else:
                    x_axis = np.array([1, 0, 0])
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)
                
                # 构建旋转矩阵
                rot_arrow = np.column_stack([x_axis, y_axis, z_axis])
                # geom.mat是3x3矩阵视图，需要直接赋值3x3矩阵
                # 根据错误信息，geom.mat期望的是shape (3,3)，直接赋值3x3矩阵
                geom.mat[:] = rot_arrow
                
                # 设置颜色
                geom.rgba[:] = color
                
                # 增加geom计数
                scene.ngeom += 1
        except Exception as e:
            # 如果绘制失败，打印错误以便调试（仅第一次）
            if not hasattr(draw_coordinate_frame, '_error_printed'):
                print(f"绘制坐标系时出错: {e}")
                draw_coordinate_frame._error_printed = True


def visualize_robot_frames(env, viewer_obj=None):
    """
    可视化所有机器人的base坐标系和末端坐标系
    
    Args:
        env: robosuite环境对象
        viewer_obj: MuJoCo viewer对象（如果使用mjviewer渲染器）
    """
    if not MUJOCO_VIEWER_AVAILABLE:
        return
    
    # 获取viewer对象和scene
    # MuJoCo viewer的Handle对象有user_scn属性，我们可以用它来添加坐标系
    scene = None
    viewer_handle = None
    
    if viewer_obj is not None:
        viewer_handle = viewer_obj
        # 尝试访问user_scn（用户自定义scene）
        if hasattr(viewer_obj, 'user_scn'):
            scene = viewer_obj.user_scn
        elif hasattr(viewer_obj, 'scene'):
            scene = viewer_obj.scene
    elif hasattr(env, 'viewer'):
        if hasattr(env.viewer, 'viewer') and env.viewer.viewer is not None:
            viewer_handle = env.viewer.viewer
            # MuJoCo viewer的Handle对象有user_scn属性
            if hasattr(env.viewer.viewer, 'user_scn'):
                scene = env.viewer.viewer.user_scn
            elif hasattr(env.viewer.viewer, 'scene'):
                scene = env.viewer.viewer.scene
        elif hasattr(env.viewer, 'scene'):
            scene = env.viewer.scene

    if scene is None:
        # 如果user_scn也不存在，尝试创建它
        if hasattr(env, 'viewer') and hasattr(env.viewer, 'viewer') and env.viewer.viewer is not None:
            try:
                # 获取model
                model = env.sim.model._model
                # 创建user_scn
                env.viewer.viewer.user_scn = mujoco.MjvScene(model, maxgeom=10000)
                scene = env.viewer.viewer.user_scn
                if not hasattr(visualize_robot_frames, '_user_scn_created'):
                    print("创建了user_scn用于绘制坐标系")
                    visualize_robot_frames._user_scn_created = True
            except Exception as e:
                if not hasattr(visualize_robot_frames, '_error_printed'):
                    print(f"创建user_scn时出错: {e}")
                    visualize_robot_frames._error_printed = True
                return
        else:
            return
        
    # 遍历所有机器人
    for robot_idx, robot in enumerate(env.robots):
        # 1. 绘制base坐标系
        base_pos = np.array(env.sim.data.get_body_xpos(robot.robot_model.root_body))
        base_rot = np.array(env.sim.data.get_body_xmat(robot.robot_model.root_body)).reshape(3, 3)
        
        # 绘制base坐标系（使用稍长的轴以便区分）
        draw_coordinate_frame(scene, base_pos, base_rot, axis_length=0.5)
        
        # 2. 绘制每个arm的末端执行器坐标系
        for arm in robot.arms:
            # 获取末端执行器位置
            eef_site_id = robot.eef_site_id[arm]
            eef_pos = np.array(env.sim.data.site_xpos[eef_site_id])
            
            # 获取末端执行器旋转矩阵
            # 优先使用grip_site（如果存在），否则使用eef_site
            try:
                pf = robot.gripper[arm].naming_prefix
                grip_site_name = f"{pf}grip_site"
                grip_site_id = env.sim.model.site_name2id(grip_site_name)
                eef_rot = np.array(env.sim.data.site_xmat[grip_site_id]).reshape(3, 3)
            except:
                # 如果grip_site不存在，使用eef_site
                eef_rot = np.array(env.sim.data.site_xmat[eef_site_id]).reshape(3, 3)
            
            # 绘制末端执行器坐标系
            draw_coordinate_frame(scene, eef_pos, eef_rot, axis_length=0.5)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="TwoArmBoxCleanup",
        help="Name of the environment to run",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the environment",
    )

    args = parser.parse_args()

    assert args.env in ENV_ROBOTS, f"Environment {args.env} not found!"

    # Create dict to hold options that will be passed to env creation call
    robots = ENV_ROBOTS[args.env]
    # 为每个机器人加载对应的控制器配置
    if len(robots) > 1 and robots[0] != robots[1]:
        # 如果两个机器人不同，为每个机器人分别加载控制器配置
        controller_configs = [
            load_composite_controller_config(robot=robots[0]),
            load_composite_controller_config(robot=robots[1]),
        ]
    else:
        # 如果机器人相同，使用第一个机器人的配置
        controller_configs = load_composite_controller_config(robot=robots[0])
    
    env_kwargs = {
        "env_name": args.env,
        "robots": robots,
        "controller_configs": controller_configs,
        "has_renderer": args.render,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": True,
        "control_freq": 20,
        "renderer": "mjviewer",  # 使用mjviewer以便绘制坐标系
    }

    # initialize the task
    env = robosuite.make(
        **env_kwargs,
    )
    env.reset()   
    
    # ============================================================
    # 调整相机位置、角度和视野范围（在环境创建后修改）
    # ============================================================
    if args.render:
        # 方法1: 使用CameraModder修改相机参数（推荐）
        # 这样可以修改相机位置、角度和FOV（视野角度）
        camera_modder = CameraModder(sim=env.sim, random_state=None)
        
        # 获取当前使用的相机名称
        # 先检查env_kwargs中是否指定了render_camera
        camera_name = env_kwargs.get("render_camera", "frontview")
        
        # 如果OpenCVViewer已经设置了相机，使用viewer中的相机名称
        if hasattr(env, 'viewer') and hasattr(env.viewer, 'camera_names'):
            if hasattr(env.viewer.camera_names, '__iter__') and len(env.viewer.camera_names) > 0:
                camera_name = env.viewer.camera_names[0]
        
        # 检查相机是否存在
        try:
            camera_id = env.sim.model.camera_name2id(camera_name)
            print(f"使用相机: '{camera_name}' (ID: {camera_id})")
        except:
            # 如果指定的相机不存在，尝试使用agentview或frontview
            for default_cam in ["agentview", "frontview"]:
                try:
                    camera_id = env.sim.model.camera_name2id(default_cam)
                    camera_name = default_cam
                    print(f"相机 '{camera_name}' 不存在，切换到: '{default_cam}' (ID: {camera_id})")
                    env.viewer.set_camera(camera_name=default_cam)
                    break
                except:
                    continue
        
        try:
            # ============================================================
            # 相机参数调整说明：
            # ============================================================
            # 1. camera_pos: [x, y, z] - 相机在世界坐标系中的位置
            #    - x: 前后方向（正值为前，负值为后）
            #    - y: 左右方向（正值为左，负值为右）
            #    - z: 上下方向（正值为上，负值为下）
            #
            # 2. camera_quat: [w, x, y, z] - 相机的四元数旋转（wxyz格式）
            #    - 控制相机朝向
            #    - 可以使用工具计算或手动调整
            #
            # 3. camera_fovy: 垂直视野角度（度）
            #    - 越大视野越广，能看到的范围越大
            #    - 典型值: 45-90度，可以根据需要调整
            # ============================================================
            
            # 示例配置0：从正面观察双臂机器人（适合TwoArmThreading）
            camera_pos = np.array([1, 0, 2.5])  # 相机位置：前方1m，高度2.5m
            camera_quat = np.array([0.653, 0.271, 0.271, 0.653])  # 更向斜下方（wxyz格式）
            camera_fovy = 65.0  # 视野角度（度），增大以获得更广的视野
            
            # 示例配置1：从斜上方观察双臂机器人（适合TwoArmBoxCleanup）
            # camera_pos = np.array([1, 1, 2.5])  # 相机位置：前方1m，高度2.5m
            # camera_quat = np.array([0.554, 0.191, 0.461, 0.665])  # 更向斜下方（wxyz格式）
            # camera_fovy = 65.0  # 视野角度（度），增大以获得更广的视野

            
            # 修改相机位置
            camera_modder.set_pos(camera_name, camera_pos)
            # 修改相机角度
            camera_modder.set_quat(camera_name, camera_quat)
            # 修改视野角度（FOV）
            camera_modder.set_fovy(camera_name, camera_fovy)
            
            print(f"\n已调整相机 '{camera_name}' 的参数:")
            print(f"  位置: {camera_pos}")
            print(f"  四元数: {camera_quat}")
            print(f"  视野角度(FOV): {camera_fovy}度")
            print(f"\n提示: 如需调整视角，可以修改上面的camera_pos、camera_quat和camera_fovy参数")
            
        except Exception as e:
            print(f"警告: 无法修改相机参数: {e}")
            print("将使用默认相机设置")
        
        env.render()
        
        # 方法2: 切换使用不同的相机（如果有多个相机可用）
        # 例如，TwoArmBoxCleanup环境可能有"agentview"、"frontview"等相机
        # 你可以通过以下方式切换：
        # env.viewer.set_camera(camera_name="agentview")  # 切换到agentview相机
        # env.viewer.set_camera(camera_name="frontview")  # 切换到frontview相机

    # ============================================================
    # 设置机器人的固定关节位置（用于确定初始位姿）
    # ============================================================
    # UR5e的关节顺序：shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    # 你可以修改这些值来测试不同的关节配置
    ur5e_joint_positions = np.array([np.pi, - np.pi / 2, - 3 *np.pi / 4, np.pi / 4, np.pi / 2, np.pi])
    # ur5e_joint_positions = np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -1.991])  # UR5e默认初始位置
    
    # Panda的关节位置（7个关节）
    # panda_joint_positions = np.array([0, 0, 0.00, -np.pi / 2.0, 0.00, np.pi, np.pi / 3 - 0.2])
    panda_joint_positions = np.array([0, 0, 0, -3 * np.pi / 4.0, 0, np.pi * 1.1, np.pi / 3 - 0.2])
    # panda_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 零位置
    
    # 设置第一个机器人（left robot，实际是UR5e）的关节位置
    if len(env.robots) > 0:
        env.robots[0].set_robot_joint_positions(ur5e_joint_positions)
        # 更新控制器状态，使其知道当前的末端执行器位置
        # 对于composite controller，需要更新每个part controller
        if hasattr(env.robots[0], 'composite_controller'):
            for part_name, part_controller in env.robots[0].composite_controller.part_controllers.items():
                if hasattr(part_controller, 'update'):
                    part_controller.update(force=True)
    
    # 设置第二个机器人（right robot，实际是Panda）的关节位置
    if len(env.robots) > 1:
        env.robots[1].set_robot_joint_positions(panda_joint_positions)
        # 更新控制器状态
        if hasattr(env.robots[1], 'composite_controller'):
            for part_name, part_controller in env.robots[1].composite_controller.part_controllers.items():
                if hasattr(part_controller, 'update'):
                    part_controller.update(force=True)
    
    # 根据ENV_ROBOTS_ACTION_SPACE初始化action向量
    assert args.env in ENV_ROBOTS_ACTION_SPACE, f"Environment {args.env} not found in ENV_ROBOTS_ACTION_SPACE!"
    action_space_config = ENV_ROBOTS_ACTION_SPACE[args.env]
    left_arm_action_dim = action_space_config[0]["arm"]
    right_arm_action_dim = action_space_config[1]["arm"]
    left_gripper_action_dim = action_space_config[0]["gripper"]
    right_gripper_action_dim = action_space_config[1]["gripper"]

    # ============================================================
    # 设置action为零增量，保持当前关节位置
    # UR5e使用OSC_POSE控制器，input_type="delta"
    # action格式：[dx, dy, dz, dax, day, daz] - 末端执行器位置/姿态增量
    # ============================================================
    left_arm_action = np.zeros(left_arm_action_dim)  # UR5e: 6维（OSC_POSE）
    right_arm_action = np.zeros(right_arm_action_dim)  # Panda: 根据配置可能是6或7维

    # 设置gripper动作
    left_gripper_action = np.array([-1.5, -1.5, -1.5, -1.5, -3, 3])  # InspireRighthand open
    right_gripper_action = np.array([0.0])  # Panda gripper open

    # 拼接所有机器人的action
    action = np.concatenate([left_arm_action, left_gripper_action, right_arm_action, right_gripper_action])
    
    # 验证拼接后的action维度与env.action_spec[0].shape一致
    expected_shape = env.action_spec[0].shape
    assert action.shape == expected_shape, (
        f"Action shape mismatch! Expected {expected_shape}, got {action.shape}. "
        f"Total action dim: {action.shape[0]}, Expected: {expected_shape[0]}"
    )
    

    # 设置viewer的render callback来绘制坐标系（如果使用mjviewer）
    render_callback_set = False
    if args.render and hasattr(env, 'viewer') and hasattr(env.viewer, 'viewer'):
        # 确保viewer已初始化
        env.viewer.update()
        
        # 尝试设置render callback
        if env.viewer.viewer is not None:
            try:
                # MuJoCo viewer可能支持render callback
                # 但我们需要检查viewer是否有这个功能
                # 实际上，MuJoCo viewer的scene是在sync时构建的
                # 我们需要在sync之后、但在渲染之前添加geoms
                render_callback_set = True
            except:
                pass

    # do visualization
    for i in range(1000):
        # 使用指定的action（可以在这里修改action的值）
        obs, reward, done, _ = env.step(action)
        if args.render:
            # 渲染环境（这会调用viewer.update()和viewer.sync()）
            env.render()
            
            # 在render之后添加坐标系到user_scn
            # user_scn是MuJoCo viewer提供的用户自定义scene，用于添加自定义可视化元素
            try:
                if hasattr(env, 'viewer') and hasattr(env.viewer, 'viewer') and env.viewer.viewer is not None:
                    # 确保user_scn已创建
                    if not hasattr(env.viewer.viewer, 'user_scn') or env.viewer.viewer.user_scn is None:
                        model = env.sim.model._model
                        env.viewer.viewer.user_scn = mujoco.MjvScene(model, maxgeom=10000)
                    
                    # 使用user_scn来绘制坐标系
                    # user_scn会在viewer渲染时自动显示
                    visualize_robot_frames(env, env.viewer.viewer)
            except Exception as e:
                # 如果绘制失败，不影响主程序运行
                if i == 0:  # 只在第一次失败时打印警告
                    print(f"警告: 无法绘制坐标系: {e}")
                    import traceback
                    traceback.print_exc()