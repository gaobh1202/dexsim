"""
加载一个双臂环境，可视化坐标轴方向
并加载初始化后得到的vision pro head/left wrist/right wrist 坐标系

用于测试和确定坐标系变换方式
"""

import os
import time
from tkinter import TRUE
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.utils import transform_utils as T
from scipy.spatial.transform import Rotation as R

# IMPORTANT: import dexmimicgen to register the environments
import dexmimicgen

# 导入VisionPro设备
try:
    from robosuite.devices.visionpro import VisionPro
    from avp_stream import VisionProStreamer
except ImportError as e:
    raise ImportError(
        f"VisionPro设备需要avp_stream模块。错误: {e}\n"
        "请安装avp_stream或确保其可用。"
    )

# 导入MuJoCo viewer相关模块用于绘制坐标系
try:
    from mujoco import viewer
    import mujoco
    MUJOCO_VIEWER_AVAILABLE = True
except ImportError:
    MUJOCO_VIEWER_AVAILABLE = False
    print("警告: MuJoCo viewer不可用，无法绘制坐标系")


ENV_ROBOTS = {
    "TwoArmBoxCleanup": ["UR5eInspireDexRH", "Panda"], # ["PandaDexRH", "PandaDexLH"]
    "Lift": ["Panda"]
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
        print(f"base_pos {robot_idx}: {base_pos}")
        base_rot = np.array(env.sim.data.get_body_xmat(robot.robot_model.root_body)).reshape(3, 3)
        
        # 绘制base坐标系（使用稍长的轴以便区分）
        draw_coordinate_frame(scene, base_pos, base_rot, axis_length=0.5)
        
        # 2. 绘制每个arm的末端执行器坐标系
        for arm in robot.arms:
            # 获取末端执行器位置
            eef_site_id = robot.eef_site_id[arm]
            eef_pos = np.array(env.sim.data.site_xpos[eef_site_id])
            eef_rot = np.array(env.sim.data.site_xmat[eef_site_id]).reshape(3, 3)
            draw_coordinate_frame(scene, eef_pos, eef_rot, axis_length=0.5)
            
            # # 获取末端执行器旋转矩阵
            # # 优先使用grip_site（如果存在），否则使用eef_site
            # try:
            #     pf = robot.gripper[arm].naming_prefix
            #     grip_site_name = f"{pf}grip_site"
            #     grip_site_id = env.sim.model.site_name2id(grip_site_name)
            #     eef_rot = np.array(env.sim.data.site_xmat[grip_site_id]).reshape(3, 3)
            # except:
            #     # 如果grip_site不存在，使用eef_site
            #     eef_rot = np.array(env.sim.data.site_xmat[eef_site_id]).reshape(3, 3)
            
            # # 绘制末端执行器坐标系
            # draw_coordinate_frame(scene, eef_pos, eef_rot, axis_length=0.5)



def transform_wrist_poses_to_head_frame(head_pose_avg, left_wrist_pose_avg, right_wrist_pose_avg):
    """
    将left_wrist和right_wrist的姿态变换到head_pose_avg坐标系下
    
    假设head_pose_avg、left_wrist_pose_avg、right_wrist_pose_avg都在同一个坐标系中
    （例如Vision Pro的世界坐标系），将它们变换到head_pose_avg坐标系中。
    
    变换公式：
    - wrist_in_head = head_pose_avg^-1 @ wrist_pose_world
    
    Args:
        head_pose_avg: head的平均姿态（4x4矩阵），作为目标坐标系
        left_wrist_pose_avg: left wrist的平均姿态（4x4矩阵），在原坐标系中
        right_wrist_pose_avg: right wrist的平均姿态（4x4矩阵），在原坐标系中
    
    Returns:
        transformed_left_wrist_pose: 变换后的left wrist姿态（4x4矩阵），在head坐标系中
        transformed_right_wrist_pose: 变换后的right wrist姿态（4x4矩阵），在head坐标系中
    """
    transformed_left_wrist_pose = np.linalg.inv(head_pose_avg) @ left_wrist_pose_avg
    transformed_right_wrist_pose = np.linalg.inv(head_pose_avg) @ right_wrist_pose_avg

    sim_head = np.eye(4)
    sim_head[:3, 3] = [-0.56, 0.0, 1.1]
    ur_right_wrist = sim_head @ transformed_right_wrist_pose
    panda_left_wrist = sim_head @ transformed_left_wrist_pose
    
    return transformed_left_wrist_pose, transformed_right_wrist_pose, ur_right_wrist, panda_left_wrist


def visualize_visionpro_poses(env, viewer_obj, head_pose_avg, left_wrist_pose_avg, right_wrist_pose_avg, 
                               apply_transform=True):
    """
    可视化VisionPro的head、left_wrist、right_wrist姿态
    
    Args:
        env: robosuite环境对象
        viewer_obj: MuJoCo viewer对象
        head_pose_avg: head的平均姿态（4x4矩阵）
        left_wrist_pose_avg: left wrist的平均姿态（4x4矩阵）
        right_wrist_pose_avg: right wrist的平均姿态（4x4矩阵）
        apply_transform: 是否将wrist姿态变换到与head一致的方向（默认True）
    """
    if not MUJOCO_VIEWER_AVAILABLE:
        return
    
    # 获取scene对象（与visualize_robot_frames相同的逻辑）
    scene = None
    
    if viewer_obj is not None:
        if hasattr(viewer_obj, 'user_scn'):
            scene = viewer_obj.user_scn
        elif hasattr(viewer_obj, 'scene'):
            scene = viewer_obj.scene
    elif hasattr(env, 'viewer'):
        if hasattr(env.viewer, 'viewer') and env.viewer.viewer is not None:
            if hasattr(env.viewer.viewer, 'user_scn'):
                scene = env.viewer.viewer.user_scn
            elif hasattr(env.viewer.viewer, 'scene'):
                scene = env.viewer.viewer.scene
        elif hasattr(env.viewer, 'scene'):
            scene = env.viewer.scene
    
    if scene is None:
        return
    
    # 如果需要应用变换，将wrist姿态变换到与head一致的方向
    if apply_transform:
        transformed_left_wrist_pose, transformed_right_wrist_pose, ur_right_wrist, panda_left_wrist = transform_wrist_poses_to_head_frame(
            head_pose_avg, left_wrist_pose_avg, right_wrist_pose_avg
        )
        # print(f"transformed_left_wrist_pose: {transformed_left_wrist_pose}")
        # print(f"transformed_right_wrist_pose: {transformed_right_wrist_pose}")
        # 只在第一次打印确认信息
        if not hasattr(visualize_visionpro_poses, '_transform_applied'):
            print("已应用坐标变换：将left_wrist和right_wrist的姿态变换到与head一致的方向")
            visualize_visionpro_poses._transform_applied = True
    else:
        transformed_left_wrist_pose = left_wrist_pose_avg
        transformed_right_wrist_pose = right_wrist_pose_avg
    
    # 绘制head pose（使用稍大的轴长度以便区分）
    if head_pose_avg is not None:
        head_pos = head_pose_avg[:3, 3]
        head_rot = head_pose_avg[:3, :3]
        # 使用稍大的轴长度来区分VisionPro的pose
        draw_coordinate_frame(scene, head_pos, head_rot, axis_length=0.3)
    
    # 绘制变换后的left wrist pose
    if transformed_left_wrist_pose is not None:
        left_wrist_pos = transformed_left_wrist_pose[:3, 3]
        left_wrist_rot = transformed_left_wrist_pose[:3, :3]
        # draw_coordinate_frame(scene, left_wrist_pos, left_wrist_rot, axis_length=0.3)
    
    # 绘制变换后的right wrist pose
    if transformed_right_wrist_pose is not None:
        right_wrist_pos = transformed_right_wrist_pose[:3, 3]
        right_wrist_rot = transformed_right_wrist_pose[:3, :3]
        # draw_coordinate_frame(scene, right_wrist_pos, right_wrist_rot, axis_length=0.3)
    

    if ur_right_wrist is not None:
        print(f"ur_right_wrist: {ur_right_wrist}")
        ur_right_wrist_pos = ur_right_wrist[:3, 3]
        ur_right_wrist_rot = ur_right_wrist[:3, :3]
        draw_coordinate_frame(scene, ur_right_wrist_pos, ur_right_wrist_rot, axis_length=0.3)
    
    # 绘制变换后的panda_left_wrist pose
    if panda_left_wrist is not None:
        print(f"panda_left_wrist: {panda_left_wrist}")
        panda_left_wrist_pos = panda_left_wrist[:3, 3]
        panda_left_wrist_rot = panda_left_wrist[:3, :3]
        draw_coordinate_frame(scene, panda_left_wrist_pos, panda_left_wrist_rot, axis_length=0.3)


def compute_avg_pose(pose_list):
    """
    辅助函数：计算一组 4x4 Pose 矩阵的平均值
    """
    if not pose_list:
        return None
    
    # 转换为 numpy 数组: (N, 4, 4)
    poses = np.array(pose_list)
    
    # 1. 计算位置平均 (Translation) -> (3,)
    # 取所有矩阵的第0-2行，第3列
    avg_pos = np.mean(poses[:, :3, 3], axis=0)
    
    # 2. 计算旋转平均 (Rotation) -> (3, 3)
    # 取所有矩阵的左上角 3x3
    rot_matrices = poses[:, :3, :3]
    
    # 使用 Scipy 处理旋转平均 (比直接平均矩阵更符合几何意义)
    # 这里使用简单的四元数平均法近似
    r = R.from_matrix(rot_matrices)
    avg_rot_mat = r.mean().as_matrix()
    
    # 3. 重新组装 4x4 矩阵
    avg_pose = np.eye(4)
    avg_pose[:3, :3] = avg_rot_mat
    avg_pose[:3, 3] = avg_pos
    
    return avg_pose


def vp_calibrate(streamer, calibration_frames=20):
    """
    初始化VisionPro设备；采集50帧数据，计算 head, left wrist, right wrist 的平均 Pose
    """
    print(f"Starting calibration... Please hold still for {calibration_frames} frames.")
    
    # 用于存储收集到的数据
    head_data = []
    left_data = []
    right_data = []

    frames_collected = 0
    
    # 循环采集数据
    while frames_collected < calibration_frames:
        try:
            data = streamer.get_latest()
            
            # 简单的有效性检查 (防止 VisionPro 还没准备好发来空数据)
            if data is None:
                time.sleep(0.01)
                continue
                
            # 提取 4x4 矩阵
            # 假设 data.head, data.left.wrist 已经是 numpy array
            head_data.append(data.head)
            left_data.append(data.left.wrist)
            right_data.append(data.right.wrist)
            
            frames_collected += 1
            
            # 稍微休眠一下，模拟帧率，避免在一个瞬间读取重复数据
            time.sleep(0.03) # 约 30Hz 采样
            
        except Exception as e:
            print(f"Warning: Failed to get Vision Pro data during calibration: {e}")
            time.sleep(0.1)
            continue

    print("Data collection complete. Computing averages...")

    # 计算平均 Pose
    head_pose_avg = compute_avg_pose(head_data)
    left_wrist_pose_avg = compute_avg_pose(left_data)
    right_wrist_pose_avg = compute_avg_pose(right_data)
    
    print("Calibration finished.")
    
    return head_pose_avg, left_wrist_pose_avg, right_wrist_pose_avg


def rotation_alone_axis(axis, angle, degrees=True):
    if degrees:
        angle = np.radians(angle)
    # 旋转矩阵4x4
    if axis == "x":
        return np.array([[1, 0, 0, 0],
                        [0, np.cos(angle), -np.sin(angle), 0],
                        [0, np.sin(angle), np.cos(angle), 0],
                        [0, 0, 0, 1]])
    elif axis == "y":
        return np.array([[np.cos(angle), 0, np.sin(angle), 0],
                        [0, 1, 0, 0],
                        [-np.sin(angle), 0, np.cos(angle), 0],
                        [0, 0, 0, 1]])
    elif axis == "z":
        return np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y' or 'z'")

def main():
    import argparse
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
        default=True,
        help="Whether to render the environment",
    )

    args = parser.parse_args()

    # ============================================
    # 1. 创建机器人环境
    # ============================================
    print("=" * 60)
    print("创建Panda机器人环境")
    print("=" * 60)

    # 设置环境机器人类型
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

    config = {
        "env_name": args.env,
        "robots": robots,
        "controller_configs": controller_configs,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.env:
        # 如果需要env_configuration，可以在这里添加
        # config["env_configuration"] = args.config
        pass

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer='mjviewer',  # mjviewer以便绘制坐标系
        has_offscreen_renderer=False,
        render_camera='frontview',
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    
    # 添加可视化包装器
    env = VisualizationWrapper(env)

    # ============================================
    # 2. 设置机器人初始位置
    # ============================================
    print("=" * 60)
    print("设置机器人初始位置")
    print("=" * 60)
    
    env.reset()
    env.render()
    
    # 设置机器人的初始关节位置
    # 这些位置使机器人处于一个便于控制的姿态
    ur5e_joint_positions = np.array([np.pi, - 2, - 1.75, 0.7, np.pi / 2, np.pi])
    panda_joint_positions = np.array([0, 0.4, 0, -1.8, 0, 3.75, 0.8])
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
    
    print("机器人已移动到初始位置")
    
    # ============================================
    # 3. 初始化VisionPro设备；设置50帧calibrating，得到一个head pose, left wrist pose, right wrist pose；用于可视化坐标轴
    # ============================================
    print("=" * 60)
    print("初始化VisionPro设备")
    print("=" * 60)
    
    # 从环境变量获取room code，或使用默认值
    room_code = os.getenv("VISIONPRO_ROOM_CODE", "SHQB-7053")
    print(f"使用room code: {room_code}")
    
    # 创建VisionPro Streamer
    streamer = VisionProStreamer(ip=room_code)
    streamer.start_webrtc()

    head_pose_avg, left_wrist_pose_avg_raw, right_wrist_pose_avg_raw = vp_calibrate(streamer)
    left_wrist_pose_avg = rotation_alone_axis("z", -90) @ left_wrist_pose_avg_raw
    right_wrist_pose_avg = rotation_alone_axis("z", -90) @ right_wrist_pose_avg_raw
    left_wrist_pose_avg = left_wrist_pose_avg @ rotation_alone_axis("y", 90) @ rotation_alone_axis("z", 180)
    right_wrist_pose_avg = right_wrist_pose_avg @ rotation_alone_axis("y", -90)

    # ============================================
    # 4. 可视化仿真环境，机器人设置0动作，可视化坐标轴方向
    # ============================================
    left_arm_action = np.zeros(6)  # UR5e: 6维（OSC_POSE） 表示以世界坐标系=机械臂base坐标系为参考，发生的位置或角度变化
    # OSC_POSE格式: [dx, dy, dz, rx, ry, rz]
    # 前3维: 位置增量（米）
    # 后3维: 旋转增量（axis-angle，弧度）
    
    # 示例1: 绕y轴旋转1度
    # left_arm_action[3] = np.pi / 180  # ry = 1度 = π/180 弧度

    right_arm_action = np.zeros(6)  # Panda: 根据配置可能是6或7维
    # right_arm_action[4] = np.pi / 180

    # 设置gripper动作
    left_gripper_action = np.array([-1.5, -1.5, -1.5, -1.5, -3, 3])  # InspireRighthand open
    right_gripper_action = np.array([0.0])  # Panda gripper open

    # 拼接所有机器人的action
    action = np.concatenate([left_arm_action, left_gripper_action, right_arm_action, right_gripper_action])
    
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
    while True:
        # 使用指定的action（可以在这里修改action的值）
        env.step(action)

        time.sleep(1)
        # data = streamer.get_latest()
        # if data is not None:
        #     head_pose_avg = data.head
        #     left_wrist_pose_avg = data.left.wrist
        #     right_wrist_pose_avg = data.right.wrist
        #     print(f"head_pose_avg: {head_pose_avg}")
        #     print(f"left_wrist_pose_avg: {left_wrist_pose_avg}")
        #     print(f"right_wrist_pose_avg: {right_wrist_pose_avg}")

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
                    
                    # 使用user_scn来绘制机器人坐标系
                    # user_scn会在viewer渲染时自动显示
                    visualize_robot_frames(env, env.viewer.viewer)
                    
                    # 绘制VisionPro的head、left_wrist、right_wrist姿态
                    visualize_visionpro_poses(
                        env, 
                        env.viewer.viewer, 
                        head_pose_avg, 
                        left_wrist_pose_avg, 
                        right_wrist_pose_avg
                    )
            except Exception as e:
                # 如果绘制失败，不影响主程序运行
                if i == 0:  # 只在第一次失败时打印警告
                    print(f"警告: 无法绘制坐标系: {e}")
                    import traceback
                    traceback.print_exc()
    env.close()
    
    # # 解析坐标变换参数（从环境变量或使用默认值）
    # # 位置轴映射：例如 "1,0,2" 表示 Vision Pro的 [y,x,z] 映射到机器人的 [x,y,z]
    # pos_axis_map_str = os.getenv("VISIONPRO_POS_AXIS_MAP", None)
    # pos_axis_map = None
    # if pos_axis_map_str is not None:
    #     try:
    #         axis_indices = [int(x.strip()) for x in pos_axis_map_str.split(',')]
    #         if len(axis_indices) == 3 and all(i in [0, 1, 2] for i in axis_indices):
    #             pos_axis_map = tuple(axis_indices)
    #             print(f"使用位置轴映射: {pos_axis_map}")
    #     except Exception as e:
    #         print(f"警告: 无法解析位置轴映射参数: {e}")
    
    # # 位置轴符号：例如 "1,1,-1" 表示反转z轴
    # pos_axis_signs_str = os.getenv("VISIONPRO_POS_AXIS_SIGNS", None)
    # pos_axis_signs = None
    # if pos_axis_signs_str is not None:
    #     try:
    #         signs = [int(x.strip()) for x in pos_axis_signs_str.split(',')]
    #         if len(signs) == 3 and all(s in [1, -1] for s in signs):
    #             pos_axis_signs = tuple(signs)
    #             print(f"使用位置轴符号: {pos_axis_signs}")
    #     except Exception as e:
    #         print(f"警告: 无法解析位置轴符号参数: {e}")
    
    # # 是否启用坐标调试
    # enable_coord_debug = os.getenv("VISIONPRO_ENABLE_COORD_DEBUG", "True").lower() == "true"
    
    # # 创建VisionPro设备
    # device = VisionPro(
    #     env=env,
    #     streamer=streamer,
    #     pos_sensitivity=1.0,
    #     rot_sensitivity=1.0,
    #     calibration_frames=50,  # 校准帧数
    #     swap_hands=False,  # 不交换左右手
    #     pos_axis_map=pos_axis_map,
    #     pos_axis_signs=pos_axis_signs,
    #     enable_coordinate_debug=enable_coord_debug,
    # )
    

    
    # ============================================
    # 4. 开始控制循环
    # ============================================
    # print("=" * 60)
    # print("开始VisionPro控制")
    # print("=" * 60)
    # print("提示：")
    # print("1. 系统将进行校准（采集50帧数据）")
    # print("2. 请保持初始姿态：手肘90度，放在身体两侧，手心朝下手指朝前")
    # print("3. 校准完成后，可以开始移动手部控制机器人")
    # print("4. 如果坐标方向不对，请调整环境变量：")
    # print("   - VISIONPRO_POS_AXIS_MAP: 轴映射（例如 '1,0,2'）")
    # print("   - VISIONPRO_POS_AXIS_SIGNS: 轴符号（例如 '1,1,-1'）")
    # print("   - VISIONPRO_ENABLE_COORD_DEBUG: 启用调试输出（'True'或'False'）")
    # print("=" * 60)
    
    # # 开始控制
    # device.start_control()
    
    # # 控制循环
    # timestep = 0
    # max_fr = 20  # 最大帧率
    
    # try:
    #     while True:
    #         start_time = time.time()
            
    #         # 获取动作
    #         action_dict = device.input2action(goal_update_mode="target")
            
    #         if action_dict is None:
    #             print("收到重置信号，退出控制循环")
    #             break
            
    #         # 构建环境动作
    #         # VisionPro已经处理了单机器人单臂的情况，直接使用返回的action_dict
    #         robot_action_dict = {}
            
    #         # 查找手臂动作（Panda通常是"right"）
    #         for arm in robot.arms:
    #             arm_delta_key = f"{arm}_delta"
    #             arm_abs_key = f"{arm}_abs"
    #             gripper_key = f"{arm}_gripper"
                
    #             # 优先使用delta动作（增量控制）
    #             if arm_delta_key in action_dict:
    #                 robot_action_dict[arm] = action_dict[arm_delta_key]
    #             elif arm_abs_key in action_dict:
    #                 robot_action_dict[arm] = action_dict[arm_abs_key]
                
    #             # 添加夹爪动作
    #             if gripper_key in action_dict:
    #                 robot_action_dict[gripper_key] = action_dict[gripper_key]
            
    #         # 创建动作向量
    #         env_action = robot.create_action_vector(robot_action_dict)
            
    #         # 执行动作
    #         env.step(env_action)
    #         env.render()
            
    #         # 打印调试信息（每100步打印一次）
    #         if timestep % 100 == 0:
    #             print(f"Timestep: {timestep}, Action norm: {np.linalg.norm(env_action):.4f}")
            
    #         timestep += 1
            
    #         # 限制帧率
    #         elapsed = time.time() - start_time
    #         diff = 1.0 / max_fr - elapsed
    #         if diff > 0:
    #             time.sleep(diff)
                
    # except KeyboardInterrupt:
    #     print("\n用户中断，退出控制循环")
    # finally:
    #     # 清理
    #     env.close()
    #     print("环境已关闭")


if __name__ == "__main__":
    main()
