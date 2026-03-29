"""
VisionPro绝对位姿控制设备驱动类

基于VisionPro设备，实现绝对位姿控制：
1. 在VisionPro坐标系中选择一个虚拟base
2. 将VisionPro中手的位姿和虚拟base位姿之间的关系映射为机械臂末端位姿和机械臂底座位姿之间的关系
3. 返回绝对位姿action而不是增量action

使用方法：
- 校准阶段：操作人员将手放在期望的初始位置，系统会记录虚拟base的位姿
- 运行阶段：计算手相对于虚拟base的位姿，映射到机器人末端相对于机器人base的位姿
"""

import numpy as np
from typing import Dict, Optional, Tuple
from copy import deepcopy

from robosuite.devices import Device
from robosuite.devices.visionpro import VisionPro
from robosuite.utils import transform_utils as T
from robosuite.controllers.parts.arm.osc import OperationalSpaceController


class VisionProAbsolute(VisionPro):
    """
    VisionPro绝对位姿控制设备类
    
    继承自VisionPro，但使用绝对位姿控制而不是增量控制。
    
    Args:
        env: 机器人环境
        streamer: VisionProStreamer实例
        pos_sensitivity: 位置敏感度
        rot_sensitivity: 旋转敏感度
        calibration_frames: 校准阶段采集的帧数
        virtual_base_calibration_frames: 虚拟base校准阶段采集的帧数
        swap_hands: 是否交换左右手数据
        pos_axis_map: 位置轴映射
        pos_axis_signs: 位置轴符号
        enable_coordinate_debug: 是否启用坐标调试
    """
    
    def __init__(
        self,
        env,
        streamer,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        calibration_frames: int = 100,
        virtual_base_calibration_frames: int = 50,
        swap_hands: bool = False,
        pos_axis_map: Optional[Tuple[int, int, int]] = None,
        pos_axis_signs: Optional[Tuple[int, int, int]] = None,
        enable_coordinate_debug: bool = False,
    ):
        # 调用父类初始化
        super().__init__(
            env=env,
            streamer=streamer,
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
            calibration_frames=calibration_frames,
            swap_hands=swap_hands,
            pos_axis_map=pos_axis_map,
            pos_axis_signs=pos_axis_signs,
            enable_coordinate_debug=enable_coordinate_debug,
        )
        
        self.virtual_base_calibration_frames = virtual_base_calibration_frames
        
        # 虚拟base位姿（在VisionPro坐标系中）
        # 格式：{hand: (position (3,), rotation_matrix (3,3))}
        self._virtual_base_pose_left = None
        self._virtual_base_pose_right = None
        
        # 虚拟base校准状态
        self._virtual_base_calibrated = False
        self._virtual_base_calibrating = False
        self._virtual_base_calibration_poses_left = []
        self._virtual_base_calibration_poses_right = []
        
        # 机器人初始末端位姿（在机器人base坐标系中）
        # 格式：{robot_idx: {arm: (position (3,), rotation_matrix (3,3))}}
        self._robot_initial_eef_poses = {}
        
        # 机器人base位姿（在世界坐标系中）
        # 格式：{robot_idx: (position (3,), rotation_matrix (3,3))}
        self._robot_base_poses = {}
    
    def _reset_internal_state(self):
        """重置内部状态"""
        super()._reset_internal_state()
        
        self._virtual_base_pose_left = None
        self._virtual_base_pose_right = None
        self._virtual_base_calibrated = False
        self._virtual_base_calibrating = False
        self._virtual_base_calibration_poses_left = []
        self._virtual_base_calibration_poses_right = []
        self._robot_initial_eef_poses = {}
        self._robot_base_poses = {}
    
    def start_control(self):
        """
        开始控制，包括虚拟base校准
        """
        # 先调用父类的start_control（这会进行手部校准）
        super().start_control()
        
        # 然后进行虚拟base校准
        if not self._virtual_base_calibrated:
            print("=" * 60)
            print("开始虚拟base校准")
            print("请将双手放在期望的初始位置（虚拟base位置）")
            print(f"将采集 {self.virtual_base_calibration_frames} 帧数据")
            print("=" * 60)
            self._virtual_base_calibrating = True
    
    def _calibrate_virtual_base(self) -> bool:
        """
        校准虚拟base位姿
        
        采集多帧数据，计算平均位姿作为虚拟base
        
        Returns:
            bool: 是否校准完成
        """
        try:
            data = self.streamer.get_latest()
        except Exception as e:
            print(f"Warning: Failed to get Vision Pro data during virtual base calibration: {e}")
            return False
        
        # 获取左右手位姿
        if self.swap_hands:
            hand_left_data = data.right
            hand_right_data = data.left
        else:
            hand_left_data = data.left
            hand_right_data = data.right
        
        pose_left = self._get_hand_pose_from_data(hand_left_data)
        pose_right = self._get_hand_pose_from_data(hand_right_data)
        
        if pose_left is not None:
            self._virtual_base_calibration_poses_left.append(pose_left)
        if pose_right is not None:
            self._virtual_base_calibration_poses_right.append(pose_right)
        
        # 检查是否采集了足够的帧数
        if (len(self._virtual_base_calibration_poses_left) >= self.virtual_base_calibration_frames and
            len(self._virtual_base_calibration_poses_right) >= self.virtual_base_calibration_frames):
            
            # 计算平均位姿
            positions_left = [p[0] for p in self._virtual_base_calibration_poses_left]
            rotations_left = [p[1] for p in self._virtual_base_calibration_poses_left]
            positions_right = [p[0] for p in self._virtual_base_calibration_poses_right]
            rotations_right = [p[1] for p in self._virtual_base_calibration_poses_right]
            
            avg_pos_left = np.mean(positions_left, axis=0)
            avg_rot_left = self._average_rotation_matrices(rotations_left)
            avg_pos_right = np.mean(positions_right, axis=0)
            avg_rot_right = self._average_rotation_matrices(rotations_right)
            
            self._virtual_base_pose_left = (avg_pos_left, avg_rot_left)
            self._virtual_base_pose_right = (avg_pos_right, avg_rot_right)
            
            # 记录机器人初始末端位姿和base位姿
            self._record_robot_initial_poses()
            
            self._virtual_base_calibrated = True
            self._virtual_base_calibrating = False
            
            print("=" * 60)
            print("虚拟base校准完成")
            print(f"虚拟base左: pos={avg_pos_left}, rot shape={avg_rot_left.shape}")
            print(f"虚拟base右: pos={avg_pos_right}, rot shape={avg_rot_right.shape}")
            print("=" * 60)
            
            return True
        
        return False
    
    def _record_robot_initial_poses(self):
        """
        记录机器人初始末端位姿和base位姿
        """
        for robot_idx, robot in enumerate(self.env.robots):
            self._robot_initial_eef_poses[robot_idx] = {}
            self._robot_base_poses[robot_idx] = {}
            
            # 获取机器人base位姿（在世界坐标系中）
            base_pos = np.array(self.env.sim.data.get_body_xpos(robot.robot_model.root_body))
            base_rot = np.array(self.env.sim.data.get_body_xmat(robot.robot_model.root_body)).reshape(3, 3)
            self._robot_base_poses[robot_idx] = (base_pos.copy(), base_rot.copy())
            
            # 获取每个arm的初始末端位姿（在机器人base坐标系中）
            for arm in robot.arms:
                # 获取末端执行器位姿（在世界坐标系中）
                eef_site_id = robot.eef_site_id[arm]
                eef_pos_world = np.array(self.env.sim.data.site_xpos[eef_site_id])
                
                # 获取末端执行器旋转（从grip_site获取，更准确）
                pf = robot.gripper[arm].naming_prefix
                grip_site_name = f"{pf}grip_site"
                grip_site_id = self.env.sim.model.site_name2id(grip_site_name)
                eef_rot_world = np.array(self.env.sim.data.site_xmat[grip_site_id]).reshape(3, 3)
                
                # 转换到机器人base坐标系
                base_pos_world, base_rot_world = self._robot_base_poses[robot_idx]
                T_world_base = T.make_pose(base_pos_world, base_rot_world)
                T_world_eef = T.make_pose(eef_pos_world, eef_rot_world)
                T_base_eef = T.pose_in_A_to_pose_in_B(T_world_eef, T.pose_inv(T_world_base))
                
                eef_pos_base, eef_rot_base = T.mat2pose(T_base_eef)
                self._robot_initial_eef_poses[robot_idx][arm] = (eef_pos_base.copy(), eef_rot_base.copy())
    
    def _compute_hand_pose_relative_to_virtual_base(
        self, 
        hand_pose: Tuple[np.ndarray, np.ndarray],
        virtual_base_pose: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算手相对于虚拟base的位姿
        
        Args:
            hand_pose: 手部当前位姿 (position, rotation_matrix)
            virtual_base_pose: 虚拟base位姿 (position, rotation_matrix)
            
        Returns:
            tuple: 手相对于虚拟base的位姿 (position, rotation_matrix)
        """
        # 构建齐次变换矩阵
        T_vbase = T.make_pose(virtual_base_pose[0], virtual_base_pose[1])
        T_hand = T.make_pose(hand_pose[0], hand_pose[1])
        
        # 计算手相对于虚拟base的位姿
        T_vbase_inv = T.pose_inv(T_vbase)
        T_hand_rel_vbase = T.pose_in_A_to_pose_in_B(T_hand, T_vbase_inv)
        
        # 提取位置和旋转
        pos_rel, rot_rel = T.mat2pose(T_hand_rel_vbase)
        
        return pos_rel, rot_rel
    
    def _map_to_robot_eef_pose(
        self,
        hand_pose_rel_vbase: Tuple[np.ndarray, np.ndarray],
        robot_idx: int,
        arm: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将手相对于虚拟base的位姿映射到机器人末端相对于机器人base的位姿
        
        Args:
            hand_pose_rel_vbase: 手相对于虚拟base的位姿 (position, rotation_matrix)
            robot_idx: 机器人索引
            arm: 手臂名称
            
        Returns:
            tuple: 机器人末端相对于机器人base的位姿 (position, rotation_matrix)
        """
        # 获取机器人初始末端位姿（相对于机器人base）
        initial_eef_pos, initial_eef_rot = self._robot_initial_eef_poses[robot_idx][arm]
        
        # 计算相对位姿的变化
        hand_pos_rel, hand_rot_rel = hand_pose_rel_vbase
        
        # 将手部相对位姿的变化映射到机器人末端位姿
        # 这里使用简单的线性映射：直接使用相对位姿作为目标位姿
        # 也可以根据需要进行缩放或变换
        
        # 目标位姿 = 初始位姿 + 相对变化
        # 对于位置：直接相加
        target_eef_pos = initial_eef_pos + hand_pos_rel
        
        # 对于旋转：组合旋转矩阵
        T_initial = T.make_pose(initial_eef_pos, initial_eef_rot)
        T_relative = T.make_pose(hand_pos_rel, hand_rot_rel)
        T_target = T_initial @ T_relative
        target_eef_pos, target_eef_rot = T.mat2pose(T_target)
        
        return target_eef_pos, target_eef_rot
    
    def _pose_to_absolute_action(
        self,
        eef_pos: np.ndarray,
        eef_rot: np.ndarray
    ) -> np.ndarray:
        """
        将末端位姿转换为绝对action格式
        
        Args:
            eef_pos: 末端位置 (3,)
            eef_rot: 末端旋转矩阵 (3, 3)
            
        Returns:
            np.ndarray: 绝对action (6,) [pos(3), rot_axis_angle(3)]
        """
        # 将旋转矩阵转换为axis-angle表示
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(eef_rot)
        rot_axis_angle = rot.as_rotvec()
        
        # 组合为6维action
        abs_action = np.concatenate([eef_pos, rot_axis_angle])
        
        return abs_action
    
    def input2action(self, mirror_actions=False, goal_update_mode="target") -> Optional[Dict]:
        """
        将VisionPro输入转换为绝对位姿action
        
        Args:
            mirror_actions: 是否镜像动作
            goal_update_mode: 目标更新模式（在绝对位姿模式下不使用）
            
        Returns:
            dict: 动作字典，包含左右手臂的绝对位姿action
        """
        # 如果正在校准虚拟base
        if self._virtual_base_calibrating:
            if self._calibrate_virtual_base():
                # 校准完成，返回零动作
                robot = self.env.robots[0]
                action_dict = {}
                for arm in robot.arms:
                    # 返回初始位姿作为绝对action
                    if 0 in self._robot_initial_eef_poses and arm in self._robot_initial_eef_poses[0]:
                        initial_pos, initial_rot = self._robot_initial_eef_poses[0][arm]
                        abs_action = self._pose_to_absolute_action(initial_pos, initial_rot)
                        action_dict[f"{arm}_abs"] = abs_action
                        action_dict[f"{arm}_delta"] = np.zeros(6)
                    else:
                        action_dict[f"{arm}_abs"] = np.zeros(6)
                        action_dict[f"{arm}_delta"] = np.zeros(6)
                    action_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
                return action_dict
            else:
                # 还在校准中，返回零动作
                robot = self.env.robots[0]
                action_dict = {}
                for arm in robot.arms:
                    action_dict[f"{arm}_abs"] = np.zeros(6)
                    action_dict[f"{arm}_delta"] = np.zeros(6)
                    action_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
                return action_dict
        
        # 如果虚拟base未校准，返回None（需要先校准）
        if not self._virtual_base_calibrated:
            print("Warning: Virtual base not calibrated. Please call start_control() first.")
            return None
        
        # 获取控制器状态（从父类获取原始数据）
        state = super().get_controller_state()
        
        # 如果重置
        if state.get("reset", False):
            return None
        
        # 获取当前手部位姿
        try:
            data = self.streamer.get_latest()
        except Exception as e:
            print(f"Warning: Failed to get Vision Pro data: {e}")
            return None
        
        # 根据swap_hands决定是否交换
        if self.swap_hands:
            hand_left_data = data.right
            hand_right_data = data.left
        else:
            hand_left_data = data.left
            hand_right_data = data.right
        
        pose_left = self._get_hand_pose_from_data(hand_left_data)
        pose_right = self._get_hand_pose_from_data(hand_right_data)
        
        # 获取gripper状态
        grasp_left = state.get("grasp_left", 0.0)
        grasp_right = state.get("grasp_right", 0.0)
        
        # 构建动作字典
        action_dict = {}
        
        # 处理多个机器人的情况
        num_robots = len(self.env.robots)
        
        if num_robots >= 2:
            # 多机器人情况：第一个机器人用左手，第二个机器人用右手
            # 处理第一个机器人（左手）
            if pose_left is not None and self._virtual_base_pose_left is not None:
                robot_left = self.env.robots[0]
                arm_names_left = robot_left.arms
                left_arm = arm_names_left[0] if len(arm_names_left) > 0 else None
                
                if left_arm is not None:
                    # 计算手相对于虚拟base的位姿
                    hand_pose_rel_vbase = self._compute_hand_pose_relative_to_virtual_base(
                        pose_left, self._virtual_base_pose_left
                    )
                    
                    # 映射到机器人末端位姿
                    eef_pos, eef_rot = self._map_to_robot_eef_pose(
                        hand_pose_rel_vbase, 0, left_arm
                    )
                    
                    # 转换为绝对action
                    abs_action = self._pose_to_absolute_action(eef_pos, eef_rot)
                    action_dict["left_abs"] = abs_action
                    action_dict["left_delta"] = np.zeros(6)  # 不使用delta
                    
                    # 设置夹爪动作
                    gripper_dof = robot_left.gripper[left_arm].dof
                    gripper_key = "left_gripper"
                    if isinstance(grasp_left, np.ndarray) and len(grasp_left) > 1:
                        if len(grasp_left) == gripper_dof:
                            action_dict[gripper_key] = grasp_left.copy()
                        else:
                            action_dict[gripper_key] = np.zeros(gripper_dof)
                    else:
                        grasp_val = 1 if (grasp_left > 0.5 if isinstance(grasp_left, (int, float, np.number)) else False) else -1
                        if hasattr(robot_left.gripper[left_arm], "grasp_qpos"):
                            gripper_action = robot_left.gripper[left_arm].grasp_qpos[grasp_val]
                            if len(gripper_action) != gripper_dof:
                                action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
                            else:
                                action_dict[gripper_key] = gripper_action
                        else:
                            action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
            
            # 处理第二个机器人（右手）
            if pose_right is not None and self._virtual_base_pose_right is not None:
                robot_right = self.env.robots[1]
                arm_names_right = robot_right.arms
                right_arm = arm_names_right[0] if len(arm_names_right) > 0 else None
                
                if right_arm is not None:
                    # 计算手相对于虚拟base的位姿
                    hand_pose_rel_vbase = self._compute_hand_pose_relative_to_virtual_base(
                        pose_right, self._virtual_base_pose_right
                    )
                    
                    # 映射到机器人末端位姿
                    eef_pos, eef_rot = self._map_to_robot_eef_pose(
                        hand_pose_rel_vbase, 1, right_arm
                    )
                    
                    # 转换为绝对action
                    abs_action = self._pose_to_absolute_action(eef_pos, eef_rot)
                    action_dict["right_abs"] = abs_action
                    action_dict["right_delta"] = np.zeros(6)  # 不使用delta
                    
                    # 设置夹爪动作
                    gripper_dof = robot_right.gripper[right_arm].dof
                    gripper_key = "right_gripper"
                    if isinstance(grasp_right, np.ndarray) and len(grasp_right) > 1:
                        if len(grasp_right) == gripper_dof:
                            action_dict[gripper_key] = grasp_right.copy()
                        else:
                            action_dict[gripper_key] = np.zeros(gripper_dof)
                    else:
                        grasp_val = 1 if (grasp_right > 0.5 if isinstance(grasp_right, (int, float, np.number)) else False) else -1
                        if hasattr(robot_right.gripper[right_arm], "grasp_qpos"):
                            gripper_action = robot_right.gripper[right_arm].grasp_qpos[grasp_val]
                            if len(gripper_action) != gripper_dof:
                                action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
                            else:
                                action_dict[gripper_key] = gripper_action
                        else:
                            action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
        
        else:
            # 单机器人情况
            robot = self.env.robots[0]
            arm_names = robot.arms
            
            # 如果有两个手臂，左手对应第一个，右手对应第二个
            if len(arm_names) >= 2:
                if pose_left is not None and self._virtual_base_pose_left is not None:
                    left_arm = arm_names[0]
                    hand_pose_rel_vbase = self._compute_hand_pose_relative_to_virtual_base(
                        pose_left, self._virtual_base_pose_left
                    )
                    eef_pos, eef_rot = self._map_to_robot_eef_pose(hand_pose_rel_vbase, 0, left_arm)
                    abs_action = self._pose_to_absolute_action(eef_pos, eef_rot)
                    action_dict[f"{left_arm}_abs"] = abs_action
                    action_dict[f"{left_arm}_delta"] = np.zeros(6)
                    
                    gripper_dof = robot.gripper[left_arm].dof
                    gripper_key = f"{left_arm}_gripper"
                    if isinstance(grasp_left, np.ndarray) and len(grasp_left) > 1:
                        action_dict[gripper_key] = grasp_left.copy() if len(grasp_left) == gripper_dof else np.zeros(gripper_dof)
                    else:
                        grasp_val = 1 if (grasp_left > 0.5 if isinstance(grasp_left, (int, float, np.number)) else False) else -1
                        if hasattr(robot.gripper[left_arm], "grasp_qpos"):
                            action_dict[gripper_key] = robot.gripper[left_arm].grasp_qpos[grasp_val]
                        else:
                            action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
                
                if pose_right is not None and self._virtual_base_pose_right is not None:
                    right_arm = arm_names[1] if len(arm_names) > 1 else arm_names[0]
                    hand_pose_rel_vbase = self._compute_hand_pose_relative_to_virtual_base(
                        pose_right, self._virtual_base_pose_right
                    )
                    eef_pos, eef_rot = self._map_to_robot_eef_pose(hand_pose_rel_vbase, 0, right_arm)
                    abs_action = self._pose_to_absolute_action(eef_pos, eef_rot)
                    action_dict[f"{right_arm}_abs"] = abs_action
                    action_dict[f"{right_arm}_delta"] = np.zeros(6)
                    
                    gripper_dof = robot.gripper[right_arm].dof
                    gripper_key = f"{right_arm}_gripper"
                    if isinstance(grasp_right, np.ndarray) and len(grasp_right) > 1:
                        action_dict[gripper_key] = grasp_right.copy() if len(grasp_right) == gripper_dof else np.zeros(gripper_dof)
                    else:
                        grasp_val = 1 if (grasp_right > 0.5 if isinstance(grasp_right, (int, float, np.number)) else False) else -1
                        if hasattr(robot.gripper[right_arm], "grasp_qpos"):
                            action_dict[gripper_key] = robot.gripper[right_arm].grasp_qpos[grasp_val]
                        else:
                            action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
            else:
                # 只有一个手臂，使用右手数据
                if pose_right is not None and self._virtual_base_pose_right is not None:
                    arm = arm_names[0]
                    hand_pose_rel_vbase = self._compute_hand_pose_relative_to_virtual_base(
                        pose_right, self._virtual_base_pose_right
                    )
                    eef_pos, eef_rot = self._map_to_robot_eef_pose(hand_pose_rel_vbase, 0, arm)
                    abs_action = self._pose_to_absolute_action(eef_pos, eef_rot)
                    action_dict[f"{arm}_abs"] = abs_action
                    action_dict[f"{arm}_delta"] = np.zeros(6)
                    
                    gripper_dof = robot.gripper[arm].dof
                    gripper_key = f"{arm}_gripper"
                    if isinstance(grasp_right, np.ndarray) and len(grasp_right) > 1:
                        action_dict[gripper_key] = grasp_right.copy() if len(grasp_right) == gripper_dof else np.zeros(gripper_dof)
                    else:
                        grasp_val = 1 if (grasp_right > 0.5 if isinstance(grasp_right, (int, float, np.number)) else False) else -1
                        if hasattr(robot.gripper[arm], "grasp_qpos"):
                            action_dict[gripper_key] = robot.gripper[arm].grasp_qpos[grasp_val]
                        else:
                            action_dict[gripper_key] = np.array([grasp_val] * gripper_dof)
        
        return action_dict
