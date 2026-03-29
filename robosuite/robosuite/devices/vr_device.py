"""
VR设备驱动类，用于通过VR手部姿态控制机器人

使用说明：
1. 实现VRTracker接口来连接你的VR SDK
2. 根据你的需求选择使用绝对位姿控制或增量控制
3. 在主脚本中添加 --device vr 选项
"""

import numpy as np
import abc
from typing import Dict, Optional, Tuple

from robosuite.devices import Device
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.utils import transform_utils as T


class VRTracker(metaclass=abc.ABCMeta):
    """
    VR跟踪器抽象基类
    你需要根据实际使用的VR SDK实现这个类
    """
    
    @abc.abstractmethod
    def get_hand_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取手部在世界坐标系下的位姿
        
        Returns:
            tuple: (position (3,), rotation_matrix (3,3))
                   position: 手部位置，单位米
                   rotation_matrix: 手部旋转矩阵
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_gripper_state(self) -> float:
        """
        获取夹爪状态（用于简单二指夹爪）
        
        Returns:
            float: 0.0 (完全打开) 到 1.0 (完全闭合)
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_gripper_action(self, gripper_dof: int) -> np.ndarray:
        """
        获取灵巧手关节动作（用于多关节灵巧手）
        
        Args:
            gripper_dof (int): 夹爪自由度数量
            
        Returns:
            np.ndarray: 夹爪关节动作，shape (gripper_dof,)
                       值域通常在 [-1, 1] 或 [0, 1]
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_reset_pressed(self) -> bool:
        """
        检查是否按下重置按钮
        
        Returns:
            bool: True表示需要重置环境
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """
        检查VR设备是否已初始化并连接
        
        Returns:
            bool: True表示设备就绪
        """
        raise NotImplementedError


class VRDevice(Device):
    """
    VR设备类，支持通过VR手部姿态控制机器人
    
    支持两种模式：
    1. 绝对位姿模式（默认）：直接将VR手部位姿作为目标
    2. 增量模式：计算VR手部运动增量（类似Keyboard）
    
    Args:
        env (RobotEnv): 机器人环境
        vr_tracker (VRTracker): VR跟踪器实例
        active_end_effector (str): 主动控制的末端执行器名称，如 "right" 或 "left"
        control_mode (str): 控制模式，"absolute" 或 "delta"
        pos_sensitivity (float): 位置敏感度（仅在delta模式下使用）
        rot_sensitivity (float): 旋转敏感度（仅在delta模式下使用）
        enable_coordinate_transform (bool): 是否启用坐标系转换
        transform_matrix (np.ndarray): VR到机器人的坐标变换矩阵 (4x4)，如果为None则使用单位矩阵
    """
    
    def __init__(
        self,
        env,
        vr_tracker: VRTracker,
        active_end_effector: str = "right",
        control_mode: str = "absolute",
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        enable_coordinate_transform: bool = True,
        transform_matrix: Optional[np.ndarray] = None,
    ):
        super().__init__(env)
        
        if not vr_tracker.is_initialized():
            raise RuntimeError("VR tracker is not initialized. Please initialize it before creating VRDevice.")
        
        self.vr_tracker = vr_tracker
        self.active_end_effector = active_end_effector
        self.control_mode = control_mode
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.enable_coordinate_transform = enable_coordinate_transform
        
        # 坐标系变换矩阵（VR坐标系 -> 机器人世界坐标系）
        if transform_matrix is None:
            # 默认单位变换矩阵（无变换）
            self.transform_matrix = np.eye(4)
        else:
            assert transform_matrix.shape == (4, 4), "Transform matrix must be 4x4"
            self.transform_matrix = transform_matrix
        
        # 增量模式需要的状态变量
        self.last_hand_pose = None  # (position, rotation_matrix)
        self.pos = np.zeros(3)
        self.last_pos = np.zeros(3)
        self.rotation = np.eye(3)
        self.raw_drotation = np.zeros(3)
        self.last_raw_drotation = np.zeros(3)
        
        self._display_controls()
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = False
    
    @staticmethod
    def _display_controls():
        """显示控制说明"""
        print("")
        print("=" * 50)
        print("VR设备控制")
        print("=" * 50)
        print("通过VR手部姿态控制机器人末端执行器")
        print("- 手部位姿直接映射到机器人末端位姿")
        print("- 手部关节姿态映射到夹爪/灵巧手动作")
        print("- 按下VR重置按钮可以重置环境")
        print("=" * 50)
        print("")
    
    def _reset_internal_state(self):
        """重置内部状态"""
        super()._reset_internal_state()
        
        # 获取当前机器人末端位姿作为初始位姿
        robot = self.env.robots[0]
        site_name = f"gripper0_{self.active_end_effector}_grip_site"
        site_id = self.env.sim.model.site_name2id(site_name)
        
        self.pos = self.env.sim.data.site_xpos[site_id].copy()
        self.last_pos = self.pos.copy()
        self.rotation = self.env.sim.data.site_xmat[site_id].copy().reshape(3, 3)
        self.raw_drotation = np.zeros(3)
        self.last_raw_drotation = np.zeros(3)
        self.last_hand_pose = None
    
    def start_control(self):
        """开始控制"""
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        # 重置VR跟踪器的状态
        self.last_hand_pose = None
    
    def get_controller_state(self):
        """
        获取控制器状态（用于增量模式）
        
        Returns:
            dict: 包含 dpos, rotation, raw_drotation, grasp, reset, base_mode
        """
        if self.control_mode != "delta":
            # 绝对模式不需要这个方法，但为了兼容性返回空字典
            return dict()
        
        # 从VR设备获取当前手部位姿
        try:
            vr_pos, vr_rot_mat = self.vr_tracker.get_hand_pose()
        except Exception as e:
            print(f"Warning: Failed to get VR pose: {e}")
            return dict(
                dpos=np.zeros(3),
                rotation=self.rotation,
                raw_drotation=np.zeros(3),
                grasp=0.0,
                reset=False,
                base_mode=0,
            )
        
        # 坐标系转换
        if self.enable_coordinate_transform:
            vr_pos, vr_rot_mat = self._transform_vr_to_robot_frame(vr_pos, vr_rot_mat)
        
        # 检查重置
        reset = self.vr_tracker.is_reset_pressed()
        if reset:
            self._reset_state = 1
            self._enabled = False
            return dict(
                dpos=np.zeros(3),
                rotation=self.rotation,
                raw_drotation=np.zeros(3),
                grasp=0.0,
                reset=True,
                base_mode=0,
            )
        
        if self.last_hand_pose is None:
            # 第一帧：初始化
            self.last_hand_pose = (vr_pos.copy(), vr_rot_mat.copy())
            self.pos = vr_pos.copy()
            self.rotation = vr_rot_mat.copy()
            dpos = np.zeros(3)
            raw_drotation = np.zeros(3)
        else:
            # 计算增量
            last_pos, last_rot = self.last_hand_pose
            
            # 位置增量
            dpos = (vr_pos - last_pos) * self.pos_sensitivity
            self.pos = vr_pos.copy()
            
            # 旋转增量：计算相对旋转
            drot_mat = vr_rot_mat @ last_rot.T
            drot_quat = T.mat2quat(drot_mat)
            drot_aa = T.quat2axisangle(drot_quat) * self.rot_sensitivity
            
            self.rotation = vr_rot_mat.copy()
            raw_drotation = drot_aa.copy()
            
            # 更新历史状态
            self.last_hand_pose = (vr_pos.copy(), vr_rot_mat.copy())
        
        # 获取夹爪状态
        grasp_value = self.vr_tracker.get_gripper_state()
        
        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=grasp_value,
            reset=False,
            base_mode=0,
        )
    
    def _transform_vr_to_robot_frame(
        self, vr_pos: np.ndarray, vr_rot_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将VR坐标系转换到机器人世界坐标系
        
        Args:
            vr_pos: VR坐标系下的位置
            vr_rot_mat: VR坐标系下的旋转矩阵
            
        Returns:
            tuple: (robot_pos, robot_rot_mat)
        """
        # 将位置和旋转组合成4x4齐次变换矩阵
        vr_pose = np.eye(4)
        vr_pose[:3, 3] = vr_pos
        vr_pose[:3, :3] = vr_rot_mat
        
        # 应用变换矩阵
        robot_pose = self.transform_matrix @ vr_pose
        
        robot_pos = robot_pose[:3, 3]
        robot_rot_mat = robot_pose[:3, :3]
        
        return robot_pos, robot_rot_mat
    
    def input2action(self, goal_update_mode: str = "target") -> Dict[str, np.ndarray]:
        """
        将VR输入转换为动作（覆盖基类方法以支持绝对位姿模式）
        
        Args:
            goal_update_mode: 目标更新模式，"target" 或 "achieved"
            
        Returns:
            dict: 动作字典
        """
        if self.control_mode == "absolute":
            return self._input2action_absolute(goal_update_mode)
        else:
            # 增量模式使用基类方法
            return super().input2action(goal_update_mode=goal_update_mode)
    
    def _input2action_absolute(self, goal_update_mode: str) -> Dict[str, np.ndarray]:
        """
        绝对位姿模式：直接将VR手部位姿作为目标
        
        Args:
            goal_update_mode: 目标更新模式
            
        Returns:
            dict: 动作字典
        """
        assert (
            goal_update_mode == "target"
        ), "goal_update_mode must be 'target' for absolute VR control"
        
        # 从VR设备获取手部位姿
        try:
            target_pos_world, target_ori_mat_world = self.vr_tracker.get_hand_pose()
        except Exception as e:
            print(f"Warning: Failed to get VR pose: {e}")
            # 返回零动作
            robot = self.env.robots[0]
            gripper_dof = robot.gripper[self.active_end_effector].dof
            return {
                f"{self.active_end_effector}_abs": np.zeros(6),
                f"{self.active_end_effector}_gripper": np.zeros(gripper_dof),
            }
        
        # 坐标系转换
        if self.enable_coordinate_transform:
            target_pos_world, target_ori_mat_world = self._transform_vr_to_robot_frame(
                target_pos_world, target_ori_mat_world
            )
        
        # 检查重置
        if self.vr_tracker.is_reset_pressed():
            return None
        
        # 获取机器人控制器配置
        robot = self.env.robots[0]
        
        # 处理不同的控制器类型
        if isinstance(robot.composite_controller, WholeBody):
            # WholeBody控制器
            ref_frame = robot.composite_controller.composite_controller_specific_config.get(
                "ik_input_ref_frame", "world"
            )
            
            if ref_frame != "world":
                # 需要转换到基座坐标系
                target_pose = np.eye(4)
                target_pose[:3, 3] = target_pos_world
                target_pose[:3, :3] = target_ori_mat_world
                target_pose = robot.composite_controller.joint_action_policy.transform_pose(
                    src_frame_pose=target_pose,
                    src_frame="world",
                    dst_frame=ref_frame,
                )
                target_pos, target_ori_mat = target_pose[:3, 3], target_pose[:3, :3]
            else:
                target_pos, target_ori_mat = target_pos_world, target_ori_mat_world
        else:
            # 单个手臂控制器
            controller = robot.part_controllers[self.active_end_effector]
            assert (
                controller.input_ref_frame == "world" and controller.input_type == "absolute"
            ), (
                f"VR device with absolute control requires controller with "
                f"input_ref_frame='world' and input_type='absolute'. "
                f"Current: input_ref_frame={controller.input_ref_frame}, "
                f"input_type={controller.input_type}"
            )
            target_pos, target_ori_mat = target_pos_world, target_ori_mat_world
        
        # 转换为axis-angle格式
        axis_angle_target = T.quat2axisangle(T.mat2quat(target_ori_mat))
        action = {}
        
        # 为所有手臂设置默认动作
        for arm in robot.arms:
            if arm == self.active_end_effector:
                action[f"{arm}_abs"] = np.concatenate([target_pos, axis_angle_target])
            else:
                # 其他手臂保持不动（使用零动作或当前目标）
                arm_action = self.get_arm_action(
                    robot,
                    arm,
                    norm_delta=np.zeros(6),
                    goal_update_mode=goal_update_mode,
                )
                action[f"{arm}_abs"] = arm_action["abs"]
            
            # 初始化夹爪动作
            action[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
        
        # 设置主动控制手臂的夹爪动作
        gripper_dof = robot.gripper[self.active_end_effector].dof
        if gripper_dof == 1:
            # 简单二指夹爪：使用get_gripper_state
            grasp_value = self.vr_tracker.get_gripper_state()
            # 映射到 [-1, 1] 范围
            grasp_action = grasp_value * 2.0 - 1.0
            action[f"{self.active_end_effector}_gripper"] = np.array([grasp_action])
        else:
            # 灵巧手：使用get_gripper_action
            gripper_action = self.vr_tracker.get_gripper_action(gripper_dof)
            # 确保动作在合理范围内
            gripper_action = np.clip(gripper_action, -1.0, 1.0)
            action[f"{self.active_end_effector}_gripper"] = gripper_action
        
        return action

