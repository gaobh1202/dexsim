"""
VisionPro设备驱动类，用于通过Apple Vision Pro检测双手姿态来控制双臂机器人

使用增量控制方式：计算手部姿态相对于初始姿态的变化，作为机器人的delta action。
类似 keyboard.py 的设计，初始姿态在 start_control() 中通过校准获得，并设置为 last_pose。

坐标变换：
- 所有wrist pose都通过SIM_HEAD变换到base坐标系中
- 在base坐标系下计算相对变化，使得delta action可以直接用于OSC_POSE控制器的delta模式
- SIM_HEAD定义了head在base坐标系中的位姿，用于将Vision Pro的wrist pose从head坐标系变换到base坐标系

初始化流程（类似 keyboard）：
1. __init__: 初始化设备，但不进行校准
2. start_control(): 执行同步校准，采集多帧数据计算平均初始姿态
   - 将初始姿态设置为 _last_pose_left 和 _last_pose_right（类似 keyboard 的 last_pos）
   - 后续计算增量时使用：dpos = current_pose - last_pose

运行阶段：
- 将当前手部姿态从Vision Pro坐标系变换到base坐标系
- 在base坐标系下计算相对于 last_pose 的变化（类似 keyboard: dpos = pos - last_pos）
- 将变化量作为delta action发送给机器人（相对于base坐标系）
- 更新 last_pose 为当前姿态，用于下一帧计算增量
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple

from robosuite.devices import Device
from robosuite.utils import transform_utils as T

SIM_HEAD = np.eye(4)
SIM_HEAD[:3, 3] = [-0.56, 0.0, 1.1]

class VisionPro(Device):
    """
    VisionPro设备类，通过Vision Pro检测双手姿态控制双臂机器人
    
    Args:
        env (RobotEnv): 机器人环境
        streamer (VisionProStreamer): VisionProStreamer实例，用于获取Vision Pro数据
        pos_sensitivity (float): 位置敏感度，控制位置变化的缩放
        rot_sensitivity (float): 旋转敏感度，控制旋转变化的缩放
        calibration_frames (int): 初始化阶段采集的帧数，用于计算平均初始姿态
    """
    
    def __init__(
        self,
        env,
        streamer,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
        calibration_frames: int = 100,
    ):
        super().__init__(env)
        
        self.streamer = streamer
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.calibration_frames = calibration_frames
        # visionpro中wrist和head的关系，目的是将wrist变成和head方向一致； t_left_wrist_rot = left_wrist_rot @ self.T_head_to_left_wrist^T
        self.T_head_to_left_wrist_rot = np.array([
            [0, 0, -1],
            [1, 0, 0],
            [0, -1, 0],
        ])
        self.T_head_to_right_wrist_rot = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ])
        # 初始化状态标志
        self._initialized = False  # 是否已完成初始化（校准完成）
        self._enabled = False
        self._reset_state = 0
        
        # 初始姿态存储（左右手）
        self._initial_pose_left = None  # (position (3,), rotation_matrix (3,3))
        self._initial_pose_right = None
        self._initial_pose_head = None
        
        # 当前和上一帧的位姿（左右手）
        self._current_pose_left = None
        self._current_pose_right = None
        self._last_pose_left = None
        self._last_pose_right = None
        
        # 校准阶段采集的位姿列表（在_reset_internal_state中初始化）
        # 位置和旋转（在_reset_internal_state中初始化）
        # 原始旋转增量（在_reset_internal_state中初始化）
        # 夹爪状态（在_reset_internal_state中初始化）
        
        self._reset_internal_state()
    
    
    def _reset_internal_state(self):
        """
        重置内部状态
        
        类似 keyboard 的 _reset_internal_state，重置运行时状态变量。
        注意：初始姿态（_initial_pose_left/right）不会被重置，它们会在 start_control() 中通过校准设置。
        last_pose 也会在 start_control() 中设置为初始姿态。
        """
        super()._reset_internal_state()
        
        self._pos_left = np.zeros(3)
        self._pos_right = np.zeros(3)
        self._last_pos_left = np.zeros(3)
        self._last_pos_right = np.zeros(3)
        
        self._rot_left = np.eye(3)
        self._rot_right = np.eye(3)
        self._last_rot_left = np.eye(3)
        self._last_rot_right = np.eye(3)
        
        self._raw_drotation_left = np.zeros(3)
        self._raw_drotation_right = np.zeros(3)
        # self._last_raw_drotation_left = np.zeros(3)
        # self._last_raw_drotation_right = np.zeros(3)
        
        self._gripper_left = 0.0
        self._gripper_right = 0.0
        
        
        # 注意：_initial_pose_left/right 和 _last_pose_left/right 不会被重置
        # 它们会在 start_control() 中通过校准重新设置
    
    # def _extract_pose_from_transform(self, transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     从4x4变换矩阵中提取位置和旋转矩阵
        
    #     Args:
    #         transform: 4x4齐次变换矩阵
            
    #     Returns:
    #         tuple: (position (3,), rotation_matrix (3,3))
    #     """
    #     position = transform[:3, 3].copy()
    #     rotation = transform[:3, :3].copy()
    #     return position, rotation
    
    def _transform_wrist_pose_to_base_frame(self, wrist_position: np.ndarray, wrist_rotation: np.ndarray, is_left: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        将wrist的位姿（位置和旋转）从Vision Pro坐标系变换到SIM_HEAD（base）坐标系
        
        这个方法将wrist pose变换到与机器人base坐标系一致的坐标系中，
        使得后续计算的delta action可以直接用于OSC_POSE控制器的delta模式。
        
        假设Vision Pro的wrist transform是相对于head的，使用SIM_HEAD将wrist pose
        从head坐标系变换到base坐标系。
        
        Args:
            wrist_position: wrist的原始位置 (3,)，在Vision Pro head坐标系中
            wrist_rotation: wrist的原始旋转矩阵 (3, 3)，在Vision Pro head坐标系中
            is_left: 是否为左手（True表示left，False表示right）
            
        Returns:
            tuple: (变换后的位置 (3,), 变换后的旋转矩阵 (3, 3))，在SIM_HEAD（base）坐标系中
        """
        # 构建wrist在Vision Pro head坐标系中的4x4变换矩阵
        wrist_pose_head = np.eye(4)
        wrist_pose_head[:3, :3] = wrist_rotation
        wrist_pose_head[:3, 3] = wrist_position
        
        # 假设Vision Pro的wrist transform是相对于head的
        # 使用SIM_HEAD将wrist pose从head坐标系变换到base坐标系
        # wrist_in_base = SIM_HEAD @ wrist_in_head
        wrist_pose_base = SIM_HEAD @ wrist_pose_head
        
        # 提取变换后的位置和旋转
        transformed_position = wrist_pose_base[:3, 3]
        transformed_rotation = wrist_pose_base[:3, :3]
        
        return transformed_position, transformed_rotation
    
    
    def _get_hand_pose_from_data(self, hand_data) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从HandData对象中获取手腕位姿
        
        Args:
            hand_data: Vision Pro的HandData对象（data.left或data.right）
            
        Returns:
            tuple: (position (3,), rotation_matrix (3,3)) 或 None（如果手部不可见）
        """
        if hand_data is None:
            return None
        
        try:
            # 使用wrist的变换矩阵（索引0）
            wrist_transform = hand_data.wrist  # (4, 4) 变换矩阵
            return self._extract_pose_from_transform(wrist_transform)
        except (AttributeError, IndexError):
            return None
    
    def _get_gripper_from_hand_data(self, hand_data, gripper: str = 'twofinger'):
        """
        从HandData对象中获取夹爪状态
        
        Args:
            hand_data: Vision Pro的HandData对象
            gripper: 夹爪类型，'dexhand'返回六维动作，'twofinger'返回一维动作
            
        Returns:
            float或np.ndarray: 
                - 当gripper='twofinger'时，返回float: 夹爪状态，0.0（打开）到1.0（闭合）
                - 当gripper='dexhand'时，返回np.ndarray: 六维动作数组
        """
        if hand_data is None:
            raise ValueError("hand_data cannot be None. Hand tracking data is required to compute gripper action.")
        
        # 根据gripper类型调用相应的计算函数
        if gripper == 'dexhand':
            return self._compute_dexhand_action(hand_data)
        else:
            return self._compute_twofinger_action(hand_data)
    
    def _compute_dexhand_action(self, hand_data) -> np.ndarray:
        """
        从HandData对象计算dexhand的六维动作
        
        Args:
            hand_data: Vision Pro的HandData对象
            
        Returns:
            np.ndarray: 六维动作数组（当前用0表示，后续实现具体计算逻辑）
        """
        # TODO: 后续实现从hand_data计算dexhand动作的逻辑
        return np.zeros(6)
    
    def _compute_twofinger_action(self, hand_data) -> float:
        """
        从HandData对象计算twofinger的一维动作
        
        Args:
            hand_data: Vision Pro的HandData对象
            
        Returns:
            float: 夹爪状态，0.0（打开）到1.0（闭合）（当前用0表示，后续实现具体计算逻辑）
        """
        # TODO: 后续实现从hand_data计算twofinger动作的逻辑
        return 0.0
    
    def start_control(self):
        """
        开始控制，包括初始化校准阶段
        
        类似 keyboard 的 start_control，在校准完成后设置初始姿态作为 last_pose，
        后续计算增量时使用当前姿态相对于初始姿态的变化。
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        
        # 执行同步校准（阻塞式），获取初始姿态
        print("=" * 60)
        print("开始Vision Pro校准")
        print("=" * 60)
        print(f"请保持初始姿态，系统将采集 {self.calibration_frames} 帧数据...")
        print("初始姿态：手肘90度，放在身体两侧，手心朝下手指朝前")
        print("=" * 60)
        
        # 同步校准：采集足够的数据后设置初始姿态
        self._perform_calibration()
        
        # 校准完成后，将初始姿态设置为 last_pose（类似 keyboard 的 last_pos）
        if self._initialized:
            self._last_pose_left = self._initial_pose_left # tuple
            self._last_pose_right = self._initial_pose_right
            self._last_pos_left = self._initial_pose_left[0].copy()
            self._last_pos_right = self._initial_pose_right[0].copy()
            self._rotation_left = self._initial_pose_left[1].copy()  # 初始的平均旋转矩阵
            self._rotation_right = self._initial_pose_right[1].copy()
            
            print("=" * 60)
            print("校准完成！可以开始操作了")
            print("=" * 60)
    
    def _perform_calibration(self):
        """
        执行同步校准：采集多帧数据计算平均初始姿态
        
        这是一个阻塞式函数，会持续采集数据直到达到指定帧数。
        校准完成后，初始姿态会被设置为 _initial_pose_left 和 _initial_pose_right，
        这些姿态将作为后续计算增量的基准（类似 keyboard 的 last_pose）。
        """
        _calibration_left_pos = []
        _calibration_left_rot = []
        _calibration_right_pos = []
        _calibration_right_rot = []
        _calibration_head_pos = []
        _calibration_head_rot = []
        
        frame_count = 0
        while frame_count < self.calibration_frames:
            # 获取最新数据
            try:
                data = self.streamer.get_latest()
            except Exception as e:
                print(f"Warning: Failed to get Vision Pro data: {e}")
                time.sleep(0.01)  # 短暂等待后重试
                continue
            
            left_wrist_pose = data.left.wrist
            right_wrist_pose = data.right.wrist
            head_pose = data.head
            # # 应用坐标变换：将wrist pose从Vision Pro坐标系变换到SIM_HEAD（base）坐标系
            # if pose_left is not None:
            #     transformed_pos_left, transformed_rot_left = self._transform_wrist_pose_to_base_frame(
            #         pose_left[0], pose_left[1], is_left=True
            #     )
            #     pose_left = (transformed_pos_left, transformed_rot_left)
            
            # if pose_right is not None:
            #     transformed_pos_right, transformed_rot_right = self._transform_wrist_pose_to_base_frame(
            #         pose_right[0], pose_right[1], is_left=False
            #     )
            #     pose_right = (transformed_pos_right, transformed_rot_right)

            _calibration_left_pos.append(left_wrist_pose[:3, 3])
            _calibration_left_rot.append(left_wrist_pose[:3, :3])
            _calibration_right_pos.append(right_wrist_pose[:3, 3])
            _calibration_right_rot.append(right_wrist_pose[:3, :3])
            _calibration_head_pos.append(head_pose[:3, 3])
            _calibration_head_rot.append(head_pose[:3, :3])
            
            frame_count += 1
            
            time.sleep(0.01)  # 控制采集频率
        
        # 计算平均初始姿态
        # 位置：直接平均
        left_positions = np.array([p for p in _calibration_left_pos])
        right_positions = np.array([p for p in _calibration_right_pos])
        head_positions = np.array([p for p in _calibration_head_pos])

        self._initial_pose_left = np.eye(4)
        self._initial_pose_left[:3, 3] = np.mean(left_positions, axis=0)
        self._initial_pose_left[:3, :3] = self._average_rotation_matrices([p for p in _calibration_left_rot])
        self._initial_pose_right = np.eye(4)
        self._initial_pose_right[:3, 3] = np.mean(right_positions, axis=0)
        self._initial_pose_right[:3, :3] = self._average_rotation_matrices([p for p in _calibration_right_rot])
        self._initial_pose_head = np.eye(4)
        self._initial_pose_head[:3, 3] = np.mean(head_positions, axis=0)
        self._initial_pose_head[:3, :3] = self._average_rotation_matrices([p for p in _calibration_head_rot])

        self._initial_pose_left, self._initial_pose_right = self.transform_wrist_pose(self._initial_pose_left, self._initial_pose_right)
        # self._initial_pose_right = self.transform_wrist_pose(self._initial_pose_right)

        # self._initial_pose_left = (
        #     np.mean(left_positions, axis=0),
        #     self._average_rotation_matrices([p for p in _calibration_left_rot])
        # )
        # self._initial_pose_right = (
        #     np.mean(right_positions, axis=0),
        #     self._average_rotation_matrices([p for p in _calibration_right_rot])
        # )
        # self._initial_pose_head = (
        #     np.mean(head_positions, axis=0),
        #     self._average_rotation_matrices([p for p in _calibration_head_rot])
        # )
        self._initialized = True
        
        print("=" * 60)
        print("校准完成！")
        print(f"左手臂初始位置（base坐标系）: {self._initial_pose_left[0]}")
        print(f"右手臂初始位置（base坐标系）: {self._initial_pose_right[0]}")
        print("注意：所有位姿已变换到base坐标系，delta action将相对于初始姿态计算")
        print("=" * 60)
    
    def _average_rotation_matrices(self, rotations: list) -> np.ndarray:
        """
        平均多个旋转矩阵
        
        使用四元数平均的方法，比直接平均旋转矩阵更准确
        
        Args:
            rotations: 旋转矩阵列表
            
        Returns:
            np.ndarray: 平均后的旋转矩阵 (3, 3)
        """
        # 转换为四元数
        quaternions = [T.mat2quat(rot) for rot in rotations]
        
        # 平均四元数（简单方法：归一化后平均）
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        
        # 转回旋转矩阵
        return T.quat2mat(avg_quat)
    
    def rotation_alone_axis(self, axis, angle, degrees=True):
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

    def transform_wrist_pose(self, left_wrist_pose, right_wrist_pose):
        left_wrist_pose = self.rotation_alone_axis("z", -90) @ left_wrist_pose
        right_wrist_pose = self.rotation_alone_axis("z", -90) @ right_wrist_pose
        left_wrist_pose = left_wrist_pose @ self.rotation_alone_axis("y", 90) @ self.rotation_alone_axis("z", 180)
        right_wrist_pose = right_wrist_pose @ self.rotation_alone_axis("y", -90)
        return left_wrist_pose, right_wrist_pose

    def get_controller_state(self) -> Dict:
        '''
        修改的get_controller_state

        参考keyboard的方式；构建如下的信息
        dpos = self.pos - self.last_pos  # delta position, (3, 3)；分别对应x,y,z轴的增量
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation  轴角的差值；shape=(3,)，drot[0]表示y轴的旋转，drot[1]表示x轴的旋转，drot[2]表示z轴的旋转
        self.last_drotation = np.array(self.raw_drotation)

        dpos表示末端在base坐标系下的变化值
        raw_drotation表示末端在base坐标系下的变化值的轴角表示
        '''

        # 获取最新数据
        try:
            data = self.streamer.get_latest()
        except Exception as e:
            print(f"Warning: Failed to get Vision Pro data: {e}")
            return None
        
        hand_data_left_raw = data.left
        hand_data_right_raw = data.right
        left_wrist_pose = data.left.wrist # [4,4]
        right_wrist_pose = data.right.wrist # [4,4]
        head_pose = data.head # [4,4]


        # 如果检测不到手部，保持上一帧状态
        if left_wrist_pose is None:
            left_wrist_pose = self._last_pose_left if self._last_pose_left is not None else self._initial_pose_left
            # 如果使用上一帧状态，dpos会是0，这是正常的
        if right_wrist_pose is None:
            right_wrist_pose = self._last_pose_right if self._last_pose_right is not None else self._initial_pose_right

        # 使用 last_pose 作为参考（类似 keyboard 的 last_pos）
        # last_pose 在 start_control() 中被设置为初始姿态，后续会更新为上一帧姿态
        last_left_wrist_pose = self._last_pose_left if self._last_pose_left is not None else self._initial_pose_left
        last_right_wrist_pose = self._last_pose_right if self._last_pose_right is not None else self._initial_pose_right

        left_wrist_pose, right_wrist_pose = self.transform_wrist_pose(left_wrist_pose, right_wrist_pose)

        left_dpos_base = left_wrist_pose[:3, 3] - last_left_wrist_pose[:3, 3]
        # left_dpos_base = np.dot(self.T_head_to_left_wrist_rot, left_dpos_vp)  # save to dict for return

        right_dpos_base = right_wrist_pose[:3, 3] - last_right_wrist_pose[:3, 3]
        # right_dpos_base = np.dot(self.T_head_to_right_wrist_rot, right_dpos_vp)  # save to dict for return

        # 是否需要转换坐标系？转换到base方向还是末端方向？根据绘制的wrist方向和末端方向来确定转换矩阵，直接变换xyz轴角？？
        # 计算相对旋转并变换到base坐标系
        # left_drot_vp = np.dot(left_wrist_pose[:3, :3], last_left_wrist_pose[:3, :3].T)
        left_drot_vp = np.dot(left_wrist_pose[:3, :3], np.linalg.inv(last_left_wrist_pose[:3, :3]))
        # left_drot_base = self.T_head_to_left_wrist_rot @ left_drot_vp @ self.T_head_to_left_wrist_rot.T
        # right_drot_vp = np.dot(right_wrist_pose[:3, :3], last_right_wrist_pose[:3, :3].T)
        right_drot_vp = np.dot(right_wrist_pose[:3, :3], np.linalg.inv(last_right_wrist_pose[:3, :3]))
        # right_drot_base = self.T_head_to_right_wrist_rot @ right_drot_vp @ self.T_head_to_right_wrist_rot.T
        
        # # head pose默认不变，中间坐标系
        # self.last_left_pos = np.array(self.left_pose)
        # self.last_right_pos = np.array(self.right_pose)
        # 不重排位置，对应input2action也不需要重排x/y位置
        # left_drot_base_axisangle = T.quat2axisangle(T.mat2quat(left_drot_base))  # save to dict for return
        left_drot_base_axisangle = T.quat2axisangle(T.mat2quat(left_drot_vp))

        # right_drot_base_axisangle = T.quat2axisangle(T.mat2quat(right_drot_base))  # save to dict for return
        right_drot_base_axisangle = T.quat2axisangle(T.mat2quat(right_drot_vp))

        # 更新状态（在计算完增量之后）
        # 类似 keyboard: self.last_pos = np.array(self.pos)
        self._pos_left = left_wrist_pose[:3, 3].copy()
        self._pos_right = right_wrist_pose[:3, 3].copy()
        self._rotation_left = left_wrist_pose[:3, :3].copy()
        self._rotation_right = right_wrist_pose[:3, 3:].copy()
        self._last_pose_left = left_wrist_pose
        self._last_pose_right = right_wrist_pose
        self._last_pos_left = self._pos_left.copy()
        self._last_pos_right = self._pos_right.copy()
        
        self._raw_drotation_left = left_drot_base_axisangle.copy()
        self._raw_drotation_right = right_drot_base_axisangle.copy()

        # 获取夹爪状态
        # 根据环境中的gripper dof判断gripper类型
        gripper_type_left = 'dexhand' if "Dex" in self.env.robots[0].name else 'twofinger'
        gripper_type_right = 'dexhand' if "Dex" in self.env.robots[1].name else 'twofinger'

        if hand_data_left_raw is not None:
            gripper_left = self._get_gripper_from_hand_data(
                hand_data_left_raw, 
                gripper=gripper_type_left
            )
        # 如果检测不到左手，保持上一帧状态
        
        if hand_data_right_raw is not None:
            gripper_right = self._get_gripper_from_hand_data(
                hand_data_right_raw, 
                gripper=gripper_type_right
            )
        # 如果检测不到右手，保持上一帧状态

        return dict(
            dpos_left=left_dpos_base,
            dpos_right=right_dpos_base,
            rotation_left=left_wrist_pose[:3, :3],
            rotation_right=right_wrist_pose[:3, :3],
            raw_drotation_left=left_drot_base_axisangle,
            raw_drotation_right=right_drot_base_axisangle,
            grasp_left=gripper_left,
            grasp_right=gripper_right,
            reset=self._reset_state,
            base_mode=int(self.base_mode),
            )


    """
    def get_controller_state(self) -> Dict:
        
        # 获取控制器状态
        
        # 类似 keyboard 的 get_controller_state，计算当前姿态相对于初始姿态（last_pose）的增量。
        # 初始姿态在 start_control() 中通过校准获得，并设置为 _last_pose_left 和 _last_pose_right。
        
        # Returns:
        #     dict: 控制器状态字典，包含位置增量、旋转等
        
        # 获取最新数据
        try:
            data = self.streamer.get_latest()
        except Exception as e:
            print(f"Warning: Failed to get Vision Pro data: {e}")
            return None
        
        hand_data_left_raw = data.left if hasattr(data, 'left') else None
        hand_data_right_raw = data.right if hasattr(data, 'right') else None
        
        pose_left = self._get_hand_pose_from_data(hand_data_left_raw)
        pose_right = self._get_hand_pose_from_data(hand_data_right_raw)
        
        # 应用坐标变换：将wrist pose从Vision Pro坐标系变换到SIM_HEAD（base）坐标系
        # 这样可以确保所有计算都在base坐标系中进行，delta action可以直接用于OSC_POSE控制器的delta模式
        if pose_left is not None:
            transformed_pos_left, transformed_rot_left = self._transform_wrist_pose_to_base_frame(
                pose_left[0], pose_left[1], is_left=True
            )
            pose_left = (transformed_pos_left, transformed_rot_left)
        
        if pose_right is not None:
            transformed_pos_right, transformed_rot_right = self._transform_wrist_pose_to_base_frame(
                pose_right[0], pose_right[1], is_left=False
            )
            pose_right = (transformed_pos_right, transformed_rot_right)
        
        # 调试信息：检查左右手数据是否被正确获取
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print("=" * 60)
            print("VisionPro左右手数据检查")
            print("=" * 60)
            print(f"data.left存在: {hand_data_left_raw is not None}")
            print(f"data.right存在: {hand_data_right_raw is not None}")
            print(f"pose_left获取成功: {pose_left is not None}")
            print(f"pose_right获取成功: {pose_right is not None}")
            if pose_left is not None:
                print(f"pose_left位置: {pose_left[0]}")
            if pose_right is not None:
                print(f"pose_right位置: {pose_right[0]}")
            print("注意：已应用坐标变换，将wrist pose从Vision Pro坐标系变换到SIM_HEAD（base）坐标系")
            print("=" * 60)
        
        # 如果检测不到手部，保持上一帧状态
        if pose_left is None:
            pose_left = self._last_pose_left if self._last_pose_left is not None else self._initial_pose_left
            # 如果使用上一帧状态，dpos会是0，这是正常的
        if pose_right is None:
            pose_right = self._last_pose_right if self._last_pose_right is not None else self._initial_pose_right
            # 如果使用上一帧状态，dpos会是0，这是正常的
        
        # 使用 last_pose 作为参考（类似 keyboard 的 last_pos）
        # last_pose 在 start_control() 中被设置为初始姿态，后续会更新为上一帧姿态
        prev_pose_left = self._last_pose_left if self._last_pose_left is not None else self._initial_pose_left
        prev_pose_right = self._last_pose_right if self._last_pose_right is not None else self._initial_pose_right
        
        # 计算位置增量（相对于上一帧的变化，用于delta控制）
        # 类似 keyboard: dpos = self.pos - self.last_pos
        dpos_left = pose_left[0] - prev_pose_left[0]
        dpos_right = pose_right[0] - prev_pose_right[0]
        
        # 计算旋转增量（相对于上一帧的变化）
        # 计算从上一帧到当前帧的相对旋转
        drot_left_mat = pose_left[1] @ prev_pose_left[1].T
        drot_right_mat = pose_right[1] @ prev_pose_right[1].T
        
        # 转换为axis-angle
        drot_left_quat = T.mat2quat(drot_left_mat)
        drot_right_quat = T.mat2quat(drot_right_mat)
        
        raw_drotation_left = T.quat2axisangle(drot_left_quat)
        raw_drotation_right = T.quat2axisangle(drot_right_quat)
        
        # 更新状态（在计算完增量之后）
        # 类似 keyboard: self.last_pos = np.array(self.pos)
        self._pos_left = pose_left[0].copy()
        self._pos_right = pose_right[0].copy()
        self._rotation_left = pose_left[1].copy()
        self._rotation_right = pose_right[1].copy()
        self._last_pose_left = pose_left
        self._last_pose_right = pose_right
        self._last_pos_left = self._pos_left.copy()
        self._last_pos_right = self._pos_right.copy()
        
        self._raw_drotation_left = raw_drotation_left.copy()
        self._raw_drotation_right = raw_drotation_right.copy()
        
        # 获取夹爪状态
        # 根据环境中的gripper dof判断gripper类型
        gripper_type_left = 'dexhand' if "Dex" in self.env.robots[0].name else 'twofinger'
        gripper_type_right = 'dexhand' if "Dex" in self.env.robots[1].name else 'twofinger'
        
        
        # 获取夹爪状态（只有在检测到手部数据时才计算，否则使用上一帧状态）
        # 注意：这里使用hand_data_left_raw和hand_data_right_raw，已经根据swap_hands处理过了
        hand_data_left = hand_data_left_raw if hand_data_left_raw is not None else None
        hand_data_right = hand_data_right_raw if hand_data_right_raw is not None else None
        
        if hand_data_left is not None:
            self._gripper_left = self._get_gripper_from_hand_data(
                hand_data_left, 
                gripper=gripper_type_left
            )
        # 如果检测不到左手，保持上一帧状态
        
        if hand_data_right is not None:
            self._gripper_right = self._get_gripper_from_hand_data(
                hand_data_right, 
                gripper=gripper_type_right
            )
        # 如果检测不到右手，保持上一帧状态
        
        return dict(
            dpos_left=dpos_left,
            dpos_right=dpos_right,
            rotation_left=self._rotation_left,
            rotation_right=self._rotation_right,
            raw_drotation_left=raw_drotation_left,
            raw_drotation_right=raw_drotation_right,
                    grasp_left=self._gripper_left,
                    grasp_right=self._gripper_right,
                    reset=self._reset_state,
                    base_mode=0,
                )
    """

    def input2action(self, mirror_actions=False, goal_update_mode="target") -> Optional[Dict]:
        """
        修改的input2action

        参考device类的input2action方法
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

        NOTIMPLEMENTED： If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action
        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.
            goal_update_mode (str): the mode to update the goal in. Can be 'target' or 'achieved'.
            If 'target', the goal is updated based on the current target goal. If 'achieved', the goal is updated based on the current achieved state.

        Returns:
            Optional[Dict]: Dictionary of actions to be fed into env.step()
                            if reset is triggered, returns None
        """
        left_robot = self.env.robots[0]
        right_robot = self.env.robots[1]
        # 多机器人时，每个robot有各自的active_arm_index
        left_arm = self.all_robot_arms[0][self.active_arm_indices[0]]
        right_arm = self.all_robot_arms[1][self.active_arm_indices[1]]
        state = self.get_controller_state()
        raw_dpos_left, raw_dpos_right, left_raw_drotation,right_raw_drotation, grasp_left, grasp_right, reset = (
            state["dpos_left"],
            state["dpos_right"],
            state["raw_drotation_left"],
            state["raw_drotation_right"],
            state["grasp_left"],
            state["grasp_right"],
            state["reset"],
        )
        if mirror_actions:
            dpos_left[0] *= -1
            dpos_left[1] *= -1
            left_raw_drotation[0] *= -1
            left_raw_drotation[1] *= -1
            dpos_right[0] *= -1
            dpos_right[1] *= -1
            right_raw_drotation[0] *= -1
            right_raw_drotation[1] *= -1

        # If we're resetting, immediately return None
        if reset:
            return None

        # Get controller reference
        controllers = (left_robot.part_controllers[left_arm], right_robot.part_controllers[right_arm])
        # controller = robot.part_controllers[active_arm]
        # grippers = (left_robot.gripper[active_arm[0]], right_robot.gripper[active_arm[1]])
        # gripper_dofs = (left_robot.gripper[active_arm[0]].dof, right_robot.gripper[active_arm[1]].dof)
        for controller in controllers:
            assert controller.name in ["OSC_POSE", "JOINT_POSITION"], "only supporting OSC_POSE and JOINT_POSITION for now"

        # 处理旋转，需要参考get_controller_state和末端执行器的方向来确定
        # 先按照以下的逻辑：先将drot_vp变换到drot_base，再根据末端和base的关系调整drot

        dpos_left = raw_dpos_left.copy()
        dpos_right = raw_dpos_right.copy()

        # dpos_left[0], dpos_left[1], dpos_left[2] = -raw_dpos_left[2], -raw_dpos_left[1], -raw_dpos_left[0]
        # dpos_right[0], dpos_right[1], dpos_right[2] = raw_dpos_right[2], raw_dpos_right[1], -raw_dpos_right[0]

        drotation_left = left_raw_drotation
        # drotation_left[0], drotation_left[1], drotation_left[2] = -left_raw_drotation[2], -left_raw_drotation[1], left_raw_drotation[0]

        drotation_right = right_raw_drotation
        # drotation_right[0], drotation_right[1], drotation_right[2] = -right_raw_drotation[2], right_raw_drotation[1], right_raw_drotation[0]

        # 缩放旋转和位置，根据设备和机器人来确定缩放系数
        dpos_left, drotation_left = self._postprocess_device_outputs(dpos_left, drotation_left)
        dpos_right, drotation_right = self._postprocess_device_outputs(dpos_right, drotation_right)

        action_dict = {}
        for arm in left_robot.arms:
            arm_action = self.get_arm_action(
                left_robot,
                arm,
                norm_delta=np.zeros(6),
                goal_update_mode=goal_update_mode,
            )
            action_dict["left_abs"] = arm_action["abs"]
            action_dict["left_delta"] = arm_action["delta"]
            action_dict["left_gripper"] = np.zeros(left_robot.gripper[arm].dof)
        for arm in right_robot.arms:
            arm_action = self.get_arm_action(
                right_robot,
                arm,
                norm_delta=np.zeros(6),
                goal_update_mode=goal_update_mode,
            )
            action_dict["right_abs"] = arm_action["abs"]
            action_dict["right_delta"] = arm_action["delta"]
            action_dict["right_gripper"] = np.zeros(right_robot.gripper[arm].dof)
        
        if left_robot.is_mobile:
            raise NotImplementedError("Mobile robot is not supported for now")
        else:
            left_arm_norm_delta = np.concatenate([dpos_left, drotation_left])
            left_device_torso_input = 0.0
        
        if right_robot.is_mobile:
            raise NotImplementedError("Mobile robot is not supported for now")
        else:
            right_arm_norm_delta = np.concatenate([dpos_right, drotation_right])
            right_device_torso_input = 0.0
        
        #=============================== left arm and left gripper ===============================
        left_arm_action = self.get_arm_action(
            left_robot,
            left_arm,
            norm_delta=left_arm_norm_delta
        )
        action_dict["left_abs"] = left_arm_action["abs"]
        action_dict["left_delta"] = left_arm_action["delta"]
    
        # 设置夹爪动作
        left_gripper_dof = left_robot.gripper[left_robot.arms[0]].dof
        left_gripper_key = "left_gripper"
        
        # 检查grasp_left是否是数组（dexhand类型）
        if isinstance(grasp_left, np.ndarray) and len(grasp_left) > 1:
            # dexhand类型：直接使用数组作为gripper action
            if len(grasp_left) == left_gripper_dof:
                action_dict[left_gripper_key] = grasp_left.copy()
            else:
                print(f"Warning: grasp_left dimension mismatch for {left_gripper_key}. "
                        f"Expected {left_gripper_dof}, got {len(grasp_left)}. Using zeros.")
                action_dict[left_gripper_key] = np.zeros(left_gripper_dof)
        else:
            # twofinger类型：标量值，转换为gripper action
            if hasattr(left_robot.gripper[left_robot.arms[0]], "grasp_qpos"):
                grasp_val = 1 if (grasp_left > 0.5 if isinstance(grasp_left, (int, float, np.number)) else False) else -1
                gripper_action = left_robot.gripper[left_robot.arms[0]].grasp_qpos[grasp_val]
                # 确保返回的gripper action维度与gripper dof匹配
                if len(gripper_action) != left_gripper_dof:
                    print(f"Warning: grasp_qpos returned wrong dimension for {left_gripper_key}. "
                            f"Expected {left_gripper_dof}, got {len(gripper_action)}. Using direct action.")
                    grasp_action = (grasp_left * 2.0 - 1.0) if isinstance(grasp_left, (int, float, np.number)) else -1.0
                    action_dict[left_gripper_key] = np.array([grasp_action] * left_gripper_dof)
                else:
                    action_dict[left_gripper_key] = gripper_action
            else:
                grasp_action = (grasp_left * 2.0 - 1.0) if isinstance(grasp_left, (int, float, np.number)) else -1.0
                action_dict[left_gripper_key] = np.array([grasp_action] * left_gripper_dof)
        
        # 验证最终left gripper action的维度
        if len(action_dict[left_gripper_key]) != left_gripper_dof:
            print(f"Error: Final gripper action dimension mismatch for {left_gripper_key}. "
                    f"Expected {left_gripper_dof}, got {len(action_dict[left_gripper_key])}. "
                    f"Fixing to correct dimension.")
            action_dict[left_gripper_key] = np.zeros(left_gripper_dof)
        # ================================================================================
        # =======================right arm and right gripper ============================
        right_arm_action = self.get_arm_action(
            right_robot,
            right_arm,
            norm_delta=right_arm_norm_delta
        )
        action_dict["right_abs"] = right_arm_action["abs"]
        action_dict["right_delta"] = right_arm_action["delta"]
    
        # 设置夹爪动作
        right_gripper_dof = right_robot.gripper[right_robot.arms[0]].dof
        right_gripper_key = "right_gripper"
        
        # 检查grasp_right是否是数组（dexhand类型）
        if isinstance(grasp_right, np.ndarray) and len(grasp_right) > 1:
            # dexhand类型：直接使用数组作为gripper action
            if len(grasp_right) == right_gripper_dof:
                action_dict[right_gripper_key] = grasp_right.copy()
            else:
                print(f"Warning: grasp_right dimension mismatch for {right_gripper_key}. "
                        f"Expected {right_gripper_dof}, got {len(grasp_right)}. Using zeros.")
                action_dict[right_gripper_key] = np.zeros(right_gripper_dof)
        else:
            # twofinger类型：标量值，转换为gripper action
            if hasattr(right_robot.gripper[right_robot.arms[0]], "grasp_qpos"):
                grasp_val = 1 if (grasp_right > 0.5 if isinstance(grasp_right, (int, float, np.number)) else False) else -1
                gripper_action = right_robot.gripper[right_robot.arms[0]].grasp_qpos[grasp_val]
                # 确保返回的gripper action维度与gripper dof匹配
                if len(gripper_action) != right_gripper_dof:
                    print(f"Warning: grasp_qpos returned wrong dimension for {right_gripper_key}. "
                            f"Expected {right_gripper_dof}, got {len(gripper_action)}. Using direct action.")
                    grasp_action = (grasp_right * 2.0 - 1.0) if isinstance(grasp_right, (int, float, np.number)) else -1.0
                    action_dict[right_gripper_key] = np.array([grasp_action] * right_gripper_dof)
                else:
                    action_dict[right_gripper_key] = gripper_action
            else:
                grasp_action = (grasp_right * 2.0 - 1.0) if isinstance(grasp_right, (int, float, np.number)) else -1.0
                action_dict[right_gripper_key] = np.array([grasp_action] * right_gripper_dof)
        
        # 验证最终right gripper action的维度
        if len(action_dict[right_gripper_key]) != right_gripper_dof:
            print(f"Error: Final gripper action dimension mismatch for {right_gripper_key}. "
                    f"Expected {right_gripper_dof}, got {len(action_dict[right_gripper_key])}. "
                    f"Fixing to correct dimension.")
            action_dict[right_gripper_key] = np.zeros(right_gripper_dof)

        # 裁剪动作
        for (k, v) in action_dict.items():
            if "abs" not in k and "gripper" not in k:
                action_dict[k] = np.clip(v, -1, 1)
        
        return action_dict
        
    """
    def input2action(self, mirror_actions=False, goal_update_mode="target") -> Optional[Dict]:
        
        # 将Vision Pro输入转换为动作（覆盖基类方法以支持双臂同时控制）
        
        # Args:
        #     mirror_actions: 是否镜像动作
        #     goal_update_mode: 目标更新模式
            
        # Returns:
        #     dict: 动作字典，包含左右手臂的动作
        
        state = self.get_controller_state()
        
        # 如果未初始化，返回零动作
        if not self._initialized:
            robot = self.env.robots[0]
            action_dict = {}
            for arm in robot.arms:
                arm_action = self.get_arm_action(
                    robot,
                    arm,
                    norm_delta=np.zeros(6),
                    goal_update_mode=goal_update_mode,
                )
                action_dict[f"{arm}_abs"] = arm_action["abs"]
                action_dict[f"{arm}_delta"] = arm_action["delta"]
                action_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
            return action_dict
        
        # 如果重置
        if state["reset"]:
            return None
        
        # 获取左右手的状态
        dpos_left_raw = state["dpos_left"]
        dpos_right_raw = state["dpos_right"]
        raw_drotation_left = state["raw_drotation_left"]
        raw_drotation_right = state["raw_drotation_right"]
        grasp_left = state["grasp_left"]
        grasp_right = state["grasp_right"]
        
        # 应用位置轴映射和符号变换（可选）
        # 注意：现在位置和旋转都已经在base坐标系中了（通过_transform_wrist_pose_to_base_frame变换）
        # 如果仍然需要调整轴的方向（例如Vision Pro的坐标系定义与机器人base坐标系不完全一致），
        # 可以使用pos_axis_map和pos_axis_signs参数进行微调
        if hasattr(self, '_transform_position_axes'):
            dpos_left = self._transform_position_axes(dpos_left_raw)
            dpos_right = self._transform_position_axes(dpos_right_raw)
        else:
            # 直接使用base坐标系中的位置增量（已经通过SIM_HEAD变换）
            dpos_left = dpos_left_raw
            dpos_right = dpos_right_raw
        
        
        # 调试信息：检查左右手数据（仅前几次）
        if not hasattr(self, '_debug_action_count'):
            self._debug_action_count = 0
        if self._debug_action_count < 5:
            print(f"Debug input2action [{self._debug_action_count}]:")
            print(f"  dpos_left norm: {np.linalg.norm(dpos_left):.6f}, dpos_right norm: {np.linalg.norm(dpos_right):.6f}")
            print(f"  raw_drotation_left norm: {np.linalg.norm(raw_drotation_left):.6f}, raw_drotation_right norm: {np.linalg.norm(raw_drotation_right):.6f}")
            self._debug_action_count += 1
        
        # 处理旋转（类似基类的处理方式）
        # 注意：旋转矩阵已经通过SIM_HEAD变换到base坐标系了，但axis-angle的轴顺序和符号
        # 需要匹配机器人控制器的期望格式（这是robosuite的标准做法）
        # 重排轴顺序：[y, x, z] -> [x, y, z]
        drotation_left = raw_drotation_left[[1, 0, 2]]
        drotation_right = raw_drotation_right[[1, 0, 2]]
        # 反转z轴符号（控制器格式要求）
        drotation_left[2] = -drotation_left[2]
        drotation_right[2] = -drotation_right[2]
        
        # 后处理
        dpos_left, drotation_left = self._postprocess_device_outputs(dpos_left, drotation_left)
        dpos_right, drotation_right = self._postprocess_device_outputs(dpos_right, drotation_right)
        
        # 构建动作字典
        action_dict = {}
        
        # 处理多个机器人的情况
        # 如果环境有多个机器人，将左手映射到第一个机器人，右手映射到第二个机器人
        # 如果只有一个机器人但有多个手臂，将左右手映射到机器人的左右手臂
        num_robots = len(self.env.robots)
        
        if num_robots >= 2:
            # 多机器人情况：第一个机器人用左手，第二个机器人用右手
            # 处理第一个机器人（左手）
            robot_left = self.env.robots[0]
            arm_names_left = robot_left.arms
            # 使用第一个手臂（通常是left或第一个）
            left_arm = arm_names_left[0] if len(arm_names_left) > 0 else None
            
            if left_arm is not None:
                arm_norm_delta = np.concatenate([dpos_left, drotation_left])
                arm_action = self.get_arm_action(
                    robot_left,
                    left_arm,
                    norm_delta=arm_norm_delta,
                    goal_update_mode=goal_update_mode,
                )
                action_dict["left_abs"] = arm_action["abs"]
                action_dict["left_delta"] = arm_action["delta"]
                
                # 设置夹爪动作
                gripper_dof = robot_left.gripper[left_arm].dof
                gripper_key = "left_gripper"
                
                # 检查grasp_left是否是数组（dexhand类型）
                if isinstance(grasp_left, np.ndarray) and len(grasp_left) > 1:
                    # dexhand类型：直接使用数组作为gripper action
                    if len(grasp_left) == gripper_dof:
                        action_dict[gripper_key] = grasp_left.copy()
                    else:
                        print(f"Warning: grasp_left dimension mismatch for {gripper_key}. "
                              f"Expected {gripper_dof}, got {len(grasp_left)}. Using zeros.")
                        action_dict[gripper_key] = np.zeros(gripper_dof)
                else:
                    # twofinger类型：标量值，转换为gripper action
                    if hasattr(robot_left.gripper[left_arm], "grasp_qpos"):
                        grasp_val = 1 if (grasp_left > 0.5 if isinstance(grasp_left, (int, float, np.number)) else False) else -1
                        gripper_action = robot_left.gripper[left_arm].grasp_qpos[grasp_val]
                        # 确保返回的gripper action维度与gripper dof匹配
                        if len(gripper_action) != gripper_dof:
                            print(f"Warning: grasp_qpos returned wrong dimension for {gripper_key}. "
                                  f"Expected {gripper_dof}, got {len(gripper_action)}. Using direct action.")
                            grasp_action = (grasp_left * 2.0 - 1.0) if isinstance(grasp_left, (int, float, np.number)) else -1.0
                            action_dict[gripper_key] = np.array([grasp_action] * gripper_dof)
                        else:
                            action_dict[gripper_key] = gripper_action
                    else:
                        grasp_action = (grasp_left * 2.0 - 1.0) if isinstance(grasp_left, (int, float, np.number)) else -1.0
                        action_dict[gripper_key] = np.array([grasp_action] * gripper_dof)
                
                # 验证最终gripper action的维度
                if len(action_dict[gripper_key]) != gripper_dof:
                    print(f"Error: Final gripper action dimension mismatch for {gripper_key}. "
                          f"Expected {gripper_dof}, got {len(action_dict[gripper_key])}. "
                          f"Fixing to correct dimension.")
                    action_dict[gripper_key] = np.zeros(gripper_dof)
            
            # 处理第二个机器人（右手）
            robot_right = self.env.robots[1]
            arm_names_right = robot_right.arms
            # 使用第一个手臂（通常是right或第一个）
            right_arm = arm_names_right[0] if len(arm_names_right) > 0 else None
            
            if right_arm is not None:
                arm_norm_delta = np.concatenate([dpos_right, drotation_right])
                arm_action = self.get_arm_action(
                    robot_right,
                    right_arm,
                    norm_delta=arm_norm_delta,
                    goal_update_mode=goal_update_mode,
                )
                action_dict["right_abs"] = arm_action["abs"]
                action_dict["right_delta"] = arm_action["delta"]
                
                # 设置夹爪动作
                gripper_dof = robot_right.gripper[right_arm].dof
                gripper_key = "right_gripper"
                
                # 检查grasp_right是否是数组（dexhand类型）
                if isinstance(grasp_right, np.ndarray) and len(grasp_right) > 1:
                    # dexhand类型：直接使用数组作为gripper action
                    if len(grasp_right) == gripper_dof:
                        action_dict[gripper_key] = grasp_right.copy()
                    else:
                        print(f"Warning: grasp_right dimension mismatch for {gripper_key}. "
                              f"Expected {gripper_dof}, got {len(grasp_right)}. Using zeros.")
                        action_dict[gripper_key] = np.zeros(gripper_dof)
                else:
                    # twofinger类型：标量值，转换为gripper action
                    if hasattr(robot_right.gripper[right_arm], "grasp_qpos"):
                        grasp_val = 1 if (grasp_right > 0.5 if isinstance(grasp_right, (int, float, np.number)) else False) else -1
                        gripper_action = robot_right.gripper[right_arm].grasp_qpos[grasp_val]
                        # 确保返回的gripper action维度与gripper dof匹配
                        if len(gripper_action) != gripper_dof:
                            print(f"Warning: grasp_qpos returned wrong dimension for {gripper_key}. "
                                  f"Expected {gripper_dof}, got {len(gripper_action)}. Using direct action.")
                            grasp_action = (grasp_right * 2.0 - 1.0) if isinstance(grasp_right, (int, float, np.number)) else -1.0
                            action_dict[gripper_key] = np.array([grasp_action] * gripper_dof)
                        else:
                            action_dict[gripper_key] = gripper_action
                    else:
                        grasp_action = (grasp_right * 2.0 - 1.0) if isinstance(grasp_right, (int, float, np.number)) else -1.0
                        action_dict[gripper_key] = np.array([grasp_action] * gripper_dof)
                
                # 验证最终gripper action的维度
                if len(action_dict[gripper_key]) != gripper_dof:
                    print(f"Error: Final gripper action dimension mismatch for {gripper_key}. "
                          f"Expected {gripper_dof}, got {len(action_dict[gripper_key])}. "
                          f"Fixing to correct dimension.")
                    action_dict[gripper_key] = np.zeros(gripper_dof)
        else:
            print(f"Error: num_robots must be >= 2, got {num_robots}")
            return None
        
        # 裁剪动作
        for (k, v) in action_dict.items():
            if "abs" not in k and "gripper" not in k:
                action_dict[k] = np.clip(v, -1, 1)
        
        return action_dict
    """

    # def _transform_position_axes(self, dpos: np.ndarray) -> np.ndarray:
    #     """
    #     变换位置坐标轴（映射和符号反转）
        
    #     Args:
    #         dpos: Vision Pro坐标系下的位置增量 (3,)
            
    #     Returns:
    #         np.ndarray: 机器人坐标系下的位置增量 (3,)
    #     """
    #     # 首先重排轴
    #     dpos_mapped = dpos[list(self.pos_axis_map)]
    #     # 然后应用符号变换
    #     dpos_transformed = dpos_mapped * np.array(self.pos_axis_signs)
    #     return dpos_transformed
    
    def _postprocess_device_outputs(self, dpos, drotation):
        """
        后处理设备输出（缩放和裁剪）
        
        Args:
            dpos: 位置增量
            drotation: 旋转增量（axis-angle）
            
        Returns:
            tuple: (处理后的dpos, 处理后的drotation)
        """
        # 应用敏感度
        dpos = dpos * self.pos_sensitivity
        drotation = drotation * self.rot_sensitivity
        
        # 缩放（类似Keyboard设备）
        # 位置缩放：Vision Pro的数据单位是米，需要适当缩放
        dpos = dpos * 20.0  # 可以根据实际情况调整
        drotation = drotation * 1.5
        
        # 裁剪到[-1, 1]
        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)
        
        return dpos, drotation
