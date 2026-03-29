import numpy as np

# 明确导入需要的基类，避免循环导入和缓存问题
from robosuite.models.robots.manipulators.panda_robot import Panda
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.models.robots.manipulators.spot_arm import SpotArm


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0 - 0.2, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.4, np.pi / 4])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-1.05, -0.1, -0.22),
            "empty": (-1.1, 0, -0.22),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, -0.22),
        }


class SpotWithArmFloating(SpotArm):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.7, -0.1, 0.0),
            "empty": (-0.6, 0, 0.0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.0),
        }


class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [-0.5, 0.5, 0.5, -0.5]}


class PandaDexLH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.01]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, -0.5]}


class UR5eInspireDexRH(UR5e):
    """
    UR5e 机器人配置，使用 InspireRightHand 灵巧手作为末端执行器
    
    这个类继承自 UR5e，并重写了 default_gripper 属性来使用 InspireRightHand，
    同时定义了 gripper_mount_pos_offset 和 gripper_mount_quat_offset 来指定
    灵巧手相对于机械臂末端的安装位置和姿态。
    
    偏移值根据 URDF 文件中的 mount_joint 定义：
    <joint name="mount_joint" type="fixed">
      <parent link="wrist_3_link"/>
      <child link="hand_base_link"/>
      <origin rpy="-1.57079 0 1.57079" xyz="0.0 0.0 -0.01"/>
    </joint>
    
    其中：
    - xyz="0.0 0.0 -0.01" 表示位置偏移：在 z 方向偏移 -0.01 米
    - rpy="-1.57079 0 1.57079" 表示欧拉角旋转（roll=-90°, pitch=0°, yaw=90°）
    """
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        """
        定义灵巧手相对于机械臂末端的位置偏移（单位：米）
        格式：[x, y, z]，相对于末端执行器坐标系
        
        根据 URDF 文件中的 mount_joint，位置偏移为 [0.0, 0.0, -0.01]
        表示在 z 方向（沿机械臂末端轴向）偏移 -0.01 米
        """
        return {"right": [0.0, 0.0, -0.01]}

    @property
    def gripper_mount_quat_offset(self):
        """
        定义灵巧手相对于机械臂末端的姿态偏移（四元数）
        格式：[w, x, y, z]，用于调整灵巧手的安装方向
        
        根据 URDF 文件中的 mount_joint，RPY 欧拉角为 (-1.57079, 0, 1.57079) 弧度
        转换为四元数：约 [0.5, -0.5, -0.5, 0.5]
        
        这个旋转将灵巧手从默认方向调整到正确的安装姿态。
        """
        # 从 URDF RPY (-1.57079, 0, 1.57079) 转换得到的四元数
        # 精确值: [0.5000031634, -0.5000000000, -0.4999968366, 0.5000000000]
        # 使用近似值: [0.5, -0.5, -0.5, 0.5]
        return {"right": [0.5, -0.5, 0.5, -0.5]}


class UR5eInspireDexLH(UR5e):
    """
    UR5e 机器人配置，使用 InspireLeftHand 灵巧手作为末端执行器
    """
    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, -0.01]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, -0.5, 0.5]}


class UR5eDex(UR5eInspireDexRH):
    """
    Alias of UR5e + InspireRightHand composition with a shorter public name.
    """
