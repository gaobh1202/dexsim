# OSC控制器绝对位姿控制说明

## 概述

OSC（Operational Space Controller）控制器**完全支持**以末端位姿为目标的绝对控制。

## 关键配置参数

### 1. `input_type`
- **`"delta"`** (默认): 增量控制，动作是相对于当前位置的变化
- **`"absolute"`**: 绝对控制，动作是目标位姿的绝对值

### 2. `input_ref_frame`
绝对位姿的参考坐标系：
- **`"base"`** (默认): 绝对位姿相对于机器人base坐标系
- **`"world"`**: 绝对位姿相对于世界坐标系

## 绝对位姿格式

绝对位姿是一个**6维向量**：`[x, y, z, rx, ry, rz]`

- **前3维**：位置（米）
  - `x, y, z`: 在参考坐标系中的位置坐标
  
- **后3维**：旋转向量（axis-angle，弧度）
  - `rx, ry, rz`: 旋转向量，表示绕某个轴的旋转角度
  - 可以通过 `scipy.spatial.transform.Rotation.from_rotvec()` 转换为旋转矩阵

## 配置示例

### 配置文件：`default_panda_absolute.json`

```json
{
    "type": "BASIC",
    "body_parts": {
        "arms": {
            "right": {
                "type": "OSC_POSE",
                "input_type": "absolute",        // 关键：设置为绝对控制
                "input_ref_frame": "base",        // 参考坐标系
                "output_max": [1.0, 1.0, 1.0, 3.14, 3.14, 3.14],
                "output_min": [-1.0, -1.0, -1.0, -3.14, -3.14, -3.14],
                "kp": 150,
                "damping_ratio": 1,
                ...
            }
        }
    }
}
```

## 代码实现原理

### 1. 设置目标位姿（`set_goal`方法）

```python
if self.input_type == "absolute":
    abs_action = goal_update  # 6维向量
    self.goal_pos = abs_action[0:3]  # 直接使用绝对位置
    if self.use_ori is True:
        # 将旋转向量转换为旋转矩阵
        self.goal_ori = Rotation.from_rotvec(abs_action[3:6]).as_matrix()
```

### 2. 坐标系转换（`run_controller`方法）

根据`input_ref_frame`将绝对位姿转换到世界坐标系：

```python
if self.input_ref_frame == "base":
    # base坐标系 -> 世界坐标系
    desired_world_pos = self.origin_pos + np.dot(self.origin_ori, self.goal_pos)
    desired_world_ori = np.dot(self.origin_ori, self.goal_ori)
elif self.input_ref_frame == "world":
    # 直接使用世界坐标系
    desired_world_pos = self.goal_pos
    desired_world_ori = self.goal_ori
```

### 3. 计算控制力矩

控制器计算位置和旋转误差，然后使用OSC算法计算所需的关节力矩：

```python
position_error = desired_world_pos - self.ref_pos
ori_error = orientation_error(desired_world_ori, self.ref_ori_mat)

# 使用PD控制计算期望力和力矩
desired_force = kp * position_error + kd * vel_error
desired_torque = kp * ori_error + kd * vel_ori_error

# 通过OSC算法转换为关节力矩
torques = J^T * F + gravity_compensation
```

## 使用示例

### 示例1：设置绝对位姿（base坐标系）

```python
import numpy as np
from scipy.spatial.transform import Rotation

# 定义目标位姿（相对于base坐标系）
target_pos = np.array([0.5, 0.0, 0.5])  # 位置：x=0.5m, y=0m, z=0.5m

# 定义目标旋转（例如：保持当前旋转，或稍微旋转）
target_rot_matrix = np.eye(3)  # 单位矩阵表示无旋转
rot = Rotation.from_matrix(target_rot_matrix)
target_rotvec = rot.as_rotvec()  # 转换为旋转向量

# 构建绝对位姿action
absolute_action = np.concatenate([target_pos, target_rotvec])

# 使用action
action_dict = {
    "right": absolute_action,
    "right_gripper": np.array([0.0])
}
env_action = robot.create_action_vector(action_dict)
env.step(env_action)
```

### 示例2：从当前位姿计算目标位姿

```python
# 获取当前末端位姿
controller.update()
current_pos = controller.ref_pos  # 世界坐标系

# 如果使用base坐标系，需要转换
if controller.input_ref_frame == "base":
    import robosuite.utils.transform_utils as T
    base_pos = np.array(env.sim.data.get_body_xpos(robot.robot_model.root_body))
    base_rot = np.array(env.sim.data.get_body_xmat(robot.robot_model.root_body)).reshape(3, 3)
    
    # 转换到base坐标系
    T_world_base = T.make_pose(base_pos, base_rot)
    T_world_eef = T.make_pose(current_pos, controller.ref_ori_mat)
    T_base_eef = T.pose_in_A_to_pose_in_B(T_world_eef, T.pose_inv(T_world_base))
    current_pos_base, current_ori_base = T.mat2pose(T_base_eef)
    
    # 在base坐标系中设置目标
    target_pos = current_pos_base + np.array([0.1, 0.0, 0.1])  # 向前0.1m，向上0.1m
    target_rot = current_ori_base
else:
    # 在世界坐标系中设置目标
    target_pos = current_pos + np.array([0.1, 0.0, 0.1])
    target_rot = controller.ref_ori_mat

# 转换为旋转向量
rot = Rotation.from_matrix(target_rot)
target_rotvec = rot.as_rotvec()

# 构建绝对位姿action
absolute_action = np.concatenate([target_pos, target_rotvec])
```

## 与增量控制的对比

| 特性 | 增量控制 (`delta`) | 绝对控制 (`absolute`) |
|------|-------------------|---------------------|
| 输入格式 | `[dx, dy, dz, drx, dry, drz]` | `[x, y, z, rx, ry, rz]` |
| 含义 | 相对于当前位置的变化 | 目标位姿的绝对值 |
| 优点 | 更直观，适合遥操作 | 精确控制，适合轨迹规划 |
| 缺点 | 可能累积误差 | 需要知道目标位姿 |
| 适用场景 | 手动控制、增量移动 | 精确轨迹、预设位姿 |

## 注意事项

1. **坐标系选择**：
   - `"base"`：适合固定base的机器人，位姿相对于机器人本体
   - `"world"`：适合移动机器人，位姿相对于世界坐标系

2. **旋转向量范围**：
   - 旋转向量通常在 `[-π, π]` 范围内
   - 如果旋转角度较大，可能需要归一化

3. **输出范围**：
   - `output_max` 和 `output_min` 定义了action的缩放范围
   - 对于绝对控制，这些值应该对应实际的工作空间范围

4. **控制器初始化**：
   - 首次调用时，如果`goal_pos`为`None`，会自动设置为当前位置
   - 使用`reset_goal()`可以重置目标到当前位置

## 测试脚本

运行测试脚本查看绝对控制效果：

```bash
python robosuite/robosuite/scripts/test_panda_absolute_control.py
```

## 总结

**OSC控制器完全支持绝对位姿控制**，只需要：
1. 设置 `input_type="absolute"`
2. 选择合适的 `input_ref_frame`（"base"或"world"）
3. 提供6维绝对位姿向量：`[x, y, z, rx, ry, rz]`

控制器会自动处理坐标系转换和OSC计算，实现精确的末端位姿控制。
