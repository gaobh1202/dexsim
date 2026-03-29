# Two-Stage Dexterous Manipulation Demo Generation

## 概述
本文档介绍了一种在 **包含两个物体的 sequential manipulation 任务** 中进行轨迹数据生成的方法。  
场景包括：
- **物体1**：可随机初始化的自由物体；
- **物体2**：固定在场景中的物体；
- **任务阶段**：
  1. 阶段1：机器人抓取物体1；
  2. 阶段2：机器人将物体1与物体2进行交互操作。

目标是在物体1随机初始化后，自动生成与原始demo轨迹一致的机器人动作，使得：
- 阶段1 保持机械臂末端与物体1的相对姿态不变；
- 阶段2 保持物体1与物体2的接触相对姿态不变。

---

## 一、阶段性约束总结

| 阶段 | 不变量 | 生成目标 | 坐标变换关系 |
|------|----------|-----------|---------------|
| 阶段1 | 末端–物体1 相对位姿不变 | 末端轨迹 | `T'_ee = T'_obj1 * (T_obj1⁻¹ * T_ee)` |
| 阶段2 | 物体1–物体2 相对位姿不变 | 末端轨迹（间接通过物体1） | `T'_ee = T'_obj1 * (T_obj1⁻¹ * T_ee)` |

注意阶段2开始时，物体1的初始姿态可能已因阶段1的随机初始化发生变化，因此需要重新计算姿态传递。

---

## 二、阶段1：保持末端–物体1 相对位姿

### 原理
对于每个时刻 \( t \)：

\[
T'_{ee}(t) = T'_{obj1}(t) \cdot (T_{obj1}(t)^{-1} \cdot T_{ee}(t))
\]

然后通过逆运动学求得新的关节角 \( q'_{arm}(t) \)。

### 实现示意

```python
for t in traj_stage1:
    T_ee_world_orig = traj[t]['T_ee_world']
    T_obj1_world_orig = traj[t]['T_obj1_world']

    # 计算原始末端与物体1的相对位姿
    T_ee_obj1 = np.linalg.inv(T_obj1_world_orig) @ T_ee_world_orig

    # 生成新的物体1位姿 (根据随机初始化)
    T_obj1_world_new = T_new_obj1_world_func(t)

    # 得到新的末端姿态
    T_ee_world_new = T_obj1_world_new @ T_ee_obj1

    q_new = solve_IK(T_ee_world_new)
    store(q_new)
