# Dexterous Manipulation Demo Generation

本文档描述两类 sequential manipulation 任务的轨迹自动生成方法：

1. **双物体两阶段任务**（通用公式，见第一节）
2. **Single-Arm HammerCleanup**（11 阶段分段策略，见第二节；实现：`autogen_dextool_demo/autogen_hammer_cleanup.py`）

---

## 一、双物体两阶段任务（通用）

### 概述

场景包括：

- **物体1**：可随机初始化的自由物体；
- **物体2**：固定在场景中的参考物体（fixture）；
- **任务阶段**：
  1. 阶段1：机器人抓取物体1；
  2. 阶段2：机器人将物体1与物体2进行交互操作。

目标是在物体1随机初始化后，自动生成与原始 demo 一致的机器人动作，使得：

- 阶段1 保持机械臂末端与物体1的相对姿态不变；
- 阶段2 保持物体1与物体2的接触相对姿态不变。

### 阶段性约束

| 阶段 | 不变量 | 生成目标 | 坐标变换关系 |
|------|----------|-----------|---------------|
| 阶段1 | 末端–物体1 相对位姿不变 | 末端轨迹 | `T'_ee = T'_obj1 @ (T_obj1⁻¹ @ T_ee)` |
| 阶段2 | 物体1–物体2 相对位姿不变 | 末端轨迹（间接通过物体1） | `T'_ee = T'_obj1 @ (T_obj1⁻¹ @ T_ee)` |

注意阶段2开始时，物体1的初始姿态可能已因阶段1的随机初始化发生变化，因此需要重新计算姿态传递。

### 实现示意

```python
for t in traj_stage1:
    T_ee_world_orig = traj[t]["T_ee_world"]
    T_obj1_world_orig = traj[t]["T_obj1_world"]

    # 原始末端相对物体1
    T_ee_obj1 = np.linalg.inv(T_obj1_world_orig) @ T_ee_world_orig

    # 新物体1位姿（随机初始化或 rollout 当前仿真）
    T_obj1_world_new = T_new_obj1_world_func(t)

    # 新末端目标
    T_ee_world_new = T_obj1_world_new @ T_ee_obj1

    q_new = solve_IK(T_ee_world_new)  # 实际用 OSC 逐步跟踪
    store(q_new)
```

---

## 二、Single-Arm HammerCleanup（11 阶段）

### 任务与增广目标

- **源数据**：一条人工 teleop demo + 逐帧 `stage_label`（0=机械臂，1=灵巧手）
- **随机化**：仅改变 **hammer 桌面初始位姿**（`T_delta`：平移 + 绕 z 轴 yaw）
- **输出**：多条新 demo，共享源轨迹语义，hammer 摆放不同

11 个语义阶段（`stage_label` 文件）：

| Stage | label | 名称 | 策略 |
|-------|-------|------|------|
| 1 | 0 | move to drawer | drawer 相对 EE + OSC |
| 2 | 1 | grasp handle | **复制 action**，物理 rollout |
| 3 | 0 | open drawer | drawer 相对 EE + OSC |
| 4 | 1 | open hand | 复制 action，物理 rollout |
| 5 | 0 | move to hammer | hammer 相对 EE + OSC |
| 6 | 1 | grasp hammer | 复制 action，物理 rollout |
| 7 | 0 | move to drawer | **段内增量重锚** + OSC |
| 8 | 1 | open hand | 复制 action，物理 rollout |
| 9 | 0 | move to handle | 段内增量重锚 + OSC |
| 10 | 1 | grasp handle | 复制 action，物理 rollout |
| 11 | 0 | close and home | drawer 相对 EE + OSC |

### 核心原则：顺序 rollout

**禁止**在生成新 demo 时整帧拷贝源 `states`（会导致物体与机械臂脱节、物体漂移）。

整条轨迹按 `t = 0 … T-1` 顺序处理：

```
init: states[0] 仅 patch hammer 桌面位姿（抓取前）
for t in 0..T-1:
    按 stage 决定 actions[t]
    env.step(actions[t])   # control_substeps 次
    states[t] = sim.get_state()   # 物体/drawer/hammer 全部由物理决定
```

- **label=1（复制）**：`actions_new[t] = actions_src[t]`（机械臂 + 灵巧手），不拷贝 `states_src[t]`
- **label=0（机械臂段）**：由 EE 目标经 OSC 生成 `actions_new[t]`，再 rollout

### 三类机械臂 EE 目标

#### A. Fixture 相对（drawer，stage 1/3/11）

drawer 在仿真中可能滑动（开/关抽屉），使用**逐帧** fixture 相对位姿：

\[
T^{target}_{ee}(t) = T^{sim}_{drawer}(t) \cdot T^{-1}_{drawer,src}(t) \cdot T_{ee,src}(t)
\]

- `T^{sim}_{drawer}(t)`：rollout 到 `t` 时仿真中的 drawer 位姿
- 抓取 hammer 之前：每步可将 hammer `qpos` pin 到 `T_delta` 后的桌面位姿

#### B. Hammer 相对（stage 5）

保持末端相对 hammer 位姿不变（与源 demo 一致，整体随 `T_delta` 变换）：

\[
T^{target}_{ee}(t) = T_\Delta \cdot T_{ee,src}(t)
\]

OSC 跟踪时 pin hammer 自由关节到变换后位姿，直至 stage 6 抓取。

#### C. 段内增量重锚（stage 7/9，post-hammer 回 drawer）

**问题**：stage 6 结束后，新 demo 的 EE 在「新 hammer 位置」；源 demo stage 7 起点仍在「旧 hammer 位置」。drawer 未动，不能直接复用源绝对 EE 或 actions。

**做法**：在段起点 `s` 读取 rollout 后的实际 EE 作为锚点，复用源 demo 在该段内的**相对运动增量**：

\[
T^{target}_{ee}(t) = T_{anchor} \cdot T^{-1}_{ee,src}(s) \cdot T_{ee,src}(t), \quad t \in [s, e]
\]

其中：

- `T_anchor = T^{sim}_{ee}(s)`（stage 6 结束后的真实 EE，保证连续）
- `t = s` 时 `T^{target}_{ee}(s) = T_anchor`
- `t = e` 时末端到达与源 demo **相同的段内相对位移**，重新对准 drawer 区域

stage 7/9 中 drawer 已打开且静止，使用世界系段内增量即可。stage 11 drawer 运动，改用公式 A。

### 复制段（label=1）语义

| 复制内容 | 不复制 |
|----------|--------|
| `actions`（臂 + 手） | 整帧 `states` |
| — | drawer 开度、hammer 位姿、接触状态 |

物体状态完全由新轨迹下的 MuJoCo 物理仿真产生。

### 实现入口

```bash
cd /home/benhua/DexSim
PYTHONPATH=robosuite:dexmimicgen MUJOCO_GL=osmesa \
python dexmimicgen/autogen_dextool_demo/autogen_hammer_cleanup.py \
  --source-hdf5 dexmimicgen/datasets/generated/single_arm_hammer_cleanup_demo_4.hdf5 \
  --labels dexmimicgen/autogen_dextool_demo/outputs/single_arm_hammer_cleanup_demo_4_demo_0_review_labels.json \
  -n 5 --seed 0
```

源 HDF5 须仅含 1 条 `demo_*`；输出合并为 `datasets/generated/single_arm_hammer_cleanup_autogen.hdf5`。

### Segment mode 对照（代码）

| `Segment.mode` | 对应 stage |
|----------------|------------|
| `replay_action` | 2, 4, 6, 8, 10 |
| `transform_drawer` | 1, 3, 11 |
| `transform_hammer` | 5 |
| `transform_drawer_anchor` | 7, 9 |

---

## 三、与两阶段公式的关系

| 场景 | 参考物体 | 公式 |
|------|----------|------|
| 通用阶段1/2 | 物体1（可动） | `T'_ee = T'_{obj1} @ T^{-1}_{obj1} @ T_ee` |
| HammerCleanup drawer 段 | drawer（可动 fixture） | 同上，obj1 → drawer |
| HammerCleanup hammer 段 | hammer（随机初始化） | `T'_ee = T_\Delta @ T_ee` |
| Post-hammer 回 drawer | drawer 静止、起点已偏 | 段内增量重锚 |

HammerCleanup 是通用 fixture-relative 公式的分段应用，并针对「hammer 偏移导致段起点不连续」增加了 **anchor 段内增量** 策略。
