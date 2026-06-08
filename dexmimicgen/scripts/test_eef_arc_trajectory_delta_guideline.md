# Guideline: 保留源轨迹趋势的 EEF 轨迹变形方法

本文档整理一种用于 demonstration 自动生成的轨迹变形原则：当目标物体位姿发生变化，或一段轨迹的起点 / 终点发生变化时，如何在满足新几何约束的同时，尽可能保留 source demonstration 的中间运动趋势，而不是简单生成一条线性插值轨迹。

适用场景包括：

- MimicGen / DexMimicGen 风格的 object-centric demonstration generation；
- 单臂或双臂机器人 EEF 轨迹重定向；
- 起点不变、终点随物体变化的 segment；
- 起点和终点都变化的 segment；
- HammerCleanup 中 `carry_hammer_to_drawer` 这类“新起点 -> 新目标”的搬运阶段。

---

## 1. 基本问题定义

假设 source demonstration 中有一段 EEF 轨迹：

```python
T_src[t],  t = 0, 1, ..., N - 1
```

其中：

```python
T_a = T_src[0]      # source 起点
T_b = T_src[-1]     # source 终点
```

在 source 场景中，终点 `T_b` 与某个物体 pose `T_o` 具有固定相对关系：

```python
T_rel = inv(T_o) @ T_b
```

现在物体发生变化：

```python
T_o -> T_o_new
```

希望得到新的终点：

```python
T_b_new = T_o_new @ T_rel
```

也就是：

```python
T_b_new = T_o_new @ inv(T_o) @ T_b
```

目标是生成新轨迹：

```python
T_new[t]
```

使得：

```python
T_new[0]  = T_a_new
T_new[-1] = T_b_new
```

同时尽可能保留 source 轨迹的中间运动趋势，例如：

- 先抬高；
- 绕开障碍物；
- 平移后下降；
- 保持某种 wrist 姿态变化；
- 保持 source demonstration 中的整体运动风格。

---

## 2. 不推荐：直接线性插值起点和终点

最简单的方式是：

```python
T_new[t] = interp_SE3(T_a_new, T_b_new, u)
```

其中：

```python
u = t / (N - 1)
```

这种方法可以保证起点和终点正确，但缺点明显：

1. 会丢失 source trajectory 的中间形状；
2. 容易把原本“先抬高再下降”的动作变成斜线直达；
3. 可能导致碰撞桌面、drawer 边缘或目标物体；
4. 对 long-horizon manipulation 不够自然。

因此，如果目标是“保留 demonstration 的趋势”，不应直接使用这种方法作为主要方案。

---

## 3. 推荐思想：对 source 轨迹施加随时间变化的 SE(3) correction

核心思想是：

> 不重新画一条从新起点到新终点的直线路径，而是以 source trajectory 为基础，对每一帧施加一个随时间变化的 SE(3) 变换。

形式为：

```python
T_new[t] = Delta_t @ T_src[t]
```

其中：

```python
Delta_t
```

是一个从起点约束逐渐过渡到终点约束的变换。

这样做的好处是：

- 每一帧仍然以 `T_src[t]` 为基础；
- source 轨迹中的中间趋势被保留；
- 起点和终点可以通过设计 `Delta_t` 严格满足；
- 过渡可以通过 smoothstep 保持平滑。

---

## 4. 情况 A：起点不变，终点随物体变化

这是最常见的 object-relative endpoint adaptation。

### 4.1 问题设定

source：

```python
T_src[t]
T_a = T_src[0]
T_b = T_src[-1]
T_o = source object pose
```

new：

```python
T_a_new = T_a
T_o_new = new object pose
```

希望终点满足：

```python
inv(T_o_new) @ T_b_new = inv(T_o) @ T_b
```

因此：

```python
T_b_new = T_o_new @ inv(T_o) @ T_b
```

### 4.2 渐进施加 object transform

物体变化带来的完整 SE(3) 变换为：

```python
Delta_obj = T_o_new @ inv(T_o)
```

如果直接对整段轨迹施加：

```python
T_new[t] = Delta_obj @ T_src[t]
```

终点是正确的，但起点会变成：

```python
T_new[0] = Delta_obj @ T_a
```

这会导致起点跳变。

因此，改为从单位变换逐渐过渡到 `Delta_obj`：

```python
Delta_t = interp_SE3(I, Delta_obj, alpha)
T_new[t] = Delta_t @ T_src[t]
```

其中：

```python
u = t / (N - 1)
alpha = smoothstep(u)
```

这样：

```python
t = 0:
    Delta_t = I
    T_new[0] = T_src[0] = T_a

t = N - 1:
    Delta_t = Delta_obj
    T_new[-1] = Delta_obj @ T_b = T_b_new
```

### 4.3 公式总结

```python
Delta_obj = T_o_new @ inv_pose(T_o)

for i, T_src_i in enumerate(T_src_seq):
    u = i / max(N - 1, 1)
    alpha = smoothstep(u)
    Delta_i = interp_SE3(I, Delta_obj, alpha)
    T_new_i = Delta_i @ T_src_i
```

这个方法不会把轨迹变成线性插值，而是“逐渐把 source 轨迹拉向新的 object-relative 终点”。

---

## 5. 情况 B：起点和终点都发生变化

这是更通用的情况。

### 5.1 问题设定

source：

```python
T_a = T_src[0]
T_b = T_src[-1]
```

new：

```python
T_a_new
T_b_new
```

希望：

```python
T_new[0]  = T_a_new
T_new[-1] = T_b_new
```

同时保留：

```python
T_src[t]
```

的中间形状。

### 5.2 两端点变换插值法

先计算起点和终点所需的 SE(3) 变换：

```python
Delta_a = T_a_new @ inv(T_a)
Delta_b = T_b_new @ inv(T_b)
```

然后对 `Delta_a` 和 `Delta_b` 随时间插值：

```python
Delta_t = interp_SE3(Delta_a, Delta_b, alpha)
```

最后作用到 source trajectory：

```python
T_new[t] = Delta_t @ T_src[t]
```

这样：

```python
t = 0:
    Delta_t = Delta_a
    T_new[0] = Delta_a @ T_a = T_a_new

t = N - 1:
    Delta_t = Delta_b
    T_new[-1] = Delta_b @ T_b = T_b_new
```

### 5.3 公式总结

```python
Delta_a = T_a_new @ inv_pose(T_a_src)
Delta_b = T_b_new @ inv_pose(T_b_src)

for i, T_src_i in enumerate(T_src_seq):
    u = i / max(N - 1, 1)
    alpha = smoothstep(u)
    Delta_i = interp_SE3(Delta_a, Delta_b, alpha)
    T_new_i = Delta_i @ T_src_i
```

这是推荐用于 HammerCleanup 第 7 阶段的主要方法。

---

## 6. Smoothstep 权重

为了避免起点和终点附近速度变化过突兀，建议使用 smoothstep：

```python
def smoothstep(u: float) -> float:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)
```

特点：

```python
smoothstep(0) = 0
smoothstep(1) = 1
```

并且在起点和终点处一阶变化更平滑。

也可以根据需求调整：

### 6.1 后半段更快收敛到目标

```python
alpha = u ** 2
```

前半段更接近 source，后半段更快转向终点。

### 6.2 更早开始转向目标

```python
alpha = 1.0 - (1.0 - u) ** 2
```

适合需要尽早避开新障碍或提前对齐目标的情况。

默认建议使用 smoothstep。

---

## 7. SE(3) 插值实现

`interp_SE3(T0, T1, alpha)` 可以用：

- translation: 线性插值；
- rotation: Slerp。

示例：

```python
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np

def interp_SE3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))

    T = np.eye(4, dtype=np.float64)

    # translation interpolation
    T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

    # rotation interpolation
    rots = R.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]], axis=0))
    slerp = Slerp([0.0, 1.0], rots)
    T[:3, :3] = slerp([alpha]).as_matrix()[0]

    return T
```

---

## 8. 通用工具函数

### 8.1 pose inverse

```python
def inv_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv
```

### 8.2 起点不变、终点随物体变化

```python
def deform_keep_start_follow_object_goal(
    T_src_seq: np.ndarray,
    T_o_src: np.ndarray,
    T_o_new: np.ndarray,
) -> np.ndarray:
    """
    Keep start unchanged, make endpoint follow object-relative goal,
    while preserving source trajectory trend.

    Args:
        T_src_seq: [N, 4, 4] source EEF trajectory.
        T_o_src:  [4, 4] source object pose.
        T_o_new:  [4, 4] new object pose.

    Returns:
        T_new_seq: [N, 4, 4] deformed EEF trajectory.
    """
    N = len(T_src_seq)
    I = np.eye(4, dtype=np.float64)

    Delta_obj = T_o_new @ inv_pose(T_o_src)

    T_new_seq = []
    for i, T_src_i in enumerate(T_src_seq):
        u = i / max(N - 1, 1)
        alpha = u * u * (3.0 - 2.0 * u)

        Delta_i = interp_SE3(I, Delta_obj, alpha)
        T_new_i = Delta_i @ T_src_i
        T_new_seq.append(T_new_i)

    return np.stack(T_new_seq, axis=0)
```

### 8.3 起点和终点都变化

```python
def deform_two_endpoint_keep_trend(
    T_src_seq: np.ndarray,
    T_a_new: np.ndarray,
    T_b_new: np.ndarray,
) -> np.ndarray:
    """
    Deform a source EEF trajectory to match new start and end poses,
    while preserving source trajectory trend.

    Args:
        T_src_seq: [N, 4, 4] source EEF trajectory.
        T_a_new:  [4, 4] new start EEF pose.
        T_b_new:  [4, 4] new end EEF pose.

    Returns:
        T_new_seq: [N, 4, 4] deformed EEF trajectory.
    """
    N = len(T_src_seq)

    T_a_src = T_src_seq[0]
    T_b_src = T_src_seq[-1]

    Delta_a = T_a_new @ inv_pose(T_a_src)
    Delta_b = T_b_new @ inv_pose(T_b_src)

    T_new_seq = []
    for i, T_src_i in enumerate(T_src_seq):
        u = i / max(N - 1, 1)
        alpha = u * u * (3.0 - 2.0 * u)

        Delta_i = interp_SE3(Delta_a, Delta_b, alpha)
        T_new_i = Delta_i @ T_src_i
        T_new_seq.append(T_new_i)

    return np.stack(T_new_seq, axis=0)
```

---

## 9. 与 object-centric transformation 的关系

MimicGen / DexMimicGen 的基础 object-centric transformation 是：

```python
T_delta = T_object_new @ inv(T_object_src)
T_eef_new[t] = T_delta @ T_eef_src[t]
```

这适用于整段轨迹都应该跟随参考物体整体移动的情况。

但如果只希望：

```text
起点不变，终点跟随 object-relative goal
```

那么直接把 `T_delta` 施加到整段轨迹会造成起点跳变。

本文档中的方法可以看作是对 object-centric transformation 的平滑推广：

```python
T_delta(t) = interp_SE3(I, T_delta, alpha)
T_eef_new[t] = T_delta(t) @ T_eef_src[t]
```

如果起点和终点都变，则进一步推广为：

```python
T_delta(t) = interp_SE3(Delta_a, Delta_b, alpha)
T_eef_new[t] = T_delta(t) @ T_eef_src[t]
```

---

## 10. 应用到 HammerCleanup 第 7 阶段

HammerCleanup 第 7 阶段是：

```text
carry_hammer_to_drawer
```

其特点是：

```text
阶段 6 结束后，EEF pose 会因为 hammer 初始 pose 变化而改变；
阶段 7 终点通常应该到达 drawer-relative placement goal；
source 阶段 7 中可能包含抬高、搬运、靠近 drawer、下降等中间趋势。
```

因此，不建议：

```python
T_new[t] = interp_SE3(T_a_new, T_b_new, u)
```

也不建议直接 replay source action。

推荐使用通用两端点版本：

```python
T_a_src = source stage 7 start EEF pose
T_b_src = source stage 7 end EEF pose

T_a_new = actual EEF pose after stage 6 rollout
T_b_new = drawer-relative placement goal
```

其中：

```python
T_rel_goal_src = inv(T_drawer_src_at_stage7_end) @ T_b_src
T_b_new = T_drawer_new_at_stage7_end @ T_rel_goal_src
```

如果 drawer 不变，则 `T_b_new` 通常接近 source stage 7 endpoint，但仍建议保留 drawer-relative 写法，方便后续扩展。

最终：

```python
Delta_a = T_a_new @ inv_pose(T_a_src)
Delta_b = T_b_new @ inv_pose(T_b_src)

for t in stage_7:
    u = (t - start) / max(end - start, 1)
    alpha = smoothstep(u)
    Delta_t = interp_SE3(Delta_a, Delta_b, alpha)
    T_eef_new[t] = Delta_t @ T_eef_src[t]
```

这会实现：

```text
1. 阶段 7 起点与阶段 6 结束后的真实 EEF pose 连续；
2. 阶段 7 终点到达 drawer-relative placement target；
3. 中间轨迹尽可能保留 source demonstration 中的搬运动作趋势。
```

---

## 11. 与 replay action 的区别

这种方法生成的是新的绝对 EEF target pose sequence：

```python
T_eef_new[t]
```

然后再由控制器转换为实际 action。

它不是直接对 source delta action 做变换。

推荐流程是：

```text
source EEF poses
    -> trajectory deformation
    -> new absolute EEF target poses
    -> controller / IK / OSC
    -> actions
    -> simulation rollout
```

而不是：

```text
source delta actions
    -> rotate / translate delta actions
    -> direct replay
```

对于需要精确保持 EEF-object relative pose 的任务，应优先使用绝对 EEF pose 变换，再转换成控制器 action。

---

## 12. 何时使用哪种方法

### 12.1 整段轨迹都应该跟随同一个 object

使用标准 object-centric transform：

```python
T_new[t] = T_object_new @ inv(T_object_src) @ T_src[t]
```

适合：

- move_to_hammer；
- grasp 前的 approach；
- 目标物体整体随机化，且整段 motion 都应该跟随物体。

### 12.2 起点不变，终点随 object 改变

使用渐进 object transform：

```python
Delta_t = interp_SE3(I, T_object_new @ inv(T_object_src), alpha)
T_new[t] = Delta_t @ T_src[t]
```

适合：

- 当前 EEF 起点必须保持；
- 终点需要对齐新的 object-relative goal；
- 希望保留 source motion trend。

### 12.3 起点和终点都变化

使用两端点变换插值：

```python
Delta_a = T_a_new @ inv(T_a_src)
Delta_b = T_b_new @ inv(T_b_src)
Delta_t = interp_SE3(Delta_a, Delta_b, alpha)
T_new[t] = Delta_t @ T_src[t]
```

适合：

- carry_hammer_to_drawer；
- handover 后继续移动；
- 从一个新 grasp pose 移动到一个新 place pose；
- 任何 variable start -> variable/fixed goal 的 segment。

### 12.4 完全不关心 source 中间趋势

才使用直接 SE(3) 插值：

```python
T_new[t] = interp_SE3(T_a_new, T_b_new, u)
```

适合非常短、无障碍、无中间姿态要求的动作。

---

## 13. 注意事项

### 13.1 轨迹趋势不是碰撞规划

该方法保留 source trajectory trend，但不保证无碰撞。若物体变化幅度较大，仍可能出现：

- 穿过 drawer 边缘；
- hammer 撞桌面；
- wrist 不可达；
- hand-object contact 不稳定。

需要通过 rollout 和 success checker 过滤失败 demo。

### 13.2 变换幅度不宜过大

如果 `Delta_a` 和 `Delta_b` 差别过大，中间 trajectory deformation 可能变得不自然。建议限制随机化范围，或增加中间 waypoint。

### 13.3 rotation interpolation 应使用 Slerp

不要直接线性插值旋转矩阵。推荐：

```python
scipy.spatial.transform.Slerp
```

或使用 SE(3) log / exp 插值。

### 13.4 生成的是 EEF target，不是最终 state

变形得到的是目标 EEF trajectory。最终训练数据中的：

- states；
- actions；
- observations；
- contact dynamics；

应通过 simulation rollout 重新生成。

### 13.5 手部动作应单独处理

对于 dexterous hand：

```text
arm / EEF trajectory: 使用本文档的方法变形
finger / hand action: 通常 replay source hand action
```

但不要直接 replay 整个 source action，因为其中可能包含旧的 arm command。

---

## 14. 最终推荐

对于需要保留 source trajectory trend 的 demonstration generation，优先使用：

```python
T_new[t] = Delta_t @ T_src[t]
```

而不是：

```python
T_new[t] = interp_SE3(T_a_new, T_b_new, u)
```

其中：

```python
Delta_t = interp_SE3(Delta_a, Delta_b, smoothstep(u))
```

这是一个简单、可实现、且与 MimicGen object-centric transformation 思想兼容的轨迹变形方法。

对于 HammerCleanup：

```text
move_to_hammer:
    标准 object-centric transform

carry_hammer_to_drawer:
    两端点变换插值，保留 source 搬运趋势

hand open / close:
    replay hand only，EEF hold

drawer fixed stages:
    可 replay 或 drawer-relative transform
```
