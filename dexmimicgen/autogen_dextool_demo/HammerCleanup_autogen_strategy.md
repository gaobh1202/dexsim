# HammerCleanup 自动生成 Demonstration 的阶段处理与轨迹变换建议

本文档整理了基于 DexMimicGen / MimicGen 思想，对单臂 `HammerCleanup` 任务进行自动化 demonstration 生成时的阶段处理策略。当前任务设定为：

- `drawer` 的初始位置通常不变；
- `hammer` 平放在桌面上；
- 生成新 demo 时，`hammer` 的初始化位姿会变化，包括：
  - 世界坐标系下的 `x-y` 平移；
  - 绕桌面法向，也就是世界 `z` 轴的 yaw 旋转；
- 新 demo 主要通过源 demo 中 EEF 与目标物体之间的相对位姿关系进行变换生成；
- 灵巧手动作优先 replay，但不应直接 replay 整个 action 中的机械臂部分。

---

## 1. DexMimicGen / MimicGen 的核心原则

DexMimicGen 继承 MimicGen 的核心思想：将一条 source demonstration 切分为多个 object-centric segments。对于每个 segment：

1. 选定该阶段的参考物体，例如 `drawer`、`hammer`、`tray` 等；
2. 记录 source demo 中参考物体在世界坐标系下的 pose；
3. 在新场景中获取该参考物体的新 pose；
4. 计算二者之间的 SE(3) 变换；
5. 将该变换施加到 source demo 中该阶段的 EEF trajectory 上；
6. 在仿真中 rollout 生成新 demonstration；
7. 最后用 success checker 过滤失败样本。

核心公式是：

```python
T_delta = T_object_new @ inv(T_object_src)
T_eef_new[t] = T_delta @ T_eef_src[t]
```

它保持的是 EEF 相对于目标物体坐标系的位姿关系：

```python
inv(T_object_new) @ T_eef_new[t] ≈ inv(T_object_src) @ T_eef_src[t]
```

对于 dexterous hand，手指动作通常可以 replay，因为手指动作主要相对于 EEF 本体执行。但在工程实现中要注意：**手部阶段只 replay 手部 / gripper / finger action，机械臂 EEF 应保持上一阶段结束时的 pose，不应直接 replay source action 中的 arm 部分。**

---

## 2. Hammer 新 pose 的定义

对于本任务，hammer 的新初始化应采用如下语义：

> hammer 在原位置附近平移 `dx, dy`，并绕自身中心在桌面上旋转 `dyaw`。

因此建议构造 hammer 的新 pose 为：

```python
R_new = Rz(dyaw) @ R_src
p_new = p_src + np.array([dx, dy, 0.0])
```

不要使用：

```python
p_new = Rz(dyaw) @ p_src + np.array([dx, dy, 0.0])
```

后者表示 hammer 的中心先绕世界原点旋转，再平移，通常不是“在桌面上改变 hammer 摆放位置和朝向”的语义。

推荐函数：

```python
def _new_hammer_pose(T_src_ref: np.ndarray, dx: float, dy: float, dyaw_deg: float) -> np.ndarray:
    th = np.deg2rad(dyaw_deg)
    c, s = np.cos(th), np.sin(th)
    R_delta = np.array(
        [
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ],
        dtype=np.float64,
    )

    T_new = np.eye(4, dtype=np.float64)
    T_new[:3, :3] = R_delta @ T_src_ref[:3, :3]
    T_new[:3, 3] = T_src_ref[:3, 3] + np.array([dx, dy, 0.0], dtype=np.float64)
    return T_new
```

更稳妥的做法是：

1. 先构造 command pose；
2. 写入 MuJoCo 的 hammer free joint qpos；
3. `sim.forward()`；
4. 再从 `body_xpos` / `body_xmat` 读取 hammer 当前真实世界坐标系 pose；
5. 用这个 actual pose 计算 `T_delta`。

示例：

```python
T_hammer_cmd = _new_hammer_pose(
    ctx.T_hammer_ref,
    hammer_dx,
    hammer_dy,
    hammer_dyaw_deg,
)

ctx.env.sim.set_state_from_flattened(ctx.states_src[ctx.hammer_arm_seg.start].copy())
_pin_hammer_free(
    ctx.env,
    ctx.hammer_qadr,
    ctx.hammer_dadr,
    T_hammer_cmd,
)
ctx.env.sim.forward()

T_hammer_new_actual = _pose_from_body(ctx.env, ctx.hammer_bid)

T_delta = _delta_from_pose_change(
    ctx.T_hammer_ref,
    T_hammer_new_actual,
)
```

---

## 3. 11 个阶段的推荐处理方式

任务流程：

1. 靠近 drawer；
2. 灵巧手抓取 handle；
3. 拉开 drawer；
4. 松开灵巧手；
5. 靠近 hammer；
6. 抓取 hammer；
7. 再次移动到 drawer；
8. 松开灵巧手；
9. 移动到 drawer handle；
10. 抓取 handle；
11. 关闭 drawer 并移动回原点。

在 `drawer` 不变、`hammer` 改变的当前设定下，推荐如下。

| 阶段 | 名称 | 推荐处理 | 说明 |
|---|---|---|---|
| 1 | move_to_drawer | 可直接 replay arm action，或 drawer-relative transform | drawer 不变时可直接复用；若未来 drawer 随机化，应改为 drawer-relative。 |
| 2 | grasp_handle | 只 replay hand action | EEF 保持阶段 1 结束 pose，不 replay source arm action。 |
| 3 | open_drawer | 可直接 replay / drawer-handle-relative | drawer 不变时可复用；更稳的是相对可动 handle 或 drawer link，而不是固定外框。 |
| 4 | open_hand | 只 replay hand action | EEF 保持阶段 3 结束 pose。 |
| 5 | move_to_hammer | hammer-relative transform | 必须根据 hammer 新 pose 变换 EEF trajectory。 |
| 6 | grasp_hammer | 只 replay hand action；若含 arm 微调则应拆分 | EEF 保持阶段 5 结束 pose；如果 source 中此段仍有明显 EEF motion，需要把该 motion 归到 arm segment 并做 hammer-relative。 |
| 7 | carry_hammer_to_drawer | 使用“两端点变换插值法” | 起点由阶段 6 后真实 EEF pose 决定，终点由 drawer-relative placement goal 决定。 |
| 8 | release_hammer | 只 replay hand action | EEF 保持阶段 7 结束 pose。 |
| 9 | move_to_handle | 可直接 replay / drawer-relative / current-anchor | hammer 已释放，目标重新回到 drawer handle。drawer 不变时可复用。 |
| 10 | grasp_handle | 只 replay hand action | EEF 保持阶段 9 结束 pose。 |
| 11 | close_and_home | close 部分 drawer-handle-relative；home 部分 replay 或显式插值 | 若 drawer 不变可复用；更稳的是关闭 drawer 后显式插值回 home pose。 |

---

## 4. 当 drawer 不变时，哪些阶段可以直接 replay action？

在 drawer 固定且 source demo 与新场景中 drawer 初始状态一致的前提下，可以直接 replay arm action 的阶段主要是：

- 阶段 1：靠近 drawer；
- 阶段 3：拉开 drawer；
- 阶段 9：移动到 drawer handle；
- 阶段 11：关闭 drawer 并回到 home 的部分动作。

但仍建议注意：

1. **阶段 3 和阶段 11 应优先参考可动 drawer body / handle site**  
   如果 `drawer_bid` 指向的是固定外框，那么 drawer 开合过程中它不会跟随 handle 运动，使用它做相对变换是不准确的。更推荐使用：
   - drawer handle site；
   - 可动 drawer link body；
   - 或直接从仿真状态读取当前 handle pose。

2. **直接 replay 只适用于 drawer 完全不变的情况**  
   如果未来也对 drawer 做位置、朝向、开合状态随机化，那么 drawer 相关阶段应该全部改为 drawer-relative 或 handle-relative transform。

3. **阶段 7 不应直接 replay**  
   即使 drawer 不变，阶段 7 的起点也随 hammer grasp pose 变化，因此 source action 不再适用。

---

## 5. 手部阶段：只 replay 手，不 replay EEF

对于阶段 2、4、6、8、10，建议规则为：

```text
hand segment:
    replay finger / gripper action
    keep EEF target equal to previous arm segment end pose
    do not replay source arm action
```

也就是说，不建议：

```python
env_action = actions_src[t].copy()
```

因为这会同时 replay 源 demo 中的 arm command 和 hand command。更稳妥的方式是构造一个新的 env action：

```python
arm_action = hold_current_eef_pose()
hand_action = source_hand_action[t]
env_action = combine(arm_action, hand_action)
```

伪代码：

```python
if seg.mode == "replay_hand_only":
    T_eef_hold = current_eef_pose_after_previous_stage
    arm_action = pose_to_arm_action(T_eef_hold)
    hand_action = extract_hand_action(actions_src[t])
    env_action = build_env_action_with_hand(env, arm_action, hand_action)
```

这样可以避免以下问题：

- hammer pose 改变后，阶段 6 replay source arm action 导致 EEF 跳回旧抓取位置；
- drawer 相关阶段结束后，阶段 2 或 10 中 source arm action 破坏当前 handle grasp pose；
- 手部动作正确，但机械臂动作错误，导致 contact 不稳定。

如果当前工具函数 `build_env_action` 只支持 arm action，需要补充一个能显式设置 dexterous hand action 的接口，或者在构造出的 action 向量中手动替换 hand 维度。

---

## 6. 阶段 7：从变化起点到固定终点

阶段 7 是本任务中最关键的特殊阶段。

它的 source 语义是：

```text
抓住 hammer 后，将 hammer 搬运到 drawer 附近或 drawer 内部。
```

但在新 demo 中：

```text
阶段 6 结束后的 EEF pose 会因为 hammer 初始 pose 变化而改变；
阶段 7 结束目标通常仍由 drawer 定义，而 drawer 不变。
```

因此阶段 7 是：

```text
variable start -> fixed / drawer-relative goal
```

不能简单直接 replay source action，也不能只使用 hammer-relative transform。

---

## 7. 方法 1：两端点变换插值法

先采用方法 1，也就是：

> 对 source segment 的起点和终点分别计算变换，然后随时间从起点变换平滑过渡到终点变换，再作用到 source trajectory 上。

设：

```python
T_a_src = source 阶段 7 起点 EEF pose
T_b_src = source 阶段 7 终点 EEF pose
T_src[t] = source 阶段 7 中第 t 帧 EEF pose

T_a_new = 新阶段 7 起点 EEF pose
T_b_new = 新阶段 7 终点 EEF pose
```

其中：

```python
T_a_new = 阶段 6 结束后仿真中真实 EEF pose
```

`T_b_new` 建议由 drawer-relative placement goal 得到：

```python
T_rel_goal_src = inv(T_drawer_src_at_stage7_end) @ T_b_src
T_b_new = T_drawer_new_at_stage7_end @ T_rel_goal_src
```

如果 drawer 不变，`T_b_new` 通常接近 source 的阶段 7 终点。

然后计算起点变换和终点变换：

```python
Delta_a = T_a_new @ inv(T_a_src)
Delta_b = T_b_new @ inv(T_b_src)
```

对 `Delta_a` 和 `Delta_b` 做插值：

```python
Delta_t = interp_SE3(Delta_a, Delta_b, u)
```

最后生成新轨迹：

```python
T_new[t] = Delta_t @ T_src[t]
```

其中：

```python
u = (t - seg.start) / (seg.end - seg.start)
```

可以使用 smoothstep 让过渡更平滑：

```python
u = u * u * (3.0 - 2.0 * u)
```

该方法满足：

```python
T_new[seg.start] ≈ T_a_new
T_new[seg.end] ≈ T_b_new
```

同时会尽量保持 source 轨迹的整体趋势。

---

## 8. 方法 1 的实现伪代码

### 8.1 SE(3) pose blending

建议使用位置线性插值 + 旋转 Slerp：

```python
from scipy.spatial.transform import Rotation as R, Slerp

def _blend_pose(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]

    rots = R.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]], axis=0))
    slerp = Slerp([0.0, 1.0], rots)
    T[:3, :3] = slerp([alpha]).as_matrix()[0]

    return T
```

### 8.2 阶段 7 target 生成

```python
def _eef_target_two_endpoint_delta_interp(
    t: int,
    seg: Segment,
    env,
    site_id: int,
    drawer_bid: int,
    eef_src: np.ndarray,
    drawer_world_src: np.ndarray,
    segment_anchors: dict[int, np.ndarray],
) -> np.ndarray:
    # 1. source endpoints
    T_a_src = eef_src[seg.start]
    T_b_src = eef_src[seg.end]

    # 2. new start: actual current EEF pose at stage start
    if seg.start not in segment_anchors:
        segment_anchors[seg.start] = _pose_from_site(env, site_id)
    T_a_new = segment_anchors[seg.start]

    # 3. new goal: drawer-relative source endpoint
    T_drawer_new = _pose_from_body(env, drawer_bid)
    T_rel_goal_src = _inv_pose(drawer_world_src[seg.end]) @ T_b_src
    T_b_new = T_drawer_new @ T_rel_goal_src

    # 4. endpoint transforms
    Delta_a = T_a_new @ _inv_pose(T_a_src)
    Delta_b = T_b_new @ _inv_pose(T_b_src)

    # 5. interpolate transform
    denom = max(seg.end - seg.start, 1)
    u = (t - seg.start) / denom
    alpha = u * u * (3.0 - 2.0 * u)

    Delta_t = _blend_pose(Delta_a, Delta_b, alpha)

    # 6. transform source trajectory
    return Delta_t @ eef_src[t]
```

### 8.3 修改 segmentation mode

建议新增：

```python
mode = "transform_two_endpoint_to_drawer"
```

在 `segments_from_labels` 中，将阶段 7 单独设置为该模式：

```python
if ref == "hammer":
    mode = "transform_hammer"
elif meta["stage_id"] == 7:
    mode = "transform_two_endpoint_to_drawer"
elif meta["stage_id"] == 9:
    mode = "transform_drawer_anchor"
else:
    mode = "transform_drawer"
```

然后在 `_compute_arm_eef_target` 中加入：

```python
if seg.mode == "transform_two_endpoint_to_drawer":
    return _eef_target_two_endpoint_delta_interp(
        t=t,
        seg=seg,
        env=env,
        site_id=site_id,
        drawer_bid=drawer_bid,
        eef_src=eef_src,
        drawer_world_src=drawer_world_src,
        segment_anchors=segment_anchors,
    )
```

---

## 9. 阶段 7 中 `T_b_new` 的选择

`T_b_new` 的定义会显著影响生成 demo 的成功率。

### 推荐版本：drawer-relative placement goal

```python
T_rel_goal_src = inv(T_drawer_src_at_stage7_end) @ T_eef_src_at_stage7_end
T_b_new = T_drawer_new @ T_rel_goal_src
```

适合：

- drawer 位置固定或轻微变化；
- 目标是把 hammer 放到 drawer 内部或 drawer 上方固定区域；
- source demo 的阶段 7 终点是可靠的 placement pose。

### 如果 drawer pose 不动

可以简化为：

```python
T_b_new = T_b_src
```

但仍建议保留 drawer-relative 写法，方便未来 drawer 随机化。

### 如果 drawer 是 articulated object

需要确认 `drawer_bid` 是否是正确的参考系。对于放置或关闭 drawer，通常应优先使用：

- drawer handle site；
- 可动 drawer link body；
- drawer interior target site；
- 而不是固定 cabinet body。

---

## 10. 其他重要注意事项

### 10.1 `skip_sim` 只能用于 debug

如果 `skip_sim=True`，但只修改 `datagen_info/eef_pose`、`target_pose` 或 hammer qpos，而没有真实 rollout robot state/action，那么生成的：

- `states`
- `actions`
- `obs`
- `datagen_info`

可能彼此不一致。

因此 `skip_sim` 只能用于检查几何变换是否合理，不应输出训练数据。

---

### 10.2 action_dict 中 6D rotation 也要同步

如果训练 pipeline 使用：

```python
action_dict/right_rel_rot_6d
```

那么不能只更新：

```python
right_rel_pos
right_rel_rot_axis_angle
```

还需要同步更新 `right_rel_rot_6d`。否则同一个 demo 中不同 action 表示会不一致。

---

### 10.3 成功过滤必不可少

自动生成的数据应通过 task success checker 过滤，例如：

- hammer 是否成功进入 drawer；
- drawer 是否最终关闭；
- robot 是否回到合理 home pose；
- hammer 是否没有掉落或穿模；
- hand 是否释放成功；
- 轨迹是否无严重碰撞。

DexMimicGen 本身也是先在仿真中执行生成轨迹，然后只保留成功的 demonstration。

---

### 10.4 碰撞和可达性需要单独处理

纯 SE(3) 变换不会显式处理碰撞与 IK 可达性。例如：

- hammer 新位置太靠近 drawer 或桌边；
- 搬运阶段穿过 drawer 边缘；
- EEF 姿态变化导致手腕不可达；
- hammer yaw 改变后 grasp pose 虽几何正确，但手指接触不稳定。

可以通过以下方式缓解：

1. 限制 hammer 随机化范围；
2. 阶段 7 使用更高的 lift waypoint；
3. 对阶段 7 加入中间 waypoint 或轨迹平滑；
4. rollout 后用 success checker 过滤；
5. 必要时加入 motion planner。

---

### 10.5 阶段 6 是否纯手部动作需要检查

如果阶段 6 只是闭合手指抓 hammer：

```text
replay hand only
hold EEF
```

是正确的。

但如果 source demo 中阶段 6 还包含：

- EEF 向 hammer 继续靠近；
- wrist 姿态调整；
- 手掌压向 hammer；
- contact 后小范围跟随；

那么阶段 6 不应整体标记为 hand replay，而应该拆成：

```text
6a final_align_to_hammer: hammer-relative arm motion
6b close_hand: hand replay only
```

---

### 10.6 阶段 3 / 11 的 drawer reference 要确认

如果 `env.obj_body_id["drawer"]` 对应的是固定 drawer cabinet，而不是可动 drawer body，那么：

```python
T_drawer_sim @ inv(T_drawer_src) @ T_eef_src
```

对拉开 / 关闭 drawer 的过程可能没有意义。应确认该 body 是否随抽屉开合移动。更稳的参考对象通常是：

- handle body；
- handle site；
- sliding drawer link；
- drawer 内部目标 site。

---

## 11. 建议的最终阶段 mode 配置

当前 drawer 不变、hammer 随机化的情况下，可以采用：

```python
STAGE_CATALOG = [
    {"stage_id": 1,  "motion_label": 0, "ref": "drawer", "name": "move_to_drawer"},
    {"stage_id": 2,  "motion_label": 1, "ref": "hand",   "name": "grasp_handle"},
    {"stage_id": 3,  "motion_label": 0, "ref": "drawer", "name": "open_drawer"},
    {"stage_id": 4,  "motion_label": 1, "ref": "hand",   "name": "open_hand"},
    {"stage_id": 5,  "motion_label": 0, "ref": "hammer", "name": "move_to_hammer"},
    {"stage_id": 6,  "motion_label": 1, "ref": "hand",   "name": "grasp_hammer"},
    {"stage_id": 7,  "motion_label": 0, "ref": "drawer", "name": "carry_hammer_to_drawer"},
    {"stage_id": 8,  "motion_label": 1, "ref": "hand",   "name": "release_hammer"},
    {"stage_id": 9,  "motion_label": 0, "ref": "drawer", "name": "move_to_handle"},
    {"stage_id": 10, "motion_label": 1, "ref": "hand",   "name": "grasp_handle"},
    {"stage_id": 11, "motion_label": 0, "ref": "drawer", "name": "close_and_home"},
]
```

对应的 mode 建议：

```python
stage 1:  replay_arm_or_transform_drawer
stage 2:  replay_hand_only
stage 3:  replay_arm_or_transform_drawer_handle
stage 4:  replay_hand_only
stage 5:  transform_hammer
stage 6:  replay_hand_only
stage 7:  transform_two_endpoint_to_drawer
stage 8:  replay_hand_only
stage 9:  replay_arm_or_transform_drawer_anchor
stage 10: replay_hand_only
stage 11: replay_arm_or_transform_drawer_then_home
```

如果希望保持和当前代码结构最接近，可以实现为：

```python
if seg.stage_id == 5:
    mode = "transform_hammer"
elif seg.stage_id == 7:
    mode = "transform_two_endpoint_to_drawer"
elif seg.motion_label == 1:
    mode = "replay_hand_only"
elif seg.ref == "drawer":
    mode = "transform_drawer"
```

当 drawer 不变时，`transform_drawer` 等价于近似 replay；但保留该模式可以让后续扩展 drawer 随机化更容易。

---

## 12. 最小修改清单

建议按以下优先级修改当前脚本：

1. 修改 `_new_hammer_pose`  
   使用 `p_new = p_src + [dx, dy, 0]`，而不是 `R_delta @ p_src + offset`。

2. 写入 hammer qpos 后重新读取 actual hammer world pose  
   用 actual pose 计算 `T_delta`。

3. 将阶段 7 改为 `transform_two_endpoint_to_drawer`  
   使用方法 1：`Delta_a -> Delta_b` 插值，然后作用到 source EEF trajectory。

4. 将 hand segment 从 `replay_action` 改为 `replay_hand_only`  
   只 replay 手部动作，EEF hold 在上一阶段结束 pose。

5. 检查 drawer 参考 body  
   确认 `drawer_bid` 是否是可动 drawer / handle，而不是固定外框。

6. 同步更新 action_dict  
   如果训练使用 6D rotation，则更新 `right_rel_rot_6d`。

7. 保留仿真 rollout 和 success filtering  
   不要将 `skip_sim=True` 的数据用于训练。

---

## 13. 当前任务的推荐总体逻辑

最终，`HammerCleanup` 的自动生成逻辑可以概括为：

```text
1. drawer 相关接近 / 开关阶段：
   drawer 不变时可 replay，最好写成 drawer-relative 以便扩展。

2. hammer 接近阶段：
   根据 hammer 新旧世界 pose 计算 SE(3) delta，变换 EEF trajectory。

3. grasp / release 等手部阶段：
   只 replay dexterous hand action，EEF 保持上一阶段末端 pose。

4. 搬运 hammer 到 drawer 的阶段：
   使用两端点变换插值法：
       起点 = 新场景中阶段 6 后实际 EEF pose
       终点 = drawer-relative placement goal
       中间 = Delta_a 到 Delta_b 的 SE(3) 插值作用于 source trajectory

5. rollout 生成真实 states/actions/obs/datagen_info。

6. 用 success checker 过滤失败 demo。
```
