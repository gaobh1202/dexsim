# VisionPro 坐标变换校准指南

## 问题描述

当使用 VisionPro 控制机械臂时，可能会遇到坐标轴映射错误的问题。例如：
- 双手向上移动时，机械臂末端向下
- 双手向前移动时，机械臂末端向后
- 双手向右移动时，机械臂末端向左

这是因为 Vision Pro 的坐标系与机器人世界坐标系不一致导致的。

## 解决方案

VisionPro 设备现在支持可配置的坐标轴映射和符号变换，通过以下参数进行调整：

### 1. 位置轴映射 (`--visionpro-pos-axis-map`)

用于调整 Vision Pro 坐标轴到机器人坐标轴的映射关系。

**格式**: `'i,j,k'`，其中 i, j, k 是索引 [0, 1, 2]

**含义**: Vision Pro 的轴 [i, j, k] 映射到机器人的轴 [x, y, z]

**示例**:
- `'0,1,2'`: 直接映射（默认）- Vision Pro [x, y, z] → 机器人 [x, y, z]
- `'1,0,2'`: Y和X交换 - Vision Pro [y, x, z] → 机器人 [x, y, z]
- `'0,2,1'`: Y和Z交换 - Vision Pro [x, z, y] → 机器人 [x, y, z]
- `'2,1,0'`: Z和X交换 - Vision Pro [z, y, x] → 机器人 [x, y, z]

### 2. 位置轴符号 (`--visionpro-pos-axis-signs`)

用于反转某个轴的方向。

**格式**: `'sx,sy,sz'`，其中每个符号是 1 或 -1

**含义**: 每个轴的符号，1 表示正方向，-1 表示反方向

**示例**:
- `'1,1,1'`: 所有轴都是正方向（默认）
- `'1,1,-1'`: Z轴反转（解决上下方向相反的问题）
- `'-1,1,1'`: X轴反转（解决左右方向相反的问题）
- `'1,-1,1'`: Y轴反转（解决前后方向相反的问题）

### 3. 调试模式 (`--visionpro-enable-coord-debug`)

启用调试模式，实时显示坐标变换前后的数据，帮助快速定位问题。

## 校准步骤

### 步骤 1: 启用调试模式

首先运行脚本并启用调试模式：

```bash
python collect_human_demonstration_in_dexmimic.py \
  --device visionpro \
  --visionpro-enable-coord-debug
```

### 步骤 2: 测试单轴运动

分别进行以下测试动作，观察机器人末端运动方向：

1. **向上移动手部** → 机器人应该向上移动（Z轴正方向）
2. **向前移动手部** → 机器人应该向前移动（Y轴正方向）
3. **向右移动手部** → 机器人应该向右移动（X轴正方向）

### 步骤 3: 根据测试结果调整

#### 情况 A: 方向相反（例如：向上移动导致向下移动）

如果某个轴的方向完全相反，只需要反转该轴的符号：

```bash
# Z轴反转（最常见的情况）
python collect_human_demonstration_in_dexmimic.py \
  --device visionpro \
  --visionpro-pos-axis-signs='1,1,-1'

# X轴反转
--visionpro-pos-axis-signs='-1,1,1'

# Y轴反转
--visionpro-pos-axis-signs='1,-1,1'
```

#### 情况 B: 轴的映射关系错误（例如：前后和左右搞反）

如果轴的映射关系不对，需要调整轴映射：

```bash
# Y和X交换（前后和左右搞反）
python collect_human_demonstration_in_dexmimic.py \
  --device visionpro \
  --visionpro-pos-axis-map='1,0,2'

# Y和Z交换（前后和上下搞反）
--visionpro-pos-axis-map='0,2,1'

# X和Z交换（左右和上下搞反）
--visionpro-pos-axis-map='2,1,0'
```

#### 情况 C: 组合调整

可以同时使用轴映射和符号反转：

```bash
# 例如：交换Y和X，并反转Z轴
python collect_human_demonstration_in_dexmimic.py \
  --device visionpro \
  --visionpro-pos-axis-map='1,0,2' \
  --visionpro-pos-axis-signs='1,1,-1'
```

### 步骤 4: 验证校准结果

重复步骤2的测试，确保所有方向都正确。如果还有问题，可以继续微调参数。

## 常见问题排查

### Q1: 如何快速确定哪个轴有问题？

A: 启用调试模式，然后依次测试单个轴的运动。观察调试输出中的"变换后位置增量"，如果某个轴的值符号错误，就需要反转该轴。

### Q2: 如何理解轴映射 `'1,0,2'`？

A: 这表示：
- Vision Pro 的 Y 轴（索引1）映射到机器人的 X 轴
- Vision Pro 的 X 轴（索引0）映射到机器人的 Y 轴
- Vision Pro 的 Z 轴（索引2）映射到机器人的 Z 轴

换句话说：`map[i]` 表示 Vision Pro 的第 i 个轴映射到机器人的第 map[i] 个轴。

### Q3: 旋转方向也有问题怎么办？

A: 当前实现只处理了位置坐标的变换。如果旋转方向也有问题，需要修改 `visionpro.py` 中的旋转处理代码（大约在第580行附近）。

### Q4: 校准后如何保存配置？

A: 可以将校准好的参数添加到启动脚本或配置文件中，避免每次都要手动输入。

## 技术细节

### Vision Pro 坐标系

Vision Pro 的坐标系通常是：
- X 轴：指向右侧
- Y 轴：指向上方
- Z 轴：指向前方（或后方，取决于具体实现）

### 机器人世界坐标系

机器人世界坐标系通常是：
- X 轴：指向右侧
- Y 轴：指向前方
- Z 轴：指向上方

因此，最常见的映射可能是：
- `--visionpro-pos-axis-map='0,2,1'`（Y和Z交换）
- `--visionpro-pos-axis-signs='1,1,1'`（或者根据实际情况调整Z的符号）

## 参考

- VisionPro 设备类：`robosuite/robosuite/devices/visionpro.py`
- 收集脚本：`robosuite/robosuite/scripts/collect_human_demonstration_in_dexmimic.py`
