# VisionPro 坐标映射问题分析

## 当前观察到的映射（从front_view视角）

| 手部动作 | Vision Pro轴 | 机械臂动作 | 机器人轴 | 问题 |
|---------|-------------|-----------|---------|------|
| 向上移动 | Z+ | 向上移动 | Z+ | ✅ 正确 |
| 向下移动 | Z- | 向下移动 | Z- | ✅ 正确 |
| 向前移动 | Y+ | 向后移动 | Y- | ❌ Y轴方向相反 |
| 向后移动 | Y- | 向左移动 | X- | ❌ Y映射到了X |
| 向右移动 | X+ | 向前移动 | Y+ | ❌ X映射到了Y |
| 向左移动 | X- | 向后移动 | Y- | ❌ X映射到了Y |

## 详细分析

根据观察结果，映射关系如下：

1. **Z轴（上下）**：✅ 正确，不需要修改
   - Vision Pro Z+ → Robot Z+
   - Vision Pro Z- → Robot Z-

2. **X轴和Y轴的映射**：
   - Vision Pro X+ (右) → Robot Y+ (前)
   - Vision Pro X- (左) → Robot Y- (后)
   - Vision Pro Y+ (前) → Robot Y- (后) 或 Robot X+ (右)？
   - Vision Pro Y- (后) → Robot X- (左)

从这些观察可以推断：
- **Vision Pro的X轴（左右）映射到了机器人的Y轴（前后）**
- **Vision Pro的Y轴（前后）映射到了机器人的X轴（左右），但方向可能相反**

## 解决方案

### 推荐方案：交换X和Y轴，并反转X轴

```bash
--visionpro-pos-axis-map='1,0,2' \
--visionpro-pos-axis-signs='-1,1,1'
```

**解释**：
- `pos_axis_map='1,0,2'`：Vision Pro的[Y, X, Z]映射到机器人的[X, Y, Z]
  - VP Y → Robot X
  - VP X → Robot Y  
  - VP Z → Robot Z
- `pos_axis_signs='-1,1,1'`：反转第一个轴（映射后的X轴，即原VP的Y轴）
  - Robot X轴反转（因为VP Y+应该映射到Robot X+，但当前是X-）
  - Robot Y轴保持（VP X+映射到Robot Y+）
  - Robot Z轴保持

### 备选方案1：如果X轴方向不对，尝试

```bash
--visionpro-pos-axis-map='1,0,2' \
--visionpro-pos-axis-signs='1,-1,1'
```

这表示反转Y轴而不是X轴。

### 备选方案2：如果映射关系不完全正确

可能需要根据实际测试结果微调。建议启用调试模式观察：

```bash
--visionpro-enable-coord-debug
```

## 测试命令

### 第一步：尝试推荐方案

```bash
python collect_human_demonstration_in_dexmimic.py \
  --device visionpro \
  --visionpro-pos-axis-map='1,0,2' \
  --visionpro-pos-axis-signs='-1,1,1' \
  --visionpro-enable-coord-debug
```

### 第二步：系统化测试

依次测试以下动作，观察机械臂末端运动：

1. **向上移动手部** → 机械臂应该向上 ✅
2. **向下移动手部** → 机械臂应该向下 ✅
3. **向前移动手部** → 机械臂应该向前（需要验证）
4. **向后移动手部** → 机械臂应该向后（需要验证）
5. **向右移动手部** → 机械臂应该向右（需要验证）
6. **向左移动手部** → 机械臂应该向左（需要验证）

### 第三步：根据结果微调

如果某个方向仍然不对：
- **方向相反**：调整对应的符号（1改为-1，或-1改为1）
- **映射错误**：调整轴映射顺序

## 调试技巧

启用调试模式后，观察控制台输出：
- "原始位置增量"：Vision Pro坐标系下的增量
- "变换后位置增量"：机器人坐标系下的增量

根据这些数值，可以更精确地判断需要调整哪些参数。
