# Panda机器人VisionPro控制测试脚本

这是一个最简单的脚本，用于测试Panda机器人通过VisionPro进行控制，并确定坐标系变换方式。

## 功能特点

- **最简单的环境**：只包含一个Panda机器人和一个方块（Lift环境）
- **VisionPro控制**：使用VisionPro设备进行手部姿态控制
- **坐标调试模式**：默认启用坐标变换调试输出，方便确定正确的坐标系映射
- **简洁直观**：代码结构清晰，易于理解和修改

## 使用方法

### 1. 基本运行

```bash
python robosuite/robosuite/scripts/test_panda_visionpro_simple.py
```

### 2. 配置参数（通过环境变量）

#### VisionPro Room Code
```bash
export VISIONPRO_ROOM_CODE="SHQB-7053"  # 替换为实际的room code
```

#### 坐标轴映射
如果VisionPro的坐标系与机器人坐标系不一致，可以通过以下参数调整：

```bash
# 位置轴映射：格式为 "i,j,k"，其中i,j,k是索引[0,1,2]
# 例如："1,0,2" 表示 Vision Pro的 [y,x,z] 映射到机器人的 [x,y,z]
export VISIONPRO_POS_AXIS_MAP="1,0,2"

# 位置轴符号：格式为 "sx,sy,sz"，其中每个值是1或-1
# 例如："1,1,-1" 表示反转z轴
export VISIONPRO_POS_AXIS_SIGNS="1,1,-1"
```

#### 坐标调试模式
```bash
# 启用坐标调试输出（默认启用）
export VISIONPRO_ENABLE_COORD_DEBUG="True"

# 禁用坐标调试输出
export VISIONPRO_ENABLE_COORD_DEBUG="False"
```

### 3. 完整示例

```bash
# 设置所有参数并运行
export VISIONPRO_ROOM_CODE="SHQB-7053"
export VISIONPRO_POS_AXIS_MAP="1,0,2"
export VISIONPRO_POS_AXIS_SIGNS="1,1,-1"
export VISIONPRO_ENABLE_COORD_DEBUG="True"

python robosuite/robosuite/scripts/test_panda_visionpro_simple.py
```

## 坐标系变换校准步骤

1. **运行脚本**：启动脚本后，系统会进行校准（采集50帧数据）
2. **保持初始姿态**：手肘90度，放在身体两侧，手心朝下手指朝前
3. **测试单轴运动**：
   - 向上移动手部 → 机器人应该向上移动（Z轴正方向）
   - 向前移动手部 → 机器人应该向前移动（Y轴正方向）
   - 向右移动手部 → 机器人应该向右移动（X轴正方向）
4. **根据测试结果调整参数**：
   - 如果某个轴方向相反，修改 `VISIONPRO_POS_AXIS_SIGNS`
   - 如果轴的映射关系不对，修改 `VISIONPRO_POS_AXIS_MAP`
5. **查看调试输出**：如果启用了坐标调试，会打印前10帧的坐标变换信息

## 常见轴映射配置

| 问题 | 解决方案 |
|------|---------|
| 前后和左右搞反 | `VISIONPRO_POS_AXIS_MAP="1,0,2"` |
| 前后和上下搞反 | `VISIONPRO_POS_AXIS_MAP="0,2,1"` |
| 左右和上下搞反 | `VISIONPRO_POS_AXIS_MAP="2,1,0"` |
| Z轴方向相反 | `VISIONPRO_POS_AXIS_SIGNS="1,1,-1"` |
| Y轴方向相反 | `VISIONPRO_POS_AXIS_SIGNS="1,-1,1"` |
| X轴方向相反 | `VISIONPRO_POS_AXIS_SIGNS="-1,1,1"` |

## 注意事项

1. **确保VisionPro已连接**：脚本启动时会尝试连接VisionPro设备
2. **校准时间**：校准阶段需要采集50帧数据，请保持稳定姿态
3. **控制频率**：脚本限制在20Hz，确保实时性
4. **退出方式**：按 `Ctrl+C` 退出控制循环

## 代码结构

```
test_panda_visionpro_simple.py
├── 1. 创建Panda环境
│   └── 使用Lift环境（最简单的单臂环境）
├── 2. 初始化VisionPro设备
│   ├── 创建VisionProStreamer
│   ├── 解析坐标变换参数
│   └── 创建VisionPro设备
├── 3. 设置机器人初始位置
│   └── 将机器人移动到便于控制的姿态
└── 4. 控制循环
    ├── 获取VisionPro动作
    ├── 构建环境动作
    └── 执行动作并渲染
```

## 故障排除

### 问题1：无法连接到VisionPro
- 检查room code是否正确
- 确保VisionPro设备已启动并显示room code

### 问题2：机器人不响应手部运动
- 检查是否完成了校准阶段
- 查看控制台是否有错误信息
- 确认VisionPro能检测到双手

### 问题3：坐标方向错误
- 启用坐标调试模式：`VISIONPRO_ENABLE_COORD_DEBUG="True"`
- 根据调试输出调整轴映射和符号参数
- 参考"常见轴映射配置"表格

## 相关文件

- `robosuite/devices/visionpro.py` - VisionPro设备实现
- `robosuite/scripts/collect_human_demonstration_in_dexmimic.py` - 完整的数据收集脚本
