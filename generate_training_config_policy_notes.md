# `generate_training_config.py` 训练配置梳理（面向后续 DP3 数据改造）

本文总结 `dexmimicgen/scripts/generate_training_config.py` 的实际训练配置逻辑，重点关注：

- policy training 使用了哪些变量；
- 这些变量是否做了标准化（normalization），以及如何做；
- 使用了哪些视觉输入；
- 你后续改 `convert_dexmimic_hdf5_to_dp3_zarr.py` 时需要对齐的关键点。

---

## 1) 脚本整体功能

该脚本基于 robomimic 的 `bc.json` 模板，自动为多个任务生成 BC-RNN 训练配置与对应训练命令。  
核心流程：

1. 定义任务级 settings（panda + humanoid）：
   - `dataset_paths`
   - `dataset_names`
   - `image_keys`
   - `low_dim_keys`
   - `horizon`
2. 调用 `make_gen(...)` 注入通用 BC-RNN 配置（序列长度、视觉编码器、obs keys、学习率等）。
3. 根据机器人类型追加 `train.action_keys` 和 `train.action_config`（这里定义动作字段和动作归一化）。
4. 输出每个任务对应的 config json 和 `python ... --config ...` 训练命令。

---

## 2) Policy Training 实际使用的变量

### 2.1 观测输入（Observation）

训练观测由两部分组成：`low_dim` + `rgb`。

- `observation.modalities.obs.low_dim`：来自每个任务 setting 里的 `low_dim_keys`
- `observation.modalities.obs.rgb`：来自每个任务 setting 里的 `image_keys`

#### Panda 任务 low-dim keys（统一）

- `robot0_eef_pos`
- `robot0_eef_quat`
- `robot0_gripper_qpos`
- `robot1_eef_pos`
- `robot1_eef_quat`
- `robot1_gripper_qpos`

#### Humanoid 任务 low-dim keys（统一）

- `robot0_right_eef_pos`
- `robot0_right_eef_quat`
- `robot0_right_gripper_qpos`
- `robot0_left_eef_pos`
- `robot0_left_eef_quat`
- `robot0_left_gripper_qpos`

---

### 2.2 动作监督目标（Action target）

脚本没有直接使用默认 `actions` 单字段，而是改成了 `action_dict/*` 的多字段动作训练：

#### Panda 动作字段

- `action_dict/right_rel_pos`
- `action_dict/right_rel_rot_axis_angle`
- `action_dict/right_gripper`
- `action_dict/left_rel_pos`
- `action_dict/left_rel_rot_axis_angle`
- `action_dict/left_gripper`

#### Humanoid 动作字段

- `action_dict/right_abs_pos`
- `action_dict/right_abs_rot_6d`
- `action_dict/left_abs_pos`
- `action_dict/left_abs_rot_6d`
- `action_dict/right_gripper`
- `action_dict/left_gripper`

> 这意味着数据集里必须存在这些 `action_dict/...` 键，否则训练读取会失败或与配置不一致。

---

### 2.3 训练超参数（与 policy 相关）

由 `set_learning_settings_for_bc_rnn(...)` 覆盖的关键项：

- `train.seq_length = 10`
- `train.frame_stack = 1`
- `algo.rnn.enabled = True`
- `algo.rnn.horizon = 10`
- `algo.rnn.hidden_dim = 1000`
- `algo.gmm.enabled = False`
- `algo.actor_layer_dims = []`
- `train.batch_size = 16`
- `train.num_epochs = 600`
- `algo.optim_params.policy.learning_rate.initial = 1e-4`
- `train.num_data_workers = 4`
- `train.hdf5_cache_mode = "low_dim"`
- `train.seed = [201, 202, 203]`

---

## 3) 标准化（Normalization）与数据增强

## 3.1 明确设置的动作标准化（核心）

动作归一化在 `train.action_config` 中显式指定：

### Panda

- `*_rel_pos`: `normalization = None`
- `*_rel_rot_axis_angle`: `normalization = None`，并声明 `format = "rot_axis_angle"`
- `*_gripper`: `normalization = "min_max"`

### Humanoid

- `*_abs_pos`: `normalization = "min_max"`
- `*_abs_rot_6d`: `normalization = None`，并声明 `format = "rot_6d"`
- `*_gripper`: `normalization = "min_max"`

结论：**只有部分动作维度启用了 min-max 标准化**，旋转类字段保持原值（但附带格式声明）。

---

## 3.2 观测标准化（obs normalization）

基于模板 `robomimic/exps/templates/bc.json`：

- `train.hdf5_normalize_obs = false`

而 `generate_training_config.py` / `config_utils.py` 没有把它改成 `true`。  
因此本脚本配置下：**未启用 robomimic 的全局观测标准化**（至少没有在该生成脚本中显式开启）。

---

## 3.3 视觉数据增强（不是 normalization，但会影响输入分布）

RGB 编码器配置中启用了：

- `obs_randomizer_class = "CropRandomizer"`
- crop 尺寸默认 `76 x 76`（来自 `crop_size=[76, 76]`）
- `num_crops = 1`
- backbone: `ResNet18Conv`
- pool: `SpatialSoftmax(num_kp=32)`

这属于训练时图像随机裁剪增强，不是数值标准化本身。

---

## 4) 使用了哪些视觉输入（按任务）

### Panda 任务

- `two_arm_box_cleanup`:  
  `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`
- `two_arm_lift_tray`:  
  `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`
- `two_arm_drawer_cleanup`:  
  `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`
- `two_arm_three_piece_assembly`:  
  `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`
- `two_arm_transport`:  
  `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`, `shouldercamera0_image`, `shouldercamera1_image`
- `two_arm_threading`:  
  `agentview_image`, `robot0_eye_in_hand_image`, `robot1_eye_in_hand_image`

### Humanoid 任务

- `two_arm_pouring_humanoid`:  
  `agentview_image`, `robot0_eye_in_left_hand_image`, `robot0_eye_in_right_hand_image`
- `two_arm_coffee_humanoid`:  
  `agentview_image`, `robot0_eye_in_left_hand_image`, `robot0_eye_in_right_hand_image`
- `two_arm_can_sort_humanoid`:  
  `frontview_image`, `robot0_eye_in_left_hand_image`, `robot0_eye_in_right_hand_image`

---

## 5) 对 `convert_dexmimic_hdf5_to_dp3_zarr.py` 的改造对齐建议

如果你要“仿照这套训练方式”，数据转换至少要对齐以下点：

1. **动作字段层面**
   - 当前转换脚本主要写了 `data/action`（来自 `/actions`）。
   - 但训练配置使用的是 `action_dict/...` 多字段，并且每个字段有不同 normalization 策略。
   - 建议在 zarr 中保留或拆分出与 `action_dict/*` 一一对应的字段（或记录可逆映射）。

2. **视觉输入层面**
   - 当前脚本固定读取双手相机并拼成 `(T,2,84,84,3)`。
   - 配置脚本中存在 3 相机、5 相机、不同命名（`frontview_image` / `shouldercamera*`）的任务。
   - 建议把相机键做成可配置列表，而不是硬编码两路眼在手相机。

3. **低维状态层面**
   - 训练配置的 low-dim 是 eef / gripper 语义键，不是你当前脚本拼接的 joint state 向量。
   - 若目标是对齐这套 robomimic 训练，建议同时导出对应语义 low-dim 字段，避免只能用 joint 拼接状态。

4. **标准化策略可追踪**
   - 将每个动作子字段的 normalization 策略（`None` / `min_max`）写入元数据（例如 `meta/action_config`），方便训练器一致读取。

5. **图像增强由训练端处理**
   - 数据侧只需保证图像语义和维度正确；`CropRandomizer(76x76)` 在训练配置里完成。

---

## 6) 一句话结论

这份训练配置是“**多模态 obs（low_dim + 多路 RGB） + action_dict 多字段监督 + 按字段动作归一化**”的 BC-RNN 配置；你后续改 zarr 转换脚本时，最关键是把 **动作字段结构** 和 **相机键集合** 做到与训练配置一致且可配置。
