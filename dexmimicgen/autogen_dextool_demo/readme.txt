1) 生成回顾视频
PYTHONPATH=robosuite:dexmimicgen MUJOCO_GL=osmesa \
python dexmimicgen/autogen_dextool_demo/export_demo_review_video.py \
  --dataset dexmimicgen/datasets/generated/single_arm_hammer_cleanup_demo_4.hdf5 --demo demo_0

2) 逐帧标注（先点击 OpenCV 窗口获得焦点；a/d 前后帧）
python dexmimicgen/autogen_dextool_demo/keyframe_labeling.py \
  --video dexmimicgen/autogen_dextool_demo/outputs/single_arm_hammer_cleanup_demo_4_demo_0_review.mp4

3) 自动生成训练用 demo（合并到一个 HDF5，格式同 single_arm_hammer_cleanup.hdf5）
PYTHONPATH=robosuite:dexmimicgen MUJOCO_GL=osmesa \
python dexmimicgen/autogen_dextool_demo/autogen_hammer_cleanup.py \
  --labels dexmimicgen/autogen_dextool_demo/outputs/single_arm_hammer_cleanup_demo_4_demo_0_review_labels.json \
  -n 5 --seed 0
# 默认输出: dexmimicgen/datasets/generated/single_arm_hammer_cleanup_autogen.hdf5
#   所有新 demo 均来自同一 --source-hdf5（文件内须仅含 1 条 demo，默认 demo_4 单条提取集）
#   data/demo_0 .. demo_{n-1}, data.attrs[env_args, total]（与手工合并集相同结构）
# datagen_info/stage_label: 0=机械臂段, 1=灵巧手段
# 生成策略见 dexmimicgen/demo_generation/trajectory_gen_pipeline.md 第二节

按键       功能
a       上一帧
d       下一帧
q       hand 段起点
e       hand 段终点
s       保存
Esc     退出并保存