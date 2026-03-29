import numpy as np
import trimesh
from pathlib import Path

# 路径按需修改
obj_path = Path("/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/meshes/simplified.obj")

# 读取网格
mesh = trimesh.load(obj_path, force="mesh")

# 计算几何中心（顶点均值）
center = mesh.vertices.mean(axis=0)
print("原始中心:", center)

# 平移顶点到原点
mesh.vertices = mesh.vertices - center

# 保存（覆盖原文件）
mesh.export(obj_path)
print("已重心归零并保存:", obj_path)

# 再次验证
mesh2 = trimesh.load(obj_path, force="mesh")
print("新中心:", mesh2.vertices.mean(axis=0))