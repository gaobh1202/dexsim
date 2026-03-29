import numpy as np
import trimesh
from pathlib import Path

# 输入/输出路径
src = Path("/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/meshes/simplified.obj")
dst = src.with_name("simplified_zup.obj")

mesh = trimesh.load(src, force="mesh")

# PCA 主轴
verts = mesh.vertices
center = verts.mean(axis=0)
verts_centered = verts - center
cov = np.cov(verts_centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, order]  # 主轴从大到小

# 选择最长轴对齐到 Z
src_axis = eigvecs[:, 0]          # 最长主轴
src_axis = src_axis / np.linalg.norm(src_axis)
dst_axis = np.array([0.0, 0.0, 1.0])

# 计算旋转矩阵（Rodrigues）
v = np.cross(src_axis, dst_axis)
c = np.dot(src_axis, dst_axis)
if np.linalg.norm(v) < 1e-8:
    # 已对齐或反向
    if c < 0:
        # 180度翻转，任选一条垂直轴
        rot = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    else:
        rot = np.eye(3)
else:
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

# 应用旋转（绕中心）
mesh.vertices = (mesh.vertices - center) @ rot.T + center

# 导出
mesh.export(dst)
print("已导出:", dst)