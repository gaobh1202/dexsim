import numpy as np
import trimesh

# 修改为你的模型路径
mesh_path = "/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/meshes/simplified_zup_1.obj"
mesh = trimesh.load(mesh_path, force="mesh")

# 1) 基本尺寸
bounds = mesh.bounds
size = bounds[1] - bounds[0]
print("AABB size:", size)

# 2) PCA 主轴（表示几何“长/宽/高”的方向）
verts = mesh.vertices
center = verts.mean(axis=0)
verts_centered = verts - center

# 协方差 & 特征分解
cov = np.cov(verts_centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)

# 按特征值从大到小排序（主轴1/2/3）
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

print("主轴方向（列向量）:")
print(eigvecs)

# 3) 可视化：模型 + 三根主轴箭头
axis_len = np.max(size) * 2  # 轴长度可调
colors = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1]])  # RGB

scene = trimesh.Scene()
scene.add_geometry(mesh)

# 画三根主轴圆柱
radius = np.max(size) * 0.01
z_axis = np.array([0.0, 0.0, 1.0])

for i in range(3):
    direction = eigvecs[:, i]
    direction = direction / np.linalg.norm(direction)
    cyl = trimesh.creation.cylinder(radius=radius, height=axis_len, sections=20)

    # 计算从 Z 轴旋转到 direction 的旋转矩阵
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    if np.linalg.norm(v) < 1e-8:
        rot = np.eye(3)
        if c < 0:
            rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = center
    cyl.apply_transform(transform)
    cyl.visual.vertex_colors = (colors[i] * 255).astype(np.uint8)
    scene.add_geometry(cyl)

try:
    scene.show()
except ModuleNotFoundError as exc:
    if "pyglet" in str(exc):
        out_path = "/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/simplified_axes.glb"
        scene.export(out_path)
        print(f"未安装 pyglet，已导出可视化文件：{out_path}")
    else:
        raise