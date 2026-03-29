import numpy as np
import trimesh

# 路径请按你的本地仓库位置调整
mesh_path = "/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/meshes/aluminum_box_1.obj"

mesh = trimesh.load(mesh_path, force='mesh')

# 原始尺寸（未缩放）
bounds = mesh.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]
size = bounds[1] - bounds[0]
center = mesh.bounding_box.centroid

print("原始尺寸 size:", size)
print("原始中心 center:", center)

# 应用 xml 中 scale
# scale = np.array([0.3, 0.3, 0.3])
scale = 1
scaled_size = size * scale
scaled_bounds = bounds * scale  # 若中心不为0，这一步仅用于近似
print("缩放后尺寸 scaled_size:", scaled_size)

# 建议 site（假设网格中心大致在原点）
bottom_z = -scaled_size[2] / 2.0
top_z = scaled_size[2] / 2.0
horizontal_radius = max(scaled_size[0], scaled_size[1]) / 2.0

print("建议 bottom_site z:", bottom_z)
print("建议 top_site z:", top_z)
print("建议 horizontal_radius:", horizontal_radius)