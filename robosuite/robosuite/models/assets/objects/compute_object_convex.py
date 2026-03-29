"""使用pybullet进行v-hacd分解，并拆分成多个凸包"""
import trimesh
import coacd
import os

# ==========================================
# 1. 定义文件与文件夹路径
# ==========================================
input_obj = '/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/meshes/e_drill_2.obj'
split_output_dir = '/home/benhua/DexSim/robosuite/robosuite/models/assets/objects/meshes/e_drill_2_collision/'

# 确保拆分后的输出文件夹存在
os.makedirs(split_output_dir, exist_ok=True)

# ==========================================
# 2. 运行 CoACD 分解
# ==========================================
print(f"🔄 正在读取原始网格...\n   输入: {input_obj}")

# force="mesh" 确保读取出来的是单个 Trimesh 对象，而不是复合场景 (Scene)
mesh = trimesh.load(input_obj, force="mesh")

print("🔄 正在启动 CoACD 凸分解...")
# 将 Trimesh 对象转换为 CoACD 所需的格式
coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)

# 运行分解算法
# 提示: 如果觉得分出来的块不够细致，可以加入参数调整，例如: 
# parts = coacd.run_coacd(coacd_mesh, threshold=0.02)  # threshold 越小，切分越细 (默认通常为 0.05)
parts = coacd.run_coacd(coacd_mesh) 

print(f"✅ CoACD 分解完成！共拆解出 {len(parts)} 个凸块。")
print("-" * 40)

# ==========================================
# 3. 使用 Trimesh 重新构建并保存凸块
# ==========================================
print("🔄 正在保存各个独立的凸块...")

for i, part in enumerate(parts):
    # CoACD 的返回值 part 是包含顶点和面的元组: (vertices, faces)
    vertices, faces = part
    
    # 将顶点和面数据重新打包为 Trimesh 对象
    part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 构造保存路径
    part_filename = f"collision_part_{i}.obj"
    part_save_path = os.path.join(split_output_dir, part_filename)
    
    # 导出为 OBJ 文件
    part_mesh.export(part_save_path)
    print(f"   [+] 成功保存凸块 {i}: {part_filename}")

print("-" * 40)
print(f"🎉 全部处理完毕！模型已成功切分并存放在: {split_output_dir}")