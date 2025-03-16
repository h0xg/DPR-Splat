import numpy as np
from scipy.spatial.transform import Rotation as R

# 计算相对变换
def compute_relative_transform(ref_pose, target_pose):
    R_ref, t_ref = ref_pose["rotation"], ref_pose["translation"]
    R_target, t_target = target_pose["rotation"], target_pose["translation"]
    
    # 相对旋转：R_relative = R_target * R_ref^T
    R_relative = R_target @ R_ref.T
    
    # 相对平移：t_relative = t_target - R_relative * t_ref
    t_relative = t_target - R_relative @ t_ref
    
    return R_relative, t_relative

# 解析 COLMAP 的 images.txt 文件
def parse_colmap_images_txt(file_path):
    """解析 COLMAP 的 images.txt 文件"""
    poses = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # 去除多余空白
            if not line or line.startswith("#"):  # 跳过空行或注释
                continue
            
            data = line.split()
            image_id = int(data[0])  # 图像 ID
            qw, qx, qy, qz = map(float, data[1:5])  # 四元数
            tx, ty, tz = map(float, data[5:8])      # 平移向量
            image_path = data[8]                   # 图像路径
            
            # 转换四元数为旋转矩阵
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            translation = np.array([tx, ty, tz])
            
            # 保存解析后的姿态
            poses[image_id] = {
                "rotation": rotation,
                "translation": translation,
                "image_path": image_path
            }
    return poses

# 示例：读取文件
file_path = "./output/images.txt"
poses = parse_colmap_images_txt(file_path)

# 打印结果
ref_pose = poses[1]  # 参考姿态（第一张图）

# 计算第二张和第三张图相对于第一张图的变换
for image_id in [2, 3]:
    R_relative, t_relative = compute_relative_transform(ref_pose, poses[image_id])
    print(f"Image {image_id} relative to Image 1:")
    print("Relative Rotation Matrix:")
    print(R_relative)
    print("Relative Translation Vector:")
    print(t_relative)
    print()