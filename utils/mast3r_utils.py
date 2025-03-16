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

def compute_pose_in_gt(R_1_to_gt, T_1_to_gt, R_1_to_sfm, T_1_to_sfm, R_i_to_sfm, T_i_to_sfm):
    """
    将图像从 SfM 坐标系转换到 GT 坐标系
    :param R_1_to_gt: 图像 1 在 GT 坐标系中的旋转矩阵 (3x3)
    :param T_1_to_gt: 图像 1 在 GT 坐标系中的平移向量 (3,)
    :param R_1_to_sfm: 图像 1 在 SfM 坐标系中的旋转矩阵 (3x3)
    :param T_1_to_sfm: 图像 1 在 SfM 坐标系中的平移向量 (3,)
    :param R_i_to_sfm: 图像 i 在 SfM 坐标系中的旋转矩阵 (3x3)
    :param T_i_to_sfm: 图像 i 在 SfM 坐标系中的平移向量 (3,)
    :return: R_i_to_gt, T_i_to_gt
    """
    # 计算 SfM 坐标系到 GT 坐标系的旋转和平移
    R_sfm_to_gt = R_1_to_gt @  np.linalg.inv(R_1_to_sfm)
    T_sfm_to_gt = T_1_to_gt - R_sfm_to_gt @ T_1_to_sfm

    # 计算图像 i 在 GT 坐标系中的旋转和平移
    R_i_to_gt = R_sfm_to_gt @ R_i_to_sfm
    T_i_to_gt = R_sfm_to_gt @ T_i_to_sfm + T_sfm_to_gt

    return R_i_to_gt, T_i_to_gt