import numpy as np
import torch
from utils.mast3r_utils import compute_pose_in_gt
import os
from utils.map_utils import torch2np, np2torch
from scipy.spatial.transform import Rotation as R

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
    # 将相机的平移增量 cam_trans_delta 和旋转增量 cam_rot_delta 合并为一个 6 维向量 tau，用于刚体变换。


    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T
    # 当前相机的位姿通过 4x4 的变换矩阵 T_w2c 表示，其中包含当前的旋转矩阵 camera.R 和平移向量 camera.T。
    new_w2c = SE3_exp(tau) @ T_w2c
    # 通过 SE3_exp(tau) 计算位姿增量的变换矩阵，并与当前相机的位姿 T_w2c 相乘，得到新的位姿矩阵 new_w2c
    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]
    # 将新的旋转矩阵 new_R 和新的平移向量 new_T 更新到相机对象中。

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    # 将旋转和平移增量重置为 0，准备下一次迭代的优化。


    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged


def parse_colmap_images_txt_to_matrices_c2w(file_path):
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
            
            # 转换四元数为旋转矩阵 (Camera-to-World)
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            translation = np.array([tx, ty, tz])
            
            # 构建 4x4 Camera-to-World 变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = translation
            
            # 保存解析后的变换矩阵
            poses[image_id] = {
                "matrix": transform_matrix,
                "image_path": image_path
            }
    return poses


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag


def start_alignment(gt_pose0, sfm_pose):
        RT_1_to_gt =  np.eye(4)
        RT_1_to_gt[:3, :3] = gt_pose0[:3, :3]
        RT_1_to_gt[:3, 3] = gt_pose0[:3, 3]
        #RT_1_to_gt = np.linalg.inv(RT_1_to_gt)
        R_1_to_gt = RT_1_to_gt[:3, :3]
        T_1_to_gt = RT_1_to_gt[:3, 3] 
        R_1_to_sfm = sfm_pose[1]["rotation"]
        T_1_to_sfm = sfm_pose[1]["translation"]
        sfm_togt = []
        for image_id, pose in sfm_pose.items():
            R_i_to_sfm = pose["rotation"]
            T_i_to_sfm = pose["translation"]

            R_i_to_gt, T_i_to_gt = compute_pose_in_gt(R_1_to_gt, T_1_to_gt, R_1_to_sfm, T_1_to_sfm, R_i_to_sfm, T_i_to_sfm)
            RT_i_to_gt = np.eye(4)
            RT_i_to_gt[:3, :3] = R_i_to_gt
            RT_i_to_gt[:3, 3] = T_i_to_gt
            sfm_togt.append(RT_i_to_gt)
        return sfm_togt


def compute_initpose(cur_frame_idx,prev_R, prev_t, colmap_path):
    prev_pose = np.eye(4)
    prev_pose[:3, :3] = torch2np(prev_R)
    prev_pose[:3, 3] = torch2np(prev_t)

    poses = parse_colmap_images_txt_to_matrices_c2w(os.path.join(colmap_path,"images.txt"))

    Trans = np.dot(poses[cur_frame_idx]["matrix"],np.linalg.inv(poses[cur_frame_idx+1]["matrix"]))
    init_pose = np.dot(np.linalg.inv(Trans),prev_pose)    
    init_R = init_pose[:3, :3]
    init_t = init_pose[:3, 3]
    return np2torch(init_R), np2torch(init_t) 
def compute_initpose_np(cur_frame_idx,prev_R, prev_t, colmap_path):
    prev_pose = np.eye(4)
    prev_pose[:3, :3] = prev_R
    prev_pose[:3, 3] = prev_t

    poses = parse_colmap_images_txt_to_matrices_c2w(os.path.join(colmap_path,"images.txt"))

    Trans = np.dot(poses[cur_frame_idx]["matrix"],np.linalg.inv(poses[cur_frame_idx+1]["matrix"]))
    init_pose = np.dot(np.linalg.inv(Trans),prev_pose)    
    init_R = init_pose[:3, :3]
    init_t = init_pose[:3, 3]
    return init_R, init_t
