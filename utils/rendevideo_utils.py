import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import torch
import imageio
import os

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def visualizer(camera_poses, colors, save_path="/mnt/data/1.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for pose, color in zip(camera_poses, colors):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]  # Corrected to use 3D translation component
        camera_positions = np.einsum(
            "...ij,...j->...i", np.linalg.inv(rotation), -translation
        )

        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")

    plt.savefig(save_path)
    plt.close()

    return save_path

def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m
def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    ###  Additional operation
    # inter_poses = []
    # for pose in poses:
    #     tmp_pose = np.eye(4)
    #     tmp_pose[:3] = np.concatenate([pose.R.T, pose.T[:, None]], 1)
    #     tmp_pose = np.linalg.inv(tmp_pose)
    #     tmp_pose[:, 1:3] *= -1
    #     inter_poses.append(tmp_pose)
    # inter_poses = np.stack(inter_poses, 0)
    # poses, transform = transform_poses_pca(inter_poses)

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points) 




def save_interpolate_pose(org_pose, n_views, outputpath):
    """
    Saves and visualizes interpolated camera poses.
    
    Parameters:
    - org_pose: list of numpy arrays, original 4x4 camera poses
    - n_views: int, number of original keyframe views
    - outputpath: Path object, directory to save output files
    """
    outputpath = Path(outputpath)
    outputpath.mkdir(parents=True, exist_ok=True)  # 确保输出路径存在

    # 可视化原始姿态
    visualizer(org_pose, ["green"] * len(org_pose), outputpath / "poses_optimized.png")

    # 计算插值的帧数
    n_interp = int(10 * 30 / n_views)  # 10 秒, FPS=30

    # 存储所有插值后的姿态
    all_inter_pose = [org_pose[0][:3, :].reshape(1, 3, 4)]  # 确保第一个姿态保留

    for i in range(n_views - 1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)

    all_inter_pose.append(org_pose[-1][:3, :].reshape(1, 3, 4))  # 确保最后一个姿态保留

    # 合并并转换为 4x4 变换矩阵
    all_inter_pose = np.vstack(all_inter_pose)  # 保证形状一致
    inter_pose_list = []

    for p in all_inter_pose:
        tmp_view = np.eye(4)  # 生成 4x4 单位矩阵
        tmp_view[:3, :3] = p[:3, :3]  # 复制旋转部分
        tmp_view[:3, 3] = p[:3, 3]    # 复制平移部分
        inter_pose_list.append(tmp_view)

    inter_pose = np.stack(inter_pose_list, axis=0)

    # 可视化插值后的姿态
    visualizer(inter_pose, ["blue"] * len(inter_pose), outputpath / "poses_interpolated.png")

    # 保存插值姿态
    np.save(outputpath / "pose_interpolated.npy", inter_pose)

    print(f"Interpolated poses saved at: {outputpath / 'pose_interpolated.npy'}")
    return inter_pose
def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

def loadCameras(poses, viewpoint_stack):

    # load optimized poses
    if poses.shape[0] == len(viewpoint_stack):
        for idx, cam in enumerate(viewpoint_stack):
            R = np.transpose(poses[idx][:3, :3])
            T = poses[idx][:3, 3]
            cam.R = R
            cam.T = T
            cam.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]

    # load interpolated poses
    elif poses.shape[0] > len(viewpoint_stack):
        repeat_times = int(np.ceil(poses.shape[0] / len(viewpoint_stack)))
        # Create repeated list instead of using np.tile
        viewpoint_stack = [copy.deepcopy(vp) for vp in viewpoint_stack * repeat_times][:poses.shape[0]]
        for idx in range(poses.shape[0]):                                 
            R = np.transpose(poses[idx][:3, :3])
            T = poses[idx][:3, 3]
            viewpoint_stack[idx].uid = idx           
            viewpoint_stack[idx].colmap_id = idx+1    
            viewpoint_stack[idx].image_name = str(idx).zfill(5)    
            viewpoint_stack[idx].R = R
            viewpoint_stack[idx].T = T            
            viewpoint_stack[idx].world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            viewpoint_stack[idx].full_proj_transform = (viewpoint_stack[idx].world_view_transform.unsqueeze(0).bmm(viewpoint_stack[idx].projection_matrix.unsqueeze(0))).squeeze(0)
            viewpoint_stack[idx].camera_center = viewpoint_stack[idx].world_view_transform.inverse()[3, :3]
    return viewpoint_stack