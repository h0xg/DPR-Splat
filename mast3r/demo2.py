from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import os
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
from depth_utils import update_metrics
import torch

def parse_extrinsics(matrix):
    """解析外参矩阵"""
    R = matrix[:3, :3]  # 旋转矩阵
    T = matrix[:3, 3]   # 平移向量
    return R, T

def ransac_triangulation(matches_im0, matches_im1, K, P1, P2, threshold=2.0, max_iter=100):
    """
    使用 RANSAC 进行鲁棒三角测量
    Args:
        matches_im0, matches_im1: 匹配的 2D 点对 (N, 2)
        K: 内参矩阵 (3, 3)
        P1, P2: 投影矩阵 (3, 4)
        threshold: 重投影误差阈值
        max_iter: 最大迭代次数
    Returns:
        inliers: 内点索引
        points_3d: 重建的 3D 点 (N, 3)
    """
    best_inliers = []
    best_points_3d = None

    for _ in range(max_iter):
        # 随机采样 8 个点对
        idx = np.random.choice(len(matches_im0), 8, replace=False)
        pts0_sample = matches_im0[idx]
        pts1_sample = matches_im1[idx]

        # 进行三角测量
        points_4d = cv2.triangulatePoints(P1, P2, pts0_sample.T, pts1_sample.T).T
        points_3d = points_4d[:, :3] / points_4d[:, 3][:, None]  # 转换为非齐次坐标

        # 计算重投影误差
        proj_pts0 = (P1 @ np.hstack((points_3d, np.ones((len(points_3d), 1)))).T).T
        proj_pts0[:, :2] /= proj_pts0[:, 2][:, None]

        proj_pts1 = (P2 @ np.hstack((points_3d, np.ones((len(points_3d), 1)))).T).T
        proj_pts1[:, :2] /= proj_pts1[:, 2][:, None]

        error0 = np.linalg.norm(proj_pts0[:, :2] - matches_im0, axis=1)
        error1 = np.linalg.norm(proj_pts1[:, :2] - matches_im1, axis=1)
        inliers = (error0 < threshold) & (error1 < threshold)

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_points_3d = points_3d

    return best_inliers, best_points_3d










def triangulate_points(matches_im0, matches_im1, K, P1, P2):
    """
    三角化匹配点以恢复3D坐标
    Args:
        matches_im0: 第一张图像中的匹配点 (N, 2)
        matches_im1: 第二张图像中的匹配点 (N, 2)
        K: 相机内参矩阵 (3, 3)
        P1: 第一张图像的投影矩阵 (3, 4)
        P2: 第二张图像的投影矩阵 (3, 4)
    Returns:
        points_3d: 重建的3D点，形状为 (N, 3)
    """
    num_points = matches_im0.shape[0]
    points_3d = []

    for i in range(num_points):
        x1, y1 = matches_im0[i]
        x2, y2 = matches_im1[i]

        # 构造三角化的线性方程
        A = np.zeros((4, 4))
        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]

        # 通过SVD求解方程
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        X /= X[3]  # 转为非齐次坐标
        points_3d.append(X[:3])

    return np.array(points_3d)

def generate_depth_map(points_3d, K, extrinsic, image_shape):
    """
    根据投影的 3D 点生成深度图
    Args:
        points_3d: 3D 点，形状为 (N, 3)
        K: 相机内参矩阵 (3, 3)
        extrinsic: 外参矩阵 (4, 4)，包括旋转和平移
        image_shape: 图像分辨率 (H, W)
    Returns:
        depth_map: 深度图，形状为 (H, W)
    """
    # 提取 R 和 T
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]

    # 初始化深度图为 NaN，表示无效深度
    depth_map = np.full(image_shape, np.nan, dtype=np.float32)

    # 投影点到图像平面
    for point in points_3d:
        # 将点从世界坐标系转换到相机坐标系
        point_camera = R @ point + T
        X, Y, Z = point_camera

        # 如果点在相机后方（Z <= 0），忽略
        if Z <= 0:
            continue

        # 投影到像素坐标系
        u = int(round(K[0, 0] * X / Z + K[0, 2]))
        v = int(round(K[1, 1] * Y / Z + K[1, 2]))

        # 检查像素是否在图像范围内
        if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
            # 如果该像素没有深度值，或者新的深度更小，则更新深度值
            if np.isnan(depth_map[v, u]) or Z < depth_map[v, u]:
                depth_map[v, u] = Z

    return depth_map

def visualize_points_3d(points_3d):
    """
    使用 Matplotlib 可视化 3D 点云
    Args:
        points_3d: 3D 点，形状为 (N, 3)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取 X, Y, Z 坐标
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    # 绘制散点图
    ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points Visualization')
    plt.show()

def restore_matches_to_original(matches_scaled, scaled_shape, original_shape):
    """
    将在缩放图片上匹配的点恢复到原始图片上的坐标。
    Args:
        matches_scaled: 缩放图片上的匹配点，形状为 (N, 2)
        scaled_shape: 缩放图片的尺寸 (H_scaled, W_scaled)
        original_shape: 原始图片的尺寸 (H_original, W_original)
    Returns:
        matches_original: 恢复到原始图片的匹配点，形状为 (N, 2)
    """
    H_scaled, W_scaled = scaled_shape
    H_original, W_original = original_shape

    # 计算缩放比例
    scale_x = W_original / W_scaled
    scale_y = H_original / H_scaled

    # 恢复到原始坐标
    matches_original = matches_scaled * np.array([scale_x, scale_y])
    return matches_original

def project_points_to_depth_map_with_extrinsic(points_3d, K, extrinsic, image_shape):
    """
    将 3D 点投影到深度图（输入为外参矩阵 Extrinsic）。
    Args:
        points_3d: 3D 点，形状为 (N, 3)
        K: 内参矩阵 (3, 3)
        extrinsic: 外参矩阵 (4, 4)
        image_shape: 图像形状 (H, W)
    Returns:
        depth_map: 深度图，形状为 (H, W)
    """
    # 提取旋转矩阵 R 和平移向量 T
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]

    # 初始化深度图
    depth_map = np.full(image_shape, np.nan)  # 使用 NaN 表示无效深度

    # 将 3D 点从世界坐标转换到相机坐标系
    points_camera = (R @ points_3d.T).T + T  # (N, 3)
    
    # 过滤掉相机视锥外的点（Z <= 0）
    valid_points = points_camera[:, 2] > 0
    points_camera = points_camera[valid_points]

    # 将相机坐标系中的点投影到图像平面
    points_homogeneous = points_camera / points_camera[:, 2][:, None]  # 归一化 (N, 3)
    pixels = (K @ points_homogeneous.T).T  # 投影到像素坐标系
    pixels[:, 0] /= pixels[:, 2]  # u = u' / z
    pixels[:, 1] /= pixels[:, 2]  # v = v' / z
    print("3D points in camera coordinates:", points_camera[:5])
    print("Pixel coordinates:", pixels[:5])    
    # 填充深度图
    for i in range(pixels.shape[0]):
        u, v = int(round(pixels[i, 0])), int(round(pixels[i, 1]))
        depth = points_camera[i, 2]
        #print(u,v,depth)
        if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:  # 检查边界
            if np.isnan(depth_map[v, u]) or depth < depth_map[v, u]:  # 更新最小深度
                depth_map[v, u] = depth
    #depth_map = np.nan_to_num(depth_map, nan=0.0) 
    return depth_map



def visualize_points_and_camera(points_3d, cameras):
    """
    可视化点云和相机位置
    Args:
        points_3d: 3D 点云，形状为 (N, 3)
        cameras: 相机外参列表，每个元素为 (R, T)，其中
                 R 是旋转矩阵 (3, 3)，T 是平移向量 (3,)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, label='3D Points')

    # 绘制相机位置和方向
    for i, (R, T) in enumerate(cameras):
        camera_position = -R.T @ T  # 计算相机位置
        ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='red', s=50, label=f'Camera {i+1}')
        
        # 绘制相机的局部坐标系
        scale = 0.1
        x_axis = camera_position + scale * R.T @ np.array([1, 0, 0])
        y_axis = camera_position + scale * R.T @ np.array([0, 1, 0])
        z_axis = camera_position + scale * R.T @ np.array([0, 0, 1])

        ax.plot([camera_position[0], x_axis[0]], [camera_position[1], x_axis[1]], [camera_position[2], x_axis[2]], 'r-')
        ax.plot([camera_position[0], y_axis[0]], [camera_position[1], y_axis[1]], [camera_position[2], y_axis[2]], 'g-')
        ax.plot([camera_position[0], z_axis[0]], [camera_position[1], z_axis[1]], [camera_position[2], z_axis[2]], 'b-')

    # 设置图例和标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Points and Camera Positions")
    plt.show()

def fit_linear_relation(dense_depth, sparse_depth, valid_mask):
    """
    根据稠密深度图和稀疏深度点，拟合线性关系 y = kx + b
    Args:
        dense_depth: 稠密深度图 (H, W)
        sparse_depth: 稀疏深度点，形状与 dense_depth 相同，但仅部分点有值 (H, W)
        valid_mask: 稀疏点的有效掩码，形状与 dense_depth 相同，为 True 表示该点有效
    Returns:
        k, b: 拟合的线性关系参数
    """
    # 提取有效点
    x = dense_depth[valid_mask]  # 稠密深度的有效点
    y = sparse_depth[valid_mask]  # 稀疏深度的有效点

    # 构造线性方程
    A = np.vstack([x, np.ones_like(x)]).T  # 系数矩阵
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]  # 最小二乘解
    return k, b




if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    sfmd_path = "/home/lingxiang/SLAM/mast3r/5images"
    sfmdfile =  sorted(os.path.join(sfmd_path, f)  for f in os.listdir(sfmd_path) if f.startswith("frame") and f.endswith(".jpg"))
    #images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)

    images = load_images([sfmdfile[0],sfmdfile[4]], size=512)
    image_shape  = (680, 1200) 
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    matches_im0_original = restore_matches_to_original(matches_im0,(288,512) ,(680,1200) )
    matches_im1_original = restore_matches_to_original(matches_im1,(288,512) , (680,1200))
    print(matches_im0_original)
    fx, fy = 600.0, 600.0
    cx, cy = 599.5, 339.5
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])


    extrinsics_list = np.load("inverted_T_list.npy", allow_pickle=True)

#     img1_extrinsics = np.array([
#     [9.062491181555123454e-01, -2.954311239679592860e-01, 3.023788796086531172e-01, -3.569159214564542326e-01],
#     [-4.227440547687880690e-01, -6.333245673155291078e-01, 6.482186796076164770e-01, -6.602722315763628336e-01],
#     [8.759610522377010340e-17, -7.152764804085384176e-01, -6.988415818870352680e-01, 8.192365926179191460e-01],
#     [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
# ])
    img1_extrinsics = extrinsics_list[0]
    img2_extrinsics = extrinsics_list[4]
   
#     img2_extrinsics = np.array([
#     [9.244909309264357500e-01, -2.501320916852852183e-01, 2.876637887255122372e-01, -2.955535868115701237e-01],
#     [-3.812040380619964863e-01, -6.066170008909974598e-01, 6.976383702142818777e-01, -6.350980390528973007e-01],
#     [9.241416745853188311e-17, -7.546189441957813493e-01, -6.561632792688449900e-01, 7.815705651857323133e-01],
#     [0.0, 0.0, 0.0, 1.0]
# ])



    R1, T1 = parse_extrinsics(img1_extrinsics)
    R2, T2 = parse_extrinsics(img2_extrinsics)

    # 构造投影矩阵
    P1 = K @ np.hstack((R1, T1.reshape(-1, 1)))
    P2 = K @ np.hstack((R2, T2.reshape(-1, 1)))

    # 进行三角化
    points_3d = triangulate_points(matches_im0_original, matches_im1_original, K, P1, P2)

    #visualize_points_3d(points_3d)
    cameras = [(R1, T1), (R2, T2)]

    # 可视化
    # visualize_points_and_camera(points_3d, cameras)

    depth_map = project_points_to_depth_map_with_extrinsic(points_3d, K, img1_extrinsics, image_shape)
    
    # print(depth_map)
    # import matplotlib.pyplot as plt
    # plt.imshow(depth_map, cmap='jet')
    # plt.colorbar(label='Depth')
    # plt.title("Depth Map")
    # plt.show()


    # depth_path = ("/home/lingxiang/datasets/replica/office0/results/depth000000.png")
    # depth_scale = 6553.5
    # depth = np.array(Image.open(depth_path)) / depth_scale

    es_depth = np.load("/home/lingxiang/datasets/replica/office0/estimation/depth/depth_frame000000.npy")

    k, b = fit_linear_relation(es_depth, depth_map, ~np.isnan(depth_map))

    final_depth = k*es_depth + b
    
    import matplotlib.pyplot as plt
    plt.imshow(final_depth, cmap='jet')
    plt.colorbar(label='Depth')
    plt.title("Depth Map")
    plt.show()


    depth_path = ("/home/lingxiang/datasets/replica/office0/results/depth000000.png")
    depth_scale = 6553.5
    real_depth = np.array(Image.open(depth_path)) / depth_scale
 
    if isinstance(real_depth, torch.Tensor):
        # PyTorch 张量
        depth_pixel_mask = (real_depth > 0.01).view(*real_depth.shape)
    elif isinstance(real_depth, np.ndarray):
        # NumPy 数组
        depth_pixel_mask = (real_depth > 0.01).reshape(*real_depth.shape)



    update_metrics(torch.from_numpy(final_depth)
               ,torch.from_numpy(real_depth),torch.from_numpy(depth_pixel_mask))
    # # visualize a few matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl
    num_matches = matches_im0.shape[0]
    n_viz = 20
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)