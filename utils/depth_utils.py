import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2
import matplotlib
from scipy.ndimage import zoom
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def update_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,):
    """
    Update metrics on CPU
    """

    assert pred.shape == target.shape

    if len(pred.shape) == 3:
        pred = pred[:, None, :, :]
        target = target[:, None, :, :]
        mask = mask[:, None, :, :]
    elif len(pred.shape) == 2:
        pred = pred[None, None, :, :]
        target = target[None, None, :, :]
        mask = mask[None, None, :, :]

    # Absolute relative error
    abs_rel_sum, valid_pics = get_absrel_err(pred, target, mask)
    abs_rel_sum = abs_rel_sum.cpu().numpy()  # 加上 .cpu() 进行修正
    valid_pics = valid_pics.cpu().numpy()  # 加上 .cpu() 进行修正

    # Squared relative error
    sqrel_sum, _ = get_sqrel_err(pred, target, mask)
    sqrel_sum = sqrel_sum.cpu().numpy()  # 加上 .cpu()

    # Root mean squared error
    rmse_sum, _ = get_rmse_err(pred, target, mask)
    rmse_sum = rmse_sum.cpu().numpy()  # 加上 .cpu()
    
    # Log root mean squared error
    log_rmse_sum, _ = get_rmse_log_err(pred, target, mask)
    log_rmse_sum = log_rmse_sum.cpu().numpy()  # 加上 .cpu()
    
    # Log10 error
    log10_sum, _ = get_log10_err(pred, target, mask)
    log10_sum = log10_sum.cpu().numpy()  # 加上 .cpu()

    # Scale-invariant log RMSE (silog)
    silog_sum, _ = get_silog_err(pred, target, mask)
    silog_sum = silog_sum.cpu().numpy()  # 加上 .cpu()

    # Ratio error (delta)
    delta1_sum, delta2_sum, delta3_sum, _ = get_ratio_err(pred, target, mask)
    delta1_sum = delta1_sum.cpu().numpy()  # 加上 .cpu()
    delta2_sum = delta2_sum.cpu().numpy()  # 加上 .cpu()
    delta3_sum = delta3_sum.cpu().numpy()  # 加上 .cpu()

    print(f"Absolute Relative Error (AbsRel): {abs_rel_sum:.4f} ↓")
    print(f"Squared Relative Error (SqRel): {sqrel_sum:.4f} ↓")
    print(f"Root Mean Squared Error (RMSE): {rmse_sum:.4f} ↓")
    print(f"Log RMSE: {log_rmse_sum:.4f} ↓")
    print(f"Log10 Error: {log10_sum:.4f} ↓")
    print(f"Scale-Invariant Log RMSE (SiLog): {silog_sum:.4f} ↓")
    print(f"Delta1: {delta1_sum:.4f} ↑")
    print(f"Delta2: {delta2_sum:.4f} ↑")
    print(f"Delta3: {delta3_sum:.4f} ↑")

def get_absrel_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes absolute relative error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    # Mean Absolute Relative Error
    rel = torch.abs(t_m - p_m) / (t_m + 1e-10) # compute errors
    abs_rel_sum = torch.sum(rel.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    abs_err = abs_rel_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(abs_err), valid_pics

def get_sqrel_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes squared relative error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    # squared Relative Error
    sq_rel = torch.abs(t_m - p_m) ** 2 / (t_m + 1e-10) # compute errors
    sq_rel_sum = torch.sum(sq_rel.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    sqrel_err = sq_rel_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(sqrel_err), valid_pics

def get_log10_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log10 error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    log10_diff = torch.abs(diff_log)
    log10_sum = torch.sum(log10_diff.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    log10_err = log10_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)
    return torch.sum(log10_err), valid_pics

def get_rmse_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    square = (t_m - p_m) ** 2
    rmse_sum = torch.sum(square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    rmse = torch.sqrt(rmse_sum / (num + 1e-10))
    valid_pics = torch.sum(num > 0)
    return torch.sum(rmse), valid_pics

def get_rmse_log_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    square = diff_log ** 2
    rmse_log_sum = torch.sum(square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    rmse_log = torch.sqrt(rmse_log_sum / (num + 1e-10))
    valid_pics = torch.sum(num > 0)
    return torch.sum(rmse_log), valid_pics

def get_silog_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes log rmse error.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """

    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred * mask

    diff_log = (torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask
    diff_log_sum = torch.sum(diff_log.reshape((b, c, -1)), dim=2) # [b, c]
    diff_log_square = diff_log ** 2
    diff_log_square_sum = torch.sum(diff_log_square.reshape((b, c, -1)), dim=2) # [b, c]
    num = torch.sum(mask.reshape((b, c, -1)), dim=2) # [b, c]
    silog = torch.sqrt(diff_log_square_sum / (num + 1e-10) - (diff_log_sum / (num + 1e-10)) ** 2)
    valid_pics = torch.sum(num > 0)
    return torch.sum(silog), valid_pics

def get_ratio_err(pred: torch.tensor,
                    target: torch.tensor,
                    mask: torch.tensor,
                    ):
    """
    Computes the percentage of pixels for which the ratio of the two depth maps is less than a given threshold.
    Tasks preprocessed depths (no nans, infs and non-positive values).
    pred, target, and mask should be in the shape of [b, c, h, w]
    """
    assert len(pred.shape) == 4, len(target.shape) == 4
    b, c, h, w = pred.shape
    mask = mask.to(torch.float)
    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)
    pred_gt = p_m / (t_m + 1e-10)
    gt_pred = gt_pred.reshape((b, c, -1))
    pred_gt = pred_gt.reshape((b, c, -1))
    gt_pred_gt = torch.cat((gt_pred, pred_gt), axis=1)
    ratio_max = torch.amax(gt_pred_gt, axis=1)

    delta_1_sum = torch.sum((ratio_max < 1.25), dim=1) # [b, ]
    delta_2_sum = torch.sum((ratio_max < 1.25 ** 2), dim=1) # [b, ]
    delta_3_sum = torch.sum((ratio_max < 1.25 ** 3), dim=1) # [b, ]
    num = torch.sum(mask.reshape((b, -1)), dim=1) # [b, ]

    delta_1 = delta_1_sum / (num + 1e-10)
    delta_2 = delta_2_sum / (num + 1e-10)
    delta_3 = delta_3_sum / (num + 1e-10)
    valid_pics = torch.sum(num > 0)    

    return torch.sum(delta_1), torch.sum(delta_2), torch.sum(delta_3), valid_pics



def depth_visualization(depth, output_path=None, show=False):
    """
    Visualize a depth map using a colormap and optionally save or display the result.

    Args:
        depth (numpy.ndarray): Input depth map (2D array).
        output_path (str, optional): Path to save the visualized image. If None, image is not saved.
        show (bool, optional): If True, displays the visualized image. Default is False.

    Returns:
        vis_depth (numpy.ndarray): Visualized depth map as a 3-channel RGB image.
    """
    # Ensure depth has valid values
    if depth.max() == depth.min():
        raise ValueError("Depth map has no variation (min == max).")

    # Normalize depth map to [0, 255]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    # Apply colormap
    cmap = plt.cm.get_cmap('Spectral')
    vis_depth = cmap(depth_normalized / 255.0)[:, :, :3]  # Remove alpha channel
    vis_depth = (vis_depth * 255).astype(np.uint8)  # Convert to 8-bit RGB

    # Convert RGB to BGR for OpenCV compatibility
    vis_depth = vis_depth[:, :, ::-1]

    # Save the visualized image if output_path is provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_depth)
        print(f"Visualized depth map saved to: {output_path}")

    # Show the visualized image if `show` is True
    if show:
        plt.imshow(vis_depth[:, :, ::-1])  # Convert BGR back to RGB for matplotlib
        plt.axis("off")
        plt.title("Depth Visualization")
        plt.show()

    return vis_depth


def reshape_and_save_depthmaps(depthmap_list, fixed_h, fixed_w,h, w, output_dir,vis_dir):
    """
    Reshape each depthmap in the list to (h, w) and save it to the output directory.

    Args:
        depthmap_list (list of numpy.ndarray): List of 1D depthmaps.
        h (int): Height of the target RGB image.
        w (int): Width of the target RGB image.
        output_dir (str): Directory to save the reshaped depthmaps.

    Returns:
        None
    """
    # Ensure output directory exists
    # Fixed intermediate size


    # Colormap for visualization
    cmap = plt.colormaps.get_cmap('Spectral_r')

    for i, depthmap in enumerate(depthmap_list):
        # Reshape to (fixed_h, fixed_w)
        reshaped_to_fixed = depthmap.reshape(fixed_h, fixed_w)

        # Reshape back to (h, w)
        scale_h = h / fixed_h
        scale_w = w / fixed_w
        reshaped_to_original = zoom(reshaped_to_fixed, (scale_h, scale_w), order=1)  # order=1 for bilinear interpolation
        # Save the reshaped depthmap as .npy
        depthmap_path = os.path.join(output_dir, f"sfmdepth_{i:04d}.npy")
        np.save(depthmap_path, reshaped_to_original)
        print(f"Saved reshaped depthmap {i} to {depthmap_path}")

        # Normalize depthmap for visualization
        normalized_depthmap = (reshaped_to_original - reshaped_to_original.min()) / (
            reshaped_to_original.max() - reshaped_to_original.min()
        )

        # Apply colormap
        color_mapped_image = cmap(normalized_depthmap)

        # Save the visualization as .png
        visualization_path = os.path.join(vis_dir, f"sfmdepth_{i:04d}.png")
        plt.imsave(visualization_path, color_mapped_image)
        print(f"Saved visualized depthmap {i} to {visualization_path}")

def depth_difference_visualization_with_legend(estimated, ground_truth, output_path=None, show=False):
    """
    Visualize the difference between the estimated depth map and ground truth depth map,
    and add a legend to indicate the error scale.

    Args:
        estimated (numpy.ndarray): Estimated depth map (2D array).
        ground_truth (numpy.ndarray): Ground truth depth map (2D array).
        output_path (str, optional): Path to save the visualized difference image. If None, image is not saved.
        show (bool, optional): If True, displays the visualized difference image. Default is False.

    Returns:
        None
    """
    # Ensure the estimated and ground_truth depth maps have the same shape
    if estimated.shape != ground_truth.shape:
        raise ValueError("Estimated and ground truth depth maps must have the same shape.")

    # Compute the depth difference
    depth_diff = estimated - ground_truth

    # Normalize the difference to [-1, 1] for visualization
    max_abs_diff = np.max(np.abs(depth_diff))
    if max_abs_diff > 0:
        depth_diff_normalized = depth_diff / max_abs_diff
    else:
        depth_diff_normalized = depth_diff  # If all differences are zero

    # Map normalized difference to colormap
    cmap = plt.cm.get_cmap('coolwarm')  # Use coolwarm colormap for positive/negative differences
    diff_vis = cmap((depth_diff_normalized + 1) / 2)  # Map [-1, 1] to [0, 1]
    diff_vis = (diff_vis[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and convert to 8-bit RGB

    # Plotting with legend
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(diff_vis)
    ax.axis("off")
    ax.set_title("Depth Difference Visualization", fontsize=14)

    # Add a colorbar as a legend
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff)),
                        ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label("Error (Estimated - Ground Truth)", fontsize=12)

    # Save the visualized difference image if output_path is provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Visualized depth difference with legend saved to: {output_path}")

    # Show the visualized difference image if `show` is True
    if show:
        plt.show()

    plt.close(fig)


def fit_ransac_transformation(es_depth, gt_depth, mask=None, max_trials=100, residual_threshold=1.0):
    """
    使用 RANSAC 方法通过线性变换将 es_depth 映射到 gt_depth，使误差最小。

    Args:
        es_depth (numpy.ndarray): 深度估计网络输出的深度图，形状为 (H, W)。
        gt_depth (numpy.ndarray): 真实深度图，形状为 (H, W)。
        mask (numpy.ndarray, optional): 有效像素的布尔掩码，形状为 (H, W)。默认为 None，表示使用所有像素。
        max_trials (int, optional): RANSAC 最大迭代次数，默认为 100。
        residual_threshold (float, optional): 判定内点的残差阈值，默认为 1.0。

    Returns:
        transformed_depth (numpy.ndarray): 经过线性变换后的深度图。
        k (float): 线性变换的斜率。
        b (float): 线性变换的截距。
        inlier_mask (numpy.ndarray): 表示内点的布尔掩码。
    """
    import numpy as np
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression
    import cv2

    # 检查输入形状是否一致
    if es_depth.shape != gt_depth.shape:
        print("Resizing gt_depth to match es_depth")
        gt_depth = cv2.resize(gt_depth, (es_depth.shape[1], es_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 展平深度图
    es_depth_flat = es_depth.flatten()
    gt_depth_flat = gt_depth.flatten()

    # 如果提供了掩码，应用掩码
    if mask is not None:
        mask_flat = mask.flatten()
        es_depth_flat = es_depth_flat[mask_flat]
        gt_depth_flat = gt_depth_flat[mask_flat]

    # 确保长度匹配
    assert es_depth_flat.shape[0] == gt_depth_flat.shape[0], "Flattened shapes are not matching!"

    # 使用 RANSAC 进行线性拟合
    ransac = RANSACRegressor(LinearRegression(), max_trials=max_trials, residual_threshold=residual_threshold)
    ransac.fit(es_depth_flat.reshape(-1, 1), gt_depth_flat)

    # 获取拟合参数
    k = ransac.estimator_.coef_[0]
    b = ransac.estimator_.intercept_

    # 应用线性变换
    transformed_depth = k * es_depth + b

    # 获取内点掩码
    inlier_mask = ransac.inlier_mask_.reshape(es_depth.shape)

    return transformed_depth

def fit_linear_transformation(es_depth, gt_depth,mask = None,):
    """
    通过线性变换 kx + b 将 es_depth 映射到 gt_depth，使得误差最小。
    
    Args:
        es_depth (numpy.ndarray): 深度估计网络输出的深度图，形状为 (H, W)。
        gt_depth (numpy.ndarray): 真实深度图，形状为 (H, W)。

    Returns:
        transformed_depth (numpy.ndarray): 经过线性变换后的深度图。
        k (float): 线性变换的斜率。
        b (float): 线性变换的截距。
    """
    # 检查输入形状是否一致
    if es_depth.shape != gt_depth.shape:
        print("Resizing gt_depth to match es_depth")
        gt_depth = cv2.resize(gt_depth, (es_depth.shape[1], es_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 展平深度图
    es_depth_flat = es_depth.flatten()
    gt_depth_flat = gt_depth.flatten()

    # 确保长度匹配
    assert es_depth_flat.shape[0] == gt_depth_flat.shape[0], "Flattened shapes are not matching!"

    # 创建线性回归矩阵
    X = np.vstack([es_depth_flat, np.ones_like(es_depth_flat)]).T  # [N, 2]
    y = gt_depth_flat  # [N]

    # 使用最小二乘法求解 k 和 b
    k, b = np.linalg.lstsq(X, y, rcond=None)[0]  # Least squares solution

    # 应用线性变换
    transformed_depth = k * es_depth + b

    return transformed_depth, k, b



def save_conf(conf_list, output_dir, fixed_h, fixed_w,h,w):
    """
    Reshape each depthmap in the list to (h, w) and save it to the output directory.

    Args:
        depthmap_list (list of numpy.ndarray): List of 1D depthmaps.
        h (int): Height of the target RGB image.
        w (int): Width of the target RGB image.
        output_dir (str): Directory to save the reshaped depthmaps.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)



    for i, depthmap in enumerate(conf_list):
        # Reshape to (fixed_h, fixed_w)
        reshaped_to_fixed = depthmap.reshape(fixed_h, fixed_w)

        # Reshape back to (h, w)
        scale_h = h / fixed_h
        scale_w = w / fixed_w
        reshaped_to_original = zoom(reshaped_to_fixed, (scale_h, scale_w), order=1)  # order=1 for bilinear interpolation
        # Save the reshaped depthmap as .npy
        output_path = os.path.join(output_dir, f"conf_{i:04d}.npy")
        np.save(output_path, reshaped_to_original)


def fit_quadratic_transformation(es_depth, gt_depth, mask=None, weight=None):
    """
    Map es_depth to gt_depth using a quadratic transformation ax^2 + bx + c
    to minimize the error.

    Args:
        es_depth (numpy.ndarray): Depth map from the depth estimation network, shape (H, W).
        gt_depth (numpy.ndarray): Ground truth depth map, shape (H, W).
        mask (numpy.ndarray): Boolean mask, shape (H, W). True indicates valid points, False for ignored points.
        weight (numpy.ndarray): Weight map, shape (H, W). Defines importance of each pixel during fitting.

    Returns:
        transformed_depth (numpy.ndarray): Depth map after applying the quadratic transformation.
        a (float): Quadratic coefficient.
        b (float): Linear coefficient.
        c (float): Constant term.
    """
    # Check if input shapes match
    if es_depth.shape != gt_depth.shape:
        print("Resizing gt_depth to match es_depth")
        gt_depth = cv2.resize(gt_depth, (es_depth.shape[1], es_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply mask
    if mask is None:
        mask = np.ones_like(es_depth, dtype=bool)  # Default to all points being valid

    # If no weight is provided, default to all ones (no weighting)
    if weight is None:
        weight = np.ones_like(es_depth)

    # Apply mask
    es_depth_flat = es_depth[mask].flatten()      # Select only points where mask is True
    gt_depth_flat = gt_depth[mask].flatten()
    weight_flat = weight[mask].flatten()

    # Ensure the lengths match
    assert es_depth_flat.shape[0] == gt_depth_flat.shape[0] == weight_flat.shape[0], "Flattened shapes do not match!"

    # Create the quadratic regression matrix
    X = np.vstack([es_depth_flat**2, es_depth_flat, np.ones_like(es_depth_flat)]).T  # [N, 3]
    y = gt_depth_flat  # [N]

    # Apply weighted least squares
    W_diag = weight_flat  # Weight vector
    XTWX = X.T @ (X * W_diag[:, np.newaxis])  # [3, 3]
    XTWY = X.T @ (y * W_diag)                # [3]

    # Solve for a, b, c
    params = np.linalg.solve(XTWX, XTWY)  # Solve [a, b, c]
    a, b, c = params

    # Apply the quadratic transformation
    transformed_depth = a * es_depth**2 + b * es_depth + c

    return transformed_depth



def align_depth_regions(reg_depth, sfm_depth, grid_size=10, mask_threshold=0.01):
    """
    Divide reg_depth and sfm_depth into grid_size * grid_size regions,
    and scale align reg_depth in each region (y = ax).

    Args:
        reg_depth (numpy.ndarray): Smooth depth map from depth estimation, shape (H, W).
        sfm_depth (numpy.ndarray): Depth map from SFM, shape (H, W).
        grid_size (int): Number of divisions along each dimension.
        mask_threshold (float): Threshold to filter invalid pixels.

    Returns:
        aligned_depth (numpy.ndarray): Scale-aligned reg_depth.
    """
    # Ensure the input dimensions match
    assert reg_depth.shape == sfm_depth.shape, "reg_depth and sfm_depth must have the same shape"

    H, W = reg_depth.shape
    aligned_depth = np.zeros_like(reg_depth)

    # Size of each region
    region_h, region_w = H // grid_size, W // grid_size

    # Iterate over each region
    for i in range(grid_size):
        for j in range(grid_size):
            # Get the indices of the current region
            h_start, h_end = i * region_h, (i + 1) * region_h
            w_start, w_end = j * region_w, (j + 1) * region_w

            # Extract depth data for the current region
            reg_region = reg_depth[h_start:h_end, w_start:w_end]
            sfm_region = sfm_depth[h_start:h_end, w_start:w_end]

            # Construct the mask
            mask = (reg_region > mask_threshold) & (sfm_region > mask_threshold)

            # Skip the region if there are too few valid pixels
            if mask.sum() < 10:
                continue

            # Extract valid regions
            reg_valid = reg_region[mask]
            sfm_valid = sfm_region[mask]

            # Compute the scale factor a
            a = (sfm_valid * reg_valid).sum() / (reg_valid**2).sum()

            # Apply scale alignment to the current region
            aligned_region = a * reg_region

            # Place the aligned region back into the result
            aligned_depth[h_start:h_end, w_start:w_end] = aligned_region

    return aligned_depth
def fit_linear_transformation(es_depth, gt_depth, mask=None, weight=None):
    """
    通过线性变换 kx + b 将 es_depth 映射到 gt_depth，使得误差最小。

    Args:
        es_depth (numpy.ndarray): 深度估计网络输出的深度图，形状为 (H, W)。
        gt_depth (numpy.ndarray): 真实深度图，形状为 (H, W)。
        mask (numpy.ndarray, optional): 掩码，形状为 (H, W)。值为 True 的位置表示使用的像素。
        weight (numpy.ndarray, optional): 权重数组，形状为 (H, W)。对不同像素赋予不同权重。

    Returns:
        transformed_depth (numpy.ndarray): 经过线性变换后的深度图。
        k (float): 线性变换的斜率。
        b (float): 线性变换的截距。
    """

    # 检查输入形状是否一致
    if es_depth.shape != gt_depth.shape:
        print("Resizing gt_depth to match es_depth")
        gt_depth = cv2.resize(gt_depth, (es_depth.shape[1], es_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply mask
    es_depth_flat = es_depth[mask].flatten()      # Select only points where mask is True
    gt_depth_flat = gt_depth[mask].flatten()

    # Apply weight if provided
    if weight is not None:
        weight_flat = weight[mask].flatten()
    else:
        weight_flat = np.ones_like(es_depth_flat)

    # 确保长度匹配
    assert es_depth_flat.shape[0] == gt_depth_flat.shape[0], "Flattened shapes are not matching!"

    # 加权线性回归
    X = np.vstack([es_depth_flat, np.ones_like(es_depth_flat)]).T  # [N, 2]
    y = gt_depth_flat  # [N]

    # 对每一列应用权重
    X_weighted = X * weight_flat[:, None]  # 权重应用到每列
    y_weighted = y * weight_flat          # 权重应用到目标值

    # 使用最小二乘法求解 k 和 b
    k, b = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]  # Least squares solution

    # 应用线性变换
    transformed_depth = k * es_depth + b

    return transformed_depth


def fit_quadratic_transformation_shared(es_depths, gt_depths, masks=None):
    """
    对多个深度图使用共同的二次拟合系数 ax^2 + bx + c。

    Args:
        es_depths (list or numpy.ndarray): 估计深度图的列表或数组，每个形状为 (H, W)。
        gt_depths (list or numpy.ndarray): 真实深度图的列表或数组，每个形状为 (H, W)。
        masks (list or numpy.ndarray, optional): 掩码的列表或数组，每个形状为 (H, W)，
            值为 True 的位置表示使用的像素。

    Returns:
        transformed_depths (list): 经过二次变换的深度图列表，与输入深度图形状一致。
        coefficients (tuple): 共享的拟合系数 (a, b, c)。
    """

    # 检查输入数量是否一致
    if len(es_depths) != len(gt_depths):
        raise ValueError("The number of estimated and ground truth depth maps must match.")
    if masks is not None and len(masks) != len(es_depths):
        raise ValueError("The number of masks must match the number of depth maps.")

    # 合并所有深度图数据
    all_es_depth = []
    all_gt_depth = []

    for i in range(len(es_depths)):
        es_depth = es_depths[i]
        gt_depth = gt_depths[i]
        mask = masks[i] if masks is not None else None

        # 检查形状是否一致
        if es_depth.shape != gt_depth.shape:
            raise ValueError(f"Shape mismatch in depth maps at index {i}: {es_depth.shape} vs {gt_depth.shape}")
        if mask is not None and mask.shape != es_depth.shape:
            raise ValueError(f"Shape mismatch in mask at index {i}: {mask.shape} vs {es_depth.shape}")

        # 应用掩码
        if mask is not None:
            es_depth = es_depth[mask]
            gt_depth = gt_depth[mask]

        # 展平并收集数据
        all_es_depth.append(es_depth.flatten())
        all_gt_depth.append(gt_depth.flatten())

    # 合并为单个数组
    all_es_depth = np.concatenate(all_es_depth)
    all_gt_depth = np.concatenate(all_gt_depth)

    # 构建二次回归矩阵
    X = np.vstack([all_es_depth**2, all_es_depth, np.ones_like(all_es_depth)]).T
    y = all_gt_depth

    # 使用最小二乘法拟合
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]

    # 使用共享系数对每个深度图进行变换
    transformed_depths = []
    for es_depth in es_depths:
        transformed_depth = a * es_depth**2 + b * es_depth + c
        transformed_depths.append(transformed_depth)

    return transformed_depths


def mlp_fit_transform(all_es_depth, all_gt_depth, es_depths, hidden_layer_sizes=(64, 64), max_iter=500):
    # 数据归一化
    scaler = StandardScaler()
    all_es_depth = all_es_depth.reshape(-1, 1)
    all_es_depth_scaled = scaler.fit_transform(all_es_depth)

    # 构建 MLP 回归器
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
    mlp.fit(all_es_depth_scaled, all_gt_depth)

    # 对每个深度图进行变换
    transformed_depths = []
    for es_depth in es_depths:
        es_depth_scaled = scaler.transform(es_depth.reshape(-1, 1))
        transformed_depth = mlp.predict(es_depth_scaled).reshape(es_depth.shape)
        transformed_depths.append(transformed_depth)

    return transformed_depths, mlp

