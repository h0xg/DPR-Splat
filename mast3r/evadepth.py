import torch
from PIL import Image
import numpy as np
import cv2
from depth_utils import update_metrics
import matplotlib
import matplotlib.pyplot as plt
import os
# 加载文件

def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and(torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat<=0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss


def fit_linear_transformation(es_depth, gt_depth):
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



flow_depth = 1/np.load("/home/lingxiang/SLAM/mast3r/depthsplat/depthsplat2.npy")
flow_depth = cv2.resize(flow_depth, (512, 288), interpolation=cv2.INTER_LINEAR)  # 宽在前，高在后
# # 打印内容

depth_path = ("/home/lingxiang/datasets/replica/office0/results/depth000000.png")
depth_scale = 6553.5
depth = np.array(Image.open(depth_path)) / depth_scale
real_depth = cv2.resize(depth, (512, 288), interpolation=cv2.INTER_LINEAR)  # 宽在前，高在后
print(flow_depth)

# es_depth = np.load("/home/lingxiang/datasets/replica/office0/estimation/depth/depth_frame000000.npy")
# es_depth = cv2.resize(es_depth, (512, 288), interpolation=cv2.INTER_LINEAR)  # 宽在前，高在后



# print(es_depth.shape)
# print(flow_depth.shape)
# transformed_depth, k, b = fit_linear_transformation(es_depth, flow_depth)

# print(k,b)

# print(transformed_depth)

# # 确保数据类型一致
if isinstance(real_depth, torch.Tensor):
    # PyTorch 张量
    depth_pixel_mask = (real_depth > 0.01).view(*real_depth.shape)
elif isinstance(real_depth, np.ndarray):
    # NumPy 数组
    depth_pixel_mask = (real_depth > 0.01).reshape(*real_depth.shape)


# print("compare depth transformed_depth and real")
# update_metrics(torch.from_numpy(transformed_depth)
#                ,torch.from_numpy(real_depth),torch.from_numpy(depth_pixel_mask))

# print("compare depth es_depth and real")
# update_metrics(torch.from_numpy(es_depth)
#                ,torch.from_numpy(real_depth),torch.from_numpy(depth_pixel_mask))

print("compare depth flow_depth and real")
update_metrics(torch.from_numpy(flow_depth)
               ,torch.from_numpy(real_depth),torch.from_numpy(depth_pixel_mask))
               
depth_visualization(flow_depth,"./depthsplat/depthsplat2.jpg")
# depth_visualization(es_depth,"./output5images/es_depth.jpg")
# depth_visualization(transformed_depth,"./output5images/transformed_depth.jpg")



# depthweight = np.load("/home/lingxiang/SLAM/mast3r/output5images/conf_0000.npy")
# print(depthweight)
# depthweight = cv2.resize(depthweight, (512, 288), interpolation=cv2.INTER_LINEAR)  # 宽在前，高在后

# depthweight_normalized = (depthweight - depthweight.min()) / (depthweight.max() - depthweight.min())

# plt.imshow(depthweight_normalized, cmap='gray')  # 使用灰度色图
# plt.colorbar(label='Depth Weight (Normalized)')  # 添加颜色条
# plt.title('Depth Weight (Normalized)')
# plt.show()

# depthmap, depthloss = optimize_depth(source=es_depth, target=flow_depth, mask=flow_depth>0.001, depth_weight=depthweight_normalized)


# print("compare depth depthmap and real")
# update_metrics(torch.from_numpy(depthmap)
#                ,torch.from_numpy(real_depth),torch.from_numpy(depth_pixel_mask))

# depth_visualization(transformed_depth,"./output/depthmap.jpg")

depth_difference_visualization_with_legend(flow_depth,real_depth,"./output/depthsplat/depthsplatdiff.jpg")

# depth_difference_visualization_with_legend(depthmap,real_depth,"./output/depdiff/depthmap.jpg")


# es_path = "/home/lingxiang/datasets/replica/office0/estimation/depth"
# es_files = sorted(
#     os.path.join(es_path, f) for f in os.listdir(es_path) if f.startswith("depth_frame") and f.endswith(".npy")
# )

# sfmd_path = "/home/lingxiang/SLAM/mast3r/output5images"
# sfmdfile =  sorted(os.path.join(sfmd_path, f)  for f in os.listdir(sfmd_path) if f.startswith("depth") and f.endswith(".npy"))



# gt_path = "/home/lingxiang/datasets/replica/office0/results"
# depth_scale = 6553.5
# gtfile =  sorted(os.path.join(gt_path, f)  for f in os.listdir(gt_path) if f.startswith("depth") and f.endswith(".png"))


# weight_path = "/home/lingxiang/SLAM/mast3r/output5images"
# weightfile =  sorted(os.path.join(weight_path, f)  for f in os.listdir(weight_path) if f.startswith("conf") and f.endswith(".npy"))

# print(sfmdfile)

# for i in range(5):
#     es_depth = np.load(es_files[i])
#     es_depth = cv2.resize(es_depth, (512, 288), interpolation=cv2.INTER_LINEAR) 
#     sfm_depth = np.load(sfmdfile[i])
#     weightile = np.load(weightfile[i])
#     gt_depth = np.array(Image.open(gtfile[i])) / depth_scale
#     gt_depth = cv2.resize(gt_depth, (512, 288), interpolation=cv2.INTER_LINEAR) 
#     depthmap, depthloss = optimize_depth(source=es_depth, target=sfm_depth, mask=sfm_depth>0.001, depth_weight=weightile)


#     if isinstance(gt_depth, torch.Tensor):
#         # PyTorch 张量
#         depth_pixel_mask = (gt_depth > 0.01).view(*gt_depth.shape)
#     elif isinstance(gt_depth, np.ndarray):
#         # NumPy 数组
#         depth_pixel_mask = (gt_depth > 0.01).reshape(*gt_depth.shape)


#     update_metrics(torch.from_numpy(depthmap)
#                ,torch.from_numpy(gt_depth),torch.from_numpy(depth_pixel_mask))

#     file_name = f"./depthmap/depthmap_{i}.npy"  # 使用 f-string 格式化
#     np.save(file_name, depthmap)  # 保存文件
#     img_name = f"./depthmap/depthmap_{i}.jpg"  # 使用 f-string 格式化

#     depth_visualization(depthmap,img_name)
