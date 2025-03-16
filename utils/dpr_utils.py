import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .map_utils import np2torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """
    Creates a 1D Gaussian kernel.

    Args:
        window_size: The size of the window for the Gaussian kernel.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The 1D Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Variable:
    """
    Creates a 2D Gaussian window/kernel for SSIM computation.

    Args:
        window_size: The size of the window to be created.
        channel: The number of channels in the image.

    Returns:
        A 2D Gaussian window expanded to match the number of channels.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: Variable, window_size: int,
          channel: int, size_average: bool = True) -> torch.Tensor:
    """
    Internal function to compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window: The Gaussian window/kernel for SSIM computation.
        window_size: The size of the window to be used in SSIM computation.
        channel: The number of channels in the image.
        size_average: If True, averages the SSIM over all pixels.

    Returns:
        The computed SSIM value.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window,
                         padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window,
                         padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window,
                       padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        img1: The first image.
        img2: The second image.
        window_size: The size of the window to be used in SSIM computation. Defaults to 11.
        size_average: If True, averages the SSIM over all pixels. Defaults to True.

    Returns:
        The computed SSIM value.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)

def get_loss_tracking_es(config, image, depth, opacity, 
                         viewpoint,usemask = True, rgb=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if rgb:
        return get_sparse_tracking_loss(config, image_ab, depth, opacity, viewpoint,usemask = usemask)
    return get_loss_tracking_rgbd_es(config, image_ab, depth, opacity, viewpoint,usemask = usemask)


def get_loss_tracking_rgb(config, image, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    
    
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()
def get_loss_normal(depth_mean, viewpoint):
    prior_normal = np2torch(viewpoint.normal,device="cuda")
    prior_normal = prior_normal.reshape(3, *depth_mean.shape[-2:]).permute(1,2,0)
    prior_normal_normalized = torch.nn.functional.normalize(prior_normal, dim=-1)

    normal_mean, _ = depth_to_normal(viewpoint, depth_mean, world_frame=False)
    tv_loss_fn = TVLoss()
    tv_loss = tv_loss_fn(normal_mean.unsqueeze(0)) 
    tv_weight = 0.1  # TVLoss 权重
    normal_weight = 1.0  # Normal Error 权重
    normal_error = 1 - (prior_normal_normalized * normal_mean).sum(dim=-1)
    normal_error[prior_normal.norm(dim=-1) < 0.2] = 0
    combined_loss = normal_weight * normal_error + tv_weight * tv_loss

    return normal_error.mean()

def get_sparse_tracking_loss(config, image, depth, opacity, viewpoint,usemask = True):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=gt_image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    max_depth_value = gt_depth.max()  # 获取 gt_depth 的最大值
    sky_mask = (gt_depth < max_depth_value).view(*depth.shape)  # sky_mask 基于最大值

    if usemask:
        rgb_region = torch.from_numpy(viewpoint.mask).cuda()
        rgb_pixel_mask = rgb_pixel_mask*rgb_region
        l1 = opacity * torch.abs(sky_mask*image * rgb_pixel_mask -sky_mask* gt_image * rgb_pixel_mask)
        return l1.mean()
    else:
        l1 = opacity * torch.abs(sky_mask*image * rgb_pixel_mask*depth_pixel_mask -sky_mask*depth_pixel_mask* gt_image * rgb_pixel_mask)
        return l1.mean()          


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    #focus on near loss
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    l1_depth = depth_normalized.detech()*l1_depth
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_tracking_rgbd_es(
    config, image, depth, opacity, viewpoint, usemask = True,initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_sparse_tracking_loss(config, image, depth, opacity, viewpoint,usemask=usemask)
    depth_mask = depth_pixel_mask * opacity_mask
    max_depth_value = gt_depth.max()  # 获取 gt_depth 的最大值
    if max_depth_value>100:
        sky_mask = (gt_depth < max_depth_value).view(*depth.shape)  # sky_mask 基于最大值
    else:
        sky_mask = torch.ones_like(gt_depth, dtype=bool).cuda()    

    if usemask:
        rgb_region = torch.from_numpy(viewpoint.mask).cuda()
        l1_depth = torch.abs(depth * rgb_region*sky_mask - sky_mask*gt_depth * rgb_region)
        return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()

    
    l1_depth = torch.abs(depth * depth_mask*sky_mask - sky_mask*gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)

def get_loss_mapping_es(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd_es(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

# def get_loss_submapping_rgbd(config, image, depth, viewpoint,opacity, initialization=True):
    
#     if initialization:
#         image_ab = image
#     else:
#         image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
#     alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
#     rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

#     gt_image = viewpoint.original_image.cuda()
#     gt_depth = torch.from_numpy(viewpoint.es_depth).to(
#         dtype=torch.float32, device=image_ab.device
#     )[None]
#     rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
#     depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
#     l1_rgb = torch.abs(image_ab * rgb_pixel_mask*depth_pixel_mask -depth_pixel_mask* gt_image * rgb_pixel_mask)
#     l1_rgb = l1_rgb.mean()*0.8+0.2*(1.0 - ssim(image_ab, gt_image))
#     l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

#     return alpha *l1_rgb.mean()+(1 - alpha) *l1_depth.mean()


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def get_loss_mapping_rgbd_es(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()


def calculate_quaternion_difference(rotation_matrix,rotation_matrix_gt):
    if not isinstance(rotation_matrix, np.ndarray):  
        rotation_matrix = rotation_matrix.detach().cpu().numpy() 

    if not isinstance(rotation_matrix_gt, np.ndarray):  
        rotation_matrix_gt = rotation_matrix_gt.detach().cpu().numpy() 


    quat = R.from_matrix(rotation_matrix).as_quat()    
    quat_gt = R.from_matrix(rotation_matrix_gt).as_quat()
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    quat_gt = quat_gt / np.linalg.norm(quat_gt, axis=-1, keepdims=True)
    # Calculate the dot product between quaternions
    dot_product = np.einsum("...i,...i->...", quat, quat_gt)  # (...,)

    # Clamp the dot product to the valid range for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # Calculate the angular difference (in radians)
    angular_difference_rad = 2 * np.arccos(np.abs(dot_product))  # Use abs to account for quaternion symmetry

    # Convert the angular difference to degrees
    angular_difference_deg = np.degrees(angular_difference_rad)

    # Convert back to PyTorch tensor
    return angular_difference_deg

def calculate_translation_difference(translation, translation_gt):
    """
    Calculates the Euclidean distance between two translation vectors.

    Args:
        translation (torch.Tensor): The estimated translation vector, shape (..., 3).
        translation_gt (torch.Tensor): The ground-truth translation vector, shape (..., 3).

    Returns:
        torch.Tensor: The Euclidean distance between the translations, shape (...).
    """
    # Ensure input tensors have the correct shape
    if len(translation.shape) == 2:
        translation = translation[0]



    if not isinstance(translation, np.ndarray):  
        translation = translation.detach().cpu().numpy() 

    if not isinstance(translation_gt, np.ndarray):  
        translation_gt = translation_gt.detach().cpu().numpy() 


    # Compute the difference between translations
    diff = translation - translation_gt

    # Compute the Euclidean norm (L2 distance)
    distance = np.linalg.norm(diff, axis=-1)

    return distance


def depths_to_points(view, depthmap, world_frame):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.0]]).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    if world_frame:
        c2w = (view.world_view_transform.T).inverse()
        rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
    else:
        rays_d = points @ intrins.inverse().T
        points = depthmap.reshape(-1, 1) * rays_d
    return points


def depth_to_normal(view, depth, world_frame=False):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth, world_frame).reshape(*depth.shape[1:], 3)
    normal_map = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return normal_map, points



class EdgeAwareTV(nn.Module):
    """Edge Aware Smooth Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, depth: Tensor, rgb: Tensor):
        """
        Args:
            depth: [batch, H, W, 1]
            rgb: [batch, H, W, 3]
        """
        grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
        grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()
    

class LogL1(nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))
        
class EdgeAwareLogL1(nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):


        logl1 = self.logl1(pred, gt)

        # 计算梯度
        grad_img_x = torch.abs(rgb[:, :, :-1] - rgb[:, :, 1:])  # 水平方向梯度
        grad_img_y = torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :])  # 垂直方向梯度

        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        # 边缘感知加权的 Log-L1 损失
        loss_x = lambda_x * logl1[:, :, :-1]
        loss_y = lambda_y * logl1[:, :-1, :]

        # 如果模式是 "per-pixel"
        if self.implementation == "per-pixel":
            if mask is not None:
                # 裁剪 mask 的宽度和高度，确保与 loss_x 和 loss_y 匹配
                mask_x = mask[:, :, :-1]  # 匹配 loss_x 的宽度
                mask_y = mask[:, :-1, :]  # 匹配 loss_y 的高度

                # 扩展 mask 的 batch size
                if mask.shape[0] == 1 and loss_x.shape[0] > 1:
                    mask_x = mask_x.expand_as(loss_x)  # 扩展 mask_x
                    mask_y = mask_y.expand_as(loss_y)  # 扩展 mask_y

                loss_x = loss_x * mask_x  # 应用裁剪后的 mask
                loss_y = loss_y * mask_y  # 应用裁剪后的 mask
            return loss_x + loss_y

        # 如果模式是 "scalar"
        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            mask_x = mask[:, :, :-1]  # 匹配 loss_x 的宽度
            mask_y = mask[:, :-1, :]  # 匹配 loss_y 的高度

            if mask.shape[0] == 1 and loss_x.shape[0] > 1:
                mask_x = mask_x.expand_as(loss_x)  # 扩展 mask_x
                mask_y = mask_y.expand_as(loss_y)  # 扩展 mask_y

            loss_x = loss_x[mask_x]
            loss_y = loss_y[mask_y]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()


        



class TVLoss(nn.Module):
    """Total Variation (TV) Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [H, W, 3]  # 去掉 batch size 维度

        Returns:
            tv_loss: 标量值
        """
        # 计算水平方向（高度方向）的差分
        h_diff = pred[:-1, :, :] - pred[1:, :, :]
        # 计算垂直方向（宽度方向）的差分
        w_diff = pred[:, :-1, :] - pred[:, 1:, :]
        # 计算总的 TV 损失
        tv_loss = torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))
        return tv_loss

        
# class DNRegularization(RegularizationStrategy):
#     """Regularization strategy as proposed in DN-Splatter

#     This consists of an EdgeAware Depth loss, a Normal loss, normal smoothing loss, and a scale loss.
#     """

#     def __init__(
#         self,
#         depth_tolerance: float = 0.1,
#         depth_loss_type: Optional[DepthLossType] = DepthLossType.EdgeAwareLogL1,
#         depth_lambda: float = 0.2,
#         normal_lambda: float = 0.1,
#     ):
#         super().__init__()
#         self.depth_tolerance = depth_tolerance
#         self.depth_loss_type = depth_loss_type
#         self.depth_loss = DepthLoss(self.depth_loss_type)
#         self.depth_lambda = depth_lambda

#         self.normal_loss_type: NormalLossType = NormalLossType.L1
#         self.normal_loss = NormalLoss(self.normal_loss_type)
#         self.normal_smooth_loss_type: NormalLossType = NormalLossType.Smooth
#         self.normal_smooth_loss = NormalLoss(self.normal_smooth_loss_type)
#         self.normal_lambda = normal_lambda

#     def get_loss(self, pred_depth, gt_depth, pred_normal, gt_normal, **kwargs):
#         """Regularization loss"""

#         depth_loss, normal_loss = 0.0, 0.0
#         if self.depth_loss is not None:
#             depth_loss = self.get_depth_loss(pred_depth, gt_depth, **kwargs)
#         if self.normal_loss is not None:
#             normal_loss = self.get_normal_loss(pred_normal, gt_normal, **kwargs)
#         scales = kwargs["scales"]
#         scale_loss = self.get_scale_loss(scales=scales)
#         return depth_loss + normal_loss + scale_loss

#     def get_depth_loss(self, pred_depth, gt_depth, **kwargs):
#         """Depth loss"""

#         valid_gt_mask = gt_depth > self.depth_tolerance
#         if self.depth_loss_type == DepthLossType.EdgeAwareLogL1:
#             gt_img = kwargs["gt_img"]
#             depth_loss = self.depth_loss(
#                 pred_depth, gt_depth.float(), gt_img, valid_gt_mask
#             )
#         elif self.depth_loss_type == DepthLossType.PearsonDepth:
#             mono_depth_loss_pearson = (
#                 self.depth_loss(pred_depth, gt_depth.float()) * valid_gt_mask.sum()
#             ) / valid_gt_mask.sum()
#             local_depth_loss = DepthLoss(DepthLossType.LocalPearsonDepthLoss)
#             mono_depth_loss_local = (
#                 local_depth_loss(pred_depth, gt_depth.float()) * valid_gt_mask.sum()
#             ) / valid_gt_mask.sum()
#             depth_loss = (
#                 mono_depth_loss_pearson + self.depth_lambda * mono_depth_loss_local
#             )

#         else:
#             depth_loss = self.depth_loss(
#                 pred_depth[valid_gt_mask], gt_depth[valid_gt_mask].float()
#             )

#         depth_loss += self.depth_lambda * depth_loss

#         return depth_loss

#     def get_normal_loss(self, pred_normal, gt_normal, **kwargs):
#         """Normal loss and normal smoothing"""
#         normal_loss = self.normal_loss(pred_normal, gt_normal)
#         normal_loss += self.normal_smooth_loss(pred_normal)

#         return normal_loss

#     def get_scale_loss(self, scales):
#         """Scale loss"""
#         # loss to minimise gaussian scale corresponding to normal direction
#         scale_loss = torch.min(torch.exp(scales), dim=1, keepdim=True)[0].mean()
#         return scale_loss    
    

def anisotropy_loss(scalings, r):
    return (scalings.max(dim=-1).values/scalings.min(dim=-1).values - r).clamp(0).mean()




def get_loss_submapping_rgbd(config, image, depth, viewpoint, opacity, initialization=True):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    # 参数配置
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    # Ground Truth 数据
    gt_image = viewpoint.original_image.cuda()
    gt_depth = torch.from_numpy(viewpoint.es_depth).to(
        dtype=torch.float32, device=image_ab.device
    )[None]

    # 像素掩码
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.1).view(*depth.shape)
    combined_mask = rgb_pixel_mask & depth_pixel_mask

    # RGB L1 损失
    l1_rgb = torch.abs(
        image_ab * combined_mask - gt_image * combined_mask
    )
    l1_rgb = l1_rgb.mean() * 0.8 + 0.2 * (1.0 - ssim(image_ab, gt_image))

    # 假设 dnloss 的计算如下
    # 你可以用 `EdgeAwareLogL1` 或其他的自定义损失函数来代替这里的 dnloss 示例
    criterion_dnloss = EdgeAwareLogL1(implementation="scalar")
    depthloss = criterion_dnloss(depth, gt_depth, gt_image, combined_mask)
    # scale_loss = torch.min(torch.exp(scales), dim=1, keepdim=True)[0].mean()
    # 综合损失
    total_loss = alpha * l1_rgb.mean() + (1 - alpha) * depthloss

    return total_loss