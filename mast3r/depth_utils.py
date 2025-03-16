import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt



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


def coarse_correction(gtesti_depth: torch.Tensor, rendered_depth: torch.Tensor) -> torch.Tensor:
    """
    估计深度尺度误差 k，并返回经过更正的 gtesti_depth，同时过滤异常值。

    Args:
        gtesti_depth (torch.Tensor): 估计深度 [H, W]。
        rendered_depth (torch.Tensor): 渲染深度 [H, W]。

    Returns:
        torch.Tensor: 经过尺度修正的 gtesti_depth。
    """
    # 保证输入是浮点类型
    gtesti_depth = gtesti_depth.detach().float()
    rendered_depth = rendered_depth.detach().float()

    # 计算均值和异常阈值
    gtesti_mean = gtesti_depth[gtesti_depth > 0].mean()
    rendered_mean = rendered_depth[rendered_depth > 0].mean()

    gtesti_threshold = gtesti_mean * 4
    rendered_threshold = rendered_mean * 4

    # 创建掩码，过滤异常值
    mask = (
        (gtesti_depth > 0) & (gtesti_depth <= gtesti_threshold) &
        (rendered_depth > 0) & (rendered_depth <= rendered_threshold)
    )
    # 过滤后的深度值
    gtesti_depth_filtered = gtesti_depth[mask]
    rendered_depth_filtered = rendered_depth[mask]

    # 如果过滤后无有效值，返回原始 gtesti_depth
    if gtesti_depth_filtered.numel() == 0 or rendered_depth_filtered.numel() == 0:
        return gtesti_depth  # 无法修正，直接返回原始深度

    # 计算比例 k = rendered_depth.sum() / gtesti_depth.sum()
    gtesti_sum = gtesti_depth_filtered.sum()
    rendered_sum = rendered_depth_filtered.sum()

    if gtesti_sum == 0:  # 防止除零
        return gtesti_depth

    k = rendered_sum / gtesti_sum

    # 应用比例 k 对 gtesti_depth 进行修正
    corrected_depth = gtesti_depth * k
    print("estiscale",k)
    return corrected_depth.unsqueeze(0), k.detach().cpu().numpy()

