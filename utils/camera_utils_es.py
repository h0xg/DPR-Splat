import torch
from torch import nn
import json
import numpy as np
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.dpr_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        gt_depth,
        es_depth,
        normal,
        mask,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.gt_depth = gt_depth
        self.es_depth = es_depth
        self.normal = normal
        self.mask = mask
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth,es_depth, normal,gt_pose,mask= dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            es_depth,
            normal,
            mask,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        # if config["Dataset"]["type"] == "replica":
        #     row, col = 32, 32
        #     multiplier = edge_threshold
        #     _, h, w = self.original_image.shape
        #     for r in range(row):
        #         for c in range(col):
        #             block = img_grad_intensity[
        #                 :,
        #                 r * int(h / row) : (r + 1) * int(h / row),
        #                 c * int(w / col) : (c + 1) * int(w / col),
        #             ]
        #             th_median = block.median()
        #             block[block > (th_median * multiplier)] = 1
        #             block[block <= (th_median * multiplier)] = 0
        #     self.grad_mask = img_grad_intensity
        # else:
        median_img_grad_intensity = img_grad_intensity.median()
        self.grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )

    def clean(self):
        self.original_image = None
        self.es_depth = None
        self.gt_depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

    def save(self, filename):
        """
        Save the Camera object to a file.

        Args:
            filename (str): The file path to save the Camera object.
        """
        torch.save(
            {
                "uid": self.uid,
                "device": self.device,
                "R": self.R,
                "T": self.T,
                "R_gt": self.R_gt,
                "T_gt": self.T_gt,
                "original_image": self.original_image,
                "gt_depth": self.gt_depth,
                "es_depth": self.es_depth,
                "grad_mask": self.grad_mask,
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
                "FoVx": self.FoVx,
                "FoVy": self.FoVy,
                "image_height": self.image_height,
                "image_width": self.image_width,
                "cam_rot_delta": self.cam_rot_delta,
                "cam_trans_delta": self.cam_trans_delta,
                "exposure_a": self.exposure_a,
                "exposure_b": self.exposure_b,
                "projection_matrix": self.projection_matrix,
            },
            filename,
        )

    @classmethod
    def load(cls, filename, device="cuda:0"):
        """
        Load a Camera object from a file.

        Args:
            filename (str): The file path to load the Camera object.
            device (str): The device to load the tensors onto.

        Returns:
            Camera: The loaded Camera object.
        """
        data = torch.load(filename, map_location=device)

        instance = cls.__new__(cls)  # Create an uninitialized instance of the class

        instance.uid = data["uid"]
        instance.device = device
        instance.R = data["R"].to(device)
        instance.T = data["T"].to(device)
        instance.R_gt = data["R_gt"].to(device)
        instance.T_gt = data["T_gt"].to(device)
        instance.original_image = data["original_image"]
        instance.gt_depth = data["gt_depth"]
        instance.es_depth = data["es_depth"]
        instance.grad_mask = data["grad_mask"]  # This may be None
        instance.fx = data["fx"]
        instance.fy = data["fy"]
        instance.cx = data["cx"]
        instance.cy = data["cy"]
        instance.FoVx = data["FoVx"]
        instance.FoVy = data["FoVy"]
        instance.image_height = data["image_height"]
        instance.image_width = data["image_width"]
        instance.projection_matrix = data["projection_matrix"].to(device)

        return instance