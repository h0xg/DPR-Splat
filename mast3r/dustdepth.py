import numpy as np
import cv2
from PIL import Image
from depth_utils import update_metrics
import torch
from evadepth import depth_visualization
dustdepths = np.load("dustdepths2.npy")
es_depth = np.load("/home/lingxiang/datasets/replica/office0/estimation/depth/depth_frame000000.npy")
es_depth = cv2.resize(es_depth, (512, 288), interpolation=cv2.INTER_LINEAR)  # 宽在前，高在后


depth_path = ("/home/lingxiang/datasets/replica/office0/results/depth000000.png")
depth_scale = 6553.5
depth = np.array(Image.open(depth_path)) / depth_scale
real_depth = cv2.resize(depth, (512, 288), interpolation=cv2.INTER_LINEAR)  # 宽在前，高在后

if isinstance(real_depth, torch.Tensor):
    # PyTorch 张量
    depth_pixel_mask = (real_depth > 0.01).view(*real_depth.shape)
elif isinstance(real_depth, np.ndarray):
    # NumPy 数组
    depth_pixel_mask = (real_depth > 0.01).reshape(*real_depth.shape)
print(dustdepths)

depth_visualization(dustdepths[0],"./output5images/dustdepth.jpg",True)
update_metrics(torch.from_numpy(dustdepths[0])
               ,torch.from_numpy(real_depth),torch.from_numpy(depth_pixel_mask))
