import os
from gaussian_splatting.utils.graphics_utils import focal2fov
from PIL import Image
import numpy as np
import torch




class SplatDataset:
    def __init__(self, root_dir,read_calibration=False,use_mask =False,eval=False):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Root directory containing the subfolders `es_depth`, `gt_depth`, `gt_pose`, `gt_rgb`.
            calibration_file (str): Path to the calibration file containing camera parameters.
        """
        self.device = "cuda:0"
        self.root_dir = root_dir
        self.eval = eval
        if self.eval:
            self.gt_pose_files = sorted(
            [os.path.join(root_dir, "eval","pose", f) for f in os.listdir(os.path.join(root_dir,  "eval","pose"))]
        )
            self.gt_rgb_files = sorted(
            [os.path.join(root_dir, "eval","rgb", f) for f in os.listdir(os.path.join(root_dir, "eval", "rgb"))]
        ) 
        
        else:
            self.reg_depth_files = sorted(
                [os.path.join(root_dir, "reg_depth", f) for f in os.listdir(os.path.join(root_dir, "reg_depth"))]
            )

            self.sfm_depth_files = sorted(
                [os.path.join(root_dir, "sfm_depth", f) for f in os.listdir(os.path.join(root_dir, "sfm_depth"))]
            )
            self.es_depth_files = sorted(
                [os.path.join(root_dir, "es_depth", f) for f in os.listdir(os.path.join(root_dir, "es_depth"))]
            )
            self.gt_depth_files = sorted(
                [os.path.join(root_dir, "gt_depth", f) for f in os.listdir(os.path.join(root_dir, "gt_depth"))]
            )
            self.gt_pose_files = sorted(
                [os.path.join(root_dir, "gt_pose", f) for f in os.listdir(os.path.join(root_dir, "gt_pose"))]
            )
            self.gt_rgb_files = sorted(
                [os.path.join(root_dir, "gt_rgb", f) for f in os.listdir(os.path.join(root_dir, "gt_rgb"))]
            ) 
            self.use_mask = use_mask
            if use_mask:       
                self.mask_files = sorted(
                    [os.path.join(root_dir, "mask", f) for f in os.listdir(os.path.join(root_dir, "mask"))]
                )
            self.normal_files = sorted(
                    [os.path.join(root_dir, "normal", f) for f in os.listdir(os.path.join(root_dir, "normal"))]
                )
        calibration_file = os.path.join(root_dir,"colmap","cameras.txt")
        # Read calibration parameters estimated by sfm
        if read_calibration:
            calibration = self.read_calibration(calibration_file)
            self.fx = calibration["fx"]
            self.fy = calibration["fy"]
            self.cx = calibration["cx"]
            self.cy = calibration["cy"]
            self.width = calibration["width"]
            self.height = calibration["height"]
        #read gt intrinsics
        else:
            gt_intrinsics = np.load(os.path.join(root_dir,"colmap","gt_intrinsics.npy"))
            self.width = int(gt_intrinsics[0])   # 图像宽度
            self.height = int(gt_intrinsics[1])  # 图像高度
            self.fx = float(gt_intrinsics[2])    # 焦距 fx
            self.fy = float(gt_intrinsics[3])    # 焦距 fy
            self.cx = float(gt_intrinsics[4])    # 主点 x 坐标
            self.cy = float(gt_intrinsics[5])    # 主点 y 坐标
        self.dtype = torch.float32
        # Calculate FOVs
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,      self.fy,  self.cy],
            [0,      0,        1]
        ], dtype=np.float32)
        # Ensure all folders have the same number of files
        self.num_files = min(
            len(self.gt_pose_files),
            len(self.gt_rgb_files),
        )
    def read_calibration(self, calibration_file):
        """
        Read camera calibration parameters from a text file.

        Args:
            calibration_file (str): Path to the calibration file.

        Returns:
            dict: Calibration parameters as a dictionary.
        """
        with open(calibration_file, "r") as file:
            line = file.readline().strip()
            parts = line.split()
            return {
                "camera_id": int(parts[0]),
                "width": int(parts[2]),
                "height": int(parts[3]),
                "fx": float(parts[4]),
                "fy": float(parts[5]),
                "cx": float(parts[6]),
                "cy": float(parts[7]),
            }
    def __len__(self):
        """Return the number of files in the dataset."""
        return self.num_files

    def __getitem__(self, idx):
        """
        Get the files for a specific index.

        Args:
            idx (int): Index of the file to retrieve.

        Returns:
            dict: A dictionary containing paths to `es_depth`, `gt_depth`, `gt_pose`, and `gt_rgb`.
        """
        if self.eval:
            image = np.array(Image.open(self.gt_rgb_files[idx]))  # 加载 RGB 图像
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )

            gt_pose = np.load(self.gt_pose_files[idx])             # origin is
            pose = torch.from_numpy(gt_pose).to(device=self.device)
            return image,None,None,None,pose,None
        else:
            if idx < 0 or idx >= self.num_files:
                raise IndexError("Index out of range.")
            image = np.array(Image.open(self.gt_rgb_files[idx]))  # 加载 RGB 图像
            image = (
                torch.from_numpy(image / 255.0)
                .clamp(0.0, 1.0)
                .permute(2, 0, 1)
                .to(device=self.device, dtype=self.dtype)
            )

            gt_depth = np.load(self.gt_depth_files[idx])           # 加载 GT 深度图
            reg_depth = np.load(self.reg_depth_files[idx])           # 加载 ES 深度图
            gt_pose = np.load(self.gt_pose_files[idx])             # origin is
            normal = np.load(self.normal_files[idx])  
            pose = torch.from_numpy(gt_pose).to(device=self.device)
            if self.use_mask:
                mask = np.load(self.mask_files[idx])  
                return  image ,gt_depth,reg_depth,normal,pose,mask   #estimated depth is reg depths
            else:
                return image ,gt_depth,reg_depth,normal,pose,None



# Example usage
if __name__ == "__main__":
    dataset = SplatDataset("/path/to/root_dir")  # Replace with your root directory

    for i in range(len(dataset)):
        data = dataset[i]
        print(f"Index {i}: ")
        print(f"  es_depth: {data['es_depth']}")
        print(f"  gt_depth: {data['gt_depth']}")
        print(f"  gt_pose: {data['gt_pose']}")
        print(f"  gt_rgb: {data['gt_rgb']}")
