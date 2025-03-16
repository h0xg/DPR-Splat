import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from utils.pose_utils import update_pose,start_alignment,compute_initpose
from mast3r.mast3r.model import AsymmetricMASt3R
import torch
import torch.multiprocessing as mp
from munch import munchify

import yaml
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.config_utils import load_config
from utils.dataset import load_dataset,get_dataset
from utils.logging_utils import Log
import cv2
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.camera_utils_es import Camera
from utils.dpr_utils import get_loss_normal,get_loss_tracking_es,depth_to_normal,ssim,\
    calculate_quaternion_difference,get_loss_tracking_rgb,calculate_translation_difference,get_loss_submapping_rgbd,anisotropy_loss
from gaussian_splatting.gaussian_renderer import render
from utils.eval_utils import eval_ate, save_gaussians
from utils.depth_utils import update_metrics,fit_quadratic_transformation_shared,\
    fit_quadratic_transformation,fit_ransac_transformation,reshape_and_save_depthmaps,save_conf,fit_linear_transformation
from random import randint
import numpy as np
from mast3r.mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.dust3r.dust3r.image_pairs import make_pairs
from mast3r.dust3r.dust3r.utils.image import load_images
from mast3r.dust3r.dust3r.utils.device import to_numpy
from mast3r.dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from mast3r.mast3r.utils.misc import mkdir_for, hash_md5
import tempfile
from contextlib import nullcontext
import trimesh
from scipy.spatial.transform import Rotation
from mast3r.dust3r_utils import   storePly, save_colmap_cameras, save_colmap_images
import shutil
from PIL import Image
import matplotlib.pyplot as plt

from utils.camera_conversion import adjust_intrinsics
import re
from utils.utils_poses.vis_pose_utils import plot_pose
from utils.splatdataset import SplatDataset
from utils.image_utils import psnr, colorize
from utils.map_utils import (torch2np, np2torch,np2ptcloud, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask,
                                    sample_pixels_based_on_gradient)
from utils.step_tracker import StepTracker
from lpipsPyTorch import lpips
from Metric3D.dn_predict import estimated_depth_and_normal
from utils.utils_poses.lie_group_helper import convert3x4_4x4
from tqdm import trange
import random
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


class DPR:
    def __init__(self, config, save_dir=None,sourcepath= None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self.device = "cuda:0"
        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        ) 

        self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.lambda_dnormal = self.config["Training"]["lambda_dnormal"]
        self.sh_degree = self.config["model_params"]["sh_degree"]
        model_params.sh_degree = self.sh_degree
        
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.depthencoder =  self.config["Depth"]["encoder"]
        #fmaily church
        self.output_path = self.save_dir 
        if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
        self.sourcepath =sourcepath
        # self.dataset_name = "Mip-NeRF 360"
        self.dataset_name = "Tanks"
        if  config["Mast3R"]["scenegraph_type"]=="swin":
            self.savemask = True
        else:
            self.savemask=False

        if self.dataset_name in ["replica","TUM","scan_net"]:
            self.dataset = load_dataset(
            self.dataset_name,model_params, model_params.source_path, config=config
        )
        
        elif self.dataset_name in  ["Tanks","Mip-NeRF 360","llff"]:
            self.dataset = load_dataset(
            dataset_name = self.dataset_name,args=  None,path = self.sourcepath,config = config
            )
            self.train_view = 12
            self.sample_rate = len(self.dataset)//   self.train_view         
            # self.train_view = 8
            # self.sample_rate = 5
        elif self.dataset_name in ["re10k"]:
            step_tracker = StepTracker()
            viewsample_config = self.config["Re10k"]["view_sampler_path"]
            viewsample_cfg = load_config(viewsample_config)
            self.config = {**self.config, **viewsample_cfg}
            self.dataset = get_dataset(self.config, "test", step_tracker)
            self.train_view = self.config["view_sampler"]["num_context_views"]
        
    def ensure_directories(self):
        self.colmap_path = os.path.join(self.output_path, 'colmap')
        self.es_depth_path = os.path.join(self.output_path, 'es_depth')
        self.sfm_depth_path = os.path.join(self.output_path, 'sfm_depth')
        self.reg_depth_path = os.path.join(self.output_path, 'reg_depth')
        self.gt_rgb_path = os.path.join(self.output_path, 'gt_rgb')
        self.gt_posepath = os.path.join(self.output_path, 'gt_pose')
        self.gt_depth_path = os.path.join(self.output_path, 'gt_depth')
        self.sfm_conf_path = os.path.join(self.output_path, 'sfm_conf')
        self.mask_path = os.path.join(self.output_path, 'mask')
        self.eval_path = os.path.join(self.output_path, 'eval')
        self.normal_path = os.path.join(self.output_path, 'normal')
        required_directories = [
            'colmap',
            'es_depth',
            'sfm_depth',
            'reg_depth',
            'gt_rgb',
            'gt_pose',
            'gt_depth',
            'sfm_conf',
            'mask',
            'eval',
            'normal'
        ]

        for directory in required_directories:
            dir_path = os.path.join(self.output_path, directory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def read_dataset(self):
        endframe = len(self.dataset)
        startframe = 0
        indexlist = list(range(startframe, endframe, self.sample_rate))[:self.train_view]
        folders = ['rgb', 'pose']
        for folder in folders:
            folder_path = os.path.join(self.eval_path, folder)
            # Check if the folder exists
            if not os.path.exists(folder_path):
                # Create the folder if it doesn't exist
                os.makedirs(folder_path)
                print(f"Created folder: {folder_path}")
            else:
                print(f"Folder already exists: {folder_path}")


        if self.dataset_name in ["replica","TUM"]:
            gt_intrinsics = np.array([int(self.dataset.width), int(self.dataset.height), self.dataset.fx,
                                    self.dataset.fy, self.dataset.cx, self.dataset.cy])
            np.save(os.path.join(self.colmap_path,"gt_intrinsics.npy"),gt_intrinsics)
            for i in indexlist:
                gt_rgb= cv2.imread(self.dataset.color_paths[i])
                rgb_path = os.path.join(self.gt_rgb_path, f"{i:04d}.png")
                cv2.imwrite(rgb_path, gt_rgb)                
                gt_depth = np.array(Image.open(self.dataset.depth_paths[i])) / self.dataset.depth_scale
                gt_depth_path = os.path.join(self.gt_depth_path, f"{i:04d}")
                np.save(gt_depth_path, gt_depth)
                #gt_pose = np.linalg.inv(self.dataset.poses[i])  #this dataset inv the extrinsics, we should inv it back...
                gt_pose = self.dataset.poses[i]  #c2w  
                gt_pose_path = os.path.join(self.gt_posepath, f"{i:04d}")
                np.save(gt_pose_path, gt_pose)
            test_index = indexlist
            for i in test_index:
                gt_rgb=  cv2.imread(self.dataset.color_paths[i])
                rgb_path = os.path.join(self.eval_path, "rgb",f"{i:04d}.png")
                cv2.imwrite(rgb_path, gt_rgb)                
                gt_pose = self.dataset.poses[i]  #c2w  
                gt_pose_path = os.path.join(self.eval_path, "pose",f"{i:04d}") 
                np.save(gt_pose_path, gt_pose) 

        elif self.dataset_name in ["Tanks","Mip-NeRF 360","llff"]:
            data0 = self.dataset[0]
            
            gt_intrinsics = np.array([int(data0.width),int(data0.height),data0.intrinsics[0][0],
                                        data0.intrinsics[1][1],data0.intrinsics[0][2],
                                        data0.intrinsics[1][2]])
            np.save(os.path.join(self.colmap_path,"gt_intrinsics.npy"),gt_intrinsics)
            for i in indexlist:
                gt_rgb= self.dataset[i].image
                imgname = self.dataset[i].image_name
                rgb_path = os.path.join(self.gt_rgb_path, imgname+".png")
                gt_rgb.save(rgb_path)
                gt_pose = np.eye(4)   #incolmap is camera to world
                gt_pose[:3,:3] = self.dataset[i].R
                gt_pose[:3,3] = self.dataset[i].T
                gt_pose_path = os.path.join(self.gt_posepath, f"{i:04d}")
                gt_pose = np.linalg.inv(gt_pose)
                np.save(gt_pose_path, gt_pose)
                gt_depth = np.array([1])
                gt_depth_path = os.path.join(self.gt_depth_path, f"{i:04d}")
                np.save(gt_depth_path, gt_depth)
                  #no gt depth to fulfill the document, set it to 1
            test_index = [i+randint(1,self.sample_rate-1) for i in indexlist]
            for i in test_index:
                gt_rgb= self.dataset[i].image
                imgname = self.dataset[i].image_name
                rgb_path = os.path.join(self.eval_path, "rgb",imgname+".png")
                gt_rgb.save(rgb_path)
                gt_pose = np.eye(4)   #incolmap is camera to world
                gt_pose[:3,:3] = self.dataset[i].R
                gt_pose[:3,3] = self.dataset[i].T
                gt_pose = np.linalg.inv(gt_pose)
                gt_pose_path = os.path.join(self.eval_path, "pose",f"{i:04d}") 
                np.save(gt_pose_path, gt_pose)          
            



    def run_sfm(self):        
        rgblist = sorted(
            [os.path.join(self.gt_rgb_path, a) for a in os.listdir(self.gt_rgb_path)],
            key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]) if re.search(r'\d+', os.path.basename(x)) else 0
        )
        image_size = self.config["Mast3R"]["image_size"]
     
        
        imgs,original_sizes, target_size = load_images(rgblist, size=image_size, verbose=True)
        scenegraph_type = config["Mast3R"]["scenegraph_type"]
        winsize = self.config["Mast3R"]["winsize"]
        scene_graph_params = [scenegraph_type]
        if scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(winsize))
        scene_graph = '-'.join(scene_graph_params)
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        optim_level = self.config["Mast3R"]["optim_level"]
        if optim_level == 'coarse':
            niter2 = 0
        # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
        weights_path = self.config["Mast3R"]["weights_path"]


        model = AsymmetricMASt3R.from_pretrained(weights_path).to(self.device)
        chkpt_tag = hash_md5(weights_path)
        tmp_dir = None
        def get_context(tmp_dir):
            return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
                else nullcontext(tmp_dir)
        with get_context(tmp_dir) as tmpdirname:
            cache_path = os.path.join(tmpdirname, chkpt_tag)
            os.makedirs(cache_path, exist_ok=True)


        cache_dir = os.path.join(self.output_path, 'cache')

        if self.config["Mast3R"]["load_intrinsic"]:
                init_K = np.load(os.path.join(self.colmap_path,"gt_intrinsics.npy"))
                init_K = adjust_intrinsics(init_K,original_sizes,target_size) 

        else:
            init_K = None
        scene = sparse_global_alignment(rgblist, pairs, cache_dir,
                                        model, lr1=self.config["Mast3R"]["lr1"], niter1=self.config["Mast3R"]["niter1"], 
                                        lr2=self.config["Mast3R"]["lr2"],
                                          niter2=self.config["Mast3R"]["niter2"], device=self.device,
                                        opt_depth='depth' in optim_level, shared_intrinsics=["shared_intrinsics"],
                                        matching_conf_thr=self.config["Mast3R"]["matching_conf_thr"],init_K= init_K,
                                        mask_path=self.mask_path,original_sizes = original_sizes,save_mask=self.savemask)
        outfile_name =os.path.join(self.output_path,"scene.glb")
        scene_state = SparseGAState(scene, None, cache_dir, outfile_name)
        if scene_state is None:
            return None
        outfile = scene_state.outfile_name
        if outfile is None:
            return None

        # get optimized values from scene
        scene = scene_state.sparse_ga
        imgs = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()
        intrinsics =  scene.get_intrinsics().cpu()
        min_conf_thr = self.config["Mast3R"]["min_conf_thr"]
        pts3d, depthmap, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
        mask = to_numpy([c > min_conf_thr for c in confs])
        assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
        pts3d = to_numpy(pts3d)
        imgs = to_numpy(imgs)
        focals = to_numpy(focals)
        cams2world = to_numpy(cams2world)
        
        scene = trimesh.Scene()
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        color_4_3dgs = (col * 255.0).astype(np.uint8)

        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
        transparent_cams = False


        storePly(os.path.join(self.output_path,"colmap", "points3D.ply"), pts, color_4_3dgs)
        save_colmap_cameras(original_sizes, intrinsics, os.path.join(self.output_path,"colmap", 'cameras.txt'))
        save_colmap_images(cams2world, os.path.join(self.output_path,"colmap", 'images.txt'), rgblist)
        np.save(os.path.join(self.output_path,"colmap", 'cams2world.npy'),cams2world)
        sfm_depth_vis_path = os.path.join(self.output_path,"sfm_depth_vis")
        if not os.path.exists(sfm_depth_vis_path):
            os.makedirs(sfm_depth_vis_path)   
        reshape_and_save_depthmaps(depthmap,target_size[1],target_size[0],original_sizes[1], original_sizes[0], self.sfm_depth_path,sfm_depth_vis_path)
        output_conf =  self.sfm_conf_path
        save_conf(confs, output_conf,target_size[1],target_size[0],original_sizes[1], original_sizes[0])

        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)

        # # add each camera
        # cam_color = None
        # for i, pose_c2w in enumerate(cams2world):
        #     if isinstance(cam_color, list):
        #         camera_edge_color = cam_color[i]
        #     else:
        #         camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        #     add_scene_cam(scene, pose_c2w, camera_edge_color,
        #                 None if transparent_cams else imgs[i], focals[i],
        #                 imsize=imgs[i].shape[1::-1], screen_width=self.config["Mast3R"]["cam_size"])

        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        scene.export(file_obj=outfile)

        if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)

    def estimated_depth_normal(self):        
            rgblist = sorted(os.listdir(self.gt_rgb_path))
            intrinsics_path = os.path.join(self.colmap_path,"gt_intrinsics.npy")
            intrinsics = np.load(intrinsics_path)
            intrinsics = [intrinsics[2],intrinsics[3],intrinsics[4],intrinsics[5]] 
            es_depth_vis_path = os.path.join(self.output_path,"es_depth_vis")
            if not os.path.exists(es_depth_vis_path):
                os.makedirs(es_depth_vis_path)   

            for i,image_name in  enumerate(rgblist) :            
                imgpath = os.path.join(self.gt_rgb_path, image_name)
                depth_name = os.path.join(self.es_depth_path, f"esdepth_{i:04d}.npy")
                normal_name = os.path.join(self.normal_path,f"normal_{i:04d}.npy")

                depth,normal = estimated_depth_and_normal(imgpath,intrinsics)
                np.save(depth_name, depth)
                np.save(normal_name,normal)


                normalized_depthmap = (depth - depth.min()) / (
                depth.max() - depth.min()
            )

                cmap = plt.colormaps.get_cmap('Spectral_r')

                color_mapped_image = cmap(normalized_depthmap)
                visualization_path = os.path.join(es_depth_vis_path, f"esdepth_{i:04d}.png")
                plt.imsave(visualization_path, color_mapped_image)
                pred_normal_vis = normal.transpose((1, 2, 0))
                pred_normal_vis = (pred_normal_vis + 1) / 2
                visualization_path = os.path.join(es_depth_vis_path, f"esnormal_{i:04d}.png")
                plt.imsave(visualization_path, pred_normal_vis)

    def depth_alignment_metricsbatch(self):
        es_depth_data = [
            np.load(path)
            for path in sorted(
                [os.path.join(self.es_depth_path, f) for f in os.listdir(self.es_depth_path) if f.endswith('.npy')]
            )
        ]

        sfm_depth_data = [
            np.load(path)
            for path in sorted(
                [os.path.join(self.sfm_depth_path, f) for f in os.listdir(self.sfm_depth_path) if f.endswith('.npy')]
            )
        ]

        gt_depth_data = [
            np.load(path)
            for path in sorted(
                [os.path.join(self.gt_depth_path, f) for f in os.listdir(self.gt_depth_path) if f.endswith('.npy')]
            )
        ]
        anything_vis = [
            cv2.imread(path)
                        for path in sorted(
                [os.path.join(self.output_path,"es_depthanything_vis", f) for f in os.listdir(os.path.join(self.output_path,"es_depthanything_vis"))]
            )
        ]

        sfm_conf_data = [
            np.load(path)
            for path in sorted(
                [os.path.join(self.sfm_conf_path, f) for f in os.listdir(self.sfm_conf_path) if f.endswith('.npy')]
            )
        ]
        mask_data = []
        skymasks = []
        for i in range(self.train_view):
            if es_depth_data[i].max()>100:
                depth_threshold = 100

                sky_mask = es_depth_data[i] > depth_threshold
            else:
                sky_mask = np.zeros_like(es_depth_data[i], dtype=bool)
            skymasks.append(sky_mask)


            mask = (es_depth_data[i]>0.01)&(sfm_depth_data[i]>0.01)&(~sky_mask)
            mask_data.append(mask)

    
        reg_depths = fit_quadratic_transformation_shared(es_depth_data, sfm_depth_data, masks = mask_data)
        maxvalue = max(reg_depths[i][~skymasks[i]].max() for i in  range(self.train_view))
        global_max = np.max(reg_depths)
        model_type = "vit_t"
        sam_checkpoint = "./weights/mobile_sam.pt"
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        for i in range(self.train_view):
            reg_depth = reg_depths[i]
            depth_vis_path = os.path.join(self.output_path,"es_depth_vis")
            es_vis = anything_vis[i]
            finalreg_depth = np.zeros_like(reg_depth)
            masks = mask_generator.generate(es_vis)
            sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
            for ann in sorted_anns:
                m = ann['segmentation']
                if m not in sky_mask[i]:
                    mask_depth = fit_ransac_transformation(es_depth_data[i], sfm_depth_data[i], mask = (reg_depth>0.01)&m,
                                                                    weight=None)
                    result = np.where(m, mask_depth, 0)
                    finalreg_depth+= result
                
            finalreg_depth[finalreg_depth == 0] = reg_depth[finalreg_depth == 0]
            finalreg_depth[skymasks[i]] = global_max
            finalreg_depth[finalreg_depth<0] = global_max
            normalized_depthmap = (finalreg_depth - finalreg_depth.min()) / (
            finalreg_depth.max() - finalreg_depth.min())

            depth_name =   os.path.join(self.reg_depth_path,f"reg_depth_{i:04d}.npy")
            np.save(depth_name,finalreg_depth)
            cmap = plt.colormaps.get_cmap('Spectral_r')
            color_mapped_image = cmap(normalized_depthmap)
            visualization_path = os.path.join(depth_vis_path, f"regdepth_{i:04d}.png")

            plt.imsave(visualization_path, color_mapped_image)


            if self.dataset_name in ["replica","TUM","scan_net"]:
                gt_depth = gt_depth_data[i]
                if isinstance(gt_depth, torch.Tensor):
                    depth_pixel_mask = (gt_depth > 0.01).view(*gt_depth.shape)
                elif isinstance(gt_depth, np.ndarray):
                    depth_pixel_mask = (gt_depth > 0.01).reshape(*gt_depth.shape)
                print(i, "compare depth es_depth and real")
                update_metrics(torch.from_numpy(es_depth_data[i])
                ,torch.from_numpy(gt_depth),torch.from_numpy(depth_pixel_mask))
                print(i, "compare depth sfm_depths and real")
                update_metrics(torch.from_numpy(sfm_depth_data[i])
                ,torch.from_numpy(gt_depth),torch.from_numpy(depth_pixel_mask))
                
                print(i, "compare depth reg_depth and real")
                update_metrics(torch.from_numpy(reg_depth),torch.from_numpy(gt_depth),torch.from_numpy(depth_pixel_mask))  
                depth_name = os.path.join(self.reg_depth_path,f"reg_depth_{i:04d}.npy")
                np.save(depth_name,reg_depth)      


    def run_splat(self):
        self.splatdataset = SplatDataset(self.output_path,use_mask=self.savemask)
        self.iteration_count =0
        self.cameras_extent = 6.0
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        #self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]/2
        self.tracking_itr_num = 100
        self.occ_aware_visibility = {}
        self.viewpoint_stack = []
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.splatdataset.fx,
            fy=self.splatdataset.fy,
            cx=self.splatdataset.cx,
            cy=self.splatdataset.cy,
            W=self.splatdataset.width,
            H=self.splatdataset.height,
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(device=self.device)

        self.gaussians = self.init_map(0,self.projection_matrix)
        single_iteration = 100
        first_frame_ratio = 0.5
        last_frame_ratio = 1/self.train_view*2
        total_iteration = 0
        for cur_frame in range(1,self.train_view):
            current_frame_ratio = first_frame_ratio - (first_frame_ratio-last_frame_ratio)/self.train_view*cur_frame
            self.gaussians,next_viewpoint = self.pose_estimation(self.gaussians,cur_frame,self.projection_matrix,rgb=True)
            self.viewpoint_stack.append(next_viewpoint)
            frame_iter = single_iteration*cur_frame
            current_frame_iters = frame_iter*current_frame_ratio
            distribution = compute_opt_views_distribution(len(self.viewpoint_stack), frame_iter, current_frame_iters)
            if total_iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()
            self.gaussians.extend_from_pcd_seq(
                next_viewpoint, kf_id=cur_frame, init=False, scale=2, depthmap=next_viewpoint.es_depth
            )
            for iteration in range(1, frame_iter):
                total_iteration+= 1
                # Update learning rat
                keyframe_id = np.random.choice(np.arange(len(self.viewpoint_stack)), p=distribution)
                viewpoint_cam = self.viewpoint_stack[keyframe_id]

                opt_params = []
                opt_params.append(
                    {
                        "params": [viewpoint_cam.cam_rot_delta],
                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                        * 0.1,
                        "name": "rot_{}".format(viewpoint_cam.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint_cam.cam_trans_delta],
                        "lr": self.config["Training"]["lr"][
                            "cam_trans_delta"
                        ]
                        * 0.1,
                        "name": "trans_{}".format(viewpoint_cam.uid),
                    }
                )

         
                pose_optimizers = torch.optim.Adam(opt_params)

                render_pkg = render(
                    viewpoint_cam, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                gt_image = viewpoint_cam.original_image.cuda()
                loss_mapping = get_loss_submapping_rgbd(self.config, image, depth, viewpoint_cam, opacity)
                loss_mapping += 1 * anisotropy_loss(self.gaussians.get_scaling,5)
                loss_mapping.backward()
                with torch.no_grad():

                    psnr_train = psnr(image, gt_image).mean().double()
                    self.just_reset = False
                    if total_iteration < self.opt_params.densify_until_iter :
                        
                        # Keep track of max radii in image-space for pruning
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                                    radii[visibility_filter])
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor, visibility_filter)
                        #larger than 500,every 100 do this
                        if total_iteration > self.opt_params.densify_from_iter and total_iteration % self.opt_params.densification_interval == 0:
                            size_threshold = 20 if total_iteration > self.opt_params.opacity_reset_interval else None
                            self.gaussians.densify_and_prune(
                                self.opt_params.densify_grad_threshold,+
                                self.gaussian_th,
                                self.gaussian_extent,
                                size_threshold,
                            )
                        
                    if total_iteration % self.opt_params.opacity_reset_interval == 0  and  (total_iteration < self.opt_params.reset_until_iter):
                        self.gaussians.reset_opacity()
                        self.just_reset = True

                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    #self.gaussians.update_learning_rate(total_iteration)
                    self.gaussians.update_learning_rate(iteration)
                    pose_optimizers.step()
                    pose_optimizers.zero_grad(set_to_none=True)
                    # Pose update
                    for cam_idx in range(len(self.viewpoint_stack)):
                        viewpoint = self.viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)

                if iteration % 100 == 0 and keyframe_id==cur_frame:
                    print({"PSNR": f"{psnr_train:.{2}f}",
                                            "Number points": f"{self.gaussians.get_xyz.shape[0]}"})




            quad_err =  calculate_quaternion_difference(next_viewpoint.R,next_viewpoint.R_gt)
            trans_err = calculate_translation_difference(next_viewpoint.T,next_viewpoint.T_gt)
            msg = f"map optimization frame_id: {1}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
            Log(msg) 
        self.global_refinement()
        save_gaussians(self.gaussians, self.output_path, 1, final=True)


    def global_refinement(self):
        Log("Starting color refinement")
        iteration_total = 2000

        for iteration in (pbar := trange(1, iteration_total + 1)):
            viewpoint_cam_idx =random.randint(0, len(self.viewpoint_stack) - 1)
            viewpoint_cam = self.viewpoint_stack[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline_params,self.background)
            image, depth,opacity = render_pkg["render"], render_pkg["depth"],render_pkg["opacity"]
            visibility_filter =  render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor =render_pkg["viewspace_points"]
            loss =get_loss_submapping_rgbd(
                self.config, image, depth, viewpoint_cam, opacity, initialization=True
            )
            loss += 1 * anisotropy_loss(self.gaussians.get_scaling,5)
            loss.backward()
            with torch.no_grad():
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.viewpoint_stack[viewpoint_cam_idx] = viewpoint_cam
            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

        Log("Map refinement done")



    def init_map(self,cur_frame_idx,projection_matrix): 
        local_gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        local_gaussians.init_lr(6.0)
        local_gaussians.training_setup(self.opt_params)

        if cur_frame_idx==0:
            viewpoint = Camera.init_from_dataset(
                self.splatdataset, cur_frame_idx, projection_matrix
            )
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            self.viewpoint_stack.append(viewpoint)
        else:
            viewpoint = self.viewpoint_stack[cur_frame_idx]
        viewpoint.compute_grad_mask(self.config)
        local_gaussians.extend_from_pcd_seq(
        viewpoint, kf_id=cur_frame_idx, init=True, scale=2.0, depthmap=viewpoint.es_depth
    )
        #init map 
        loss_init = 0
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, local_gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )            
            loss_init = get_loss_submapping_rgbd(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init += 1 * anisotropy_loss(local_gaussians.get_scaling,5)

            loss_init.backward()

            with torch.no_grad():
                local_gaussians.max_radii2D[visibility_filter] = torch.max(
                    local_gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                local_gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    local_gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,  
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    local_gaussians.reset_opacity()

                local_gaussians.optimizer.step()
                local_gaussians.optimizer.zero_grad(set_to_none=True)
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return local_gaussians

    def pose_estimation(self,local_gaussians,cur_frame_idx,projection_matrix,next_viewpoint = None,rgb=True):
        viewpoint = self.viewpoint_stack[-1]
        if next_viewpoint == None:
            next_viewpoint = Camera.init_from_dataset(
                    self.splatdataset, cur_frame_idx, projection_matrix
                )
            next_viewpoint.compute_grad_mask(self.config)
            init_R,init_t = compute_initpose(cur_frame_idx,viewpoint.R, viewpoint.T,self.colmap_path)
            next_viewpoint.update_RT(init_R, init_t)

        quad_err =  calculate_quaternion_difference(next_viewpoint.R,next_viewpoint.R_gt)
        trans_err = calculate_translation_difference(next_viewpoint.T,next_viewpoint.T_gt)
        msg = f"before optimization1 : frame_id: {cur_frame_idx}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
        Log(msg)
        opt_params = []
        opt_params.append(
            {
                "params": [next_viewpoint.cam_rot_delta],
                "lr": 0.8*self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )   
        opt_params.append(
            {
                "params": [next_viewpoint.cam_trans_delta],
                "lr": 0.8*self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(next_viewpoint.uid),
            }
        )
       
        pose_optimizer = torch.optim.Adam(opt_params)
        self.optimize_pose(local_gaussians,pose_optimizer,next_viewpoint,cur_frame_idx,rgb=rgb)


        return local_gaussians,next_viewpoint
    
  

    def optimize_pose(self,local_gaussians,pose_optimizer,next_viewpoint,cur_frame_idx,rgb):
        loss_tracking = 0
        for tracking_itr in range(self.tracking_itr_num): 
            # render image  
            render_pkg = render(
                next_viewpoint, local_gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            loss_tracking = get_loss_tracking_es(
                self.config, image, depth, opacity, next_viewpoint,rgb = rgb,usemask=self.savemask
            )
            loss_tracking.backward()
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(next_viewpoint)
                pose_optimizer.zero_grad(set_to_none=True)


            if converged:
                print(converged)
                break
        quad_err =  calculate_quaternion_difference(next_viewpoint.R,next_viewpoint.R_gt)
        trans_err = calculate_translation_difference(next_viewpoint.T,next_viewpoint.T_gt)
        msg = f"rgb optimization frame_id: {cur_frame_idx}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
        Log(msg)
        return render_pkg

 

    def loadgs(self,output):        
        output_dir = os.path.join(output,"viewpoint")
        viewpoints = []
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith(".pth"):
                vp = Camera.load(os.path.join(output_dir, filename))
                viewpoints.append(vp)

        print(f"Loaded {len(viewpoints)} viewpoints.")

        gaussians = []
        output_dir = os.path.join(output,"gaussian")
        for filename in sorted(os.listdir(output_dir)):
            print(os.path.join(output_dir, filename))
            gaussian = GaussianModel(self.sh_degree)
            gaussian.load_ply(os.path.join(output_dir, filename))
            gaussians.append(gaussian)
        new_gaussian = GaussianModel(self.sh_degree)

        # Concatenate all Gaussian parameters
        new_gaussian._xyz = torch.cat([g._xyz for g in gaussians], dim=0)
        new_gaussian._features_dc = torch.cat([g._features_dc for g in gaussians], dim=0)
        new_gaussian._features_rest = torch.cat([g._features_rest for g in gaussians], dim=0)
        new_gaussian._scaling = torch.cat([g._scaling for g in gaussians], dim=0)
        new_gaussian._rotation = torch.cat([g._rotation for g in gaussians], dim=0)
        new_gaussian._opacity = torch.cat([g._opacity for g in gaussians], dim=0)
        new_gaussian.max_radii2D = torch.cat([g.max_radii2D for g in gaussians], dim=0)
        save_gaussians(new_gaussian, self.output_path, 3, final=True)


        
 
    def evaluation(self,gaussian):
        self.splatdataset = SplatDataset(self.output_path,eval=False)
        psnr_eval = 0
        ssim_eval = 0
        lpips_eval = 0
        end_frame = len(self.splatdataset)
        with torch.no_grad():
            for cur_frame_idx in range(end_frame):
                viewpoint_cam = self.viewpoint_stack[cur_frame_idx]
                gt_image = viewpoint_cam.original_image.cuda()
                render_dict = render(
                    viewpoint_cam, 
                    gaussian, 
                    self.pipeline_params, 
                    self.background
                )

                render_dict["render"]  = torch.clamp(render_dict["render"] , max=1)
                render_dict["render"] = (torch.exp(viewpoint_cam.exposure_a)) * render_dict["render"] + viewpoint_cam.exposure_b
                      
                current_psnr = psnr(render_dict["render"], gt_image).mean().double()
                
                print(f"Current PSNR: {current_psnr}")
                psnr_eval += current_psnr

                # 计算并打印 SSIM
                current_ssim = ssim(render_dict["render"], gt_image).mean().double()
                print(f"Current SSIM: {current_ssim}")
                ssim_eval += current_ssim

                # 计算并打印 LPIPS
                current_lpips = lpips(render_dict["render"], gt_image, net_type="vgg").mean().double()
                print(f"Current LPIPS: {current_lpips}")
                lpips_eval += current_lpips
                self.visualize(viewpoint_cam,render_dict,
                            f"{self.output_path}/eval/eval_out/{cur_frame_idx:04d}.png",
                            save_ply=True)

            with open(f"{self.output_path}/eval/eval_out/test.txt", 'w') as f:
                f.write('PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
                        psnr_eval / end_frame,
                        ssim_eval / end_frame,
                        lpips_eval / end_frame))
                f.close()

            print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
                0,
                end_frame,
                psnr_eval / end_frame,
                ssim_eval / end_frame,
                lpips_eval / end_frame))   



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--sourcepath", type=str, required=True, help="Path to source files")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save directory")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    # Load configuration
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    # Initialize DPR with sourcepath and save_dir
    dpr = DPR(config, save_dir=args.save_dir, sourcepath=args.sourcepath)

    # Run pipeline
    dpr.ensure_directories()
    dpr.read_dataset()
    dpr.run_sfm()
    dpr.estimated_depth_normal()
    dpr.depth_alignment_metricsbatch()
    dpr.run_splat()
