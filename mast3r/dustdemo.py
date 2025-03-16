# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
from dust3r.dust3r.model import AsymmetricCroCo3DStereo

from dust3r.dust3r.inference import inference
from dust3r.dust3r.image_pairs import make_pairs
from dust3r.dust3r.utils.image import load_images, rgb
from dust3r.dust3r.utils.device import to_numpy
from dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import sys
sys.path.append('./dust3r/dust3r')

import matplotlib.pyplot as pl


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                         "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                         "DUSt3R_ViTLarge_BaseDecoder_224_linear"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser


def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

device = "cuda"

weights_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to("cuda")
scenegraph_type = "swin"
filepath = "/home/lingxiang/SLAM/mast3r/5images"
filelist=  sorted(os.path.join(filepath, f)  for f in os.listdir(filepath) if f.startswith("frame") and f.endswith(".jpg"))
silent = False
winsize = 1
imgs = load_images(filelist, size=512, verbose=not silent)
if len(imgs) == 1:
    imgs = [imgs[0], copy.deepcopy(imgs[0])]
    imgs[1]['idx'] = 1
if scenegraph_type == "swin":
    scenegraph_type = scenegraph_type + "-" + str(winsize)
elif scenegraph_type == "oneref":
    scenegraph_type = scenegraph_type + "-" + str(refid)

pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

output = inference(pairs, model, device, batch_size=1, verbose=not silent)


view1, pred1 = output['view1'], output['pred1']
view2, pred2 = output['view2'], output['pred2']
print(list(view1.keys()))
print(list(pred1.keys()))


print(view1['img'].shape)
print(pred1['pts3d'].shape)

mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
# focal = [256,256,256,256,256]
# pp = [[255.7867,143.7882],[255.7867,143.7882],[255.7867,143.7882],[255.7867,143.7882],[255.7867,143.7882]]
# pp_numpy = np.array(pp)
# scene.preset_focal(focal)
# scene.preset_principal_point(pp_numpy)
# w2c = np.load("camera_ext.npy")
# c2w = np.linalg.inv(w2c)
# scene.preset_pose(c2w)
lr = 0.01

if mode == GlobalAlignerMode.PointCloudOptimizer:
    loss = scene.compute_global_alignment(init='mst', niter=300, schedule="linear", lr=lr)

# outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
#                                     clean_depth, transparent_cams, cam_size)

# also return rgb, depth and confidence imgs
# depth is normalized with the max value for all images
# we apply the jet colormap on the confidence maps
rgbimg = scene.imgs
depths = to_numpy(scene.get_depthmaps())
np.save("dustdepths2.npy", np.array(depths), allow_pickle=True)
confs = to_numpy([c for c in scene.im_conf])
cmap = pl.get_cmap('jet')
depths_max = max([d.max() for d in depths])
depths = [d / depths_max for d in depths]
confs_max = max([d.max() for d in confs])
confs = [cmap(d / confs_max) for d in confs]

imgs = []
for i in range(len(rgbimg)):
    imgs.append(rgbimg[i])
    imgs.append(rgb(depths[i]))
    imgs.append(rgb(confs[i]))




