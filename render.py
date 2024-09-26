import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
import random
import clip
from os import makedirs
from gaussian_renderer import render, render_static
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from sklearn.decomposition import PCA
from utils.sh_utils import RGB2SH
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.transform import resize
from scipy.ndimage import binary_fill_holes
import cv2

# python render.py -s data/marble -m output/test

def segment_hand(rendering_mask, rendering_image, output_path, manual_thresh=0.1):
    # Convert images to numpy arrays for further processing
    mask_np = np.array(rendering_mask)
    original_image_np = np.array(rendering_image)

    # mask_image= rendering_mask
    # original_image= rendering_image

    # Convert images to numpy arrays for further processing
    mask_np = np.array(mask_np)
    original_image_np = np.array(original_image_np)

    # Remove the alpha channel from mask if present
    if mask_np.shape[-1] == 4:
        mask_np = mask_np[:, :, :3]

    # Convert mask to grayscale
    mask_gray = rgb2gray(mask_np)

    # Apply Gaussian blur to the grayscale mask
    mask_blur = cv2.GaussianBlur(mask_gray, (5, 5), 0)

    # Apply Otsu's threshold to create a binary mask or use manual threshold if provided
    if manual_thresh is None:
        thresh = threshold_otsu(mask_blur)
    else:
        thresh = manual_thresh
    binary_mask = mask_blur > thresh

    # Fill the holes inside the binary mask to create a complete mask
    filled_mask = binary_fill_holes(binary_mask).astype(np.uint8)

    # Perform morphological operations to clean up the mask
    binary_mask_cleaned = binary_closing(filled_mask, disk(3))
    binary_mask_cleaned = remove_small_objects(binary_mask_cleaned, min_size=500)

    # Resize the binary mask to match the dimensions of the original image
    binary_mask_resized = resize(binary_mask_cleaned.astype(float), original_image_np.shape[:2], anti_aliasing=False)
    binary_mask_resized = binary_mask_resized > 0.5  # Convert to binary

    # Apply the resized binary mask to the original image to segment the hand
    segmented_image_resized = original_image_np.copy()
    segmented_image_resized[~binary_mask_resized] = 0  # Set background pixels to 0

    # Save the segmented image
    segmented_image = Image.fromarray(segmented_image_resized.astype(np.uint8))
    segmented_image.save(output_path)
    
def render_set_train(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    hands_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hands_mask")
    fix_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fix")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    fix_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fix_mask")
    hands_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hands_segement")
    fix_hands_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fix_hands_mask")

    static_path = os.path.join(model_path, name, "ours_{}".format(iteration), "static_scene")
    fix_static_path =  os.path.join(model_path, name, "ours_{}".format(iteration), "fix_static_scene")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(hands_mask_path, exist_ok=True)
    os.makedirs(fix_render_path, exist_ok=True)
    os.makedirs(fix_mask_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(hands_path, exist_ok=True)
    os.makedirs(fix_hands_path, exist_ok=True)


    os.makedirs(static_path, exist_ok=True)
    os.makedirs(fix_static_path, exist_ok=True)

    t_list = []

    fix_view = views[68]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
            fix_view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        mask = results["feature_map"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)  # Normalize the depth values
        depth = 1 - depth  # Invert the normalized depth values

        results_static= render_static(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        static_rendering = results_static["render"]
        fix_static_results = render_static(fix_view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        fix_static_rendering = fix_static_results["render"]

        fix_view_results =  render(fix_view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        fix_rendering =  fix_view_results["render"]
        fix_mask = fix_view_results["feature_map"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask, os.path.join(hands_mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fix_rendering, os.path.join(fix_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fix_mask, os.path.join(fix_mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(static_rendering, os.path.join(static_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fix_static_rendering, os.path.join(fix_static_path , '{0:05d}'.format(idx) + ".png"))  

        # Load the rendering mask 
        image_path = os.path.join(render_path, '{0:05d}'.format(idx) + ".png")
        segement_mask_path = os.path.join(hands_mask_path, '{0:05d}'.format(idx) + ".png")
        fix_view_path = os.path.join(fix_render_path, '{0:05d}'.format(idx) + ".png")
        fix_segement_path = os.path.join(fix_mask_path, '{0:05d}'.format(idx) + ".png")

        loaded_rendering_image = Image.open(image_path) 
        loaded_rendering_mask = Image.open(segement_mask_path)   
        loaded_rendering_fix_image = Image.open(fix_view_path)
        loaded_rendering_fix_segement = Image.open(fix_segement_path)


        segment_hand(loaded_rendering_mask , loaded_rendering_image, os.path.join(hands_path, '{0:05d}_segmented.png'.format(idx)))
        segment_hand(loaded_rendering_fix_segement, loaded_rendering_fix_image, os.path.join(fix_hands_path, '{0:05d}_segmented.png'.format(idx)))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')


def render_set_test(model_path, load2gpu_on_the_fly, is_6dof, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    hands_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hands_mask")
    fix_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fix")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    fix_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fix_mask")
    hands_path = os.path.join(model_path, name, "ours_{}".format(iteration), "hands_segement")
    fix_hands_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fix_hands_mask")

    static_path = os.path.join(model_path, name, "ours_{}".format(iteration), "static_scene")
    fix_static_path =  os.path.join(model_path, name, "ours_{}".format(iteration), "fix_static_scene")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(hands_mask_path, exist_ok=True)
    os.makedirs(fix_render_path, exist_ok=True)
    os.makedirs(fix_mask_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(hands_path, exist_ok=True)
    os.makedirs(fix_hands_path, exist_ok=True)


    os.makedirs(static_path, exist_ok=True)
    os.makedirs(fix_static_path, exist_ok=True)

    t_list = []

    fix_view = views[10]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpu_on_the_fly:
            view.load2device()
            fix_view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz

        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        rendering = results["render"]
        mask = results["feature_map"]
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)  # Normalize the depth values
        depth = 1 - depth  # Invert the normalized depth values

        results_static= render_static(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        static_rendering = results_static["render"]
        fix_static_results = render_static(fix_view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        fix_static_rendering = fix_static_results["render"]

        fix_view_results =  render(fix_view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)
        fix_rendering =  fix_view_results["render"]
        fix_mask = fix_view_results["feature_map"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mask, os.path.join(hands_mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fix_rendering, os.path.join(fix_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fix_mask, os.path.join(fix_mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        torchvision.utils.save_image(static_rendering, os.path.join(static_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fix_static_rendering, os.path.join(fix_static_path , '{0:05d}'.format(idx) + ".png"))  

        # Load the rendering mask 
        image_path = os.path.join(render_path, '{0:05d}'.format(idx) + ".png")
        segement_mask_path = os.path.join(hands_mask_path, '{0:05d}'.format(idx) + ".png")
        fix_view_path = os.path.join(fix_render_path, '{0:05d}'.format(idx) + ".png")
        fix_segement_path = os.path.join(fix_mask_path, '{0:05d}'.format(idx) + ".png")

        loaded_rendering_image = Image.open(image_path) 
        loaded_rendering_mask = Image.open(segement_mask_path)   
        loaded_rendering_fix_image = Image.open(fix_view_path)
        loaded_rendering_fix_segement = Image.open(fix_segement_path)


        segment_hand(loaded_rendering_mask , loaded_rendering_image, os.path.join(hands_path, '{0:05d}_segmented.png'.format(idx)))
        segment_hand(loaded_rendering_fix_segement, loaded_rendering_fix_image, os.path.join(fix_hands_path, '{0:05d}_segmented.png'.format(idx)))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)

        torch.cuda.synchronize()
        t_start = time.time()

        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, is_6dof)

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {xyz.shape[0]}')


def render_sets(dataset: ModelParams, opt: OptimizationParams, iteration: int, pipeline: PipelineParams,  skip_train: bool, skip_test: bool, frame : int, points : list, thetas : list, prompt : str, novel_views : int):
                
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, semantic_feature_dim = dataset.semantic_dimension)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(dataset.is_blender, dataset.is_6dof)
        deform.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
            render_set_train(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "train", scene.loaded_iter,
                    scene.getTrainCameras(), gaussians, pipeline, background, deform,)
        if not skip_test:        
            render_set_test(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, "test", scene.loaded_iter,
                    scene.getTestCameras(), gaussians, pipeline, background, deform,)
            


        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    optim = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--points', nargs='+', default=None)
    parser.add_argument('--thetas', nargs='+', default=None)
    parser.add_argument('--prompt', nargs='+', default=None)
    args, _ = parser.parse_known_args()
    args.sh_degree = 3
    args.images = 'images'
    args.data_device = "cuda"
    args.resolution = -1
    args.eval = True

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), optim.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test,
        args.frame, args.points, args.thetas, args.prompt, args.novel_views)
