#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous
import numpy as np
from scipy.ndimage import gaussian_filter

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def gaussian_separate(gaussians, percentage=75):
    
    mask_feature = gaussians.get_semantic_feature
    # Take the absolute value of each element in mask_feature
    abs_mask_feature = torch.abs(mask_feature)
    
    # Sum each row along the third dimension (axis=2)
    summed_data = torch.sum(abs_mask_feature, dim=2)
    
    # Move the tensor to the CPU and convert to NumPy
    summed_data_np = summed_data.cpu().numpy()
    
    # Flatten the summed data
    flattened_summed_data = summed_data_np.flatten()
    
    # Calculate the 95% upper bound
    upper_bound = np.percentile(flattened_summed_data,percentage)
    
    # Create boolean masks for within and outside the principal distribution range
    within_bound_mask = summed_data_np <= upper_bound
    outside_bound_mask = summed_data_np > upper_bound
    
    # Convert boolean masks to indices
    within_bound_indices = np.where(within_bound_mask)[0]
    outside_bound_indices = np.where(outside_bound_mask)[0]

    within_bound_indices = torch.tensor(within_bound_indices, dtype=torch.long)
    outside_bound_indices = torch.tensor(outside_bound_indices, dtype=torch.long)
    
    return within_bound_indices, outside_bound_indices  



def apply_high_pass_filter(data, sigma):
    # Apply a low-pass Gaussian filter
    low_pass = gaussian_filter(data.cpu().numpy(), sigma=sigma)
    low_pass = torch.tensor(low_pass, device=data.device)
    
    # Compute the high-pass filter by subtracting the low-pass filtered data
    high_pass = data - low_pass
    return high_pass

def gaussian_separate_with_filter(gaussians, sigma=1.0, percentage=70):
    """
    Separate the Gaussians based on their semantic features using a high-pass filter.

    Args:
        gaussians (GaussianModel): The Gaussian model containing the semantic features.
        sigma (float): The standard deviation for the Gaussian kernel used in the high-pass filter.
        percentage (float): The percentile threshold to separate the Gaussians.

    Returns:
        within_bound_indices (torch.Tensor): Indices of Gaussians within the principal distribution range.
        outside_bound_indices (torch.Tensor): Indices of Gaussians outside the principal distribution range.
    """
    # Extract the semantic features from the Gaussian model
    mask_feature = gaussians.get_semantic_feature
    
    # Apply high-pass filter to the semantic features
    high_pass_feature = apply_high_pass_filter(mask_feature, sigma=sigma)
    
    # Take the absolute value of each element in high_pass_feature
    abs_high_pass_feature = torch.abs(high_pass_feature)
    
    # Sum each row along the third dimension (axis=2)
    summed_data = torch.sum(abs_high_pass_feature, dim=2)
    
    # Move the tensor to the CPU and convert to NumPy
    summed_data_np = summed_data.cpu().numpy()
    
    # Flatten the summed data
    flattened_summed_data = summed_data_np.flatten()
    
    # Calculate the percentile upper bound
    upper_bound = np.percentile(flattened_summed_data, percentage)
    
    # Create boolean masks for within and outside the principal distribution range
    within_bound_mask = summed_data_np <= upper_bound
    outside_bound_mask = summed_data_np > upper_bound
    
    # Convert boolean masks to indices
    within_bound_indices = np.where(within_bound_mask)[0]
    outside_bound_indices = np.where(outside_bound_mask)[0]

    # Convert indices to torch tensors
    within_bound_indices = torch.tensor(within_bound_indices, dtype=torch.long)
    outside_bound_indices = torch.tensor(outside_bound_indices, dtype=torch.long)
    
    return within_bound_indices, outside_bound_indices

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    semantic_feature = pc.get_semantic_feature


    # # Separate Gaussian indices
    # within_bound_indices, outside_bound_indices = gaussian_separate(pc)
    # # Ensure indices are correct
    # within_bound_indices = within_bound_indices.squeeze()
    # outside_bound_indices =outside_bound_indices.squeeze()
    # # Select only the Gaussians within the principal distribution range
    # means3D = means3D[within_bound_indices].squeeze()
    # means2D = means2D[within_bound_indices].squeeze()
    # opacity = opacity[within_bound_indices].squeeze()
    # if scales is not None:
    #     scales = scales[within_bound_indices].squeeze()
    # if rotations is not None:
    #     rotations = rotations[within_bound_indices].squeeze()
    # if cov3D_precomp is not None:
    #     cov3D_precomp = cov3D_precomp[within_bound_indices].squeeze()
    # if shs is not None:
    #     shs = shs[within_bound_indices].squeeze()
    # if colors_precomp is not None:
    #     colors_precomp = colors_precomp[within_bound_indices].squeeze()
    # semantic_feature = semantic_feature[within_bound_indices].squeeze()

    # means3D = means3D[outside_bound_indices].squeeze()
    # means2D = means2D[outside_bound_indices].squeeze()
    # opacity = opacity[outside_bound_indices].squeeze()
    # if scales is not None:
    #     scales = scales[outside_bound_indices].squeeze()
    # if rotations is not None:
    #     rotations = rotations[outside_bound_indices].squeeze()
    # if cov3D_precomp is not None:
    #     cov3D_precomp = cov3D_precomp[outside_bound_indices].squeeze()
    # if shs is not None:
    #     shs = shs[outside_bound_indices].squeeze()
    # if colors_precomp is not None:
    #     colors_precomp = colors_precomp[outside_bound_indices].squeeze()
    # semantic_feature = semantic_feature[outside_bound_indices].squeeze()


    rendered_image, depth, hands_mask_map, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        semantic_feature=semantic_feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            'feature_map': hands_mask_map}


def render_static(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    semantic_feature = pc.get_semantic_feature


    # Separate Gaussian indices
    within_bound_indices, outside_bound_indices = gaussian_separate_with_filter(pc)
    # Ensure indices are correct
    within_bound_indices = within_bound_indices.squeeze()
    outside_bound_indices =outside_bound_indices.squeeze()
    # Select only the Gaussians within the principal distribution range
    means3D = means3D[within_bound_indices].squeeze()
    means2D = means2D[within_bound_indices].squeeze()
    opacity = opacity[within_bound_indices].squeeze()
    if scales is not None:
        scales = scales[within_bound_indices].squeeze()
    if rotations is not None:
        rotations = rotations[within_bound_indices].squeeze()
    if cov3D_precomp is not None:
        cov3D_precomp = cov3D_precomp[within_bound_indices].squeeze()
    if shs is not None:
        shs = shs[within_bound_indices].squeeze()
    if colors_precomp is not None:
        colors_precomp = colors_precomp[within_bound_indices].squeeze()
    semantic_feature = semantic_feature[within_bound_indices].squeeze()

    # means3D = means3D[outside_bound_indices].squeeze()
    # means2D = means2D[outside_bound_indices].squeeze()
    # opacity = opacity[outside_bound_indices].squeeze()
    # if scales is not None:
    #     scales = scales[outside_bound_indices].squeeze()
    # if rotations is not None:
    #     rotations = rotations[outside_bound_indices].squeeze()
    # if cov3D_precomp is not None:
    #     cov3D_precomp = cov3D_precomp[outside_bound_indices].squeeze()
    # if shs is not None:
    #     shs = shs[outside_bound_indices].squeeze()
    # if colors_precomp is not None:
    #     colors_precomp = colors_precomp[outside_bound_indices].squeeze()
    # semantic_feature = semantic_feature[outside_bound_indices].squeeze()


    rendered_image, depth, feature_map, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        semantic_feature=semantic_feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            'feature_map': feature_map}


def render_dynamic(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None, high_pass_sigma=1.0, percentage=85):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    semantic_feature = pc.get_semantic_feature


    # Separate Gaussian indices
    within_bound_indices, outside_bound_indices = gaussian_separate_with_filter(pc, sigma=high_pass_sigma, percentage=percentage)
    # Ensure indices are correct
    within_bound_indices = within_bound_indices.squeeze()
    outside_bound_indices =outside_bound_indices.squeeze()
    # Select only the Gaussians within the principal distribution range
    # means3D = means3D[within_bound_indices].squeeze()
    # means2D = means2D[within_bound_indices].squeeze()
    # opacity = opacity[within_bound_indices].squeeze()
    # if scales is not None:
    #     scales = scales[within_bound_indices].squeeze()
    # if rotations is not None:
    #     rotations = rotations[within_bound_indices].squeeze()
    # if cov3D_precomp is not None:
    #     cov3D_precomp = cov3D_precomp[within_bound_indices].squeeze()
    # if shs is not None:
    #     shs = shs[within_bound_indices].squeeze()
    # if colors_precomp is not None:
    #     colors_precomp = colors_precomp[within_bound_indices].squeeze()
    # semantic_feature = semantic_feature[within_bound_indices].squeeze()

    means3D = means3D[outside_bound_indices].squeeze()
    means2D = means2D[outside_bound_indices].squeeze()
    opacity = opacity[outside_bound_indices].squeeze()
    if scales is not None:
        scales = scales[outside_bound_indices].squeeze()
    if rotations is not None:
        rotations = rotations[outside_bound_indices].squeeze()
    if cov3D_precomp is not None:
        cov3D_precomp = cov3D_precomp[outside_bound_indices].squeeze()
    if shs is not None:
        shs = shs[outside_bound_indices].squeeze()
    if colors_precomp is not None:
        colors_precomp = colors_precomp[outside_bound_indices].squeeze()
    semantic_feature = semantic_feature[outside_bound_indices].squeeze()


    rendered_image, depth, feature_map, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        semantic_feature=semantic_feature,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            'feature_map': feature_map}