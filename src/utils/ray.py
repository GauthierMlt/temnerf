import torch
from kornia import create_meshgrid
import numpy as np

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # Added in the 0.5 pixel correction
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i+0.5-W/2.)/focal, -(j+0.5-H/2.)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
        
    return directions, (i+0.5)/W, (j+0.5)/H


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def create_geometry_vectors_parallel(angles: np.ndarray, 
                                     size: tuple,
                                     detector_spacing_X:float=1.0, 
                                     detector_spacing_Y:float=1.0) -> np.ndarray:

    vectors = np.zeros((len(angles), 12))

    scale_factor_X = size[0]
    scale_factor_Y = size[1]

    for i in range(len(angles)):

        # ray direction
        vectors[i, 0] = np.sin(angles[i])
        vectors[i, 1] = -np.cos(angles[i])
        vectors[i, 2] = 0

        # center of detector
        vectors[i, 3] = 0.5
        vectors[i, 4] = 0.5
        vectors[i, 5] = 0.5

        # vector from detector pixel (0,0) to (0,1)
        vectors[i, 6] = np.cos(angles[i]) * detector_spacing_X / scale_factor_X
        vectors[i, 7] = np.sin(angles[i]) * detector_spacing_X / scale_factor_X
        vectors[i, 8] = 0

        # vector from detector pixel (0,0) to (1,0)
        vectors[i, 9] = 0
        vectors[i, 10] = 0
        vectors[i, 11] = detector_spacing_Y / scale_factor_Y

    return torch.tensor(vectors, dtype=torch.float32)

def compute_rays(angles: list, size: int | tuple, geometry="parallel"):

    if geometry != "parallel":
        raise NotImplementedError()

    n_projections = len(angles)

    size = (size, size) if isinstance(size, int) else size

    u, v = torch.meshgrid(torch.arange(-size[0]//2, size[0]//2), 
                          torch.arange(-size[1]//2, size[1]//2), indexing='ij')

    u = u.reshape(-1)
    v = v.reshape(-1)
    
    geometry_vectors = create_geometry_vectors_parallel(angles, size)

    origins    = torch.zeros((n_projections, size[0] * size[1], 3), dtype=torch.float32)
    directions = torch.zeros((n_projections, size[0] * size[1], 3), dtype=torch.float32)

    for p in range(n_projections):
        ray_direction = geometry_vectors[p, :3]
        det_center    = geometry_vectors[p, 3:6]
        proj_ox       = geometry_vectors[p, 6:9]
        proj_oy       = geometry_vectors[p, 9:12]
        
        pixel_locations = det_center + u[:, None] * proj_ox + v[:, None] * proj_oy

        origins[p]    = pixel_locations 
        directions[p] = -ray_direction 

    origins    = origins.reshape(-1, 3)
    directions = directions.reshape(-1, 3)

    directions_norm = torch.norm(directions, dim=1, keepdim=True)
    directions = directions / directions_norm

    return origins, directions

def get_ray_directions_orthographic(H, W):
    """
    Get ray directions for all pixels in camera coordinate for orthographic projection.
    
    Inputs:
        H, W: image height, width
    
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([torch.zeros_like(i), torch.zeros_like(j), -torch.ones_like(i)], -1)
    return directions, (i+0.5)/W, (j+0.5)/H

def rotation_matrix_y(angle):
    """
    Compute the rotation matrix for a given angle around the Y-axis.

    Inputs:
        angle: Rotation angle in radians.

    Outputs:
        rotation_matrix: (3, 3) rotation matrix around the Y-axis.
    """
    cos_angle = torch.cos(angle).item()
    sin_angle = torch.sin(angle).item()

    rotation_matrix = torch.tensor([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ], dtype=torch.float32)
    return rotation_matrix

