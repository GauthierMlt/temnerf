from torch.utils.data import Dataset
from PIL import Image
from utils.image import get_n_channels
from utils.ray import get_ray_directions_orthographic, \
                      rotation_matrix_y, \
                      get_rays_orthographic, \
                      create_geometry_vectors
import torch
import numpy as np
from data import transforms
from utils.data import load_angles, load_images

class EMDataset(Dataset):

    def __init__(self, 
                 device: torch.device, 
                 images_path: str, 
                 angles_path: str,
                 target_size: int,
                 split: str="train"):
        
        super().__init__()

        self.device: torch.device = device
        self.target_size: int = target_size
        self.data: torch.Tensor = None
        self.geometry_vectors: np.ndarray = None
        self.split: str = split
        self.images_tensor: torch.Tensor = self.load_images(images_path, target_size)
        self.angles_array: np.ndarray    = load_angles(angles_path)

        self.compute_rays_torch()

    def load_images(self, images_path, target_size):
        return torch.tensor( load_images(images_path, target_size) )

    @torch.no_grad
    def compute_rays_torch(self):
        
        self.geometry_vectors = create_geometry_vectors(self.angles_array)

        size = self.images_tensor.shape[2]
        # factor = ( size * np.sqrt(3))  /2
        factor = 10
            
        u, v = torch.meshgrid(torch.arange(-size//2, size//2), torch.arange(-size//2, size//2), indexing='ij')
        # u, v = torch.meshgrid(torch.arange(0, size), torch.arange(0, size), indexing='ij')
        u = u.reshape(-1)
        v = v.reshape(-1)

        u = (u + size // 2) / size
        v = (v + size // 2) / size

        # u = (u / (size // 2)).float()
        # v = (v / (size // 2)).float()
        
        vecs = torch.tensor(self.geometry_vectors, dtype=torch.float32)
        n_projections = vecs.shape[0]

        origins = torch.zeros((n_projections, size * size, 3), dtype=torch.float32)
        directions = torch.zeros((n_projections, size * size, 3), dtype=torch.float32)
        
        for p in range(n_projections):
            ray_direction = vecs[p, :3]
            det_center = vecs[p, 3:6]
            # det_center = torch.tensor([0.5, 0.5, 0.5])
            width_vec = vecs[p, 6:9]
            height_vec = vecs[p, 9:12]
            
            pixel_locations = det_center + u[:, None] * width_vec + v[:, None] * height_vec
            origins[p] = pixel_locations + ray_direction * factor
            directions[p] = -ray_direction
    
        origins    = origins.reshape(-1, 3)
        directions = directions.reshape(-1, 3)

        directions_norm = torch.norm(directions, dim=1, keepdim=True)
        directions = directions / directions_norm

        # Optionally, normalize origins
        origins_norm = torch.norm(origins, dim=1, keepdim=True)
        origins = origins / origins_norm

        # pixel_values = self.images_tensor.permute(0, 2, 3, 1).reshape(-1, self.images_tensor.shape[1])
        # pixel_values = self.images_tensor.flatten()

        # from utils.debug import plot_rays
        # plot_rays(origins.cpu(), directions.cpu(), show_directions=True, directions_scale=10, directions_subsample=500, show_box=False)

        # print("*"*50)
        # print(torch.max(self.images_tensor))

        self.data = torch.cat([
            origins,
            directions,
            self.images_tensor.flatten().unsqueeze(1)
        ], dim=1).to(self.device)


    def __len__(self):
        return self.images_tensor.numel()

    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx]
        else:
            rays_per_image = self.images_tensor.shape[2] * self.images_tensor.shape[3]
            start_idx = idx * rays_per_image
            end_idx = start_idx + rays_per_image
            return self.data[start_idx:end_idx].cpu()
    


    # @torch.no_grad
    # def compute_rays(self):
    #     self.geometry_vectors = EMDataset.create_geometry_vectors(self.angles_array)

    #     size = self.images_tensor.shape[2]
    #     factor = (size * np.sqrt(3)) / 2

    #     u, v = np.meshgrid(np.arange(-size//2, size//2), np.arange(-size//2, size//2), indexing='ij')
    #     u = u.reshape(-1)
    #     v = v.reshape(-1)
        
    #     origins = np.zeros((self.geometry_vectors.shape[0], size*size, 3))
    #     directions = np.zeros((self.geometry_vectors.shape[0], size*size, 3))
        
    #     for p in range(self.geometry_vectors.shape[0]):

    #         ray_direction   = self.geometry_vectors[p, :3]
    #         det_center      = self.geometry_vectors[p, 3:6]
    #         width_vec       = self.geometry_vectors[p, 6:9]
    #         height_vec      = self.geometry_vectors[p, 9:12]
            
    #         pixel_locations = det_center + u[:, None] * width_vec + v[:, None] * height_vec
    #         origins[p] = pixel_locations + ray_direction * factor
    #         directions[p] = -ray_direction
        
    #     origins    = origins.reshape(-1, 3)
    #     directions = directions.reshape(-1, 3)

    #     # pixel_values = self.images_tensor.permute(0, 2, 3, 1).reshape(-1, self.images_tensor.shape[1])
    #     self.data = torch.cat([
    #         torch.tensor(origins, dtype=torch.float32),
    #         torch.tensor(directions, dtype=torch.float32),
    #         self.images_tensor.flatten().unsqueeze(1)
    #     ], dim=1).to(self.device)


    # def compute_rays(self):

    # @torch.no_grad
    # def compute_rays(self):
    #     self.geometry_vectors = EMDataset.create_geometry_vectors(self.angles_array)

    #     size = self.images_tensor.shape[2]
    #     factor = (size * np.sqrt(3)) / 2

    #     u, v = np.meshgrid(np.arange(-size//2, size//2), np.arange(-size//2, size//2), indexing='ij')
    #     u = u.reshape(-1)
    #     v = v.reshape(-1)
        
    #     origins = np.zeros((self.geometry_vectors.shape[0], size*size, 3))
    #     directions = np.zeros((self.geometry_vectors.shape[0], size*size, 3))
        
    #     for p in range(self.geometry_vectors.shape[0]):

    #         ray_direction   = self.geometry_vectors[p, :3]
    #         det_center      = self.geometry_vectors[p, 3:6]
    #         width_vec       = self.geometry_vectors[p, 6:9]
    #         height_vec      = self.geometry_vectors[p, 9:12]
            
    #         pixel_locations = det_center + u[:, None] * width_vec + v[:, None] * height_vec
    #         origins[p] = pixel_locations + ray_direction * factor
    #         directions[p] = -ray_direction
        
    #     origins    = origins.reshape(-1, 3)
    #     directions = directions.reshape(-1, 3)

    #     # pixel_values = self.images_tensor.permute(0, 2, 3, 1).reshape(-1, self.images_tensor.shape[1])
    #     self.data = torch.cat([
    #         torch.tensor(origins, dtype=torch.float32),
    #         torch.tensor(directions, dtype=torch.float32),
    #         self.images_tensor.flatten().unsqueeze(1)
    #     ], dim=1).to(self.device)


    # def compute_rays(self):
    #     self.directions, self.pix_x, self.pix_y = get_ray_directions_orthographic(self.images_tensor.shape[2], self.images_tensor.shape[3])

    #     self.all_imgs = self.images_tensor.permute(2, 3, 0, 1).reshape(-1).unsqueeze(1)
    #     num_rays_per_image = self.images_tensor.shape[2] * self.images_tensor.shape[3]  # h * w
    #     total_rays = len(self.angles_array) * num_rays_per_image

    #     self.all_rays = torch.zeros(total_rays, 6, device=self.images_tensor.device)

    #     start_idx = 0
    #     for i in range(len(self.angles_array)):
    #         rays_o, rays_d = self.process_rays(self.angles_array[i])
    #         rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
    #         end_idx = start_idx + num_rays_per_image
    #         self.all_rays[start_idx:end_idx, :] = rays
    #         start_idx = end_idx

    #     self.data = torch.cat((self.all_rays[:,:6],self.all_imgs[:,:]),1).to(self.device)
        

    # def process_rays(self, angle):
    #     """
    #     Process rays for a given frame angle around the object.

    #     Inputs:
    #         angle: Rotation angle in radians around the Y-axis.

    #     Outputs:
    #         rays_o: (h*w, 3), the origin of the rays in world coordinate.
    #         rays_d: (h*w, 3), the normalized direction of the rays in world coordinate.
    #     """
        
    #     angle_rad = torch.deg2rad(angle)
    #     rotation = rotation_matrix_y(angle_rad)

    #     # camera 10 unit away from the origin along the Z-axis
    #     translation = torch.tensor([0, 0, -10], dtype=torch.float32).view(3, 1)
    #     c2w = torch.cat((rotation, translation), dim=1)  # (3, 4)
        
    #     rays_o, rays_d = get_rays_orthographic(self.directions, c2w)  # both (h*w, 3)

    #     # Rescale scene so that object fits in a [0, 1] box
    #     rays_o = 0.5 * rays_o / self.scale + 0.5 
    #     rays_d = rays_d / self.scale
    
    #     return rays_o, rays_d

    # def process_rays(self, angle):
    #     """
    #     Process rays for a given frame angle around the object.

    #     Inputs:
    #         angle: Rotation angle in radians around the Y-axis.

    #     Outputs:
    #         rays_o: (h*w, 3), the origin of the rays in world coordinate.
    #         rays_d: (h*w, 3), the normalized direction of the rays in world coordinate.
    #     """
        
    #     angle_rad = torch.deg2rad(angle)
    #     rotation = rotation_matrix_y(angle_rad)

    #     # camera 10 unit away from the origin along the Z-axis
    #     translation = torch.tensor([0, 0, -10], dtype=torch.float32).view(3, 1)
    #     c2w = torch.cat((rotation, translation), dim=1)  # (3, 4)
        
    #     rays_o, rays_d = get_rays_orthographic(self.directions, c2w)  # both (h*w, 3)

    #     # Rescale scene so that object fits in a [0, 1] box
    #     rays_o = 0.5 * rays_o / self.scale + 0.5 
    #     rays_d = rays_d / self.scale
    
    #     return rays_o, rays_d