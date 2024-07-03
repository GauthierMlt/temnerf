from utils.data import load_angles, load_images
from torch.utils.data import Dataset
from utils.ray import compute_rays
import numpy as np
import torch

class EMDataset(Dataset):

    def __init__(self, 
                 device: torch.device, 
                 images_path: str, 
                 angles_path: str,
                 target_size: int,
                 n_projections=None,
                 split: str="train"):
        
        super().__init__()

        self.n_projections: int = n_projections
        self.device: torch.device = device
        self.target_size: int = target_size
        self.split: str = split

        self.geometry_vectors: np.ndarray = None
        self.data: torch.Tensor = None
        self.images_tensor: torch.Tensor = None
        self.angles_array: np.ndarray    = None
        
        self.read_data(images_path, angles_path, n_projections)
        self.init_data()

    def read_data(self, images_path, angles_path, n_projections):
        
        self.images_tensor = load_images(images_path, self.target_size)
        self.angles_array  = load_angles(angles_path)

        if n_projections:

            if n_projections >= len(self.angles_array):
                return

            indices = np.linspace(0, len(self.angles_array)-1, n_projections, dtype=int)
            
            self.images_tensor = self.images_tensor[indices]
            self.angles_array  = self.angles_array[indices]
    
    @torch.no_grad
    def init_data(self):


        ray_origins, ray_directions = compute_rays(self.angles_array, self.target_size)

        self.data = torch.cat([
            ray_origins,
            ray_directions,
            self.images_tensor.flatten().unsqueeze(1)
        ], dim=1).to(self.device)


    # @torch.no_grad
    # def compute_rays_torch(self):
        
    #     self.geometry_vectors = create_geometry_vectors(self.angles_array, size=self.images_tensor.shape[2])

    #     size = self.images_tensor.shape[2]
            
    #     u, v = torch.meshgrid(torch.arange(-size//2, size//2), torch.arange(-size//2, size//2), indexing='ij')

    #     u = u.reshape(-1)
    #     v = v.reshape(-1)
        
    #     vecs = torch.tensor(self.geometry_vectors, dtype=torch.float32)
    #     n_projections = vecs.shape[0]

    #     origins = torch.zeros((n_projections, size * size, 3), dtype=torch.float32)
    #     directions = torch.zeros((n_projections, size * size, 3), dtype=torch.float32)
        
    #     for p in range(n_projections):
    #         ray_direction = vecs[p, :3]
    #         det_center = vecs[p, 3:6]
    #         proj_ox = vecs[p, 6:9]
    #         proj_oy = vecs[p, 9:12]
            
    #         pixel_locations = det_center + u[:, None] * proj_ox + v[:, None] * proj_oy
    #         origins[p] = pixel_locations 
    #         directions[p] = -ray_direction 

    #     origins    = origins.reshape(-1, 3)
    #     directions = directions.reshape(-1, 3)

    #     directions_norm = torch.norm(directions, dim=1, keepdim=True)
    #     directions = directions / directions_norm
    #     from utils.debug import plot_rays
    #     # plot_rays(origins.cpu()[::20], directions.cpu()[::20], show_directions=True, directions_scale=2, directions_subsample=20, show_box=False, coordinate_frame_size=0.1)

    #     # print( self.images_tensor.flatten().max(), self.images_tensor.flatten().min() )
    #     self.data = torch.cat([
    #         origins,
    #         directions,
    #         self.images_tensor.flatten().unsqueeze(1)
    #     ], dim=1).to(self.device)


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
        
    