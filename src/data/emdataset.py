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

        self.n_projections: int     = n_projections
        self.target_size: int       = target_size
        self.device: torch.device   = device
        self.split: str             = split

        self.images: torch.Tensor   = None
        self.angles: np.ndarray     = None
        self.data: torch.Tensor     = None
        
        self.read_data(images_path, angles_path, n_projections)
        self.init_data()


    def read_data(self, images_path, angles_path, n_projections):
        
        self.images = load_images(images_path, self.target_size)
        self.angles = load_angles(angles_path)

        if n_projections:

            if n_projections >= len(self.angles):
                return

            indices = np.linspace(0, len(self.angles)-1, n_projections, endpoint=True, dtype=int)
            
            self.images = self.images[indices]
            self.angles = self.angles[indices]


    @torch.no_grad
    def init_data(self):

        ray_origins, ray_directions = compute_rays(self.angles, self.target_size)

        self.data = torch.cat([
            ray_origins,
            ray_directions,
            self.images.flatten().unsqueeze(1)
        ], dim=1)


    def __len__(self):
        return self.images.numel()


    def __getitem__(self, idx):
        if self.split == "train":
            return self.data[idx]
        else:
            size = self.images.shape[2] * self.images.shape[3]
            start_idx = idx * size
            end_idx   = start_idx + size
            return self.data[start_idx:end_idx].cpu()
        
    