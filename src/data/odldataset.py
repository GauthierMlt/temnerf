from torch.utils.data import Dataset
from utils.data import load_angles, load_images
import numpy as np
import torch

def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


class ODLDataset(Dataset):

    def __init__(self, images_path, angles_path, new_proj_size):
        super().__init__()

        self.images: torch.Tensor = torch.tensor( load_images(images_path, new_proj_size) )
        self.angles: np.ndarray    = load_angles(angles_path)
        self.new_proj_size: tuple = (new_proj_size, new_proj_size, new_proj_size)

        self.images = self.images.permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        grid = create_grid_3d(*self.new_proj_size)
        return grid, self.images
    
    def __len__(self):
        return 1