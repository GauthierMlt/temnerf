import numpy as np
from skimage import exposure, transform # supports 16-bit tiff
import time
from torchvision.transforms import v2
import torch


def walnut(img, shape):
    # Walnut data needs to be transformed 5 pixel to the left
    img = np.roll(img, -5, axis=1)
    img = -np.log(img)

    # img = transform.resize(img, shape, anti_aliasing=True)

    # New version: much faster for smaller images
    img = v2.Resize(shape)(torch.from_numpy(img).reshape(1, 1, *img.shape)).squeeze().numpy()

    # subtract off background values
    background = 0.25*(img[0,0]+img[shape[0]-1,0]+img[shape[0]-1,shape[1]-1]+img[0,shape[1]-1])
    img -= background
    # CLAMP BACKGROUND VALUES TO ZERO
    # nerf is overfitting to noise, so clamp
    # any shot noise to zero by zeroing small values
    dark_floor = 0.05 # determine empirically

    img = exposure.rescale_intensity(img, in_range=(dark_floor, 1.8), out_range=(0.,1.))

    return img

def jaw(img, shape):
    img = transform.resize(img, shape, anti_aliasing=False)
    img = np.float32(img)
    return img

def au_ag(img: np.ndarray, shape: int | tuple, normalize=False):
    
    img = v2.Resize(shape)(torch.from_numpy(img).reshape(1, 1, *img.shape)).squeeze()

    factor = 255. if normalize else 1.

    img = (img.to(device='cuda').to(dtype=torch.float32)/factor).contiguous()
    
    # img = torch.permute(img, (1, 0))
    

    if len(img.shape) == 2:
        img = img.unsqueeze(0) # (C, H, W) 

    return img
