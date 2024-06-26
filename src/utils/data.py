import numpy as np
from data import transforms
from utils.image import get_n_channels
from PIL import Image
import torch
import os
from torchvision.transforms import ToPILImage

def load_images(images_path: str, target_size: int) -> np.ndarray:

    with Image.open(images_path) as im:

        assert im.mode == "L", \
                f"Image mode '{im.mode}' different than the expected 'L' (8-bit grayscale)"

        n_channels   =  get_n_channels(im.mode)
        tensor_shape = (im.n_frames, n_channels, target_size, target_size)

        images = np.zeros(tensor_shape, dtype=np.uint8) # (frame, C, H, W)

        for i in range(im.n_frames):
            im.seek(i)
            np_array = np.array(im)  # (H,W) if 1 channel, else (H, W, C) 
            image = transforms.au_ag(np_array, target_size)
                
            images[i] = image.transpose(2, 0, 1) #(C, H, W)

    
    # assert np.all((images >= 0.0) & (images <= 1.0)), \
    #         'Tensor values are not normalized properly'
    
    return images


def load_angles(angles_path: str) -> np.ndarray:

    with open(angles_path, 'r') as file:
        lines = file.readlines()

    return np.deg2rad(np.array([float(line.strip()) for line in lines], dtype=np.float64))


def save_slices_as_tiff(train_output, iteration, output_dir):

    with torch.no_grad():
        z_slice_index = train_output.shape[1] // 2  # Middle slice along z-axis
        y_slice_index = train_output.shape[2] // 2  # Middle slice along y-axis
        x_slice_index = train_output.shape[3] // 2  # Middle slice along x-axis

        z_slice = train_output[0, z_slice_index, :, :, 0].clamp(0,1).cpu().numpy()
        y_slice = train_output[0, :, y_slice_index, :, 0].clamp(0,1).cpu().numpy()
        x_slice = train_output[0, :, :, x_slice_index, 0].clamp(0,1).cpu().numpy()

        # Convert slices to PIL Images
        z_slice_img = Image.fromarray((z_slice * 255).astype('uint8'))
        y_slice_img = Image.fromarray((y_slice * 255).astype('uint8'))
        x_slice_img = Image.fromarray((x_slice * 255).astype('uint8'))

        # Define file paths
        z_slice_path = os.path.join(output_dir, f'z_slice_{iteration + 1}.tiff')
        y_slice_path = os.path.join(output_dir, f'y_slice_{iteration + 1}.tiff')
        x_slice_path = os.path.join(output_dir, f'x_slice_{iteration + 1}.tiff')

        # Save images as TIFF files
        z_slice_img.save(z_slice_path)
        y_slice_img.save(y_slice_path)
        x_slice_img.save(x_slice_path)

def save_tensor_as_image(tensor, iter, directory, prefix="train_proj", file_format="png"):
    os.makedirs(directory, exist_ok=True)
    
    # Normalize the tensor to 0-1 range
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Convert tensor to PIL Image and save each image
    to_pil_image = ToPILImage()
    batch_size, num_images, height, width = tensor.size()
    
    for i in range(batch_size):
        for j in range(num_images):
            img = to_pil_image(tensor[i, j])
            img.save(os.path.join(directory, f"{iter:4g}_{prefix}_batch{i}_img{j}.{file_format}"))