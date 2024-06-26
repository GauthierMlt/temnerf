import numpy as np
from tifffile import imsave, imwrite
import matplotlib.pyplot as plt
# Load the reconstructed volume from the file
reconstructed_volume = np.load('reconstructed_volume.npy')

# Save specific slices of the reconstructed volume as TIFF files
# Specify the slices you want to save
slices_to_save = [
    reconstructed_volume.shape[0] // 4,
    reconstructed_volume.shape[0] // 2,
    3 * reconstructed_volume.shape[0] // 4
]

# Save each slice as a TIFF file
for idx, slice_data in enumerate(reconstructed_volume[10::5]):
# for idx in range(0, len(reconstructed_volume[0]), 10):
    file_name = f'reconstructed_slice_{idx}.tiff'
    # imsave(file_name, slice_data)
    # plt.imshow(reconstructed_volume[:,:, idx])
    plt.imshow(slice_data)
    plt.show()
    # print(f'Saved slice {idx} to {file_name}')

# imwrite(f'reconstructed_slices.tiff', reconstructed_volume[::50], bigtiff=True)