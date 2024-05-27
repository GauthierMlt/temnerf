from skimage import io
from PIL import Image
import numpy as np

class NerfImage():
    def __init__(self, path, img_transform, noise_level, im_wh):
        numpy_image = io.imread(path)
        dtype = numpy_image.dtype
        assert (dtype == 'uint8') or (dtype == 'uint16'), 'unknown datatype'
        if numpy_image.ndim == 3:
            numpy_image = numpy_image[:,:,1] # only keep the first colour channel if RGB
        h, w = numpy_image.shape 
        assert h > w, 'data might be wrong way round!'

        img = img_transform(numpy_image, im_wh)

        if noise_level > 0:
            noise_sd = 64 # noise centred at 128, sd can be changed if desired
            noise = Image.effect_noise((im_wh[1], im_wh[0]), noise_sd) # pillow and numpy shapes are swapped round
            noise = np.array(noise, dtype=np.float32)  /256. # rescale
            img = (img + noise_level*(noise - 0.5)).clip(0,1) # noise centred at 0.5

        self.image = img
        self.h, self.w = self.image.shape
        assert self.h > self.w, 'data might be wrong way round!'

    def get_pixel_normalised(self, x, y):
        # read 5 pixels to right (circular)
        assert 0. <= x <= 1., 'img coords must be normalised'
        assert 0. <= y <= 1., 'img coords must be normalised'

        x *= self.w
        y *= self.h

        return self.image[int(y), int(x)]