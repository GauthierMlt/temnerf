import numpy as np
import os
from PIL import Image
import astra
import cupy as cp
from calculations import make_projection, backprojection, gradient, divergence

data_path = "../data/Au_Ag" 
projections_path = os.path.join(data_path, "Serie_3_ali.tif")
angles_path = os.path.join(data_path, "TiltAngle_ali.txt")
mrc_path = os.path.join(data_path, "Serie_3.mrc")

# projections_path = "/home/multari/temnerf/data/shepp_projections/shepp_projections_512_120_8bit.tiff"
# angles_path = "/home/multari/temnerf/data/shepp_projections/angles_120.txt"


number_of_projections = 241
initial_width = 512
initial_height = 512
d_type = np.uint8

target_size = 512
width = 512
height = 512
slices = 512

def read_angles_file(filepath: str):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    return np.array([float(line.strip()) for line in lines], dtype=np.float32)

def load_images(images_path: str, target_size=128):
    with Image.open(images_path) as im:
        tensor_shape = (im.n_frames, target_size, target_size)
        images_tensor = np.zeros(tensor_shape, dtype=np.uint8)  # (frame, H, W)

        for i in range(im.n_frames):
            im.seek(i)
            img = im.resize((target_size, target_size), Image.LANCZOS)
            img = np.array(img)

            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)  # (H, W, C)

            images_tensor[i] = img.squeeze()

    return images_tensor

def SIRT_FISTA_TV(projections,size_image,proj_geom, vol_geom, lambd, tau, alpha, nbiter, nbiter2) : 
    #SIRT algorithm


    f=cp.zeros(size_image, dtype = 'float32') #n^3
    inv_C = backprojection(cp.ones(projections.shape),vol_geom,proj_geom)
    inv_R = make_projection(cp.ones(size_image),vol_geom,proj_geom) 
    C=1/inv_C #2n^3
    R=1/inv_R
    del inv_C
    del inv_R
    
    t = 1
    
    for _ in range(nbiter):
        
        f_pre = f.copy() #3n^3
        proj = make_projection(f,vol_geom,proj_geom)
        bp_term = R*(projections-proj)
        del proj
        backp=backprojection(bp_term,vol_geom,proj_geom) #4n^3
        f+=lambd*C*backp
        del backp #3n^3
        del bp_term

        f[f<0]=0
        
        #denoising    
        div = cp.zeros_like(f, dtype = 'float32') #4n^3
        j = 0
        assert len(f.shape) == 2 or len(f.shape) == 3, "wrong dimension"
        if len(f.shape) == 2:
            phi = cp.zeros((2,f.shape[0],f.shape[1]))
        elif len(f.shape) == 3:
            phi= cp.zeros((3,f.shape[0],f.shape[1],f.shape[2]))    #7n^3   
        while j < nbiter2 :
            v = div - f/alpha #8n^3
            grad = gradient(v) #11n^3
            del v #10n^3
            denom = 1 + tau*cp.sqrt(cp.sum(grad**2, axis = 0)) #11n^3
            p_update = (phi + tau*grad)/denom #12n^3
            del denom #11n^3
            del grad #8n^3
            phi = p_update
            del p_update #7n^3
            div = divergence(phi) #up to 10n^3
            j += 1
        
        del phi #4n^3
        f =f-alpha*div   
        del div #3n^3
        
        t_pre = t
        t = (1+cp.sqrt(1+4*t**2))/2
        f = f + (t_pre - 1)/t*(f - f_pre)

    return f

angles = np.deg2rad( read_angles_file(angles_path) )
images = load_images(projections_path, target_size=target_size)

pathto = "test_results"

nbiter=100 #maximum number of iterations for the main algorithm

#parameter of regularization
# alpha=0.0005
alpha=0.000

#only for EM_TV and SIRT_TV algorithms
nbiter2=0 #number or iterations in the second loop 

lambd = 1
tau = 0.006

vol_size = target_size

detector_size_y = images.shape[1]
detector_size_x = images.shape[2]

images = np.transpose(images, (1, 0, 2))

vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)


proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, detector_size_y, detector_size_x, angles)
# proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 128, 128, angles, 553.74, 553.74)
imshape = (slices, height, width)

im_SIRTTV = SIRT_FISTA_TV(cp.asarray(images),imshape,proj_geom, vol_geom, lambd, tau, alpha, nbiter, nbiter2)