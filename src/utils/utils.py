import numpy as np
import torch
import random
import os
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def list_available_devices():
    if torch.cuda.is_available():
        gpu_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        gpu_devices = []

    cpu_device = "CPU"

    devices = {
        "CPU": cpu_device,
        "CUDA": gpu_devices
    }

    return devices

def desktop_environment_available():
    if os.name == 'nt':
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0) != 0
    else:
        return os.getenv('DISPLAY') is not None
    
def get_model(encoding_config, network_config):
    
    try:
        import tinycudann as tcnn
        TINY_CUDA = True
    except ImportError:
        print("tiny-cuda-nn not found.  Performance will be significantly degraded.")
        print("You can install tiny-cuda-nn from https://github.com/NVlabs/tiny-cuda-nn")
        print("============================================================")
        TINY_CUDA = False

    from models.nerf import Nerf
    
    if TINY_CUDA:
        return tcnn.NetworkWithInputEncoding(n_input_dims=3, 
                                              n_output_dims=1, 
                                              encoding_config=encoding_config, 
                                              network_config=network_config)
    else: 
        return Nerf(encoding_config, network_config)
