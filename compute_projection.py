import os
import argparse
import yaml
import torch
from utils.utils import get_model
from utils.ray import compute_rays
from utils.render import render_image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():

    class ParseValues(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):

            if os.path.isfile(values):
                with open(values, 'r') as file:
                    content = file.read()
                    float_values = list(map(float, content.split()))
            else:
                try:
                    float_values = list(map(float, values.split(",")))
                except:
                    parser.exit(1,
                                f"ERROR: The argument given for the angles ({values}) is neither a comma-separated "
                                "list of numbers nor a filepath containing angles (separated by spaces or newlines)")
            
            setattr(namespace, self.dest, float_values)

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint',  type=str, help="Path to the model's weight")
    parser.add_argument('--config', type=str, help="Path to the config")
    parser.add_argument('--angles', 
                        action=ParseValues,
                        help="A comma-separated list of floats (e.g 20,30.5,0) or path to a text file containing"
                             "the angles (separated by spaces or newlines)")
    parser.add_argument('--uint8', type=bool, default=False, help="Convert the values to uint8")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Loading libraries...", end="\r", flush=True)
    import torch
    from utils.utils import get_model
    from utils.ray import compute_rays
    from utils.render import render_image
    print("Loading libraries: Done")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    output_dir = os.path.join("Projections", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(output_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["img_size"]

    model = get_model(cfg["encoding"], cfg["network"])
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    near = -1 / (np.sqrt(2))
    far  =  1 / (np.sqrt(2))

    # near = -1 
    # far  =  1 

    projections = torch.zeros((len(args.angles), size, size), dtype=torch.float)
    
    for angle in args.angles:

        ray_origins, ray_directions = compute_rays([np.deg2rad(angle)], size)
        data = torch.cat([ray_origins, ray_directions], dim=1)
        
        print(f"Computing projection at angle {angle} degrees", end="\r", flush=True)

        img = render_image( model, 
                            data, 
                            **{
                                "device": device,
                                "H": size,
                                "W": size,
                                "hf": far,
                                "hn": near,
                                "nb_bins": cfg["output"]["samples_per_ray"]
                            })
        
        img = img.clamp(0., 1.).reshape(size, size)

        if args.uint8:
            img = (img*255.).to(torch.uint8)
        
        save_path = os.path.join(output_dir, f"{angle:3g}_deg.tiff")
        plt.imsave(save_path, img, cmap="Greys_r")
        plt.show()
        print(f"Computing projection at angle {angle} degrees: Done")