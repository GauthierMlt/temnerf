# import tinycudann as tcnn
from data.odldataset import ODLDataset
from torch.utils.data import DataLoader
import os
import sys
import torch
import torchvision.utils as vutils

from models.odlnerf import SIREN
from models.encoders import Positional_Encoder
from utils.parallelbeamprojector import ParallelBeam3DProjector
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import utils

from utils.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.data import save_slices_as_tiff, save_tensor_as_image

@hydra.main(config_path="config", config_name="example_odl", version_base="1.2")
def run(_cfg: DictConfig):

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    tf_writer = SummaryWriter(log_dir=out_dir)

    set_seed(_cfg.optim.seed)

    if torch.cuda.is_available():
        print(f"Found {torch.cuda.get_device_name()}.")
    else:
        print(f"Could not find a CUDA device.")
        sys.exit()

    device = torch.device("cuda:0")

    encoder = Positional_Encoder(_cfg.encoding)
    model   = SIREN(_cfg.network).to(device)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)

    batch_size = 1

    target_size = 128
    size_tuple = (target_size,) * 3

    dataset = ODLDataset(_cfg.data.images_path, 
                         _cfg.data.angles_path, 
                         new_proj_size=target_size)

    data_loader = DataLoader(dataset=dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             drop_last=True, 
                             num_workers=0)
    
    max_iter = 500
    proj_idx = list(range(0, 75))
    display_image_num = 8

    ct_projector = ParallelBeam3DProjector(size_tuple, size_tuple, dataset.angles)

    for it, (grid, projs) in enumerate(data_loader):
    # Input coordinates (x,y) grid and the projections
    
        grid = grid.cuda()  # [bs, z, x, y, 3], [0, 1]
        projs = projs.cuda()  # [bs, n, h, w, 1], [0, 1]

        print(grid.shape, projs.shape)

        train_data = (grid, projs)  # [bs, n, h, w]


        # Train model
        for iterations in range(max_iter):
            model.train()
            optimizer.zero_grad()

            train_embedding = encoder.embedding(train_data[0])  # [bs, z, x, y, embedding*2]
            train_output = model(train_embedding)

            train_projs = ct_projector.forward_project(train_output.transpose(1, 4).squeeze(1)) # [bs, n, h, w]
            train_loss = 0.5 * loss(train_projs, train_data[1])

            train_loss.backward()
            optimizer.step()
            # Compute training psnr
            if (iterations + 1) % 10 == 0:
                train_psnr = -10 * torch.log10(2 * train_loss).item()
                train_loss = train_loss.item()
                
                print("[Iteration: {}/{}] Train loss: {:.4g} | Train psnr: {:.04}".format(iterations + 1, max_iter, train_loss, train_psnr))
            
            if (iterations + 1) % 50 == 0:
                save_slices_as_tiff(train_output, iterations, out_dir)

            if (iterations + 1) % 100 == 0:
                save_tensor_as_image(train_projs, iterations, directory="train_projs_images")
                
            
            # Save final model
            if (iterations + 1) % 1000 == 0:
                model_name = os.path.join(out_dir, 'model_%06d.pt' % (iterations + 1))
                torch.save({'net': model.state_dict(), \
                            'enc': encoder.B, \
                            'opt': optimizer.state_dict(), \
                            }, model_name)
    


if __name__ == "__main__":
    run()