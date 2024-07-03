import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
try:
	import tinycudann as tcnn
	print("--- tiny-cuda-neural-net found ---")
	TINY_CUDA = True
except ImportError:
	print("tiny-cuda-nn not found.  Performance will be significantly degraded.")
	print("You can install tiny-cuda-nn from https://github.com/NVlabs/tiny-cuda-nn")
	print("============================================================")
	TINY_CUDA = False
	
from data.projectionsdataset import ProjectionsDataset
from data.emdataset import EMDataset
from models.nerf import Nerf
from data.datasets import get_params
from test import *
from utils.phantom import get_sigma_gt
from utils.render import get_points_along_rays, get_pixel_values, render_slice, get_ray_sigma, test_model, build_3d_model
from utils.chart_writer import linear_to_db, write_imgs, get_px_values
from utils.utils import set_seed
from utils.debug import plot_rays, plot_n_exit
from utils.data import init_output_dir, write_slices

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path="config", config_name="example", version_base="1.2")
def run(_cfg: DictConfig):

	out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

	tf_writer = SummaryWriter(log_dir=out_dir)

	set_seed(_cfg.optim.seed)

	device = _cfg["hardware"]["train"]

	if torch.cuda.is_available():
		print(f"Found {torch.cuda.get_device_name()}.  Will use hardware accelerated training.")
	else:
		print(f"No GPU acceleration available")

	network_config  = OmegaConf.to_container(_cfg.network)
	encoding_config = OmegaConf.to_container(_cfg.encoding)
	
	if TINY_CUDA:
		model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=1, encoding_config=encoding_config, network_config=network_config).to(device)
	else: 
		model = Nerf(encoding_config, network_config).to(device)

	data_config = _cfg["data"]

	data_name = data_config["dataset"]
	h = data_config["img_size"]

	c, radius, object_size, aspect_ratio = get_params(data_name)

	w = int(h*aspect_ratio)

	if "au" in data_name:
		near = -1 / (np.sqrt(2))
		far  =  1 / (np.sqrt(2))

		# near = -1 
		# far  = 1 
	else:
		near = 0.5*radius - 0.5*object_size 
		far  = 0.5*radius + 0.5*object_size 


	object_center = 0.5 if "au" in data_name else 0.5

	checkpoint_path, slices_path = init_output_dir(out_dir)

	if not _cfg.optim.get("checkpoint", False):
		optim = OmegaConf.to_container(_cfg.optim)

		print(f"Output will be written to {out_dir}. \nLoading Data...")

		if "au" in data_name:
			training_dataset = EMDataset(device=device,
								         images_path=data_config["images_path"],
										 angles_path=data_config["angles_path"],
										 target_size=data_config["img_size"],
										 n_projections=data_config["n_images"])

			# plot_n_exit( training_dataset.images_tensor[0].squeeze() )
		else:
			training_dataset = ProjectionsDataset(data_name, 
										          data_config["transforms_file"], 
												  split="train", 
												  device=device,img_wh=(w,h),
												  scale=object_size, 
												  n_chan=c, 
												  noise_level=data_config["noise_level"], 
												  n_train=data_config["n_images"])	

			# plot_n_exit( training_dataset.all_rgbs[:128*121].reshape(128, 121).cpu() )
		if optim["pixel_importance_sampling"]:
			pixel_weights = training_dataset.get_pixel_values()
			sampler = WeightedRandomSampler(pixel_weights, len(pixel_weights))
			data_loader = DataLoader(training_dataset, batch_size=optim["batchsize"], sampler=sampler)
		else:
			data_loader = DataLoader(training_dataset, optim["batchsize"], shuffle=True)

		optimizer = torch.optim.Adam(model.parameters(), lr=optim["learning_rate"])
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=optim["milestones"], gamma=optim["gamma"])

		print(f"Done.\nTraining model on {device}...")

		loss_function=nn.MSELoss().to(device)

		total_batches = optim["training_epochs"] * len(data_loader)
		train_psnr = torch.zeros(total_batches, dtype=torch.float16, device=device)

		early_stop = False
		with tqdm(range(optim["training_epochs"]), desc="Epochs") as t:
			for ep in t:

				ep_idx = ep * len(data_loader) 
				for batch_num, batch in enumerate(tqdm(data_loader, leave=False, desc="Batch")):

					ray_origins 		   = batch[:, :3]
					ray_directions 		   = batch[:, 3:6]
					ground_truth_px_values = batch[:, 6]
					# plot_rays(ray_origins.cpu(), ray_directions.cpu(), exit_program=True, show_directions=True, directions_scale=2, directions_subsample=20)
					regenerated_px_values = get_pixel_values(model, ray_origins, ray_directions, hn=near, hf=far, nb_bins=optim["samples_per_ray"])
					# print(regenerated_px_values.min(), regenerated_px_values.max())
					loss = loss_function(regenerated_px_values*255., ground_truth_px_values)
					
					if loss.isnan():
						print("\nLoss in NaN. Ending training")
						early_stop = True
						break

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()


					batch_idx = ep_idx + batch_num
					train_psnr[batch_idx] = linear_to_db(loss).item()

					if (batch_num % 500) == 0:
						tf_writer.add_scalar("loss", loss, batch_idx) 
						tf_writer.add_scalar("PSNR (db)", train_psnr[batch_idx], batch_idx) 
				
				if early_stop:
					break

				scheduler.step()
				torch.save(model.state_dict(), os.path.join(checkpoint_path, f'nerf_model_{ep}.pt'))
				
				write_slices(model, device, ep, batch_num, _cfg["output"], slices_path, object_center) 
				t.set_postfix(PSNR=f"{train_psnr[batch_idx]:.1f}dB", lr=f"{optimizer.param_groups[0]['lr']}")

		trained_model = model
		print(f"Training complete.")
	else:
		checkpoint = _cfg.optim.checkpoint
		print(f"Loading model from {checkpoint}")
		train_psnr = torch.zeros(1)
		trained_model = model
		trained_model.load_state_dict(torch.load(checkpoint))
		trained_model.eval()

	has_gt = True if _cfg["data"]["dataset"] == "jaw" else False

	if has_gt:
		phantom = np.load("data/jaw/jaw_phantom.npy")

	test_device = _cfg["hardware"]["test"]
	trained_model.to(test_device)

	output = _cfg["output"]
	
	if output["images"]:

		if "au" in data_name:
			testing_dataset = EMDataset(device=device,
								         images_path=data_config["images_path"],
										 angles_path=data_config["angles_path"],
										 target_size=data_config["img_size"],
										 n_projections=data_config["n_images"],
										 split="test")
		else:
			testing_dataset = ProjectionsDataset(data_name, data_config["transforms_file"], device="cpu", split="test",img_wh=(w,h), scale=object_size, n_chan=c)
		
		
		for img_index in range(1):
			test_loss, imgs = test_model(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=test_device, nb_bins=output["samples_per_ray"], H=h, W=w)
			cpu_imgs = [img.data.reshape(h, w).clamp(0.0, 255.0).detach().cpu().numpy() for img in imgs]
			# cpu_imgs = [img.data.reshape(h, w).detach().cpu().numpy() for img in imgs]


			view = testing_dataset[img_index]
			
			NB_RAYS = 5
			ray_ids = torch.randint(0, h*w, (NB_RAYS,)) # 5 random rays
			px_vals = get_px_values(ray_ids, w) 
		
			ray_origins = view[ray_ids,:3].squeeze(0)
			ray_directions = view[ray_ids,3:6].squeeze(0)

			points, delta = get_points_along_rays(ray_origins, ray_directions, hn=near, hf=far, nb_bins=output["samples_per_ray"])
			sigma = get_ray_sigma(trained_model, points, device=test_device)
			
			if has_gt:
				sigma_gt = get_sigma_gt(points.cpu().numpy(), phantom)
				sigma_gt = sigma_gt.reshape(NB_RAYS,-1)
			else:
				sigma_gt = np.zeros(0)

			sigma = sigma.data.cpu().numpy().reshape(NB_RAYS,-1)
			text = f"Test PSNR: {test_loss}"
			# write_slices(model, device, 9999, 9999, _cfg["output"], slices_path, object_center) 

			# build_3d_model(model, device, _cfg, os.getcwd())
			write_imgs((cpu_imgs, train_psnr.tolist(), sigma, sigma_gt, px_vals), f'{out_dir}/loss_{img_index}.png', text, show_training_img=False)

	# if output["slices"]:
	# 	# stich together slices into a video
	# 	sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/{run_name}_x_%04d.png -vf \"hflip\" out/{run_name}_slices_x.mp4"
	# 	os.system(sys_command)
	# 	sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/{run_name}_y_%04d.png -vf \"hflip\" out/{run_name}_slices_y.mp4"
	# 	os.system(sys_command)
	# 	sys_command = f"ffmpeg -hide_banner -loglevel error -r 5 -i tmp/{run_name}_z_%04d.png -vf \"hflip\" out/{run_name}_slices_z.mp4"
	# 	os.system(sys_command)


if __name__ == '__main__':
	run()