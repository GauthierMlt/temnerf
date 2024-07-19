import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import torchmetrics
	
from data.projectionsdataset import ProjectionsDataset
from data.emdataset import EMDataset
from data.datasets import get_params
from utils.phantom import get_sigma_gt
from utils.render import get_points_along_rays, get_pixel_values, get_ray_sigma, test_model, build_volume, get_density_mip_nerf
from utils.chart_writer import compute_psnr, write_imgs, get_px_values
from utils.utils import set_seed, get_model
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
		print(f"Found {torch.cuda.get_device_name()}")
	else:
		print(f"No GPU found")
		
	model = get_model(OmegaConf.to_container(_cfg.encoding), 
				      OmegaConf.to_container(_cfg.network))
	
	model.to(device)
	data_config = _cfg["data"]

	optim = OmegaConf.to_container(_cfg.optim)
	data_name = data_config["dataset"]
	h = data_config["img_size"]

	c, radius, object_size, aspect_ratio = get_params(data_name)

	w = int(h*aspect_ratio)
	
	object_center = 0.5 

	if "tem" in data_name:
		near = -1 / (np.sqrt(2)) 
		far  =  1 / (np.sqrt(2))

		# near = -1
		# far  = 1
	else:
		near = 0.5*radius - 0.5*object_size 
		far  = 0.5*radius + 0.5*object_size 


	checkpoint_path, slices_path, volume_path = init_output_dir(out_dir)

	factor = optim["factor"]

	if not _cfg.optim.get("checkpoint", False):

		print(f"Output will be written to {out_dir}. \nLoading Data...")

		if "tem" in data_name:
			training_dataset = EMDataset(device=device,
								         images_path=data_config["images_path"],
										 angles_path=data_config["angles_path"],
										 target_size=data_config["img_size"],
										 n_projections=data_config["n_images"])
			
		else:
			training_dataset = ProjectionsDataset(data_name, 
										          data_config["transforms_file"], 
												  split="train", 
												  device=device,img_wh=(w,h),
												  scale=object_size, 
												  n_chan=c, 
												  noise_level=data_config["noise_level"], 
												  n_train=data_config["n_images"])	

		if optim["pixel_importance_sampling"]:
			pixel_weights = training_dataset.get_pixel_values()
			sampler = WeightedRandomSampler(pixel_weights, len(pixel_weights))
			data_loader = DataLoader(training_dataset, batch_size=optim["batchsize"], sampler=sampler)
		else:
			data_loader = DataLoader(training_dataset, optim["batchsize"], shuffle=True, num_workers=0)

		optimizer = torch.optim.Adam(model.parameters(), lr=optim["learning_rate"], betas=(0.9, 0.999))
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=optim["milestones"], gamma=optim["gamma"])

		loss_function=nn.MSELoss().to(device)

		total_batches = optim["epochs"] * len(data_loader)
		train_psnr = torch.zeros(total_batches, dtype=torch.float16, device=device)

		early_stop = False
		with tqdm(range(optim["epochs"]), desc="Iter") as t:
			for ep in t:

				ep_idx = ep * len(data_loader) 

				bar = tqdm(data_loader, leave=False, desc="Epoch")
				for batch_num, batch in enumerate(bar):

					ray_origins 		   = batch[:, :3].to(device)
					ray_directions 		   = batch[:, 3:6].to(device)
					ground_truth_px_values = batch[:, 6].to(device)

					regenerated_px_values = get_pixel_values(model, ray_origins, ray_directions, hn=near, hf=far, nb_bins=optim["samples_per_ray"], debug=False)
					# regenerated_px_values = get_density_mip_nerf(model, ray_origins, ray_directions, hn=near, hf=far, nb_bins=optim["samples_per_ray"], mip_level=1, debug=False)

					loss = loss_function(regenerated_px_values*factor, ground_truth_px_values)

					if loss.isnan():
						print("\nLoss in NaN. Ending training")
						early_stop = True
						break

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					batch_idx = ep_idx + batch_num
					train_psnr[batch_idx] = compute_psnr(loss).item()

					if (batch_idx % 5) == 0:
						bar.set_description(f"Pred/GT min: {regenerated_px_values.min().item():.4f}/{ground_truth_px_values.min().item():.4f}"
						  					f" | max: {regenerated_px_values.max().item():.4f}/{ground_truth_px_values.max().item():.4f} | PSNR:{train_psnr[batch_idx]:.2f} | lr: {optimizer.param_groups[0]['lr']:.2e}")
					if (batch_idx % 10) == 0:
						# print(f"\n{regenerated_px_values.min().item():2f}, {regenerated_px_values.max().item():2f}, {ground_truth_px_values.min().item():2f}, {ground_truth_px_values.max().item():2f}")
						
						tf_writer.add_scalar("loss", loss, batch_idx) 
						tf_writer.add_scalar("PSNR (db)", train_psnr[batch_idx], batch_idx) 

					# torch.cuda.empty_cache()
				if early_stop:
					break

				scheduler.step()
				# SAVE ONLY BEST MODEL !!!
				torch.save(model.state_dict(), os.path.join(checkpoint_path, f'nerf_model_{ep}.pt'))
				
				write_slices(model, device, ep, batch_idx, _cfg["output"], slices_path, object_center) 
				# t.set_postfix(PSNR=f"{train_psnr[batch_idx]:.1f}dB")

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

		if "tem" in data_name:
			testing_dataset = EMDataset(device=device,
								         images_path=data_config["images_path"],
										 angles_path=data_config["angles_path"],
										 target_size=data_config["img_size"],
										 n_projections=data_config["n_images"],
										 split="test")
		else:
			testing_dataset = ProjectionsDataset(data_name, data_config["transforms_file"], device="cpu", split="test",img_wh=(w,h), scale=object_size, n_chan=c)
		
		
		for img_index in range(1):


			with torch.no_grad():
				test_loss, imgs = test_model(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=test_device, nb_bins=output["samples_per_ray"], H=h, W=w, factor=factor)
			cpu_imgs = [img.data.reshape(h, w).clamp(0.0, 1.0).detach().cpu().numpy() for img in imgs]
			# cpu_imgs = [img.data.reshape(h, w).detach().cpu().numpy() for img in imgs]


			view = testing_dataset[img_index]
			
			NB_RAYS = 5
			ray_ids = torch.randint(0, h*w, (NB_RAYS,)) # 5 random rays
			px_vals = get_px_values(ray_ids, w) 
		
			ray_origins    = view[ray_ids, 0:3].squeeze(0)
			ray_directions = view[ray_ids, 3:6].squeeze(0)

			points, _ = get_points_along_rays(ray_origins, ray_directions, hn=near, hf=far, nb_bins=output["samples_per_ray"])
			sigma = get_ray_sigma(trained_model, points, device=test_device)
			
			if has_gt:
				sigma_gt = get_sigma_gt(points.cpu().numpy(), phantom)
				sigma_gt = sigma_gt.reshape(NB_RAYS,-1)
			else:
				sigma_gt = np.zeros(0)

			sigma = sigma.data.cpu().numpy().reshape(NB_RAYS,-1)
			text = f"PSNR: {test_loss:.2f}"

			if _cfg.output.get("volume", False):
				build_volume(model, device, _cfg, volume_path, factor)

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