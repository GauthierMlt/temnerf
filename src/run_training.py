import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
import time, json, uuid
import os
import shutil
import random
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
from utils.render import get_points_along_rays, get_pixel_values, render_slice, get_ray_sigma, test_model
import utils.chart_writer as writer
from utils.chart_writer import linear_to_db
from utils.utils import set_seed
import commentjson as json
import hydra
from hydra import utils
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
import subprocess


def write_slices(model, device, epoch, sub_epoch, output, out_dir):
		MAX_BRIGHTNESS = 10.
		if output["slices"]:
			resolution = (output["slice_resolution"], output["slice_resolution"])
			for axis, name in enumerate(['x','y','z']):
				img = render_slice(model=model, dim=axis, device=device, resolution=resolution, voxel_grid=False, samples_per_point = output["rays_per_pixel"])
				img = img.data.cpu().numpy().reshape(resolution[0], resolution[1])/MAX_BRIGHTNESS
				writer.write_img(img, f'{out_dir}/slice_{name}_{epoch:04}_{sub_epoch:04}.png', verbose=True)


def parse_args():
	parser = argparse.ArgumentParser(description="Train model")

	parser.add_argument('config', nargs="?", default="config/test_config.json", help="Configuration parameters")
	parser.add_argument("--load_checkpoint", default='', help="Load uuid and bypass training")

	return parser.parse_args()

def init_output_dir(out_path):

	checkpoint_path = os.path.join(out_path, "checkpoints")
	slices_path = os.path.join(out_path, "slices")

	os.mkdir(checkpoint_path)
	os.mkdir(slices_path)

	return checkpoint_path, slices_path

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
	near = 0.5*radius - 0.5*object_size 
	far  = 0.5*radius + 0.5*object_size 

	if not _cfg.optim.get("checkpoint", False):
		optim = OmegaConf.to_container(_cfg.optim)
		
		interval = _cfg["output"].get("interval", 10)
		n_images = _cfg["data"].get("n_images")
		intermediate_slices = _cfg["output"].get("intermediate_slices", True)
		print(f"Output will be written to {out_dir}.")

		if "au" in data_name:
			training_dataset = EMDataset(device,data_config["images_path"],data_config["angles_path"],data_config["img_size"])
		else:
			training_dataset = ProjectionsDataset(data_name, data_config["transforms_file"], split="train", device=device,img_wh=(w,h), scale=object_size, n_chan=c, noise_level=data_config["noise_level"], n_train=data_config["n_images"])	
		

		if optim["pixel_importance_sampling"]:
			pixel_weights = training_dataset.get_pixel_values()
			sampler = WeightedRandomSampler(pixel_weights, len(pixel_weights))
			data_loader = DataLoader(training_dataset, batch_size=optim["batchsize"], sampler=sampler)
		else:
			data_loader = DataLoader(training_dataset, optim["batchsize"], shuffle=True)

		# training_im = training_dataset[:w*h,6:]
		# writer.write_img(training_im.data.reshape(h, w).clamp(0.0, 1.0).detach().cpu().numpy(), f"{out_dir}/training-img.png")

		model_optimizer = torch.optim.Adam(model.parameters(), lr=optim["learning_rate"])
		scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=optim["milestones"], gamma=optim["gamma"])

		print(f"Finished loading training data.  Training model on {device}...")
		now = time.monotonic()

		# TODO: implement speedups in here:
		# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
		# training_loss_db = []
		loss_function=nn.MSELoss().to(device)

		total_batches = optim["training_epochs"] * len(data_loader)
		training_loss_db = torch.zeros(total_batches, dtype=torch.float16, device=device)
		
		checkpoint_path, slices_path = init_output_dir(out_dir)

		with tqdm(range(optim["training_epochs"]), desc="Epochs") as t:
			for ep in t:

				ep_idx = ep * len(data_loader) 
				for batch_num, batch in enumerate(tqdm(data_loader, leave=False, desc="Batch")):
					ray_origins 		   = batch[:, :3]
					ray_directions 		   = batch[:, 3:6]
					ground_truth_px_values = batch[:, 6]

					regenerated_px_values = get_pixel_values(model, ray_origins, ray_directions, hn=near, hf=far, nb_bins=optim["samples_per_ray"])
					
					loss = loss_function(regenerated_px_values, ground_truth_px_values)
					model_optimizer.zero_grad()
					loss.backward()
					model_optimizer.step()

					batch_idx = ep_idx + batch_num
					training_loss_db[ batch_idx] = linear_to_db(loss).item()

					if (batch_num % 500) == 0:
						tf_writer.add_scalar("loss", loss, batch_idx) 
						tf_writer.add_scalar("PSNR (db)", training_loss_db[batch_idx], batch_idx) 

				scheduler.step()
				torch.save(model.state_dict(), os.path.join(checkpoint_path, f'nerf_model_{ep}.pt')) # save after each epoch
				# model.to(device)
				
				t.set_postfix(loss=f"{training_loss_db[batch_idx]:.1f} dB per pixel")
				write_slices(model, device, ep, batch_num, _cfg["output"], slices_path) 
				# print(f"{time.time()-start}")
		
		# write_slices(model, device, ep, batch_num, _cfg["output"], out_dir) 

		# training_time = time.monotonic() - now
		# timestamp = time.strftime("%Y_%m_%d_%H:%M:%S")

		# snapshot_path = f"{out_dir}/checkpoint.pt"
		# torch.save(model.state_dict(), snapshot_path)
		trained_model = model
		print(f"Training complete.")
	else:
		snapshot = _cfg.optim.checkpoint
		print(f"Loading model from {snapshot}")
		training_loss_db = torch.zeros(1) # no info
		trained_model = model
		trained_model.load_state_dict(torch.load(snapshot))
		trained_model.eval()

	has_gt = True if _cfg["data"]["dataset"] == "jaw" else False

	if has_gt:
		phantom = np.load("data/jaw/jaw_phantom.npy")

	test_device = _cfg["hardware"]["test"]
	trained_model.to(test_device)

	output = _cfg["output"]
	
	if output["images"]:
		# no noise in test data

		if "au" in data_name:
			testing_dataset = EMDataset(device,data_config["images_path"],data_config["angles_path"],data_config["img_size"], split="test")
		else:
			testing_dataset = ProjectionsDataset(data_name, data_config["transforms_file"], device="cpu", split="test",img_wh=(w,h), scale=object_size, n_chan=c)
		
		
		for img_index in range(10):
			test_loss, imgs = test_model(model=trained_model, dataset=testing_dataset, img_index=img_index, hn=near, hf=far, device=test_device, nb_bins=output["samples_per_ray"], H=h, W=w)
			cpu_imgs = [img.data.reshape(h, w).clamp(0.0, 1.0).detach().cpu().numpy() for img in imgs]
			# train_img = training_im.reshape(h, w, 3)
			# cpu_imgs.append(train_img)
			view = testing_dataset[img_index]
			
			NB_RAYS = 5
			ray_ids = torch.randint(0, h*w, (NB_RAYS,)) # 5 random rays
			px_vals = writer.get_px_values(ray_ids, w) 
		
			ray_origins = view[ray_ids,:3].squeeze(0)
			ray_directions = view[ray_ids,3:6].squeeze(0)

			points, delta = get_points_along_rays(ray_origins, ray_directions, hn=near, hf=far, nb_bins=output["samples_per_ray"])
			sigma = get_ray_sigma(trained_model, points, device=test_device)
			
			if has_gt:
				sigma_gt = get_sigma_gt(points.cpu().numpy(), phantom)
				sigma_gt = sigma_gt.reshape(NB_RAYS,-1)
			else:
				sigma_gt = np.zeros(0) # essentially empty array, so won't be shown

			sigma = sigma.data.cpu().numpy().reshape(NB_RAYS,-1)
			# text = f"test_loss: {test_loss:.1f}dB, training_loss: {float(final_training_loss_db):.1f}dB, lr: {lr:.2E}, loss function: {loss}, training noise level (sd): {args.noise} ({args.noise*(args.noise_sd/256)})\nepochs: {epochs}, layers: {layers}, neurons: {neurons}, embed_dim: {embed_dim}, training time (h): {training_time/3600:.2f}\nnumber of training images: {args.n_train}, img_size: {img_size}, samples per ray: {samples}, pixel importance sampling: {args.importance_sampling}\n"
			text = f"test loss: {test_loss}"

			writer.write_imgs((cpu_imgs, training_loss_db.tolist(), sigma, sigma_gt, px_vals), f'{out_dir}/loss_{img_index}.png', text, show_training_img=False)
		

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