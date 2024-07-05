import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from data.datasets import get_params
from utils.chart_writer import *
from utils.sampling import get_samples_around_point
from utils.debug import plot_rays
import utils.chart_writer as writer

class IndexedDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, idx):
		return self.data[idx], idx

	def __len__(self):
		return self.data.shape[0]

class SamplesDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.shape[0] 

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.data[idx,:]


def compute_accumulated_transmittance(alphas):
	accumulated_transmittance = torch.cumprod(alphas, 1)
	return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
					  accumulated_transmittance[:, :-1]), dim=-1)

def get_voxels_in_slice(z0, device, resolution):
	# initially z=0 slice
	# actually z slices are square!
	# JAW DIMENSIONS
	xs = torch.linspace(0, 275, steps=resolution[0], device=device)
	ys = torch.linspace(0, 275, steps=resolution[1], device=device)
	n_pixels = resolution[0] * resolution[1]
	x,y = torch.meshgrid(xs, ys, indexing='xy')
	z = torch.tensor([z0], device=device).expand(n_pixels).reshape(x.shape)
	voxs = torch.stack([z,y,x],dim=2).reshape(-1, 3)
	return voxs 

def get_points_in_slice(dim, device, resolution, center_offset=0.5):
	half_dx = 1./resolution[0]
	half_dy = 1./resolution[1]
	d1s = torch.linspace(0.+half_dx, 1.-half_dx, steps=resolution[0], device=device)
	d2s = torch.linspace(0.+half_dy, 1.-half_dy, steps=resolution[1], device=device)
	n_pixels = resolution[0] * resolution[1]
	d1, d2 = torch.meshgrid(d1s, d2s, indexing='xy')
	d0 = torch.zeros_like(d1) # ([z0], device=device).expand(n_pixels).reshape(x.shape)
	d0 = d0 + center_offset # object centred at 0.5
	if dim == 0: # x=0
		points = torch.stack([d0,d1,d2], dim=2).reshape(-1, 3)
	if dim == 1: # y=0
		points = torch.stack([d1,d0,d2], dim=2).reshape(-1, 3)
	if dim == 2: # z=0
		points = torch.stack([d1,d2,d0], dim=2).reshape(-1, 3)
	return points

def get_all_slices_along_z(device, resolution, center_offset=0.5):
    half_dx = 1. / resolution
    half_dy = 1. / resolution
    half_dz = 1. / resolution
    
    d1s = torch.linspace(0. + half_dx, 1. - half_dx, steps=resolution, device=device)
    d2s = torch.linspace(0. + half_dy, 1. - half_dy, steps=resolution, device=device)
    d3s = torch.linspace(0. + half_dz, 1. - half_dz, steps=resolution, device=device)
    
    d1, d2, d3 = torch.meshgrid(d1s, d2s, d3s, indexing='ij')
    d1, d2, d3 = d1.flatten(), d2.flatten(), d3.flatten()
    
    points = torch.stack([d1, d2, d3], dim=1)
    points += center_offset
    
    return points

def get_slice_along_z(slice_index, device, resolution, num_slices, center_offset=0.5):
    half_dx = 1. / resolution[0]
    half_dy = 1. / resolution[1]
    dz = 1. / num_slices
    z = dz * slice_index + dz / 2.0
    
    d1s = torch.linspace(0. + half_dx, 1. - half_dx, steps=resolution[0], device=device) + center_offset - 0.5
    d2s = torch.linspace(0. + half_dy, 1. - half_dy, steps=resolution[1], device=device) + center_offset - 0.5
    z = z + center_offset - 0.5
    
    d1, d2 = torch.meshgrid(d1s, d2s, indexing='ij')
    d1, d2 = d1.flatten(), d2.flatten()
    d3 = torch.full_like(d1, z)
    
    points = torch.stack([d1, d2, d3], dim=1)
    
    return points

@torch.no_grad()
def render_slice_from_points(model, points, device, resolution, samples_per_point):
	nb_points = points.shape[0]
	delta = 1. / resolution[0]

	# In general we have too many points to put directly on gpu (res**2 * samples_per_point), so put them on cpu then calculate on gpu in batches
	# points = points.to('cpu') 
	samples = get_samples_around_point(points, delta, samples_per_point) # [nb_samples, nb_points, 3]
	sigma = torch.empty((0,1), device=device)
	samples = samples.reshape(nb_points*samples_per_point,3)

	sigma = model(samples)

	sigma = sigma.reshape(samples_per_point, -1) # [nb_points, samples_per_point]
	sigma = torch.mean(sigma, dim=0)
	return sigma # single channel

def build_volume(model, device, config, out_dir):
	resolution = config["output"]["slice_resolution"]
	num_slices = resolution
	volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
	MAX_BRIGHTNESS = 10.0

	from tqdm import tqdm
	import os

	for i in tqdm(range(num_slices), desc="Building volume"):
		points = get_slice_along_z(i, device, (resolution, resolution), num_slices)
		img = render_slice_from_points(model=model, points=points, device=device, resolution=(resolution, resolution), samples_per_point=config["output"]["rays_per_pixel"])
		img = img.cpu().numpy().reshape(resolution, resolution) / MAX_BRIGHTNESS
		volume[:, :, i] = img

	filename = f'{config["network"]["n_hidden_layers"]}_{config["network"]["n_neurons"]}.npy'
	filepath = os.path.join(out_dir, filename)
	np.save(filepath, volume)
	print(f"3D model saved to {os.path.join(out_dir, filepath)}")

@torch.no_grad
def get_points_along_rays(ray_origins, ray_directions, hn, hf, nb_bins):
	device = ray_origins.device
	t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)

	# Perturb sampling along each ray.
	mid = (t[:, :-1] + t[:, 1:]) / 2.
	lower = torch.cat((t[:, :1], mid), -1)
	upper = torch.cat((mid, t[:, -1:]), -1)
	u = torch.rand(t.shape, device=device)
	t = lower + (upper - lower) * u  # [batch_size, nb_bins]
	delta = t[:, 1:] - t[:, :-1] # [batch_size, nb_bins-1]
	x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]

	# sampled_points_vect = torch.cat((x[:,  0, :]  , x[:, -1, :]  ), dim=0)
	# plot_rays(ray_origins.cpu(), ray_directions.cpu(), sampled_points_vect.cpu(), show_origins=False, show_directions=False, directions_scale=1, show_coordinate_frame=True, box_size=1/np.sqrt(2), box_origin=0.5, show_box=True, coordinate_frame_size=0.1, exit_program=False)

	# print(f"ray near/far is {x[1,0,:]}/{x[1,-1,:]	}")
	return x.reshape(-1, 3), delta


def get_pixel_values(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins):
	x, delta = get_points_along_rays(ray_origins, ray_directions, hn, hf, nb_bins)
	sigma = nerf_model(x) # [batch_size*(nb_bins-1), 1]
	sigma = sigma.reshape(-1, nb_bins)

	# interpolate: nb_bins values, nb_bins - 1 intervals
	# --- | ------- | ----- | --- | --------
	# --- | ------- | ----- | --- | --------
	sigma_mid = (sigma[:, 1:] + sigma[:, :-1])/2
	alpha = (sigma_mid * delta).unsqueeze(2)
	# single channel
	c = alpha.sum(dim=1).squeeze()/nb_bins
	# print(c)
	return c


@torch.no_grad()
def get_ray_sigma(model, points, device):
	model = model.to(device)
	points = points.to(device)
	sigma = model(points)
	return sigma

@torch.no_grad()
def render_slice(model, dim, device, resolution, voxel_grid, samples_per_point, object_center):
	points = get_points_in_slice(dim, device, resolution, center_offset=object_center)
	nb_points = points.shape[0]
	delta = 1. / resolution[0]

	# In general we have too many points to put directly on gpu (res**2 * samples_per_point), so put them on cpu then calculate on gpu in batches
	# points = points.to('cpu') 
	samples = get_samples_around_point(points, delta, samples_per_point) # [nb_samples, nb_points, 3]
	sigma = torch.empty((0,1), device='cuda')
	samples = samples.reshape(nb_points*samples_per_point,3)

	# try:
	sigma = model(samples)
	# except RuntimeError as e:
	# 		if 'out of memory' in str(e):
	# 			print('| WARNING: ran out of memory, retrying batch')
	# for batch in tqdm(samples_loader, leave=False, desc="Rendering slice"):
	# 	batch = batch.to(device)
	# 	batch_sigma = model(batch)
	# 	sigma = torch.cat((sigma,batch_sigma),0)
	# 	# TODO: don't need to keep all samples, can do the
	# 	# averaging here
	# 	del batch

	sigma = sigma.reshape(samples_per_point, -1) # [nb_points, samples_per_point]
	sigma = torch.mean(sigma, dim=0)
	return sigma # single channel

@torch.no_grad()	
def render_image(model, frame, **params):
	device = params["device"]
	H = params["H"]
	W = params["W"]
	hf = params["hf"]
	hn = params["hn"]
	nb_bins = params["nb_bins"]

	dataset = IndexedDataset(frame)
	data = DataLoader(dataset, batch_size=2_000_000)
	
	img_tensor = torch.zeros(H*W) # single channel
	for batch, idx in data:
		ray_origins = batch[...,:3].squeeze(0).to(device)
		ray_directions = batch[...,3:6].squeeze(0).to(device)

		# from utils.debug import plot_rays
		# plot_rays(ray_origins.cpu(), ray_directions.cpu(), show_box=False, show_directions=True)

		regenerated_px_values = get_pixel_values(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
		img_tensor[idx,...] = regenerated_px_values.cpu()

	return img_tensor



@torch.no_grad()
def test_model(model, dataset, img_index, **render_params):
	frame = dataset[img_index]

	img_tensor = render_image(model, frame, **render_params)

	gt = frame[...,6].squeeze(0)

	if gt.max() > 1.:
		gt /= 255.
	
	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() 
	test_loss = linear_to_db(loss)
	return test_loss, [img_tensor,gt,diff]
	
