import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from data.datasets import get_params
from utils.chart_writer import *
from utils.sampling import get_samples_around_point
from utils.debug import plot_rays
import utils.chart_writer as writer

from tqdm import tqdm
import os

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
    half_dx = 1.0 / resolution[0]
    half_dy = 1.0 / resolution[1]
    
    d1s = torch.linspace(0.0 + half_dx, 1.0 - half_dx, steps=resolution[0], device=device)
    d2s = torch.linspace(0.0 + half_dy, 1.0 - half_dy, steps=resolution[1], device=device)
    
    d1, d2 = torch.meshgrid(d1s, d2s, indexing='xy')
    
    d0 = torch.full_like(d1, center_offset)
    
    if dim == 0:  # x=0
        points = torch.stack([d0, d1, d2], dim=2).reshape(-1, 3)
    elif dim == 1:  # y=0
        points = torch.stack([d1, d0, d2], dim=2).reshape(-1, 3)
    elif dim == 2:  # z=0
        points = torch.stack([d1, d2, d0], dim=2).reshape(-1, 3)
    else:
        raise ValueError("dim should be 0, 1, or 2")
    
    return points

def sample_volume(device, resolution, num_slices, center_offset=0.5):
    half_dx = 1. / resolution[0]
    half_dz = 1. / resolution[1]
    dy = 1. / num_slices

    d1s = torch.linspace(0. + half_dx, 1. - half_dx, steps=resolution[0], device=device) + center_offset - 0.5
    d3s = torch.linspace(0. + half_dz, 1. - half_dz, steps=resolution[1], device=device) + center_offset - 0.5
    ys = torch.linspace(dy / 2.0, 1.0 - dy / 2.0, steps=num_slices, device=device) + center_offset - 0.5

    d1, d3 = torch.meshgrid(d1s, d3s, indexing='xy')
    d1, d3 = d1.flatten(), d3.flatten()
    
    d1 = d1.unsqueeze(0).repeat(num_slices, 1).flatten()
    d3 = d3.unsqueeze(0).repeat(num_slices, 1).flatten()
    d2 = ys.repeat_interleave(resolution[0] * resolution[1])
    
    points = torch.stack([d1, d2, d3], dim=1)
    
    return points.reshape(num_slices, -1, 3)

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

@torch.no_grad()
def build_volume(model, device, config, out_dir):
	resolution = config["output"]["slice_resolution"]
	num_slices = resolution
	volume = np.zeros((resolution, resolution, resolution), dtype=np.float32)
	slices = sample_volume(device, (resolution, resolution), num_slices)

	for i in tqdm(range(num_slices), desc="Building volume"):
		points = slices[i]
		# from utils.debug import plot_rays
		# plot_rays(torch.zeros(1), torch.zeros(1), additional_points=slices.reshape(-1, 3)[::10].cpu(), show_origins=False, show_directions=False)

		img = render_slice_from_points(model=model, points=points, device=device, resolution=(resolution, resolution), samples_per_point=config["output"]["rays_per_pixel"])
		
		img = img.clamp(0., 255.).reshape(resolution, resolution).cpu().numpy()
		volume[:, i, :] = img

	filename = f'{config["network"]["n_hidden_layers"]}_{config["network"]["n_neurons"]}.npy'
	filepath = os.path.join(out_dir, filename)
	
	np.save(filepath, volume)
	print(f"3D model saved to {os.path.join(out_dir, filepath)}")

	try:
		os.system(f'xdg-open {os.path.realpath(out_dir)}')
	except:
		pass

def get_points_along_rays_mip(ray_origins, ray_directions, hn, hf, nb_bins, mip_level, debug=False):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)

    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = t[:, 1:] - t[:, :-1]  # [batch_size, nb_bins-1]

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]

    if debug:
        sampled_points_vect = torch.cat((x[:, 0, :], x[:, -1, :]), dim=0)
        plot_rays(ray_origins.cpu(), ray_directions.cpu(), sampled_points_vect[::].cpu(),
                  show_origins=False, show_directions=False, show_coordinate_frame=True,
                  box_size=1 / np.sqrt(2), show_box=True, coordinate_frame_size=0.1, exit_program=False)

    # Compute the Gaussian parameters for each segment
    means = (x[:, :-1, :] + x[:, 1:, :]) / 2  # Midpoints
    covariances = (t[:, 1:] - t[:, :-1]) ** 2 / 12  # Variance for uniform distribution along the segment

    return means, covariances, delta


def get_density_mip_nerf(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins, mip_level, debug=True):
    means, covariances, delta = get_points_along_rays_mip(ray_origins, ray_directions, hn, hf, nb_bins, mip_level, debug)
    batch_size, num_samples, _ = means.shape

    # Flatten means and covariances to pass through the model
    means = means.reshape(-1, 3)
    covariances = covariances.reshape(-1, 1)

    # Predict densities
    density_inputs = torch.cat([means, covariances], dim=-1)  # Combine means and variances
    sigma = nerf_model(density_inputs)  # [batch_size*(nb_bins-1), 1]
    sigma = sigma.reshape(batch_size, num_samples)

    # Integrate densities
    alpha = (1.0 - torch.exp(-sigma * delta))  # Alpha from densities
    T = torch.cumprod(torch.cat([torch.ones((batch_size, 1), device=sigma.device), 1.0 - alpha + 1e-10], dim=-1), dim=-1)
    weights = alpha * T[:, :-1]

    return weights.sum(dim=1)  # Sum the weights along the ray

@torch.no_grad()
def get_points_along_rays(ray_origins, ray_directions, hn, hf, nb_bins, debug=False):
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

	if debug:
	# # print(f"ray near/far is {x[1,0,:]}/{x[1,-1,:]	}")
		sampled_points_vect = torch.cat((x[:,  0, :]  , x[:, -1, :]  ), dim=0)
		plot_rays(ray_origins.cpu(), ray_directions.cpu(), sampled_points_vect[::].cpu(), show_origins=False, show_directions=False, show_coordinate_frame=True, box_size=1/np.sqrt(2), show_box=True, coordinate_frame_size=0.1, exit_program=False)

	return x.reshape(-1, 3), delta


def get_pixel_values(nerf_model, ray_origins, ray_directions, hn, hf, nb_bins, debug=False):
	x, delta = get_points_along_rays(ray_origins, ray_directions, hn, hf, nb_bins, debug)
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

@torch.no_grad
def render_slice(model, dim, device, resolution, voxel_grid, samples_per_point):
	points = get_points_in_slice(dim, device, resolution)
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

# def render_slice(model, dim, device, resolution, voxel_grid, samples_per_point):
#     points = get_points_in_slice(dim, device, resolution)
#     nb_points = points.shape[0]
#     delta = 1. / resolution[0]

#     # Compute samples around each point, and the corresponding Gaussian parameters
#     samples = get_samples_around_point(points, delta, samples_per_point) # [nb_samples, nb_points, 3]
#     samples = samples.reshape(nb_points * samples_per_point, 3)

#     means = samples
#     covariances = torch.full((nb_points * samples_per_point, 1), delta**2 / 12, device=device)

#     density_inputs = torch.cat([means, covariances], dim=-1)

#     batch_size = 1024  # Define a reasonable batch size to avoid OOM
#     sigma = torch.empty((0, 1), device=device)
    
#     for i in range(0, density_inputs.shape[0], batch_size):
#         batch = density_inputs[i:i + batch_size].to(device)
#         batch_sigma = model(batch)
#         sigma = torch.cat((sigma, batch_sigma), dim=0)

#     sigma = sigma.reshape(samples_per_point, -1) # [samples_per_point, nb_points]
#     sigma = torch.mean(sigma, dim=0)  # Averaging across samples_per_point
#     return sigma  # single channel


@torch.no_grad()
def render_image(model, frame, **params):
	device = params["device"]
	H = params["H"]
	W = params["W"]
	hf = params["hf"]
	hn = params["hn"]
	nb_bins = params["nb_bins"]

	dataset = IndexedDataset(frame)
	data = DataLoader(dataset, batch_size=5_000)
	
	img_tensor = torch.zeros(H*W) # single channel
	for batch, idx in tqdm(data):
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

	# img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor))
		
	# plt.imshow(img_tensor.reshape(render_params["H"],render_params["W"]).cpu(), cmap="gray")
	# plt.show()
	gt = frame[...,6].squeeze(0)

	# gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))

	diff = (gt - img_tensor).abs()
	loss = (diff ** 2).mean() 
	test_loss = compute_psnr(loss)
	return test_loss, [img_tensor,gt,diff]
	
