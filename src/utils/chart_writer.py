import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from cycler import cycler # to get colours to match in different plots
import torch

def get_px_values(ray_ids, W):
	cols = np.remainder(ray_ids, W)
	rows = np.floor(ray_ids / W)
	px_vals = np.stack((cols,rows), axis=1)
	return px_vals

def moving_average(vals, window=10) :
	n_v = len(vals)
	idx = np.arange(window)
	kernel = np.exp(-0.2*idx)
	kernel /= kernel.sum()
	av = np.convolve(vals, kernel, mode='valid')
	return av

def forward(x):
	return np.maximum(0,x)**(1/2)

def inverse(x):
	return x**2

def save_training_report(data, path, title=None, show_training_img=False):
	fig = plt.figure(tight_layout=True, figsize=(40., 20.))
	n_cols = 5 if show_training_img else 4
	gs = gridspec.GridSpec(2, n_cols)
	imgs, curve, rays, rays_gt, px_vals = data

	# colors = np.random.rand(5)
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] #, '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
	custom_cycler = cycler(color=colors)

	# images
	titles = ["Pred","Ground truth","diff"]
	if show_training_img:
		titles.append("example training")
	for i, sub_tit in enumerate(titles):
		ax = fig.add_subplot(gs[0, i])
		if i < 2:
			im = ax.imshow(imgs[i],cmap='Greys_r')
		else:
			im = ax.imshow(imgs[i],cmap='viridis')
		plt.colorbar(im, ax=ax) 
		ax.set_title(sub_tit)

	# curve
	if curve:
		ax = fig.add_subplot(gs[1, :])
		ax.plot(moving_average(curve,10))
		ax.set_ylabel('dB')
		ax.invert_yaxis()

	# ray
	ax = fig.add_subplot(gs[0, n_cols-1])
	ax.set_prop_cycle(custom_cycler)
	ax.set_yscale('function', functions=(forward, inverse))
	ax.plot(rays.transpose(), '-')
	if rays_gt.any():
		ADJUSTED_BRIGHTNESS = 1
		# add some jitter to distinguish values
		jitter = 0.005*np.random.rand(rays_gt.shape[0],1)
		rays_gt = rays_gt + jitter
		ax.plot(ADJUSTED_BRIGHTNESS*rays_gt.transpose(), '--')
	ax.set_title("density along rays")

	fig.suptitle(title, fontsize=18, fontweight="bold")
	plt.savefig(path)
	plt.close()