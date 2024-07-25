import tinycudann as tcnn
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.render import build_volume
from utils.data import compute_psnr

encoding_config = {
    "otype": "HashGrid",
    "type": "Hash",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 19,
    "base_resolution": 16,
    "per_level_scale": 2.0,
    "n_frequencies": 16
}

network_config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 128,
    "n_hidden_layers": 6
}

model =  tcnn.NetworkWithInputEncoding(n_input_dims=3, 
                                        n_output_dims=1, 
                                        encoding_config=encoding_config, 
                                        network_config=network_config).double()

ref = np.load("data/benchmark/au_ag_sirt.npy")


[plt.imsave(f"{i}.png", np.moveaxis(ref, i, 0)[ref.shape[i]//2]) for i in range(3)]


model = model.cuda()
ref_tensor = torch.tensor(ref, dtype=torch.float16, device="cuda")

criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dim_x, dim_y, dim_z = ref.shape

targets = ref_tensor.flatten().cuda()

x = np.linspace(0, 1, dim_x)
y = np.linspace(0, 1, dim_y)
z = np.linspace(0, 1, dim_z)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1) # X and Z axis are inverted 
coords = torch.tensor(coords, dtype=torch.float16, device="cuda")


dataset = TensorDataset(coords, targets)
batch_size = 512
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

n_epochs = 1
for epoch in range(n_epochs):
    model.train()
    
    for batch_coords, batch_targets in tqdm(dataloader):
        optimizer.zero_grad()
        outputs = model(batch_coords)
        outputs = outputs.view(-1)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "nerf_volume_identity.pt")

    
    if (epoch % max(1, n_epochs // 100)) == 0:
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

config = {
    "output":{
        "slice_resolution": ref.shape[0],
        "rays_per_pixel": 128
    },
    "network":{
        "n_hidden_layers": 6,
        "n_neurons":128
    }
}

volume = build_volume(model, "cuda", config, ".").cuda()
np.save("nerf_volume_identity.npy", volume.cpu())
print(f"GT Volume min {ref_tensor.min()} | max: {ref_tensor.max()} | avg {ref_tensor.mean()} ")
volume_psnr = compute_psnr(torch.mean((ref_tensor-volume)**2), ref_tensor.max())
print(f"Volume PSNR = {volume_psnr}")