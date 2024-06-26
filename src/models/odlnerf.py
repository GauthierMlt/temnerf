import torch.nn as nn
import torch
import numpy as np

    
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['depth']
        hidden_dim = params['width']
        input_dim  = params['input_size']
        output_dim = params['output_size']

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out