import torch
from torch import nn


class ToTensor(nn.Module):
    def forward(self, x, md):
        x = torch.tensor(x)
        md['y'] = torch.tensor(md['y'])
        return x, md
