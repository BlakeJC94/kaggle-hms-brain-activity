import torch
from torch import nn


class ToTensor(nn.Module):
    def forward(self, x, md):
        x = torch.tensor(x)
        md['y'] = torch.tensor(md['y'])
        return x, md


class VotesToProbabilities(nn.Module):
    def forward(self, x, md):
        y = md['y']
        y = y / y.sum(axis=0).unsqueeze(0)
        md['y'] = y
        return x, md
