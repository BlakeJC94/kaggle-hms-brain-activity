import torch
from torch import nn


class ToTensor(nn.Module):
    def forward(self, x, md):
        x = torch.tensor(x)
        md["y"] = torch.tensor(md["y"])
        return x, md


class VotesToProbabilities(nn.Module):
    def forward(self, x, md):
        y = md["y"]
        y = y / y.sum(axis=0).unsqueeze(0)
        md["y"] = y
        return x, md


class TanhClipTensor(nn.Module):
    def __init__(self, abs_bound: float):
        super().__init__()
        self.abs_bound = abs_bound

    def forward(self, x, md):
        x = torch.tanh(x / self.abs_bound) * self.abs_bound
        return x, md


class ScaleEEG(nn.Module):
    def __init__(self, scalar: float):
        super().__init__()
        self.scalar = scalar

    def forward(self, x, md):
        x[:-1, ...] = x[:-1, ...] / self.scalar
        return x, md
