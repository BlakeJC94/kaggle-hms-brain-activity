import abc
import random
from typing import List, Tuple

import torch
from torch import nn

from hms_brain_activity.globals import CHANNEL_NAMES
from hms_brain_activity.utils import saggital_flip_channel

class _BaseTransform(nn.Module, abc.ABC):
    @abc.abstractmethod
    def compute(x, md):
        return x, md

    def forward(self, x, md=None):
        x, md = self.compute(x, md)
        if md is None:
            return x
        return x, md



class ToTensor(_BaseTransform):
    def __init__(self, nan_to_num=0):
        super().__init__()
        self.nan_to_num = nan_to_num

    def compute(self, x, md):
        x = torch.nan_to_num(torch.tensor(x), self.nan_to_num)
        md["y"] = torch.nan_to_num(torch.tensor(md["y"]), self.nan_to_num)
        return x, md


class VotesToProbabilities(_BaseTransform):
    def compute(self, x, md):
        y = md["y"]
        y = y / y.sum(axis=0).unsqueeze(0)
        md["y"] = y
        return x, md


class TanhClipTensor(_BaseTransform):
    def __init__(self, abs_bound: float):
        super().__init__()
        self.abs_bound = abs_bound

    def compute(self, x, md):
        x = torch.tanh(x / self.abs_bound) * self.abs_bound
        return x, md


class _BaseScaleChannels(_BaseTransform, abc.ABC):
    def __init__(self, scalar: float):
        super().__init__()
        self.scalar = scalar

    def compute(self, x, md):
        x_slice = [slice(None)] * x.ndim
        x_slice[-2] = self.ch_slice
        x[x_slice] = x[x_slice] / self.scalar
        return x, md


class ScaleEEG(_BaseScaleChannels):
    ch_slice = slice(-1)


class ScaleECG(_BaseScaleChannels):
    ch_slice = slice(-1, None)


class _BaseMontage(_BaseTransform, abc.ABC):
    montage: List[Tuple[str, str]]

    def __init__(self):
        super().__init__()
        n_channels = len(CHANNEL_NAMES)
        montage_mat = torch.zeros((n_channels, n_channels))
        for j, (ch_1, ch_2) in enumerate(self.montage):
            ch_idx_1 = CHANNEL_NAMES.index(ch_1) if ch_1 in CHANNEL_NAMES else None
            if ch_idx_1 is not None:
                montage_mat[ch_idx_1, j] = 1

            ch_idx_2 = CHANNEL_NAMES.index(ch_2) if ch_2 in CHANNEL_NAMES else None
            if ch_idx_2 is not None:
                montage_mat[ch_idx_2, j] = -1

        self.register_buffer("montage_mat", montage_mat)

    def compute(self, x, md):
        x = torch.matmul(
            x.transpose(-2, -1),
            self.montage_mat,
        ).transpose(-2, -1)
        return x, md


class DoubleBananaMontage(_BaseMontage):
    montage = [
        ("Fp1", "F7"),
        ("F7", "T3"),
        ("T3", "T5"),
        ("T5", "O1"),
        ("Fp2", "F8"),
        ("F8", "T4"),
        ("T4", "T6"),
        ("T6", "O2"),
        ("Fp1", "F3"),
        ("F3", "C3"),
        ("C3", "P3"),
        ("P3", "O1"),
        ("Fp2", "F4"),
        ("F4", "C4"),
        ("C4", "P4"),
        ("P4", "O2"),
        ("Fz", "Cz"),
        ("Cz", "Pz"),
        ("EKG", ""),
    ]


class RandomSaggitalFlip(_BaseMontage):
    montage = [(saggital_flip_channel(ch), "") for ch in CHANNEL_NAMES]

    def compute(self, x, md):
        if random.random() < 0.5:
            x, md = super().compute(x, md)
        return x, md


class RandomScale(_BaseTransform):
    def __init__(self, min_scale=0.75, max_scale=1.25, per_channel=True):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.per_channel = per_channel

    def compute(self, x, md):
        size = x.shape[:-1] if self.per_channel else x.shape[:-2]
        scale = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(size)
        x = x * scale.unsqueeze(-1)
        return x, md
