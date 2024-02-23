import abc
import random
from typing import List, Tuple, Literal

import torch
import numpy as np
from torch import nn
from scipy import signal

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

class FillNanNpArray(_BaseTransform):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def compute(self, x, md):
        x = np.nan_to_num(x, self.val)
        md["y"] = np.nan_to_num(md["y"].copy(), self.val)
        return x, md


class PadNpArray(_BaseTransform):
    def __init__(
        self,
        module: nn.Module,
        padlen: int,
        mode: Literal["odd", "even", "const"] = "odd",
        val: float = 0.0,
    ):
        super().__init__()
        self.module = module
        self.padlen = int(padlen)
        self.mode = mode
        self.val = val

    @staticmethod
    def odd_ext(x, n):
        left_end = x[..., :1]
        left_ext = np.flip(x[..., 1 : n + 1], axis=-1)

        right_end = x[..., -1:]
        right_ext = np.flip(x[..., -(n + 1) : -1], axis=-1)

        return np.concatenate(
            (
                2 * left_end - left_ext,
                x,
                2 * right_end - right_ext,
            ),
            axis=-1,
        )

    @staticmethod
    def even_ext(x, n):
        left_ext = np.flip(x[..., 1 : n + 1], axis=-1)
        right_ext = np.flip(x[..., -(n + 1) : -1], axis=-1)
        return np.concatenate(
            (
                left_ext,
                x,
                right_ext,
            ),
            axis=-1,
        )

    @staticmethod
    def _pad_const(x, n, val=0):
        ext = val * np.ones_like(x)[..., :n]
        return np.concatenate(
            (
                ext,
                x,
                ext,
            ),
            axis=-1,
        )

    def compute(self, x, md):
        if self.mode == "odd":
            x = self.odd_ext(x, self.padlen)
        elif self.mode == "even":
            x = self.even_ext(x, self.padlen)
        else:
            x = self.const_ext(x, self.padlen, self.val)
        x, md = self.module(x, md)
        return x[..., self.padlen : -self.padlen], md


class _BaseFilterNpArray(_BaseTransform, abc.ABC):
    btype: Literal["lowpass", "highpass", "band", "bandstop"]

    def __init__(
        self,
        order: int,
        cutoff: int | List[int],
        sample_rate: float,
    ):
        super().__init__()
        self.sos = self.get_filter_coeffs(
            order,
            cutoff,
            sample_rate,
        )

    def get_filter_coeffs(
        self,
        order: int,
        cutoffs: int | List[int],
        sample_rate: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return signal.butter(
            order, cutoffs, btype=self.btype, output="sos", fs=sample_rate
        )

    def compute(self, x, md):
        x = signal.sosfiltfilt(self.sos, x, axis=-1)
        return x, md


class LowPassNpArray(_BaseFilterNpArray):
    btype = "lowpass"

    def __init__(self, cutoff: float, sample_rate: float, order: int = 2):
        super().__init__(order, cutoff, sample_rate)


class HighPassNpArray(_BaseFilterNpArray):
    btype = "highpass"

    def __init__(self, cutoff: float, sample_rate: float, order: int = 2):
        super().__init__(order, cutoff, sample_rate)


class BandPassNpArray(_BaseFilterNpArray):
    btype = "band"

    def __init__(
        self,
        cutoff_low: float,
        cutoff_high: float,
        sample_rate: float,
        order: int = 2,
    ):
        super().__init__(order, (cutoff_low, cutoff_high), sample_rate)


class NotchNpArray(_BaseFilterNpArray):
    btype = "bandstop"

    def __init__(
        self,
        cutoff_low: float,
        cutoff_high: float,
        sample_rate: float,
        order: int = 2,
    ):
        super().__init__(order, (cutoff_low, cutoff_high), sample_rate)


class ToTensor(_BaseTransform):
    def __init__(self, dtype_x=torch.float32, dtype_y=torch.float32):
        super().__init__()
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y

    def compute(self, x, md):
        x = torch.tensor(x.copy(), dtype=self.dtype_x)
        md["y"] = torch.tensor(md["y"].copy(), dtype=self.dtype_y)
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
        montage_mat = torch.zeros((n_channels, len(self.montage)))
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
