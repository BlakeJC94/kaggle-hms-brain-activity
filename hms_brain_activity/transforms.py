import abc
import random
from typing import List, Tuple, Literal, Any, Iterable, Callable

import torch
import numpy as np
from torch import nn
from scipy import signal
from torchvision.transforms.v2 import Compose

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


class TransformIterable(_BaseTransform):
    def __init__(self, apply_to: List[Any], transform: Callable):
        super().__init__()
        self.transform = transform
        self.apply_to = apply_to

    def compute(self, x: Iterable, md=None):
        for i in self.apply_to:
            x[i], md = self.transform(x[i], md)
        return x, md


class TransformCompose(_BaseTransform):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms
        self.transform = Compose(transforms)

    def compute(self, x, md):
        return self.transform(x, md)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, i):
        foo = self.transform.transforms[i]
        if isinstance(foo, list):
            return Compose(foo)
        return foo


class FillNanNpArray(_BaseTransform):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def compute(self, x, md):
        x = np.nan_to_num(x, self.val)
        if "y" in md:
            md["y"] = np.nan_to_num(md["y"].copy(), self.val)
        return x, md


class Pad(_BaseTransform):
    def __init__(
        self,
        padlen: int,
        mode: Literal["odd", "even", "const"] = "odd",
        val: float = 0.0,
    ):
        super().__init__()
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
        return x, md


class Unpad(_BaseTransform):
    def __init__(self, padlen: int):
        super().__init__()
        self.padlen = int(padlen)

    def compute(self, x, md):
        return x[..., self.padlen : -self.padlen], md


class JoinArrays(_BaseTransform):
    def compute(self, x, md):
        return np.concatenate([v for _, v in x.items()], axis=0), md


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
        return signal.bessel(
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
        if "y" in md:
            md["y"] = torch.tensor(md["y"].copy(), dtype=self.dtype_y)
        return x, md


class VotesToProbabilities(_BaseTransform):
    def compute(self, x, md):
        y = md["y"]
        y = y / y.sum(axis=0, keepdim=True)
        y = y.squeeze(-1)
        md["y"] = y
        return x, md


class TanhClipNpArray(_BaseTransform):
    def __init__(self, abs_bound: float):
        super().__init__()
        self.abs_bound = abs_bound

    def compute(self, x, md):
        x = np.tanh(x / self.abs_bound) * self.abs_bound
        return x, md


class Scale(_BaseTransform, abc.ABC):
    def __init__(self, scalar: float):
        super().__init__()
        self.scalar = scalar

    def compute(self, x, md):
        if not isinstance(self.scalar, dict):
            x = x * self.scalar
        else:
            for k, v in self.scalar.keys():
                x[k] = x[k] * self.scalar
        return x, md


class _BaseScaleChannels(_BaseTransform, abc.ABC):
    def __init__(self, scalar: float):
        super().__init__()
        self.scalar = scalar

    def compute(self, x, md):
        x_slice = tuple(
            self.ch_slice if i == x.ndim - 2 else slice(None) for i in range(x.ndim)
        )
        x[x_slice] = x[x_slice] / self.scalar
        return x, md


class ScaleEEG(_BaseScaleChannels):
    ch_slice = slice(-1)


class ScaleECG(_BaseScaleChannels):
    ch_slice = slice(-1, None)


class _BaseMontageNpArray(_BaseTransform, abc.ABC):
    montage: List[Tuple[str, str]]

    def __init__(self):
        super().__init__()
        eeg_channel_names = CHANNEL_NAMES[:-1]
        n_channels = len(eeg_channel_names)
        montage_mat = np.zeros((n_channels, len(self.montage)))
        for j, (ch_1, ch_2) in enumerate(self.montage):
            ch_idx_1 = (
                eeg_channel_names.index(ch_1) if ch_1 in eeg_channel_names else None
            )
            if ch_idx_1 is not None:
                montage_mat[ch_idx_1, j] = 1

            ch_idx_2 = (
                eeg_channel_names.index(ch_2) if ch_2 in eeg_channel_names else None
            )
            if ch_idx_2 is not None:
                montage_mat[ch_idx_2, j] = -1

        self.montage_mat = montage_mat

    def compute(self, x, md):
        x = np.matmul(
            np.swapaxes(x, -2, -1),
            self.montage_mat,
        )
        x = np.swapaxes(x, -2, -1)
        return x, md


class DoubleBananaMontageNpArray(_BaseMontageNpArray):
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
    ]


class RandomSaggitalFlipNpArray(_BaseMontageNpArray):
    montage = [(saggital_flip_channel(ch), "") for ch in CHANNEL_NAMES[:-1]]

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
        x = x * np.expand_dims(scale, -1)
        return x, md
