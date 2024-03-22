"""Apply efficientnet_v2_m to aggregated spectrograms of EEG."""

import os
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.signal.windows import dpss
from torch import nn, optim
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from torchmetrics import MeanSquaredError
from torchvision.models.efficientnet import efficientnet_v2_m

from core.modules import PredictModule, TrainModule
from core.transforms import DataTransform, TransformCompose, TransformIterable, _BaseTransform
from hms_brain_activity import metrics as m
from hms_brain_activity import transforms as t
from hms_brain_activity.callbacks import SubmissionWriter
from hms_brain_activity.datasets import HmsDataset, PredictHmsDataset
from hms_brain_activity.globals import VOTE_NAMES


class MultiTaperSpectrogram(nn.Module):
    """Create a multi-taper spectrogram transform for time series."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        frequency_resolution: float = 1.0,
        **spectrogram_kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.frequency_resolution = frequency_resolution
        self.spectrogram_kwargs = {}

        # Get tapers
        time_bandwidth, num_tapers = self.calculate_num_tapers(
            self.n_fft,
            self.sample_rate,
            frequency_resolution=self.frequency_resolution,
        )

        # Initialise spectrograms
        self.taper_spectrograms = nn.ModuleList()
        for i in range(num_tapers):
            self.taper_spectrograms.append(
                Spectrogram(
                    self.n_fft,
                    **spectrogram_kwargs,
                    window_fn=lambda n, idx, **kwargs: torch.from_numpy(
                        dpss(n, **kwargs).copy()[i]
                    ),
                    wkwargs={"idx": i, "NW": time_bandwidth, "Kmax": num_tapers},
                )
            )

        # Copy common attributes of Spectrogram
        _spect = self.taper_spectrograms[0]
        self.win_length = _spect.win_length
        self.hop_length = _spect.hop_length
        self.pad = _spect.pad
        self.power = _spect.power
        self.normalized = _spect.normalized
        self.center = _spect.center
        self.pad_mode = _spect.pad_mode
        self.onesided = _spect.onesided

    def n_frames(self, n_timesteps: int) -> int:
        if self.center:
            n_timesteps += 2 * self.pad if self.pad else 0
            return n_timesteps // self.hop_length + 1
        else:
            return (n_timesteps - self.win_length) // self.hop_length + 1

    @staticmethod
    def calculate_num_tapers(
        n_fft: int, sample_rate: float, frequency_resolution: float
    ) -> Tuple[int, int]:
        window_size_seconds = n_fft / sample_rate
        time_halfbandwidth_product = max(window_size_seconds * frequency_resolution, 1)
        num_tapers = int(2 * (time_halfbandwidth_product) - 1)
        return int(time_halfbandwidth_product), num_tapers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_channels, n_timesteps = x.shape[-2:]

        n_freq = self.n_fft // 2 + 1
        n_frames = self.n_frames(n_timesteps)
        if x.ndim == 3:
            batch_size = x.shape[0]
            xtaper = torch.zeros(batch_size, n_channels, n_freq, n_frames)
        else:
            xtaper = torch.zeros(n_channels, n_freq, n_frames)
        xtaper = xtaper.to(x)  # Ensure tensor is on correct device

        for spectrogram in self.taper_spectrograms:
            xtaper += spectrogram(x)

        return xtaper / len(self.taper_spectrograms)


class SpectrogramPower(nn.Module):
    def forward(self, x):
        # Set near-0 values to a fixed floor to prevent log(tiny value) creating large negative
        # values that obscure the actual meaningful signal
        x[x < 1e-6] = 1e-6
        x = torch.log(x)
        return x


class TrimMaxFreq(nn.Module):
    def __init__(self, sample_rate, max_frequency):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_frequency = max_frequency

    def forward(self, x):
        num_freqs, _num_timesteps = x.shape[-2:]

        # Trim unwanted frequencies
        frequencies = np.linspace(0, self.sample_rate / 2, num_freqs)
        frequency_mask = frequencies <= self.max_frequency
        # Leave batch, channel & time dims; slice the frequency dim
        x = x[..., frequency_mask, :]

        return x


class AggregateChannels(nn.Module):
    def forward(self, x):
        out = [
            torch.nanmean(x[..., sl, :, :], dim=-3, keepdim=True)
            for sl in [
                slice(0, 4),
                slice(4, 8),
                slice(8, 12),
                slice(12, 16),
                slice(16, 18),  # Sagittal plane EEG
                # slice(18, 19),  # ECG
            ]
        ]
        return torch.cat(out, dim=-3)


class AggregateFrequencies(nn.Module):
    def __init__(self, sample_rate, bands: List[Tuple[float, float]]):
        super().__init__()
        self.sample_rate = sample_rate
        self.bands = bands

    def forward(self, x):
        num_freqs, _num_timesteps = x.shape[-2:]
        frequencies = np.linspace(0, self.sample_rate / 2, num_freqs)

        out = []
        for f_low, f_high in self.bands:
            frequency_mask = (frequencies >= f_low) & (frequencies < f_high)
            out.append(torch.nanmean(x[..., frequency_mask, :], dim=-2, keepdim=True))

        return torch.cat(out, dim=-2)


def model_config(hparams):
    n_channels = 5
    n_classes = len(VOTE_NAMES)

    # Create Network
    net = efficientnet_v2_m(num_classes=n_classes)

    # Replace first convolution
    _conv0_prev = net.features[0][0]
    _conv0 = nn.Conv2d(
        n_channels,
        _conv0_prev.out_channels,
        _conv0_prev.kernel_size,
        stride=_conv0_prev.stride,
        padding=_conv0_prev.padding,
        bias=_conv0_prev.bias,
    )
    _conv0.weight = nn.init.kaiming_normal_(_conv0.weight, mode="fan_out")
    net.features[0][0] = _conv0

    return nn.Sequential(
        MultiTaperSpectrogram(
            int(hparams["config"]["sample_rate"]),
            int(
                hparams["config"]["sample_rate"] / hparams["config"]["freq_resolution"]
            ),
            hop_length=int(hparams["config"]["sample_rate"]) // 4,
            center=False,
            power=2,
        ),
        SpectrogramPower(),
        TrimMaxFreq(hparams["config"]["sample_rate"], max_frequency=80),
        AggregateChannels(),
        AggregateFrequencies(
            hparams["config"]["sample_rate"],
            bands=[
                (0.5, 4),  # Delta
                (4, 8),  # Theta
                (8, 13),  # Alpha
                (13, 30),  # Beta
                (30, 100),  # Gamma
            ],
        ),
        nn.BatchNorm2d(num_features=n_channels),
        net,
    )


def transforms(hparams):
    return [
        *[
            TransformIterable(["EEG", "ECG"], transform)
            for transform in [
                t.Pad(padlen=2 * hparams["config"]["sample_rate"]),
                t.HighPassNpArray(
                    hparams["config"]["bandpass_low"],
                    hparams["config"]["sample_rate"],
                ),
                t.LowPassNpArray(
                    hparams["config"]["bandpass_high"],
                    hparams["config"]["sample_rate"],
                ),
                t.NotchNpArray(
                    45,
                    55,
                    hparams["config"]["sample_rate"],
                ),
                t.NotchNpArray(
                    55,
                    65,
                    hparams["config"]["sample_rate"],
                ),
                t.Unpad(padlen=2 * hparams["config"]["sample_rate"]),
            ]
        ],
        TransformIterable(["EEG"], t.DoubleBananaMontageNpArray()),
        t.JoinArrays(),
        t.ToTensor(),
    ]


def metrics(hparams):
    return {
        "mse": m.MetricWrapper(
            lambda y_pred, y: (nn.functional.softmax(y_pred, dim=1), y),
            MeanSquaredError(),
        ),
        "mean_y_pred": m.MetricWrapper(
            lambda y_pred, y: (nn.functional.softmax(y_pred, dim=1), y),
            m.MeanProbability(class_names=VOTE_NAMES),
        ),
        "mean_y": m.MetricWrapper(
            lambda y_pred, y: (y, y_pred),
            m.MeanProbability(class_names=VOTE_NAMES),
        ),
        "cross_entropy": m.MetricWrapper(
            lambda y_pred, y: (nn.functional.softmax(y_pred, dim=1), y),
            m.PooledMean(
                nn.CrossEntropyLoss(),
            ),
        ),
        "prob_distribution": m.MetricWrapper(
            lambda y_pred, y: (nn.functional.softmax(y_pred, dim=1), y),
            m.ProbabilityDistribution(class_names=VOTE_NAMES),
        ),
    }


class LossFunction(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, y_pred, y):
        y_pred = nn.functional.log_softmax(y_pred, dim=1)
        return self.kldiv(y_pred, y)


def optimizer_factory(hparams, *args, **kwargs):
    return optim.AdamW(
        *args,
        lr=hparams["config"]["learning_rate"],
        weight_decay=hparams["config"]["weight_decay"],
        **kwargs,
    )


def scheduler_factory(hparams, *args, **kwargs):
    return {
        "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
            *args,
            T_0=7,
            eta_min=1e-6,
            **kwargs,
        ),
        "monitor": hparams["config"]["monitor"],
    }


def train_config(hparams):
    train_dataset = HmsDataset(
        data_dir=hparams["config"]["data_dir"],
        annotations=pd.read_csv(hparams["config"]["train_ann"]),
        augmentation=TransformCompose(
            TransformIterable(["EEG"], t.AddGaussianNoise(0.15)),
            TransformIterable(["EEG"], t.RandomSaggitalFlipNpArray(0.3)),
        ),
        transform=TransformCompose(
            *transforms(hparams),
        ),
    )

    val_dataset = HmsDataset(
        data_dir=hparams["config"]["data_dir"],
        annotations=pd.read_csv(hparams["config"]["val_ann"]),
        transform=TransformCompose(
            *transforms(hparams),
        ),
    )

    return dict(
        model=TrainModule(
            model_config(hparams),
            loss_function=LossFunction(hparams),
            metrics=metrics(hparams),
            optimizer_factory=partial(optimizer_factory, hparams),
            scheduler_factory=partial(scheduler_factory, hparams),
        ),
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=num_workers(hparams),
            shuffle=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=num_workers(hparams),
            shuffle=False,
        ),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=hparams["config"]["monitor"],
                min_delta=0.0001,
                patience=hparams["config"]["patience"],
                verbose=True,
                mode="min",
            ),
        ],
    )


def num_workers(hparams) -> int:
    return min(
        hparams["config"].get("num_workers", os.cpu_count() or 0),
        os.cpu_count() or 0,
    )


def output_transforms(hparams):
    return [
        lambda y_pred, md: (y_pred.to(torch.double), md),
        lambda y_pred, md: (torch.softmax(y_pred, axis=1), md),
    ]


class Ensemble(nn.ModuleList):
    def forward(self, x):
        out = []
        for model in self:
            out.append(model(x))
        out = torch.cat(out, dim=-1)
        return torch.mean(dim=-1)


def predict_config(hparams, predict_args):
    *weights_paths, data_dir = predict_args

    ensemble = []
    for weights_path in weights_paths:
        weights_path = Path(weights_path)
        ckpt = torch.load(weights_path, map_location="cpu")
        model = model_config(hparams)
        model.load_state_dict(ckpt["state_dict"]["model"])
        ensemble.append(model)

    module = PredictModule(
        Ensemble(ensemble),
        transform=TransformCompose(
            *output_transforms(hparams),
            lambda y_pred, md: (y_pred.cpu().numpy(), md),
        ),
    )

    data_dir = Path(data_dir).expanduser()
    predict_dataset = PredictHmsDataset(
        data_dir=data_dir,
        transform=TransformCompose(*transforms(hparams)),
    )

    return dict(
        model=module,
        predict_dataloaders=DataLoader(
            predict_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=num_workers(hparams),
            shuffle=False,
        ),
        callbacks=[
            SubmissionWriter("./"),
        ],
    )
