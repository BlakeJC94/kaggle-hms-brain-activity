"""Apply efficientnet_v2_s to aggregated spectrograms of EEG.

- Augment EEGs with random saggital flip
- Filter and scale EEG
- Double Banana montage
- Scale ECG
- Tanh clip values
- Compute spectrograms
- Average across electrode groups
- Append asymmetric spectrograms across sagittal plane

TODO
- Multi-taper spectrogram
- Better filtering
- Heart rate feature

"""

import os
from functools import partial
from pathlib import Path
from math import floor, ceil, sqrt
from typing import *

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from core.modules import PredictModule, TrainModule
from core.transforms import TransformCompose, TransformIterable
from hms_brain_activity import logger
from hms_brain_activity import metrics as m
from hms_brain_activity import transforms as t
from hms_brain_activity.callbacks import SubmissionWriter
from hms_brain_activity.datasets import HmsDataset, PredictHmsDataset
from hms_brain_activity.globals import VOTE_NAMES
from hms_brain_activity.paths import DATA_PROCESSED_DIR
from scipy.signal.windows import dpss
from torch import nn, optim
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from torchmetrics import MeanSquaredError
from torchvision.models.efficientnet import efficientnet_v2_s

logger = logger.getChild(Path(__file__).parent.name)


## Spectrogram transforms
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


class PostProcessSpectrograms(nn.Module):
    def __init__(self, sample_rate, max_frequency):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_frequency = max_frequency

    def forward(self, x):
        num_freqs, _num_timesteps = x.shape[-2:]

        # Set near-0 values to a fixed floor to prevent log(tiny value) creating large negative
        # values that obscure the actual meaningful signal
        x[x < 1e-10] = 1e-10
        x = torch.log(x)

        # Trim unwanted frequencies
        frequencies = np.linspace(0, self.sample_rate / 2, num_freqs)
        frequency_mask = frequencies <= self.max_frequency
        # Leave batch, channel & time dims; slice the frequency dim
        x = x[..., frequency_mask, :]

        return x


class KLDivWithLogitsLoss(nn.KLDivLoss):
    def forward(self, y_hat, y):
        y_hat = nn.functional.log_softmax(y_hat, dim=1)
        return super().forward(y_hat, y)


class AggregateSpectrograms(nn.Module):
    def forward(self, x):
        out = [
            torch.nanmean(x[..., sl, :, :], dim=-3, keepdim=True)
            for sl in [
                slice(0, 4),
                slice(4, 8),
                slice(8, 12),
                slice(12, 16),
                # slice(16, 18),  # Sagittal plane EEG
                slice(18, 19),  # ECG
            ]
        ]
        return torch.cat(out, dim=-3)


## Time series model
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int,
        kernel_size: int,
        **conv_kwargs,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, features, kernel_size, **conv_kwargs)
        self.bn1 = nn.BatchNorm1d(num_features=features)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(features, features, kernel_size, **conv_kwargs)
        self.bn2 = nn.BatchNorm1d(num_features=features)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        length = out.shape[-1]
        start = (x.shape[-1] - length) // 2
        x = torch.narrow(x, dim=-1, start=start, length=length)
        return self.act(out + x)


class DownConvBlock(nn.Sequential):
    def __init__(self, in_channels, features, pool):
        super().__init__()
        self.in_channels = in_channels
        self.features = features
        self.scale = pool

        self.conv = nn.Conv1d(in_channels, features, kernel_size=pool, stride=pool)
        self.bn = nn.BatchNorm1d(features)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int,
        kernel_size: int,
        pool: int,
        **conv_kwargs,
    ):
        super().__init__()
        self.pool = pool

        self.expand = DownConvBlock(in_channels, features, pool)
        self.block = BasicBlock(features, features, kernel_size)

    def forward(self, x) -> torch.Tensor:
        x = self.expand(x)
        x = self.block(x)
        return x


class UpConvBlock(nn.Sequential):
    def __init__(self, in_channels, features, scale):
        super().__init__()
        self.in_channels = in_channels
        self.features = features
        self.scale = scale

        self.conv = nn.Conv1d(in_channels, features, scale, padding="valid")
        self.bn = nn.BatchNorm1d(num_features=features)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = nn.functional.interpolate(
            x,
            size=self.scale * (x.shape[-1] + 1) - 1,
            mode="nearest-exact",
        )
        return super().forward(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int,
        scale: int,
        kernel_size: int,
        **conv_kwargs,
    ):
        super().__init__()
        self.scale = scale

        self.upconv = UpConvBlock(in_channels, features, scale)
        self.contract = nn.Sequential(
            nn.Conv1d(2 * features, features, kernel_size),
            nn.BatchNorm1d(features),
        )
        self.block = BasicBlock(features, features, kernel_size, **conv_kwargs)

    def forward(self, x, res) -> torch.Tensor:
        x = self.upconv(x)
        x = torch.cat(self.centre_crop(x, res), dim=1)
        x = self.contract(x)
        return self.block(x)

    @staticmethod
    def centre_crop(x, res) -> Tuple[torch.Tensor, torch.Tensor]:
        x_len, res_len = x.shape[-1], res.shape[-1]
        if x_len == res_len:
            return x, res

        l_crop = floor(int(abs(x_len - res_len)) / 2)
        r_crop = ceil(int(abs(x_len - res_len)) / 2)

        if x_len < res_len:
            res = res[..., l_crop:(-r_crop)]
        else:
            x = x[..., l_crop:(-r_crop)]

        return x, res


class Backbone(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dilation: int = 1,
        kernel_size: int = 9,
        n_features: int = 32,
        padding: str = "valid",
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classses = n_classes
        self.kernel_size = kernel_size
        self.n_features = n_features
        self.padding = padding

        pools = [2, 2, 2, 2]
        expansions = [sqrt(2) for _ in pools]
        filters = self._get_filters(n_features, expansions=expansions)

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(n_channels, filters[0], kernel_size, 1, padding=padding),
                EncoderBlock(
                    filters[0], filters[1], kernel_size, pools[0], padding=padding
                ),
                EncoderBlock(
                    filters[1], filters[2], kernel_size, pools[1], padding=padding
                ),
                EncoderBlock(
                    filters[2], filters[3], kernel_size, pools[2], padding=padding
                ),
            ]
        )

        self.bridge = EncoderBlock(
            filters[3], filters[4], kernel_size, pools[3], padding=padding
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    filters[4], filters[3], pools[3], kernel_size, padding=padding
                ),
                DecoderBlock(
                    filters[3], filters[2], pools[2], kernel_size, padding=padding
                ),
                DecoderBlock(
                    filters[2], filters[1], pools[1], kernel_size, padding=padding
                ),
                DecoderBlock(
                    filters[1], filters[0], pools[0], kernel_size, padding=padding
                ),
            ]
        )

        dense_blocks = [
            nn.Conv1d(
                in_channels=filters[0],
                out_channels=n_classes,
                kernel_size=1,
            ),
            nn.Tanh(),
        ]
        self.dense = nn.Sequential(*dense_blocks)

        self.apply(self._init_conv_weight_bias)
        self.apply(self._init_bn_weight)

    def _init_conv_weight_bias(self, module):
        if isinstance(module, (torch.nn.Conv1d)):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
            nn.init.constant_(module.bias, 0.01)

    def _init_bn_weight(self, module):
        if isinstance(module, BasicBlock):
            nn.init.constant_(module.bn2.bias, 0)

    @staticmethod
    def _get_filters(n_features, expansions):
        fs = [n_features]
        for expansion in expansions:
            fs.append(fs[-1] * expansion)
        return [int(f) for f in fs]

    def forward(self, x):
        _n_batches, _n_channels, n_timesteps = x.shape

        residuals = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            residuals.append(x)

        x = self.bridge(x)

        for i, (block, residual) in enumerate(zip(self.decoder, residuals[::-1])):
            x = block(x, residual)

        x = self.dense(x)
        return x


## Ensemble
class MyModel(nn.Module):
    def __init__(self, n_channels, n_classes, spectrogram_transform):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.spectrogram_transform = spectrogram_transform

        # Create network for time series
        self.model_time_series = nn.Sequential(
            Backbone(
                n_channels=n_channels,
                n_classes=n_classes,
                padding="valid",
            ),
            nn.AdaptiveAvgPool1d(1),
        )

        # Create network for spectrogram (replace first conv layer)
        net = efficientnet_v2_s(num_classes=n_classes)
        _conv0_prev = net.features[0][0]
        _conv0 = nn.Conv2d(
            5,
            _conv0_prev.out_channels,
            _conv0_prev.kernel_size,
            stride=_conv0_prev.stride,
            padding=_conv0_prev.padding,
            bias=_conv0_prev.bias,
        )
        _conv0.weight = nn.init.kaiming_normal_(_conv0.weight, mode="fan_out")
        net.features[0][0] = _conv0
        self.model_spectrogram = net

        # Create network for head
        self.head = nn.Sequential(
            nn.Conv1d(n_classes * 2, n_classes * 2, kernel_size=1),
            nn.BatchNorm1d(n_classes * 2),
            nn.LeakyReLU(),
            nn.Conv1d(n_classes * 2, n_classes, kernel_size=1),
        )

    def forward(self, x):
        xs = self.spectrogram_transform(x)
        y_hat_sp = self.model_spectrogram(xs)

        y_hat_ts = self.model_time_series(x).squeeze(-1)

        y_hat = torch.concat([y_hat_sp, y_hat_ts], dim=-1).unsqueeze(-1)
        y_hat = self.head(y_hat)
        return y_hat.squeeze(-1)


## Config
def model_config(hparams):
    n_channels = 19  # 18 bipolar EEG chs, 1 ECG ch
    n_classes = 1

    spectrogram_transform = nn.Sequential(
        MultiTaperSpectrogram(
            int(hparams["config"]["sample_rate"]),
            int(hparams["config"]["sample_rate"]),
            hop_length=int(hparams["config"]["sample_rate"]) // 2,
            center=False,
            power=2,
        ),
        PostProcessSpectrograms(hparams["config"]["sample_rate"], max_frequency=80),
        AggregateSpectrograms(),
        nn.BatchNorm2d(num_features=n_channels),
    )

    return MyModel(
        n_channels=n_channels,
        n_classes=n_classes,
        spectrogram_transform=spectrogram_transform,
    )


def transforms(hparams):
    return [
        *[
            TransformIterable(["EEG"], transform)
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
        t.Scale({"ECG": 1 / 1e4, "EEG": 1 / (35 * 1.5)}),
        t.JoinArrays(),
        t.TanhClipNpArray(4),
        t.ToTensor(),
    ]


def metrics(hparams):
    class_names = VOTE_NAMES
    return {
        "mse": m.MetricWrapper(
            lambda y_pred, y: (torch.softmax(y_pred, dim=1), y),
            MeanSquaredError(),
        ),
        "mean_y_pred": m.MetricWrapper(
            lambda y_pred, y: (torch.softmax(y_pred, dim=1), y),
            m.MeanProbability(class_names=class_names),
        ),
        "mean_y": m.MetricWrapper(
            lambda y_pred, y: (y, y_pred),
            m.MeanProbability(class_names=class_names),
        ),
        # "cross_entropy": m.MetricWrapper(
        #     lambda y_pred, y: (torch.softmax(y_pred, dim=1), y),
        #     m.PooledMean(
        #         nn.CrossEntropyLoss(),
        #     ),
        # ),
        "prob_distribution": m.MetricWrapper(
            lambda y_pred, y: (torch.softmax(y_pred, dim=1), y),
            m.ProbabilityDensity(class_names=class_names),
        ),
    }


def train_config(hparams):
    optimizer_factory = partial(
        optim.Adam,
        lr=hparams["config"]["learning_rate"],
    )

    scheduler_factory = lambda opt: {
        "scheduler": optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=hparams["config"]["milestones"],
            gamma=hparams["config"]["gamma"],
        ),
        "monitor": hparams["config"]["monitor"],
    }

    module = TrainModule(
        model_config(hparams),
        loss_function=KLDivWithLogitsLoss(reduction="batchmean"),
        metrics=metrics(hparams),
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )

    data_dir = "./data/hms/train_eegs"

    train_dataset = HmsDataset(
        data_dir=data_dir,
        annotations=pd.read_csv(DATA_PROCESSED_DIR / "train.csv"),
        augmentation=TransformCompose(
            TransformIterable(["EEG"], t.RandomSaggitalFlipNpArray())
        ),
        transform=TransformCompose(
            *transforms(hparams),
            t.VotesToProbabilities(),
        ),
    )

    val_dataset = HmsDataset(
        data_dir=data_dir,
        annotations=pd.read_csv(DATA_PROCESSED_DIR / "val.csv"),
        transform=TransformCompose(
            *transforms(hparams),
            t.VotesToProbabilities(),
        ),
    )

    return dict(
        model=module,
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


def predict_config(hparams, predict_args):
    module = PredictModule(
        model_config(hparams),
        transform=TransformCompose(
            *output_transforms(hparams),
            lambda y_pred, md: (y_pred.cpu().numpy(), md),
        ),
    )

    weights_path, *dataset_args = predict_args
    weights_path = Path(weights_path)
    ckpt = torch.load(weights_path, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"])

    data_dir = Path(dataset_args[0]).expanduser()
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
