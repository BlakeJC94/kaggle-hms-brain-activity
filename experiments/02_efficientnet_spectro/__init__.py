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

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hms_brain_activity import metrics as m
from hms_brain_activity import transforms as t
from hms_brain_activity.core.modules import PredictModule, TrainModule
from hms_brain_activity.core.transforms import (
    DataTransform,
    TransformCompose,
    TransformIterable,
    _BaseTransform,
)
from hms_brain_activity.datasets import HmsDataset, PredictHmsDataset
from hms_brain_activity.globals import VOTE_NAMES
from hms_brain_activity.paths import DATA_PROCESSED_DIR
from torch import nn, optim
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from torchmetrics import MeanSquaredError
from torchvision.models.efficientnet import efficientnet_v2_s


class PostProcessSpectrograms(_BaseTransform):
    def __init__(self, sample_rate, max_frequency):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_frequency = max_frequency

    def compute(self, x, md):
        _num_channels, num_freqs, _num_timesteps = x.shape
        x = x / self.sample_rate

        # Set near-0 values to a fixed floor to prevent log(tiny value) creating large negative
        # values that obscure the actual meaningful signal
        x[x < 1e-10] = 1e-10
        x = torch.log(x)

        # Trim unwanted frequencies
        frequencies = np.linspace(0, self.sample_rate / 2, num_freqs)
        frequency_mask = frequencies <= self.max_frequency
        # Leave batch, channel & time dims; slice the frequency dim
        x = x[..., frequency_mask, :]

        return x, md


class AggregateSpectrograms(_BaseTransform):
    def compute(self, x, md):
        out = [
            torch.nanmean(x[..., sl, :, :], dim=-3, keepdim=True)
            for sl in [
                slice(0, 4),
                slice(4, 8),
                slice(8, 12),
                slice(12, 16),
                # slice(16, 18),  # Sagittal plane EEG
                # slice(18, 19),  # ECG
            ]
        ]
        return torch.cat(out, dim=-3), md


def model_config(hparams):
    n_channels = 4
    n_classes = len(VOTE_NAMES)

    # Create Network
    net = efficientnet_v2_s(num_classes=n_classes)

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
        net,
        nn.LogSoftmax(dim=1),
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
        t.Scale({"EEG": 1 / (35 * 1.5), "ECG": 1 / 1e4}),
        TransformIterable(["EEG"], t.DoubleBananaMontageNpArray()),
        t.JoinArrays(),
        t.TanhClipNpArray(4),
        t.ToTensor(),
        DataTransform(
            Spectrogram(
                int(hparams["config"]["sample_rate"]),
                hop_length=int(hparams["config"]["sample_rate"]),
                center=False,
                power=2,
            ),
        ),
        PostProcessSpectrograms(hparams["config"]["sample_rate"], max_frequency=80),
        AggregateSpectrograms(),
    ]


def metrics(hparams):
    return {
        "mse": MeanSquaredError(),
        "mean_y_pred": m.MetricWrapper(
            TransformCompose(*output_transforms(hparams)),
            m.MeanProbability(class_names=VOTE_NAMES),
        ),
        "mean_y": m.MetricWrapper(
            lambda y_pred, y: (y, y_pred),
            m.MeanProbability(class_names=VOTE_NAMES),
        ),
        "cross_entropy": m.MetricWrapper(
            TransformCompose(*output_transforms(hparams)),
            m.PooledMean(
                nn.CrossEntropyLoss(),
            ),
        ),
        "prob_distribution": m.MetricWrapper(
            TransformCompose(*output_transforms(hparams)),
            m.ProbabilityDistribution(class_names=VOTE_NAMES),
        ),
        "prob_density": m.MetricWrapper(
            TransformCompose(*output_transforms(hparams)),
            m.ProbabilityDistribution(class_names=VOTE_NAMES),
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
        loss_function=nn.KLDivLoss(reduction="batchmean"),
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
        lambda y_pred, md: (torch.exp(y_pred), md),
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
    )
