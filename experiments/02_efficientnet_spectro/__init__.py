import os
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision.transforms.v2 import Compose
from torchvision.models.efficientnet import efficientnet_v2_s
from torchaudio.transforms import Spectrogram

from hms_brain_activity.module import TrainModule, PredictModule
from hms_brain_activity.datasets import HmsDataset, HmsPredictDataset
from hms_brain_activity import transforms as t
from hms_brain_activity.paths import DATA_PROCESSED_DIR


class AggregateSpectrograms(nn.Module):
    def forward(self, x):
        out = [
            torch.nanmean(x[:, sl, :, :], dim=1, keepdim=True)
            for sl in [
                slice(0, 4),
                slice(4, 8),
                slice(8, 12),
                slice(12, 16),
                slice(16, 18),
            ]
        ]
        return torch.cat(out, dim=1)


class AsymmetricSpectrograms(nn.Module):
    def forward(self, x):
        out = [x]
        for i, j in [
            (1, 2),
            (3, 4),
        ]:
            res = (x[:, j, ...] / (x[:, i, ...] + x[:, j, ...]) * 100) - 50
            out.append(res.unsqueeze(1))
        return torch.cat(out, dim=1)


def model_config(hparams):
    num_channels = 19
    num_classes = 6

    # Create Network
    net = efficientnet_v2_s(num_classes=num_classes)

    # Replace first convolution
    _conv0_prev = net.features[0][0]
    _conv0 = nn.Conv2d(
        num_channels,
        _conv0_prev.out_channels,
        _conv0_prev.kernel_size,
        stride=_conv0_prev.stride,
        padding=_conv0_prev.padding,
        bias=_conv0_prev.bias,
    )
    _conv0.weight = nn.init.kaiming_normal_(_conv0.weight, mode="fan_out")
    net.features[0][0] = _conv0

    return nn.Sequential(
        Spectrogram(
            int(hparams['config']["sample_rate"]),
            hop_length=int(hparams['config']["sample_rate"]),
            center=False,
            power=2,
        ),
        AggregateSpectrograms(),
        AsymmetricSpectrograms(),
        net,
        nn.LogSoftmax(dim=1),
    )


def transforms(hparams):
    return [
        *[
            t.TransformIterable(transform, apply_to=["EEG"])
            for transform in [
                t.Pad(padlen=hparams["config"]["sample_rate"]),
                t.BandPassNpArray(
                    hparams["config"]["bandpass_low"],
                    hparams["config"]["bandpass_high"],
                    hparams["config"]["sample_rate"],
                ),
                t.Unpad(padlen=hparams["config"]["sample_rate"]),
                t.Scale(1 / (35 * 1.5)),
                t.DoubleBananaMontageNpArray(),
            ]
        ],
        t.TransformIterable(
            t.Scale(1 / 1e4),
            apply_to=["ECG"],
        ),
        t.JoinArrays(),
        t.TanhClipNpArray(4),
        t.ToTensor(),
    ]


def train_config(hparams):
    module = TrainModule(
        model_config(hparams),
        loss_function=nn.KLDivLoss(reduction="batchmean"),
        metrics={
            "mse": MeanSquaredError(),
        },
        optimizer_factory=partial(
            optim.Adam,
            lr=hparams["config"]["learning_rate"],
        ),
        scheduler_factory=lambda opt: {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=hparams["config"]["milestones"],
                gamma=hparams["config"]["gamma"],
            ),
            "monitor": hparams["config"]["monitor"],
        },
    )

    data_dir = "./data/hms/train_eegs"

    train_dataset = HmsDataset(
        data_dir=data_dir,
        annotations=pd.read_csv(DATA_PROCESSED_DIR / "train.csv"),
        augmentation=t.TransformCompose(
            t.TransformIterable(
                t.RandomSaggitalFlipNpArray(),
                apply_to=["EEG"]
            )
        ),
        transform=t.TransformCompose(
            *transforms(hparams),
            t.VotesToProbabilities(),
        ),
    )

    val_dataset = HmsDataset(
        data_dir=data_dir,
        annotations=pd.read_csv(DATA_PROCESSED_DIR / "val.csv"),
        transform=t.TransformCompose(
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


def predict_config(hparams):
    module = PredictModule(
        model_config(hparams),
        transform=Compose(
            [
                lambda y_pred, md: (y_pred.squeeze(-1), md),
                lambda y_pred, md: (torch.exp(y_pred), md),
                lambda y_pred, md: (y_pred.to(torch.double), md),
                lambda y_pred, md: (torch.softmax(y_pred, axis=1), md),
                lambda y_pred, md: (y_pred.cpu().numpy(), md),
            ]
        ),
    )

    weights_path = Path(hparams["predict"]["weights_path"])
    ckpt = torch.load(weights_path, map_location="cpu")
    module.load_state_dict(ckpt["state_dict"], strict=False)

    data_dir = Path(hparams["predict"]["data_dir"])
    annotations = pd.DataFrame(
        {"eeg_id": [fp.stem for fp in data_dir.glob("*.parquet")]}
    )
    predict_dataset = HmsPredictDataset(
        data_dir=data_dir,
        annotations=annotations,
        transform=Compose(transforms(hparams)),
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
