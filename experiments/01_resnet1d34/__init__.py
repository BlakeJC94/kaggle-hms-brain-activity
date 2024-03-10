import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision.transforms.v2 import Compose

from hms_brain_activity.module import TrainModule, PredictModule
from hms_brain_activity.datasets import HmsDataset, HmsPredictDataset
from hms_brain_activity import transforms as t
from hms_brain_activity.paths import DATA_PROCESSED_DIR


class BasicBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()  # Could equivalently use F.relu()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        if in_channels != out_channels or stride != 1:
            self.projection_shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm1d(num_features=out_channels),
            )
        else:
            self.projection_shortcut = lambda x: x

    def forward(self, x):
        identity = self.projection_shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x


class ResNet1d34Backbone(nn.Sequential):
    channels = (64, 128, 256, 512)

    def __init__(self, in_channels: int):
        super().__init__()

        # Layer 1
        conv1 = OrderedDict(
            conv1=nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.channels[0],
                kernel_size=7,
                stride=2,
                padding=0,
                bias=False,
            ),
            bn=nn.BatchNorm1d(num_features=self.channels[0]),
            relu=nn.ReLU(),
        )
        self.conv1 = nn.Sequential(conv1)

        # Layer 2
        in_channels2 = self.channels[0]
        conv2 = OrderedDict(
            mp=nn.MaxPool1d(kernel_size=3, stride=2, padding=0),
            conv2_1=BasicBlock1d(self.channels[0], self.channels[0], stride=1),
            conv2_2=BasicBlock1d(in_channels2, self.channels[0], stride=1),
            conv2_3=BasicBlock1d(in_channels2, self.channels[0], stride=1),
        )
        self.conv2 = nn.Sequential(conv2)

        # Layer 3
        in_channels3 = self.channels[1]
        conv3 = OrderedDict(
            conv3_1=BasicBlock1d(in_channels2, self.channels[1], stride=2),
            conv3_2=BasicBlock1d(in_channels3, self.channels[1], stride=1),
            conv3_3=BasicBlock1d(in_channels3, self.channels[1], stride=1),
            conv3_4=BasicBlock1d(in_channels3, self.channels[1], stride=1),
        )
        self.conv3 = nn.Sequential(conv3)

        # Layer 4
        in_channels4 = self.channels[2]
        conv4 = OrderedDict(
            conv4_1=BasicBlock1d(in_channels3, self.channels[2], stride=2),
            conv4_2=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_3=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_4=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_5=BasicBlock1d(in_channels4, self.channels[2], stride=1),
            conv4_6=BasicBlock1d(in_channels4, self.channels[2], stride=1),
        )
        self.conv4 = nn.Sequential(conv4)

        # Layer 5
        in_channels5 = self.channels[3]
        conv5 = OrderedDict(
            conv5_1=BasicBlock1d(in_channels4, self.channels[3], stride=2),
            conv5_2=BasicBlock1d(in_channels5, self.channels[3], stride=1),
            conv5_3=BasicBlock1d(in_channels5, self.channels[3], stride=1),
        )
        self.conv5 = nn.Sequential(conv5)
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weight_bias)

    def _init_weight_bias(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)


class ClassificationHead1d(nn.Sequential):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(num_channels, num_classes, 1)


def model_config(hparams):
    num_channels = 19
    num_classes = 6
    return nn.Sequential(
        ResNet1d34Backbone(num_channels),
        ClassificationHead1d(ResNet1d34Backbone.channels[-1], num_classes),
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
