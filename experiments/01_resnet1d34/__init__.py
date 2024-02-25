import os
from collections import OrderedDict
from functools import partial

import pytorch_lightning as pl
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision.transforms.v2 import Compose

from hms_brain_activity.module import MainModule
from hms_brain_activity.datasets import HmsLocalClassificationDataset
from hms_brain_activity import transforms as t
from hms_brain_activity.utils import split_annotations_across_patients


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


def config(hparams):
    num_channels = 19
    num_classes = 6

    module = MainModule(
        nn.Sequential(
            t.DoubleBananaMontage(),
            t.ScaleEEG(1 / (35 * 1.5)),
            t.ScaleECG(1 / 1e4),
            t.TanhClipTensor(4),
            ResNet1d34Backbone(num_channels),
            ClassificationHead1d(ResNet1d34Backbone.channels[-1], num_classes),
            nn.LogSoftmax(dim=1),
        ),
        loss_function=nn.KLDivLoss(reduction="batchmean"),
        metrics_preprocessor=lambda y_pred, y: (y_pred.squeeze(-1), y.squeeze(-1)),
        metrics={
            "mse": MeanSquaredError(),
        },
        optimizer_factory=partial(
            optim.AdamW,
            lr=hparams["config"]["learning_rate"],
            weight_decay=hparams["config"]["weight_decay"],
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

    annotations = pd.read_csv("./data/hms/train.csv")

    train_annotations, val_annotaions = split_annotations_across_patients(
        annotations,
        test_size=0.2,
        random_state=0,
    )

    data_dir = "./data/hms/train_eegs"

    train_dataset = HmsLocalClassificationDataset(
        data_dir=data_dir,
        annotations=train_annotations,
        transform=Compose(
            [
                t.FillNanNpArray(0),
                t.PadNpArray(
                    t.BandPassNpArray(
                        hparams["config"]["bandpass_low"],
                        hparams["config"]["bandpass_high"],
                        hparams["config"]["sample_rate"],
                    ),
                    padlen=hparams["config"]["sample_rate"],
                ),
                t.ToTensor(),
                t.RandomSaggitalFlip(),
                t.RandomScale(),
                t.VotesToProbabilities(),
            ]
        ),
    )

    val_dataset = HmsLocalClassificationDataset(
        data_dir=data_dir,
        annotations=val_annotations,
        transform=Compose(
            [
                t.FillNanNpArray(0),
                t.PadNpArray(
                    t.BandPassNpArray(
                        hparams["config"]["bandpass_low"],
                        hparams["config"]["bandpass_high"],
                        hparams["config"]["sample_rate"],
                    ),
                    padlen=hparams["config"]["sample_rate"],
                ),
                t.ToTensor(),
                t.VotesToProbabilities(),
            ]
        ),
    )

    return dict(
        model=module,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=hparams["config"].get("num_workers", os.cpu_count()) or 0,
            shuffle=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=hparams["config"].get("num_workers", os.cpu_count()) or 0,
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
