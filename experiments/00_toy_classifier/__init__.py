import os
from functools import partial

import pytorch_lightning as pl
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision.transforms.v2 import Compose

from hms_brain_activity.module import MainModule
from hms_brain_activity.datasets import HmsClassificationDataset
from hms_brain_activity import transforms as t
from hms_brain_activity.utils import split_annotations_across_patients


class PlaceholderModel(nn.Module):
    def __init__(self, num_channels, num_classes, num_filters=32):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(num_features=num_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(num_features=num_filters)
        self.conv3 = nn.Conv1d(
            in_channels=num_filters, out_channels=num_classes, kernel_size=1, bias=True
        )
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avg(x)
        x = self.conv3(x)
        return x


def config(hparams):
    num_channels = 19
    num_classes = 6

    module = MainModule(
        nn.Sequential(
            t.DoubleBananaMontage(),
            t.ScaleEEG(1 / (35 * 1.5)),
            t.ScaleECG(1 / 1e4),
            t.TanhClipTensor(4),
            PlaceholderModel(num_channels=num_channels, num_classes=num_classes),
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

    train_annotations, val_annotations = split_annotations_across_patients(
        annotations,
        test_size=0.2,
        random_state=0,
    )

    data_dir = "./data/hms/train_eegs"

    train_dataset = HmsClassificationDataset(
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

    val_dataset = HmsClassificationDataset(
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
