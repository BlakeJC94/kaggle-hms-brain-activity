import os
from pathlib import Path
from functools import partial

import pytorch_lightning as pl
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms.v2 import Compose

from hms_brain_activity.metadata_classes import ModelConfig
from hms_brain_activity.module import MainModule
from hms_brain_activity.datasets import HmsLocalClassificationDataset
from hms_brain_activity import transforms as t


class PlaceholderModel(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(num_channels, num_classes, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        return nn.functional.softmax(x, dim=1)


def config(hparams) -> ModelConfig:
    num_channels = 20
    num_classes = 6

    module = MainModule(
        PlaceholderModel(num_channels=num_channels, num_classes=num_classes),
        loss_function=nn.MSELoss(),
        metrics={
            "accuracy": MulticlassAccuracy(num_classes=num_classes),
        },
        optimizer_factory=partial(
            optim.AdamW,
            lr=hparams["config"]["learning_rate"],
            weight_decay=hparams["config"]["weight_decay"],
        ),
        scheduler_factory=lambda opt: {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=[20, 40, 60],
                gamma=0.2,
            ),
            "monitor": hparams["config"]["monitor"],
        },
    )

    annotations = pd.read_csv("./data/hms/train.csv")

    # TODO Find a better subsampling strategy
    val_patient_ids = set(annotations['patient_id'].unique().sample(frac=0.2, random_seed=0))
    val_annotations = annotations[annotations['patient_ids'].isin(val_patient_ids)]
    train_annotations = annotations[~annotations['patient_ids'].isin(val_patient_ids)]

    data_dir = "./data/hms/train_eegs"

    train_dataset = HmsLocalClassificationDataset(
        data_dir=data_dir,
        annotations=train_annotations,
        transform=Compose(
            [
                t.ToTensor(),
            ]
        ),
    )

    val_dataset = HmsLocalClassificationDataset(
        data_dir=data_dir,
        annotations=val_annotations,
        transform=Compose(
            [
                t.ToTensor(),
            ]
        ),
    )

    return ModelConfig(
        project=hparams["task"]["init"]["project_name"],
        experiment_name="-".join(Path(__file__).parts[-2:]),
        model=module,
        train_dataloader=DataLoader(
            train_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=hparams["config"].get("num_workers") or (os.cpu_count() or 0),
            shuffle=True,
        ),
        val_dataloader=DataLoader(
            val_dataset,
            batch_size=hparams["config"]["batch_size"],
            num_workers=hparams["config"].get("num_workers") or (os.cpu_count() or 0),
            shuffle=False,
        ),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="loss/val",
                min_delta=0.0001,
                patience=hparams["config"]["patience"],
                verbose=True,
                mode="min",
            ),
        ],
    )
