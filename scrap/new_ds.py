from pathlib import Path

import pandas as pd
import numpy as np
from torchvision.transforms.v2 import Compose

from hms_brain_activity.datasets import HmsDataset
from hms_brain_activity.vis import dataset_dash, plot_channels
from hms_brain_activity.paths import DATA_DIR
from hms_brain_activity import transforms as t


data_dir = Path("./data/hms/train_eegs")

annotations = pd.read_csv(DATA_DIR / "hms/train.csv")
annotations = annotations[
    annotations["eeg_id"].isin([int(fp.stem) for fp in data_dir.glob("*.parquet")])
]

class HmsViewDataset(HmsDataset):
    sample_secs = 10

ds = HmsViewDataset(
    data_dir=data_dir,
    annotations=annotations,
    augmentation=t.TransformCompose(
        t.TransformIterable(
            t.RandomSaggitalFlipNpArray(),
            apply_to=["EEG"]
        )
    ),
    transform=t.TransformCompose(
        *[
            t.TransformIterable(transform, apply_to=["EEG"])
            for transform in [
                t.Pad(padlen=200),
                t.BandPassNpArray(
                    0.3,
                    45,
                    200,
                ),
                t.Unpad(padlen=200),
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
        t.VotesToProbabilities(),
    ),
)


# %%
dataset_dash(ds)

# i = 0
# x, y = ds.get_raw(i)

# fig = plot_channels(x)

