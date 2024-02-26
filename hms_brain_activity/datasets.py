import logging
from pathlib import Path
from typing import Tuple, Optional, Callable, Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .globals import CHANNEL_NAMES, VOTE_NAMES


logger = logging.getLogger(__name__)


class _BaseHmsDataset(Dataset):
    sample_rate = 200.0  # Hz
    sample_secs = 50.0  # s
    channel_names = CHANNEL_NAMES
    vote_names = VOTE_NAMES

    def get_data(self, eeg_path: str | Path, start: int = 0, duration: int = 50 * 200):
        data = pd.read_parquet(eeg_path)
        data = data[self.channel_names]
        data = data.iloc[start : start + duration].to_numpy().transpose()
        metadata = {"eeg_id": Path(eeg_path).stem}
        return data, metadata


class HmsPredictDataset(_BaseHmsDataset):
    def __init__(self, data_dir: str | Path, transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.filepaths = list(self.data_dir.glob("*.parquet"))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i: int):
        data, metadata = self.get_data(self.filepaths[i])
        if self.transform:
            data, metadata = self.transform(data, metadata)
        return data, metadata


class HmsClassificationDataset(_BaseHmsDataset):
    def __init__(
        self,
        data_dir: str | Path,
        annotations: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.annotations = annotations
        self.transform = transform
        assert self.data_dir.exists()

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        annotation = self.annotations.iloc[idx]
        start_secs = annotation.get("eeg_label_offset_secs", 0)
        start = int(start_secs * self.sample_rate)
        duration = int(self.sample_rate * self.sample_secs)

        eeg_id = annotation["eeg_id"]
        eeg_path = self.data_dir / f"{eeg_id}.parquet"
        data, metadata = self.get_data(eeg_path, start, duration)

        label = np.zeros(len(self.vote_names))
        if all(col in annotation for col in self.vote_names):
            label = annotation[self.vote_names].to_numpy()

        metadata = {
            **metadata,
            "y": np.expand_dims(label.astype(int), -1),
            "patient_id": annotation.get("patient_id"),
        }

        if self.transform:
            data, metadata = self.transform(data, metadata)

        return data, metadata


# class HmsStrideDataset(_BaseHmsDataset):
#     def __init__(self, hop_secs: Optional[float] = _BaseHmsDataset.sample_secs)
#         ...
#     ...
