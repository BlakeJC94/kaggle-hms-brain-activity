import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from core.datasets import BaseDataset, CollateData
from hms_brain_activity.globals import CHANNEL_NAMES, VOTE_NAMES

logger = logging.getLogger(__name__)


class HmsReaderMixin(Dataset):
    sample_rate = 200.0  # Hz
    sample_secs = 50.0  # s
    channel_names = CHANNEL_NAMES
    vote_names = VOTE_NAMES
    channel_groups = {
        "EEG": CHANNEL_NAMES[:-1],
        "ECG": CHANNEL_NAMES[-1:],
    }
    nan_val = 0

    def read_data(self, eeg_path: str | Path, start: int = 0, duration: int = 50 * 200):
        data = pd.read_parquet(eeg_path)
        data = data[self.channel_names]
        data = data.iloc[start : start + duration].to_numpy().transpose()
        data = np.nan_to_num(data, self.nan_val)
        data = {
            "EEG": data[:-1, :],
            "ECG": data[-1:, :],
        }
        return data

    def read_label(self, annotation):
        label = np.zeros(len(self.vote_names))
        if all(col in annotation for col in self.vote_names):
            label = annotation[self.vote_names].to_numpy()

        label = np.nan_to_num(label, self.nan_val)
        return label.astype(int)


class HmsDataset(BaseDataset, HmsReaderMixin):
    def __init__(
        self,
        data_dir: str | Path,
        annotations: pd.DataFrame,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.annotations = annotations
        self.augmentation = augmentation
        self.transform = transform
        assert self.data_dir.exists()

    def __len__(self):
        return len(self.annotations)

    def get_raw_data(self, i: int) -> Any:
        annotation = self.annotations.iloc[i]

        start_secs = annotation.get("eeg_label_offset_secs", 0)
        start = int(start_secs * self.sample_rate)
        duration = int(self.sample_rate * self.sample_secs)

        eeg_id = annotation["eeg_id"]
        eeg_path = self.data_dir / f"{eeg_id}.parquet"
        return self.read_data(eeg_path, start, duration)

    def get_raw_label(self, i: int) -> Any:
        annotation = self.annotations.iloc[i]
        return self.read_label(annotation)

    def get_additional_metadata(self, i: int) -> Dict[str, CollateData]:
        annotation = self.annotations.iloc[i]
        return {
            "patient_id": annotation.get("patient_id", "None"),
            "eeg_id": annotation.get("eeg_id", "None"),
        }

    def get_data(self, eeg_path: str | Path, start: int = 0, duration: int = 50 * 200):
        data = pd.read_parquet(eeg_path)
        data = data[self.channel_names]
        data = data.iloc[start : start + duration].to_numpy().transpose()
        metadata = {"eeg_id": Path(eeg_path).stem}
        return data, metadata


class PredictHmsDataset(HmsDataset):
    def __init__(self, data_dir: str | Path, transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        self.annotations = pd.DataFrame(
            {"eeg_id": [fp.stem for fp in data_dir.glob("*.parquet")]}
        )

    def get_raw_label(self, i: int) -> Any:
        return None
