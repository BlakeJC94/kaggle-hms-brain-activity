import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Callable, Any, Dict, TypeAlias

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hms_brain_activity.globals import CHANNEL_NAMES, VOTE_NAMES


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
            "patient_id": annotation.get("patient_id", "None"),
        }

        if self.transform:
            data, metadata = self.transform(data, metadata)

        return data, metadata


# class HmsStrideDataset(_BaseHmsDataset):
#     def __init__(self, hop_secs: Optional[float] = _BaseHmsDataset.sample_secs)
#         ...
#     ...


# ---

CollateData: TypeAlias = int | float | str | np.ndarray | torch.Tensor


class BaseDataset(Dataset, ABC):
    transform = None
    augmentation = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_raw_data(self, i: int) -> Any:
        pass

    @abstractmethod
    def get_raw_label(self, i: int) -> Any:
        pass

    @abstractmethod
    def get_additional_metadata(self, i: int) -> Dict[str, CollateData]:
        pass

    def get_raw(self, i: int) -> Any:
        data = self.get_raw_data(i)
        metadata = {
            **self.get_additional_metadata(i),
            "idx": i,
        }
        label = self.get_raw_label(i)
        if label is not None:
            metadata["y"] = self.get_raw_label(i)
        return data, metadata

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        x, y = self.get_raw(i)

        if self.augmentation is not None:
            x, y = self.augmentation(x, y)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y


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
        return np.expand_dims(label.astype(int), -1)

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
