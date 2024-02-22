import logging
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .globals import CHANNEL_NAMES, VOTE_NAMES


logger = logging.getLogger(__name__)


class HmsLocalClassificationDataset(Dataset):
    """Dataset class for loading time series data"""

    sample_rate = 200.0  # Hz
    sample_secs = 50.0  # s
    channel_names = CHANNEL_NAMES
    vote_names = VOTE_NAMES

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

        self.filepaths = list(self.data_dir.glob("*.parquet"))

        n_files = len(self.filepaths)
        n_eeg_ids = self.annotations["eeg_id"].nunique()
        if n_files < n_eeg_ids:
            logger.warning(
                f"Only {n_files} / {n_eeg_ids} were found, filtering annotations list"
            )
            found_eeg_ids = set(int(fp.stem) for fp in self.filepaths)
            self.annotations = self.annotations[
                self.annotations["eeg_id"].isin(found_eeg_ids)
            ]

        if len(self.annotations) == 0:
            raise ValueError("No samples found.")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        annotation = self.annotations.iloc[idx]
        start = int(annotation["eeg_label_offset_seconds"] * self.sample_rate)
        duration = int(self.sample_rate * self.sample_secs)

        eeg_id = annotation["eeg_id"]
        eeg_path = self.data_dir / f"{eeg_id}.parquet"
        data = pd.read_parquet(eeg_path)
        data = data[self.channel_names]
        data = data.iloc[start : start + duration].to_numpy().transpose()

        label = annotation[self.vote_names].to_numpy()
        metadata = {
            "y": label,
            "patient_id": annotation['patient_id'],
        }

        if self.transform:
            data, metadata = self.transform(data, metadata)

        return data, label
