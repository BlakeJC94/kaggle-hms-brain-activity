from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, TypeAlias

from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset

CollateData: TypeAlias = int | float | str | ndarray | Tensor


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
