from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader


@dataclass
class ModelConfig:
    """A container class to bundle together a PyTorch Lightning Module, DataLoaders, and various parameters for automatic logging to ClearML, such as project and
    experiment name.

    Args:
        project: The ClearML project/local folder to save experiment artefacts to, e.g. 'epilepsy'.
            NB ClearML can create nested project folders, e.g. for 'epilepsy/multilabel'.
        experiment_name: The name of the experiment in ClearML. Also included the path for saving
            artefacts locally.
        model: A model object to train.
        train_dataloader: A data loader to use for training.
        val_dataloader: A data loader to use for validation.
        test_dataloader: A data loader to use for testing.
        callbacks: An optional list of `pytorch_lightning.callbacks.Callback` objects to use with
            the `pytorch_lightning.Trainer` class.
        tags: Optional tags to attach to the ClearML experiment.
        user_properties: Optional dict of {key: value} pairs, to be logged as additional user
            properties in ClearML.
        class_names: Optional dict of {class_name: int} for logging to ClearML,
            e.g. {'normal': 0, 'epileptiform': 1}
    """

    project: str
    experiment_name: str
    model: pl.LightningModule = field(repr=False)
    train_dataloader: Optional[DataLoader] = field(repr=False, default=None)
    val_dataloader: Optional[DataLoader] = field(repr=False, default=None)
    test_dataloader: Optional[DataLoader] = field(repr=False, default=None)
    callbacks: List[pl.callbacks.Callback] = field(repr=False, default_factory=list)
    tags: Sequence[str] = field(default_factory=list)
    user_properties: Mapping[str, str] = field(default_factory=lambda: {})
    class_names: Optional[Mapping[str, int]] = None
