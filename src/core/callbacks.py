import logging
import os
from pathlib import Path

import psutil
import pytorch_lightning as pl
import torch


logger = logging.getLogger(__name__)


class EpochProgress(pl.Callback):
    """Dead simple callback to print a message when an epoch completes (a quieter alternative to the
    progress bar).
    """

    @staticmethod
    def num_batches(val):
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        return val

    def on_train_start(self, trainer, module):
        logger.info(
            f"Starting training with {self.num_batches(trainer.num_training_batches)} batches"
        )

    def on_validation_start(self, trainer, module):
        logger.info(
            f"Starting validation with {self.num_batches(trainer.num_val_batches)} batches"
        )

    def on_train_epoch_end(self, trainer, module):
        logger.info(f"Finished epoch {module.current_epoch + 1:04}")


class NanMonitor(pl.Callback):
    """Raise if any Nans are encountered"""

    def check(self, batch_idx, batch, outputs=None):
        outputs = outputs or {}
        to_check = {
            "x": batch[0],
            "y": batch[1]["y"],
            "y_pred": outputs.get("y_pred"),
            "loss": outputs.get("loss"),
            "out": outputs.get("out"),
        }
        for k, v in to_check.items():
            if v is None:
                continue
            v = torch.isnan(v)
            if v.ndim > 1:
                v = v.flatten(1).any(1)
            if v.any():
                nan_idxs_str = ""
                if v.ndim > 0:
                    nan_idxs = [i for i, b in enumerate(v) if b]
                    nan_idxs_str = ", ".join([str(idx) for idx in nan_idxs[:5]])
                    if len(nan_idxs) > 5:
                        nan_idxs_str += f", ... [{len(nan_idxs)}]"
                raise ArithmeticError(
                    f"Encountered NaN in '{k}' for batch {batch_idx} (samples {nan_idxs_str})"
                )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)


class PidMonitor(pl.Callback):
    FILENAME = "./.train.{i}.pid"

    def __init__(self):
        super().__init__()
        running_pids = psutil.pids()

        globpat = self.FILENAME.split("/")[-1].replace("{i}", "*")
        for fp in Path(".").glob(globpat):
            pid = int(fp.read_text())
            if pid in running_pids:
                fp.unlink()

        prev_files = Path(".").glob(globpat)
        max_pid_i = -1
        for fp in prev_files:
            pid_i = int(fp.suffixes[0].removeprefix("."))
            max_pid_i = max(max_pid_i, pid_i)
        self.filename = Path(self.FILENAME.format(i=max_pid_i + 1))

    def on_fit_start(self, trainer, pl_module):
        with open(self.filename, "w") as f:
            f.write(str(os.getpid()))

    def on_fit_end(self, trainer, pl_module):
        if self.filename.exists():
            self.filename.unlink()

    def on_exception(self, trainer, pl_module, exception):
        if self.filename.exists():
            self.filename.unlink()
