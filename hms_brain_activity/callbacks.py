import logging
import pytorch_lightning as pl
from pathlib import Path

import pandas as pd
import numpy as np

from hms_brain_activity.globals import VOTE_NAMES


logger = logging.getLogger(__name__)


class EpochProgress(pl.Callback):
    """Dead simple callback to print a message when an epoch completes (a quieter alternative to the
    progress bar).
    """

    def on_train_start(self, trainer, module):
        logger.info(f"Starting training with {trainer.num_training_batches} batches")

    def on_validation_start(self, trainer, module):
        logger.info(f"Starting validation with {trainer.num_val_batches} batches")

    def on_train_epoch_end(self, trainer, module):
        logger.info(f"Finished epoch {module.current_epoch + 1:04}")


class SubmissionWriter(pl.callbacks.BasePredictionWriter):
    filename = "submission.csv"

    def __init__(self, output_dir: Path):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        assert self.output_dir.is_dir()
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_path = self.output_dir / self.filename
        if self.output_path.exists():
            self.output_path.unlink()

    def write_on_batch_end(
        self,
        _trainer,
        _pl_module,
        prediction,
        _batch_indices,
        _batch,
        _batch_idx,
        _dataloader_idx=0,
    ):
        out, md = prediction["out"], prediction["md"]
        out = out.cpu().squeeze(-1).numpy()
        eeg_ids = md['eeg_id']

        rows = pd.DataFrame(
            { "eeg_id": eeg_ids, **{col: out[:,i] for i, col in enumerate(VOTE_NAMES)} }
        )
        rows = rows[["eeg_id", *VOTE_NAMES]]
        sums = rows[VOTE_NAMES].sum(axis=1)
        rows[VOTE_NAMES] = rows[VOTE_NAMES] / np.expand_dims(sums.to_numpy(), -1)

        mode = "w" if not self.output_path.exists() else "a"
        rows.to_csv(self.output_path, index=False, mode=mode)
