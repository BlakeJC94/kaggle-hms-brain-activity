import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl

from src.hms_brain_activity.globals import VOTE_NAMES

logger = logging.getLogger(__name__)


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
            logger.warning(f"Removing existing '{str(self.output_path)}'.")
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
        out, mds = prediction["out"], prediction["md"]

        rows = pd.DataFrame(
            {
                "eeg_id": mds["eeg_id"],
                **{col: out[:, i] for i, col in enumerate(VOTE_NAMES)},
            }
        )
        rows = rows[["eeg_id", *VOTE_NAMES]]

        opts = (
            dict(mode="w", header=True)
            if not self.output_path.exists()
            else dict(mode="a", header=False)
        )
        rows.to_csv(self.output_path, index=False, **opts)
