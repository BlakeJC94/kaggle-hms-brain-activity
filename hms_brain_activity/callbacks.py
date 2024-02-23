import logging
import pytorch_lightning as pl

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
