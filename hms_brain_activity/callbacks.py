import pytorch_lightning as pl


class EpochProgress(pl.Callback):
    """Dead simple callback to print a message when an epoch completes (a quieter alternative to the
    progress bar).
    """

    def on_train_epoch_end(self, trainer, module):
        print(f"Finished epoch {module.current_epoch + 1:04}")
