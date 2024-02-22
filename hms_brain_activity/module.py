from copy import deepcopy
from typing import Callable, Optional, TypeAlias, Dict, List, Any

import pytorch_lightning as pl
import torch
from clearml import Logger
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import Metric, MeanMetric

from hms_brain_activity.metrics import PooledMean


LRScheduler: TypeAlias = lr_scheduler._LRScheduler


class MainModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: Optional[nn.Module],
        optimizer_factory: Callable,
        scheduler_factory: Optional[Callable] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        metrics_preprocessor: Optional[Callable] = None,
        hyperparams_ignore: Optional[List[str]] = None,
    ):
        """

        Args:
            model: PyTorch module to call in the forward method.
            loss_function: Loss function to call for training and validation batches.
            optimizer_factory: Callable that returns an optimizer to use for training. It should
                expect a single argument, `self.parameters()`.
            scheduler_factory: Optional callable that returns a learning rate scheduler. It should
                expect a single argument, the registered optimizer.
            metrics: A dict of {metric_name: function}. Functions should accept 2 args:
                predictions and labels, and return a scalar number.
            metrics_preprocessor: Optional callable to apply to preds (e.g. if preds are logits)
                before calculating metrics (by default no operation is applied).
            hyperparams_ignore: A list of attribute strings to add to the `ignore` list in
                `save_hyperparameters`.
        """
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        metrics = metrics or {}
        metrics = {
            k: metric if isinstance(metric, Metric) else PooledMean(metric)
            for k, metric in metrics.items()
        }
        self.metrics = nn.ModuleDict(
            {
                k: nn.ModuleDict(deepcopy(metrics))
                for k in ["train", "sanity_check", "validate", "test", "predict"]
            }
        )
        self.metrics_preprocessor = metrics_preprocessor

        hyperparams_ignore = hyperparams_ignore or []
        self.save_hyperparameters(
            ignore=[
                "loss_function",
                "optimizer_factory",
                "scheduler_factory",
                "metrics",
                "model",
                "metrics_preprocessor",
                *hyperparams_ignore,
            ],
        )

    def configure_optimizers(self) -> Dict[str, Optimizer | LRScheduler]:
        """Return your favourite optimizer."""
        out = {}
        if self.optimizer_factory:
            out["optimizer"] = self.optimizer_factory(
                filter(lambda p: p.requires_grad, self.parameters())
            )
        if self.scheduler_factory:
            out["lr_scheduler"] = self.scheduler_factory(out["optimizer"])
        return out

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    ## Metric and loss logging methods
    def loss_calculate_and_log(self, y_pred: Any, md: Any) -> torch.Tensor:
        y = md["y"]
        stage = self.get_stage()
        loss = self.loss_function(y_pred, y)
        if isinstance(loss, dict):
            self._log_metric(loss, "loss", stage, batch_size=len(y_pred))
            loss = torch.stack(list(loss.values())).mean()
        self.log(
            f"loss/{stage}",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(y_pred),
        )
        return loss

    @torch.no_grad()
    def metrics_update_and_log(self, y_pred: Any, md: Any) -> None:
        """Calculate and log metrics for a batch of predictions against target labels."""
        y = md["y"]
        stage = self.get_stage()
        if self.metrics_preprocessor is not None:
            y_pred, y = self.metrics_preprocessor(y_pred, y)

        metrics = self.metrics.get(stage, {})
        for metric_name, metric in metrics.items():
            if isinstance(metric, MeanMetric):
                metric.update(y_pred)
            else:
                metric.update(y_pred, y)
            if getattr(metric, "compute_on_batch", True):
                self._log_metric(metric, metric_name, stage, batch_size=len(y_pred))

    @torch.no_grad()
    def metrics_compute_and_log(self):
        stage = self.get_stage()
        metrics = getattr(self.metrics, f"{stage}_metrics", {})
        for metric_name, metric in metrics.items():
            self._log_metric(metric, metric_name, stage, epoch=True)
            metric.reset()

    def _log_metric(self, metric, name, stage, batch_size=None, epoch=False):
        result = metric.compute() if isinstance(metric, Metric) else metric

        # Metrics used purely for side-effects (e.g., plotting) can return None and won't be logged
        # If metrics return a dict of results, train/val metrics are separated into different plots
        if isinstance(result, dict):
            for k, v in result.items():
                self.log(f"{name} ({stage})/{k}", v, batch_size=batch_size)
        elif result is not None:
            self.log(f"{name}/{stage}", result, batch_size=batch_size)

        if epoch and hasattr(metric, "plot") and (clearml_logger := Logger.current_logger()) is not None:
            clearml_logger.report_plotly(
                f"{name} ({stage})",
                stage,
                metric.plot(),
                iteration=self.current_epoch,
            )

    def get_stage(self) -> str:
        return self.trainer.state.stage.value

    ## Train methods
    def training_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self.forward(x)
        loss = self.loss_calculate_and_log(y_pred, md)
        self.metrics_update_and_log(y_pred, md)
        return {"loss": loss, "metadata": md}

    def on_train_epoch_end(self):
        self.metrics_compute_and_log()

    ## Val methods
    def validation_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self.forward(x)
        loss = self.loss_calculate_and_log(y_pred, md)
        self.metrics_update_and_log(y_pred, md)
        return {"loss": loss, "metadata": md}

    def on_validation_epoch_end(self):
        self.metrics_compute_and_log()

    ## Test methods
    def test_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self.forward(x)
        loss = self.loss_calculate_and_log(y_pred, md)
        self.metrics_update_and_log(y_pred, md)
        return {"loss": loss, "metadata": md}

    def on_test_epoch_end(self):
        self.metrics_compute_and_log()
