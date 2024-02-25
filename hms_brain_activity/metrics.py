from typing import Callable

import torch
from torchmetrics import Metric


class MetricWrapper(Metric):
    def __init__(self, preprocessor: Callable, metric: Metric):
        super().__init__()
        self.preprocessor = preprocessor
        self.metric = metric

    def update(self, y_pred, y):
        y_pred, y = self.preprocessor(y_pred, y)
        self.metric.update(y_pred, y)

    def compute(self):
        return self.metric.compute()


class PooledMean(Metric):
    """Base class for converting a standard callable into a pooled mean metric.
    This class will store results from calls so that the overall mean can be calculated in
    MapReduce fashion.

    E.g.,
    First __call__: fn(y_pred, y) -> 10.4, len(y) -> 32
    Second __call__: fn(y_pred, y) -> 3.4, len(y) -> 24
    compute: (10.4 * 32 + 3.4 * 24) / (32 + 24)

    Args:
        metric: A callable that returns a torch scalar metric value. It should have the
            signature `fn(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor`.
    """

    def __init__(self, metric: Callable, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.add_state("value_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.value_sum = self.value_sum + self.metric(preds, target) * target.shape[0]
        self.count = self.count + target.shape[0]

    def compute(self) -> torch.Tensor:
        return self.value_sum / self.count
