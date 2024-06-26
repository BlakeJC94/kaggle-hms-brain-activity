from itertools import product
from typing import Callable, List

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
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

    def plot(self):
        return self.metric.plot()


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


class _BaseProbabilityPlotMetric(Metric):
    def __init__(
        self,
        class_names: List[str],
        n_bins: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_classes = len(class_names)
        self.n_bins = n_bins
        self.class_names = class_names

        self.add_state(
            "histogram",
            default=torch.zeros(self.n_classes, 2, self.n_bins),
            dist_reduce_fx="sum",
        )

    def compute(self):
        pass

    def plot(self) -> go.Figure:
        num_labels = len(self.class_names)
        num_bins = self.histogram.shape[-1]
        fig = make_subplots(
            rows=1, cols=num_labels * 2, horizontal_spacing=0.001, shared_yaxes=True
        )
        for label_id, label_name in enumerate(self.class_names):
            for target_id in range(2):
                target_name = f"Target = {target_id}"
                histogram = self.histogram[label_id, target_id].cpu()
                col = label_id * 2 + target_id + 1
                fig.add_trace(
                    go.Bar(
                        x=histogram,
                        y=np.linspace(start=0, stop=1, num=num_bins, endpoint=False),
                        name=target_name,
                        orientation="h",
                        marker=dict(color="blue" if target_id == 0 else "orange"),
                    ),
                    row=1,
                    col=col,
                )

                # Reverse axes and remove tick labels to create back-to-back plots
                x_max = histogram.max()
                fig.update_xaxes(
                    range=[x_max, 0] if target_id == 0 else [0, x_max], row=1, col=col
                )
                fig.update_yaxes(
                    showticklabels=True if col == 1 else False, row=1, col=col
                )

                # Calculate and plot estimates for Q1, Median and Q3
                cdf = np.cumsum(histogram)
                bin_inds = np.searchsorted(cdf, cdf[-1] * np.array([0.25, 0.5, 0.75]))
                quantiles = (bin_inds / num_bins * 1).tolist()
                fig.add_trace(
                    go.Scatter(
                        x=[x_max / 2] * 3,
                        y=quantiles,
                        mode="markers",
                        marker=dict(color="black"),
                        line=dict(color="black"),
                        text=[
                            f"{name}: {round(yval,2)}"
                            for name, yval in zip(["Q1", "Median", "Q3"], quantiles)
                        ],
                        hoverinfo="text",
                    ),
                    row=1,
                    col=col,
                )
            fig.add_annotation(
                dict(
                    font=dict(size=16),
                    x=(label_id + 0.5) / num_labels,
                    y=-0.12,
                    showarrow=False,
                    text=label_name,
                    xanchor="center",
                    xref="paper",
                )
            )
        fig.update_xaxes(visible=False)
        fig.update_layout(yaxis_title="Probability")
        for trace in fig["data"][3:] + fig["data"][1:2]:
            trace["showlegend"] = False
        return fig


class ProbabilityDensity(_BaseProbabilityPlotMetric):
    def update(self, y_pred, y):
        for class_idx in range(self.n_classes):
            # Hist prob across each class (left preds, right targets)
            self.histogram[class_idx, 0, :] += torch.histc(
                y_pred[:, class_idx], self.n_bins, 0, 1
            )
            self.histogram[class_idx, 1, :] += torch.histc(
                y[:, class_idx], self.n_bins, 0, 1
            )


class ProbabilityDistribution(_BaseProbabilityPlotMetric):
    def update(self, y_pred, y):
        for class_idx, hist_idx in product(range(self.n_classes), range(2)):
            # Histogram pos/negative class (left preds where target class is highest, right preds else)
            mask = y[:, class_idx] > 0.5
            if hist_idx == 1:
                mask = ~mask
            self.histogram[class_idx, hist_idx, :] += torch.histc(
                y_pred[mask, class_idx], self.n_bins, 0, 1
            )


class MeanProbability(Metric):
    def __init__(self, class_names: List[str], **kwargs):
        super().__init__(**kwargs)

        self.class_names = class_names

        self.add_state(
            "probabilities",
            default=torch.zeros(len(self.class_names)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_samples",
            default=torch.zeros(()),
            dist_reduce_fx="sum",
        )

    def update(self, y_pred, _):
        self.probabilities += y_pred.sum(0)
        self.n_samples += y_pred.shape[0]

    def compute(self):
        result = self.probabilities / self.n_samples
        out = {}
        for i, class_name in enumerate(self.class_names):
            out[class_name] = result[i]
        return out
