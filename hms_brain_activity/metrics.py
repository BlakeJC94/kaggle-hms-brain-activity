from itertools import product
from typing import Callable, Optional, List

import torch
import pandas as pd
from torchmetrics import Metric
from plotly import graph_objects as go


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
        self.class_names = self.class_names

        self.add_state(
            "histogram",
            default=torch.zeros(self.n_classes, 2, self.n_bins),
            dist_reduce_fx="sum",
        )

    def compute(self):
        pass

    def plot(self) -> go.Figure:
        fig = go.Figure()
        boundaries = torch.linspace(0, 1, self.n_bins + 1)[:-1]

        # TODO try to repeat the values, not scale them!
        left_histogram = self.histogram[:, 0, :].cpu()
        left_density = 1000 * left_histogram / left_histogram.sum(0, keepdim=True)
        left_data = left_density * boundaries.unsqueeze(0)
        left_data = left_data.numpy()

        right_histogram = self.histogram[:, 1, :].cpu()
        right_density = 1000 * right_histogram / right_histogram.sum(0, keepdim=True)
        right_data = right_density * boundaries.unsqueeze(0)
        right_data = right_data.numpy()

        df = pd.DataFrame()
        for class_idx in range(self.n_classes):
            n_left = left_data.shape[-1]
            n_right = right_data.shape[-1]
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "class": [self.class_names[class_idx]] * n_left,
                            "prob": left_data[class_idx],
                            "left": [True] * n_left,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "class": [self.class_names[class_idx]] * n_right,
                            "prob": right_data[class_idx],
                            "left": [False] * n_right,
                        }
                    ),
                ]
            )

        fig.add_trace(
            go.Violin(
                x=df["class"][df["left"] == True],
                y=df["prob"][df["left"] == True],
                legendgroup="Yes",
                scalegroup="Yes",
                name="Yes",
                side="negative",
                line_color="blue",
            )
        )
        fig.add_trace(
            go.Violin(
                x=df["class"][df["left"] == False],
                y=df["prob"][df["left"] == False],
                legendgroup="No",
                scalegroup="No",
                name="No",
                side="positive",
                line_color="orange",
            )
        )
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode="overlay")
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
            mask = y[:, :].argmax(1) == class_idx
            if hist_idx == 1:
                mask = ~mask
            self.histogram[class_idx, j, :] += torch.histc(
                y_pred[mask, class_idx], self.num_bins, 0, 1
            )


class MeanProbability(Metric):
    def __init__(self, class_names: List[str], **kwargs):
        super().__init__(**kwargs)

        self.class_names = class_names

        self.add_state(
            "probabilities",
            default=torch.zeros(len(self.class_names), 2),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_samples",
            default=torch.zeros(()),
            dist_reduce_fx="sum",
        )

    def update(self, y_pred, y):
        self.probabilities[:, 0] += y_pred.sum(0)
        self.probabilities[:, 1] += y.sum(0)
        self.n_samples += y_pred.shape[0]

    def compute(self):
        result = self.probabilities / self.n_samples
        out = {}
        for i, class_name in enumerate(self.class_names):
            out[class_name] = result[i, 0]
            out[f"{class_name}_true"] = result[i, 1]
        return out
