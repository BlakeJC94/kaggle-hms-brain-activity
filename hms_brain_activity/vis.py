from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List, Tuple, Callable, TypeAlias
from warnings import warn

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html, Dash
from dash.dependencies import Input, Output
from plotly import graph_objects as go

from hms_brain_activity.core.datasets import BaseDataset
from hms_brain_activity.core.transforms import TransformIterable, TransformCompose

TabularData: TypeAlias = Union[pd.DataFrame, np.ndarray]
Control: TypeAlias = Union[dbc.Label, html.Div, dcc.Slider]


def extract_transforms(
    transforms: Callable,
    starting_ind: int = 0,
) -> List[Callable]:
    """Extract transforms recursively."""
    transform_list = []
    if transforms:
        # Ensure transforms is a Compose object
        if not isinstance(transforms, TransformCompose):
            transforms = TransformCompose(transforms)

        ind = starting_ind
        for transform in transforms.transforms:
            # If transform is a nested Compose object, expand it out by calling this function
            # recursively
            if isinstance(transform, TransformCompose):
                # Recurse into Compose object transforms
                transform_list_i = extract_transforms(
                    transform,
                    ind,  # Ensure consistently increasing transform count throughout recursion
                )
                transform_list.extend(transform_list_i)
                ind += len(transform_list_i)
            else:
                # Add transform to dashboard controls
                transform_list.append(transform)
                ind += 1

    return transform_list


def plot_channels(
    data: TabularData,
    labels: Optional[pd.DataFrame] = None,
    spacing: float = 100.0,
    scale: Union[Dict[str, float], float] = 6,
) -> go.Figure:
    """Plot a dataframe with multiple timeseries channels.

    Args:
        data: A timeseries dataframe or numpy array where all columns represent data channels. For
            dataframes, channels will be named with column names.
        labels: An optional dataframe in either sparse (a row per label) or timeseries (a row per
            time sample) format.
        spacing: The vertical spacing in scaled units between each channel on the plot.
        scale: The scaling factor to apply to the data before plotting. Applied as 10 ^ scale.

    Returns:
        A plotly figure object.
    """
    if not isinstance(data, dict):
        data = {"data": data}
    if not isinstance(scale, dict):
        scale = {k: scale for k in data.keys()}

    if labels is not None:
        if not isinstance(labels, pd.DataFrame):
            warn("Plotting labels is only supported for Pandas DataFrames.")
            labels = None
        if labels is not None and "tag" not in labels:
            data["labels"] = labels.copy()
            scale["labels"] = np.log10(spacing * 0.75)
            labels = None

    # Can't localise labels within sample if data is a numpy array
    # TODO: get start time from metadata to localise on numpy arrays?
    if not all(isinstance(data[k], pd.DataFrame) for k in data):
        warn("Plotting labels against data stored as numpy arrays is not supported.")
        labels = None

    # Convert numpy arrays is data to dataframes
    for k in data:
        if not isinstance(data[k], pd.DataFrame):
            # If data is a numpy array, we'll use integer channel names and time index
            data[k] = pd.DataFrame(
                data[k].transpose(), columns=np.arange(data[k].shape[0]).astype(str)
            )

    fig = go.Figure()
    data_traces, x_range, y_range, y_offsets = _get_data_traces(data, spacing, scale)

    label_traces = []
    if labels is not None:
        label_traces = _get_label_traces(labels, y_range, spacing, rgb=(70, 130, 180))

    for trace in [*data_traces, *label_traces]:
        fig.add_trace(trace)

    # Remove background & legend, set grid color, and add channel labels
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        autosize=True,
        hoverdistance=10,
        hoverlabel=dict(
            bgcolor="rgb(70, 130, 180)",
            font_size=14,
        ),
    )
    fig.update_xaxes(
        gridcolor="lightgrey",
        linecolor="lightgrey",
        range=x_range,
    )

    tickvals = [o for offsets in y_offsets.values() for o in offsets]
    ticktext = [str(c) + "   " for d in data.values() for c in d.columns]
    fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, range=y_range)

    return fig


def _get_data_traces(
    data: Dict[str, TabularData],
    spacing: float,
    scale: Dict[str, float],
) -> Tuple[List[go.Scatter], Dict[str, List[float]], float]:
    traces = []
    data = deepcopy(data)
    y_offsets = {}
    max_offset = 0.0
    for k in data:
        data[k].index = pd.to_datetime(data[k].index.values, unit="ms")

        # Scale data to appropriate units and determine the spacing between traces
        data[k] = data[k] * 10 ** scale[k]
        y_offsets[k] = [max_offset + i * spacing for i in range(len(data[k].columns))]
        max_offset = max(y_offsets[k]) + spacing * 1.5

        for i, (channel_name, channel) in enumerate(data[k].items()):
            trace = go.Scattergl(
                x=channel.index,
                y=channel + y_offsets[k][i],
                name=channel_name,
                line=dict(color="#63A295", width=1),
                text=channel,
                hoverinfo="name+text",
            )
            traces.append(trace)

    y_range = [-spacing * 1.5, max_offset]

    x_range = [
        min([min(d.index) for d in data.values()]),
        max([max(d.index) for d in data.values()]),
    ]
    return traces, x_range, y_range, y_offsets


def _get_label_traces(
    labels: pd.DataFrame,
    y_range: List[float],
    spacing: float,
    rgb: Tuple[int],
) -> List[go.Scatter]:
    traces = []
    labels["start"] = pd.to_datetime(labels["start"], unit="ms")
    labels["end"] = pd.to_datetime(labels["end"], unit="ms")
    labels = labels.groupby(["start", "end"]).agg({"tag": ", ".join})
    for label_time, label in zip(labels.index.values, labels["tag"]):
        x0 = label_time[0]
        x1 = label_time[1]
        if (x1 - x0).total_seconds() < 0.1:
            x1 = x0 + pd.Timedelta("100 milli")
        y0 = y_range[0] + spacing * 0.02
        y1 = y_range[1] - spacing * 0.02
        trace = go.Scatter(
            x=[x0, x0, x1, x1, x0],
            y=[y0, y1, y1, y0, y0],
            fill="toself",
            text=label,
            name="",
            fillcolor="rgba(" + ", ".join(map(str, rgb)) + ", 0.2)",
            line=dict(
                color="rgb(" + ", ".join(map(str, rgb)) + ")", dash="dash", width=2
            ),
            mode="lines",
        )
        traces.append(trace)
    return traces


@dataclass
class DashState:
    current_x: Any
    current_y: Any
    current_ind: int


class DashUIElements:
    title = "EEG Data Visualizer"

    def __init__(self, dataset):
        self.dataset = dataset

        transforms = []
        if dataset.augmentation is not None:
            transforms.append(dataset.augmentation)
        if dataset.transform is not None:
            transforms.append(dataset.transform)
        if len(transforms) == 0:
            self.data_transform_list = []
        else:
            self.data_transform_list = extract_transforms(TransformCompose(*transforms))

        self.header = self.get_header()

        self.navigation_controls = self.get_navigation_controls()
        self.scaling_controls = self.get_scaling_controls()
        self.data_transform_controls = self.get_transform_controls()

    def get_header(self):
        return dbc.Navbar(
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand(self.title, className="pt-3 pb-0")),
                ],
                align="center",
            ),
            color="secondary",
            className="sticky-top",
        )

    def get_navigation_controls(self) -> List[Control]:
        return [
            dbc.Label("Label index:"),
            html.Div(
                html.Ul(
                    [
                        html.Li(
                            html.A(
                                "Prev",
                                href="#",
                                id="prev_label",
                                className="page-link m-1",
                            )
                        ),
                        html.Li(
                            html.A(
                                "Next",
                                href="#",
                                id="next_label",
                                className="page-link m-1",
                            )
                        ),
                    ],
                    className="pagination mt-2",
                )
            ),
            dbc.Label("Dataset index:"),
            html.Div(
                html.Ul(
                    [
                        html.Li(
                            html.A(
                                "Prev", href="#", id="prev", className="page-link m-1"
                            )
                        ),
                        html.Li(
                            html.A(
                                "Next", href="#", id="next", className="page-link m-1"
                            )
                        ),
                    ],
                    className="pagination mt-2",
                )
            ),
            dcc.Slider(
                id="index-slider",
                min=0,
                max=len(self.dataset) - 1,
                marks={i: str(i) for i in [0, len(self.dataset) - 1]},
                step=1,
                value=0,
                tooltip=dict(always_visible=True, placement="bottom"),
                className="p-0 pb-4",
            ),
        ]

    def get_scaling_controls(self):
        scaling_controls = []
        for channel_group in self.dataset.channel_groups:
            scaling_controls.append(dbc.Label([f"{channel_group}: 10", html.Sup("x")]))
            scaling_controls.append(
                dcc.Slider(
                    id=f"{channel_group}-scale-slider",
                    min=-6,
                    max=9,
                    step=0.1,
                    marks={i: str(i) for i in [-6, 9]},
                    value=0,
                    tooltip=dict(always_visible=True, placement="bottom"),
                    className="p-0 pb-4",
                )
            )
        return scaling_controls

    def get_transform_controls(self):
        controls = []
        for ind, transform in enumerate(self.data_transform_list):
            label = transform.__class__.__name__
            if isinstance(transform, TransformIterable):
                label = f"{transform.transform.__class__.__name__}-{transform.apply_to}"
            controls.append(
                dbc.Checklist(
                    options=[dict(label=label, value=ind)],
                    value=[],
                    id=f"transform-{ind}",
                    switch=True,
                    inline=True,
                    className="m-2",
                )
            )

        controls.append(html.Br())
        return controls
        return [
            *[
                dbc.Checklist(
                    options=[dict(label=transform.__class__.__name__, value=ind)],
                    value=[],
                    id=f"transform-{ind}",
                    switch=True,
                    inline=True,
                    className="m-2",
                )
                for ind, transform in enumerate(self.data_transform_list)
            ],
            controls.append(html.Br()),
        ]

    def settings_cards(self) -> List[dbc.Card]:
        navigation_controls = self.navigation_controls
        scaling_controls = self.scaling_controls
        data_transform_controls = self.data_transform_controls

        navigation_card = dbc.Card(
            [
                dbc.CardHeader("Navigation"),
                dbc.CardBody([dbc.Col(navigation_controls)], className="p-3"),
            ],
            className="my-3",
        )

        data_transform_card = dbc.Card(
            [
                dbc.CardHeader("Data Transforms"),
                dbc.CardBody([dbc.Col(data_transform_controls)], className="p-3"),
            ],
            className="my-3",
        )

        scaling_card = dbc.Card(
            [
                dbc.CardHeader("Scaling"),
                dbc.CardBody([dbc.Col(scaling_controls)], className="p-3"),
            ],
            className="my-3",
        )

        settings_cards = [
            navigation_card,
            data_transform_card,
            scaling_card,
        ]

        return settings_cards


def dataset_dash(
    dataset: BaseDataset,
    host: str = "localhost",
    port: int = 8050,
):
    state = DashState(
        current_x=None,
        current_y=None,
        current_ind=0,
    )
    dash_ui_elements = DashUIElements(dataset)

    app = Dash(
        external_stylesheets=[dbc.themes.FLATLY],
        meta_tags=[
            dict(name="viewport", content="width=device-width, initial-scale=1")
        ],
    )
    app.title = dash_ui_elements.title

    # Define page layout
    settings_cards = dash_ui_elements.settings_cards()
    plot_div = dcc.Graph(
        id="timeseries",
        style=dict(
            height="90vh",
            width="100%",
            minHeight=600,
            minWidth=600,
        ),
    )
    app.layout = dbc.Container(
        [
            dash_ui_elements.header,
            dbc.Row(
                [
                    dbc.Col(settings_cards, lg=3),
                    dbc.Col(plot_div, lg=9),
                ],
                className="mx-3",
            ),
        ],
        fluid=True,
        className="p-0",
    )

    data_transform_list = dash_ui_elements.data_transform_list
    data_transform_inputs = [
        Input(f"transform-{ind}", "value") for ind in range(len(data_transform_list))
    ]

    # Navigation and data transform callback
    @app.callback(
        [
            Output("timeseries", "figure"),
        ],
        [
            Input("index-slider", "value"),
            *[
                Input(f"{channel_group}-scale-slider", "value")
                for channel_group in dataset.channel_groups
            ],
            *data_transform_inputs,
        ],
    )
    def _(index, *scale_and_transform_inds):
        """Navigation and data transform callback."""
        # Parse variable length args
        scales = scale_and_transform_inds[: len(dataset.channel_groups)]
        transform_inds = scale_and_transform_inds[len(dataset.channel_groups) : -1]

        n_data_transforms = len(data_transform_inputs)
        data_transform_inds = transform_inds[:n_data_transforms]

        # Determine if triggered by an update to the index slider
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]["prop_id"] == "index-slider.value":
            state.current_x = None
            state.current_y = None

        state.current_ind = index

        ds_index = index

        # Load new raw sample only if required
        if state.current_x is None:
            state.current_x, state.current_y = dataset.get_raw(ds_index)

        # Get current raw sample and output
        x = state.current_x.copy()
        y = state.current_y.copy()

        # Apply data transforms
        data_transform_inds = [i for ii in data_transform_inds for i in ii]
        for data_transform_ind in data_transform_inds:  # TODO try/catch?
            x, y = data_transform_list[data_transform_ind](x, y)

        scales = dict(zip(x, scales)) if isinstance(x, dict) else scales[0]
        fig = plot_channels(x, labels=y["y"], scale=scales)
        fig.update_layout(margin=dict(t=20, b=20, l=0, r=20))

        return (fig,)

    # Navigation callback
    @app.callback(
        [
            Output("index-slider", "value"),
            Output("index-slider", "max"),
        ],
        [
            Input("prev", "n_clicks"),
            Input("next", "n_clicks"),
            Input("next_label", "n_clicks"),
            Input("prev_label", "n_clicks"),
        ],
    )
    def _(*_):
        """Navigation callback."""
        new_ind = state.current_ind
        max_ind = len(dataset) - 1

        # Check if the callback was triggered by the navigation controls
        ctx = dash.callback_context
        if ctx.triggered:
            if ctx.triggered[0]["prop_id"] == "prev.n_clicks":
                # Triggered by the previous button
                new_ind = max(state.current_ind - 1, 0)
            elif ctx.triggered[0]["prop_id"] == "next.n_clicks":
                # Triggered by the next button
                new_ind = min(state.current_ind + 1, max_ind)
            # elif ctx.triggered[0]["prop_id"] == "prev_label.n_clicks":
            #     # Triggered by the previous label button
            #     lower_labels = [
            #         i[0]
            #         for i in dataset.sample_labels[study_idx]
            #         if i[0] < state.current_ind
            #     ]
            #     if lower_labels:
            #         new_ind = max(lower_labels)
            # elif ctx.triggered[0]["prop_id"] == "next_label.n_clicks":
            #     # Triggered by the next label button
            #     higher_labels = [
            #         i[0]
            #         for i in dataset.sample_labels[study_idx]
            #         if i[0] > state.current_ind
            #     ]
            #     if higher_labels:
            #         new_ind = min(higher_labels)

        return new_ind, max_ind

    app.run_server(host=host, port=port)
