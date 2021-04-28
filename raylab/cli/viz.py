"""Utilities for visualization."""
from __future__ import annotations

import bokeh
import numpy as np
import pandas as pd
from bokeh.models import (
    BoxZoomTool,
    ColumnDataSource,
    HelpTool,
    HoverTool,
    PanTool,
    ResetTool,
    SaveTool,
    WheelZoomTool,
)
from bokeh.plotting import figure

from raylab.utils.exp_data import ExperimentData, Selector


def time_series(
    x_key: str, y_key: str, groups: list[Selector], labels: list[str], config: dict
):
    """Plot time series with error bands per group."""
    # pylint:disable=too-many-function-args
    kwargs = dict(y_axis_type="log") if config["log_scale"] else {}
    pic = figure(title="Plot", **kwargs)
    pic.xaxis.axis_label = x_key
    pic.yaxis.axis_label = y_key

    pic.tools = [
        PanTool(),
        BoxZoomTool(),
        WheelZoomTool(dimensions="height"),
        WheelZoomTool(dimensions="width"),
        SaveTool(),
        ResetTool(),
        HelpTool(),
    ]
    pic.add_tools()
    if config["individual"]:
        pic.add_tools(HoverTool(tooltips=[("y", "@y"), ("x", "@x{a}"), ("id", "@id")]))
    else:
        pic.add_tools(HoverTool(tooltips=[("y", "@y_mean"), ("x", "@x{a}")]))

    for label, group, color in zip(labels, groups, bokeh.palettes.cividis(len(labels))):
        data = group.extract()
        progresses = [d.progress for d in data if y_key in d.progress.columns]
        if not progresses:
            continue

        x_all, all_ys = filter_and_interpolate(x_key, y_key, progresses)

        if config["individual"]:
            plot_individual(pic, x_all, all_ys, data, label, color)
        else:
            plot_mean_dispersion(
                pic,
                x_all,
                all_ys,
                label,
                color,
                standard_error=config["standard_error"],
            )

    pic.legend.location = "bottom_left"
    pic.legend.click_policy = "hide"
    return pic


def filter_and_interpolate(
    x_key: str, y_key: str, progresses: list[pd.DataFrame]
) -> tuple[np.ndarray, list[np.ndarray]]:
    # pylint:disable=missing-function-docstring
    # Filter NaN values from plots
    masks = [~np.isnan(p[y_key]) for p in progresses]
    xs_ = [p[x_key][m] for m, p in zip(masks, progresses)]
    ys_ = [p[y_key][m] for m, p in zip(masks, progresses)]
    x_all = np.unique(np.sort(np.concatenate(xs_)))
    all_ys = [
        np.interp(x_all, x, y, left=np.nan, right=np.nan) for x, y in zip(xs_, ys_)
    ]
    return x_all, all_ys


def plot_individual(
    pic,
    x_all: np.ndarray,
    all_ys: list[np.ndarray],
    data: ExperimentData,
    label: str,
    color: str,
):
    # pylint:disable=missing-function-docstring,too-many-arguments
    for datum, y_i in zip(data, all_ys):
        identifier = str(datum.params["id"])
        dataframe = pd.DataFrame({"x": x_all, "y": y_i, "id": identifier})
        source = ColumnDataSource(data=dataframe)
        pic.line(
            x="x",
            y="y",
            source=source,
            legend_label=label,
            color=color,
        )


def plot_mean_dispersion(
    pic,
    x_all: np.ndarray,
    all_ys: list[np.ndarray],
    label: str,
    color: str,
    standard_error: bool = False,
):
    # pylint:disable=missing-function-docstring,too-many-arguments
    y_mean = np.nanmean(all_ys, axis=0)
    dispersion = np.nanstd(all_ys, axis=0)
    if standard_error:
        dispersion /= np.sqrt(np.sum(1 - np.isnan(y_mean), axis=0))
    dataframe = pd.DataFrame(
        {
            "x": x_all,
            "y_mean": y_mean,
            "y_low": y_mean - dispersion,
            "y_high": y_mean + dispersion,
            "label": label,
        }
    )
    source = ColumnDataSource(data=dataframe)
    pic.line(x="x", y="y_mean", source=source, legend_label=label, color=color)
    pic.varea(
        x="x",
        y1="y_low",
        y2="y_high",
        source=source,
        fill_alpha=0.25,
        legend_label=label,
        color=color,
    )
