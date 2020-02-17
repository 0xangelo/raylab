"""Experiment monitoring with Streamlit."""
import streamlit as st
import numpy as np
import bokeh
from bokeh.plotting import figure
from raylab.cli.viskit import core

# pylint:disable=invalid-name
"""
# Raylab
"""


_NUMERIC_KINDS = set("uifc")


def is_numeric(array):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS


def is_increasing_key(key, exps_data):
    for exp in exps_data:
        if key in exp.progress and not is_increasing(exp.progress[key]):
            return False
    return True


def is_increasing(arr):
    arr = np.asarray(arr)
    return (
        is_numeric(arr)
        and np.all(arr[1:] - arr[:-1] >= 0)
        and np.max(arr) >= np.min(arr)
    )


def get_plottable_keys(exps_data):
    return sorted(
        list(
            set(
                col
                for exp in exps_data
                for col in exp.progress.columns.to_list()
                if is_numeric(exp.progress[col])
            )
        )
    )


def get_x_plottable_keys(plottable_keys):
    return [key for key in plottable_keys if is_increasing_key(key, exps_data)]


path = st.text_input(
    label="Enter experiment directory", value="data/SoftAC/20200215/IndustrialBenchmark"
)

if path:
    # st.write(f"Path chosen: {path}")
    exps_data = core.load_exps_data(path.split(" "))
    # st.write(f"Number of logs: {len(exps_data)}")
    selector = core.Selector(exps_data)
    distinct_params = dict(sorted(core.extract_distinct_params(exps_data)))

    include = []
    params = st.multiselect("Filter by:", list(distinct_params.keys()))
    if params:
        for param in params:
            values = st.multiselect(
                f"{param} values:",
                distinct_params[param],
                default=distinct_params[param][0],
            )
            for value in values:
                include.append((param, value))
    for key, val in include:
        selector = selector.where(key, val)

    exps_data = selector.extract()
    if exps_data:
        plottable_keys = get_plottable_keys(exps_data)
        x_plottable_keys = get_x_plottable_keys(plottable_keys)
        x_key = st.selectbox("X axis:", x_plottable_keys)
        y_key = st.selectbox("Y axis:", plottable_keys)

        distinct_params = dict(sorted(core.extract_distinct_params(exps_data)))
        split = st.selectbox("Split axis:", list(distinct_params.keys()))
        if split:
            values = distinct_params.get(split, [])
            labels = list(map(str, values))
            groups = [selector.where(split, v) for v in values]
        else:
            labels = ["plot"]
            groups = [selector]

        p = figure(title="Plot")
        p.xaxis.axis_label = x_key
        p.yaxis.axis_label = y_key
        palette = bokeh.palettes.Viridis256
        step = len(palette) // len(labels)
        for idx, (label, group) in enumerate(zip(labels, groups)):
            data = group.extract()
            all_xs = np.unique(
                np.sort(np.concatenate([d.progress.get(x_key, []) for d in data]))
            )
            progresses = [
                np.interp(
                    all_xs,
                    d.progress[x_key],
                    d.progress[y_key],
                    left=np.nan,
                    right=np.nan,
                )
                for d in data
            ]
            mean_ys = np.nanmean(progresses, axis=0)
            std_ys = np.nanstd(progresses, axis=0)
            lower = mean_ys - std_ys
            upper = mean_ys + std_ys
            p.line(all_xs, mean_ys, legend_label=label, color=palette[idx * step])
            p.varea(
                all_xs, y1=lower, y2=upper, fill_alpha=0.25, color=palette[idx * step]
            )
        st.bokeh_chart(p)
