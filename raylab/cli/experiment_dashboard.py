"""Experiment monitoring with Streamlit."""
import streamlit as st
import numpy as np
import bokeh
from bokeh.plotting import figure
from raylab.utils import exp_data as exp_util

# pylint:disable=invalid-name,missing-docstring,pointless-string-statement
"""
# Raylab
"""


_NUMERIC_KINDS = set("uifc")


@st.cache
def load_data(directories):
    return exp_util.load_exps_data(directories)


@st.cache
def get_exp_root_folders(directories):
    return exp_util.get_folders_with_target_files(directories, is_experiment_root)


def is_experiment_root(path):
    return path.startswith("experiment_state") and path.endswith(".json")


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


def get_x_plottable_keys(plottable_keys, exps_data):
    return [key for key in plottable_keys if is_increasing_key(key, exps_data)]


def dict_value_multiselect(mapping, name=None):
    items = []

    keys = st.multiselect(f"{name}:", list(mapping.keys()), key=name)
    if keys:
        for key in keys:
            choices = list(mapping[key])
            values = st.multiselect(f"{key} values:", choices, key=name)
            for val in values:
                items.append((key, val))

    return items


def time_series(x_key, y_key, groups, labels):
    p = figure(title="Plot")
    p.xaxis.axis_label = x_key
    p.yaxis.axis_label = y_key
    palette = bokeh.palettes.cividis(len(labels))
    for idx, (label, group) in enumerate(zip(labels, groups)):
        data = group.extract()
        all_xs = np.unique(
            np.sort(np.concatenate([d.progress.get(x_key, []) for d in data]))
        )
        progresses = [
            np.interp(
                all_xs, d.progress[x_key], d.progress[y_key], left=np.nan, right=np.nan
            )
            for d in data
        ]
        mean_ys = np.nanmean(progresses, axis=0)
        std_ys = np.nanstd(progresses, axis=0)
        lower = mean_ys - std_ys
        upper = mean_ys + std_ys
        p.line(all_xs, mean_ys, legend_label=label, color=palette[idx])
        p.varea(
            all_xs,
            y1=lower,
            y2=upper,
            fill_alpha=0.25,
            legend_label=label,
            color=palette[idx],
        )
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"
    return p


def main():
    import sys

    directories = tuple(sys.argv[1:])
    root_exp_folders = [f.path for f in get_exp_root_folders(directories)]
    folders = st.sidebar.multiselect(
        "Filter experiments:", root_exp_folders, default=root_exp_folders
    )

    if folders:
        exps_data = load_data(tuple(folders))
        selector = exp_util.Selector(exps_data)
        distinct_params = dict(sorted(exp_util.extract_distinct_params(exps_data)))

        include = dict_value_multiselect(distinct_params, name="Include")
        exclude = dict_value_multiselect(distinct_params, name="Exclude")

        [selector], _ = exp_util.filter_and_split_experiments(
            exps_data, include=include, exclude=exclude
        )
        exps_data = selector.extract()
        if exps_data:
            plottable_keys = get_plottable_keys(exps_data)
            x_plottable_keys = get_x_plottable_keys(plottable_keys, exps_data)
            x_key = st.selectbox("X axis:", x_plottable_keys)
            y_key = st.selectbox("Y axis:", plottable_keys)

            distinct_params = dict(sorted(exp_util.extract_distinct_params(exps_data)))
            split = st.selectbox(
                "Group by:",
                [""] + list(distinct_params.keys()),
                format_func=lambda x: "(none)" if x == "" else x,
            )
            if split:
                values = distinct_params.get(split, [])
                labels = list(map(str, values))
                groups = [selector.where(split, v) for v in values]
            else:
                labels = ["experiment"]
                groups = [selector]

            st.bokeh_chart(time_series(x_key, y_key, groups, labels))


if __name__ == "__main__":
    main()
