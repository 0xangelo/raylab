"""Experiment monitoring with Streamlit."""
import streamlit as st

from raylab.cli.viz import time_series
from raylab.utils import exp_data as exp_util

# pylint:disable=invalid-name,missing-docstring,pointless-string-statement
# pylint:disable=no-value-for-parameter
"""
# Raylab
"""


# https://discuss.streamlit.io/t/how-can-i-clear-a-specific-cache-only/1963/6
@st.cache(allow_output_mutation=True)
def load_data(directories, include_errors=False):
    return [exp_util.load_exps_data(directories, include_errors=include_errors)]


@st.cache
def get_exp_root_folders(directories):
    return exp_util.get_folders_with_target_files(directories, is_experiment_root)


def is_experiment_root(path):
    return path.startswith("experiment_state") and path.endswith(".json")


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


def add_sidebar_options():
    return {
        "include_errors": st.sidebar.checkbox("Include experiments with errors"),
        "legend_location": st.sidebar.selectbox(
            "legend location",
            """
            top_left
            top_center
            top_right
            center_right
            bottom_right
            bottom_center
            bottom_left
            center_left
            center
            """.split(),
        ),
        "legend_orientation": st.sidebar.selectbox(
            "legend orientation", "vertical horizontal".split()
        ),
        "individual": st.sidebar.checkbox("Show individual curves"),
        "standard_error": st.sidebar.checkbox("Use standard error"),
        "log_scale": st.sidebar.checkbox("Log scale"),
    }


def get_selector(folders, sidebar_options):
    data_wrapper = load_data(
        tuple(folders), include_errors=sidebar_options["include_errors"]
    )
    if st.button("Reload Data"):
        data_wrapper.clear()
        data_wrapper.append(
            exp_util.load_exps_data(
                tuple(folders), include_errors=sidebar_options["include_errors"]
            )
        )
    exps_data = data_wrapper[0]
    distinct_params = dict(sorted(exp_util.extract_distinct_params(exps_data)))

    include = dict_value_multiselect(distinct_params, name="Include")
    exclude = dict_value_multiselect(distinct_params, name="Exclude")

    [selector], _ = exp_util.filter_and_split_experiments(
        exps_data, include=include, exclude=exclude
    )
    return selector


def select_axis_keys(exps_data):
    plottable_keys = exp_util.get_plottable_keys(exps_data)
    x_plottable_keys = exp_util.get_x_plottable_keys(plottable_keys, exps_data)
    x_key = st.selectbox(
        "X axis:", x_plottable_keys, index=x_plottable_keys.index("timesteps_total"),
    )
    y_key = st.selectbox(
        "Y axis:", plottable_keys, index=plottable_keys.index("episode_reward_mean"),
    )
    return x_key, y_key


def split_and_label_groups(exps_data, selector):
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

    return groups, labels, split


def main():
    import sys

    directories = tuple(sys.argv[1:])
    root_exp_folders = [f.path for f in get_exp_root_folders(directories)]
    folders = st.sidebar.multiselect(
        "Filter experiments:", root_exp_folders, default=root_exp_folders
    )
    sidebar_options = add_sidebar_options()

    if folders:
        selector = get_selector(folders, sidebar_options)
        exps_data = selector.extract()
        if exps_data:
            x_key, y_key = select_axis_keys(exps_data)
            groups, labels, split = split_and_label_groups(exps_data, selector)

            chart = time_series(x_key, y_key, groups, labels, sidebar_options)

            chart.legend.title = split
            chart.legend.title_text_font_style = "bold"
            chart.legend.location = sidebar_options["legend_location"]
            chart.legend.orientation = sidebar_options["legend_orientation"]
            st.bokeh_chart(chart)


if __name__ == "__main__":
    main()
