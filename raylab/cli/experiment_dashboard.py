"""Experiment monitoring with Streamlit."""
import streamlit as st

from raylab.utils import exp_data as exp_util
from raylab.cli.viz import time_series

# pylint:disable=invalid-name,missing-docstring,pointless-string-statement
# pylint:disable=no-value-for-parameter
"""
# Raylab
"""


@st.cache
def load_data(directories, include_errors=False):
    return exp_util.load_exps_data(directories, include_errors=include_errors)


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


def main():
    # pylint:disable=too-many-locals
    import sys

    directories = tuple(sys.argv[1:])
    root_exp_folders = [f.path for f in get_exp_root_folders(directories)]
    folders = st.sidebar.multiselect(
        "Filter experiments:", root_exp_folders, default=root_exp_folders
    )
    include_errors = st.sidebar.checkbox("Include experiments with errors")

    if folders:
        exps_data = load_data(tuple(folders), include_errors=include_errors)
        distinct_params = dict(sorted(exp_util.extract_distinct_params(exps_data)))

        include = dict_value_multiselect(distinct_params, name="Include")
        exclude = dict_value_multiselect(distinct_params, name="Exclude")

        [selector], _ = exp_util.filter_and_split_experiments(
            exps_data, include=include, exclude=exclude
        )
        exps_data = selector.extract()
        if exps_data:
            plottable_keys = exp_util.get_plottable_keys(exps_data)
            x_plottable_keys = exp_util.get_x_plottable_keys(plottable_keys, exps_data)
            x_key = st.selectbox(
                "X axis:",
                x_plottable_keys,
                index=x_plottable_keys.index("timesteps_total"),
            )
            y_key = st.selectbox(
                "Y axis:",
                plottable_keys,
                index=plottable_keys.index("episode_reward_mean"),
            )

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

            ts_config = {
                "individual": st.checkbox("Show individual curves"),
                "standard_error": st.checkbox("Use standard error"),
                "log_scale": st.sidebar.checkbox("Log scale"),
            }
            chart = time_series(x_key, y_key, groups, labels, ts_config)
            chart.legend.title = split
            chart.legend.title_text_font_style = "bold"
            chart.legend.location = st.sidebar.selectbox(
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
            )
            chart.legend.orientation = st.sidebar.selectbox(
                "legend orientation", "vertical horizontal".split()
            )
            st.bokeh_chart(chart)


if __name__ == "__main__":
    main()
