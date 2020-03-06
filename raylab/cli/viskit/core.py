# pylint: disable=missing-docstring
from ast import literal_eval

import pandas as pd

from raylab.utils.exp_data import extract_distinct_params


def lineplot_instructions(selectors, titles, **lineplot_kwargs):
    plots = []
    for selector, title in zip(selectors, titles):
        exps_data = selector.extract()
        if not exps_data:
            continue

        distinct_params = dict(sorted(extract_distinct_params(exps_data)))
        plots.append(
            dict(
                title=str(title),
                lineplot_kwargs=dict(
                    data=pd.concat(
                        [exp.progress for exp in exps_data],
                        ignore_index=True,
                        sort=False,
                    ),
                    hue_order=distinct_params.get(lineplot_kwargs.get("hue")),
                    size_order=distinct_params.get(lineplot_kwargs.get("size")),
                    style_order=distinct_params.get(lineplot_kwargs.get("style")),
                    **lineplot_kwargs,
                ),
            )
        )
    return plots


def substitute_key(dictionary, key, subs_key):
    val = None
    if key in dictionary:
        val = dictionary[key]
        del dictionary[key]
    dictionary[subs_key] = val


def substitute_val(dictionary, val, subs_val):
    for key in dictionary.keys():
        if dictionary.get(key) == val:
            dictionary[key] = subs_val


def rename_params(exps_data, subskey=(), subsval=()):
    def process(arg):
        try:
            return literal_eval(arg)
        except (ValueError, SyntaxError):
            return arg

    for exp_data in exps_data:
        for key, subs_key in map(lambda t: tuple(map(process, t)), subskey):
            substitute_key(exp_data.flat_params, key, subs_key)
            substitute_key(exp_data.params, key, subs_key)
        for val, subs_val in map(lambda t: tuple(map(process, t)), subsval):
            substitute_val(exp_data.flat_params, val, subs_val)
            substitute_val(exp_data.params, val, subs_val)


def insert_params_dataframe(exps_data, *param_keys):
    for exp_data in exps_data:
        params, dataframe = exp_data.flat_params, exp_data.progress
        for key in filter(None, param_keys):
            if key in dataframe:
                continue
            dataframe.insert(len(dataframe.columns), str(key), params.get(key))
