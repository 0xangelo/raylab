"""Utilities for loading and processing experiment results."""
# pylint:disable=missing-docstring
import itertools
import json
import os
from ast import literal_eval
from collections import namedtuple
from functools import reduce

import numpy as np
import pandas as pd


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
    if not is_numeric(arr):
        return False

    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return False

    return np.all(np.less_equal(arr[:-1], arr[1:])) and np.max(arr) >= np.min(arr)


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


ExperimentData = namedtuple("ExperimentData", ["progress", "params", "flat_params"])
Folder = namedtuple("Folder", ["path", "files"])


def load_progress(progress_path, verbose=True):
    if verbose:
        print("Reading {}".format(progress_path))

    if progress_path.endswith(".csv"):
        return pd.read_csv(progress_path, index_col=None, comment="#")

    dicts = []
    with open(progress_path, "rt") as file:
        for line in file:
            dicts.append(json.loads(line))
    return pd.DataFrame(dicts)


def flatten_dict(dic):
    flat_params = dict()
    for key, val in dic.items():
        if isinstance(val, dict):
            val = flatten_dict(val)
            for subk, subv in flatten_dict(val).items():
                flat_params[key + "/" + subk] = subv
        else:
            flat_params[key] = val
    return flat_params


def load_params(params_json_path):
    with open(params_json_path, "r") as file:
        data = json.load(file)
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-3]
    return data


def first_that(criterion, lis):
    return next((x for x in lis if criterion(x)), None)


def unique(lis):
    return list(set(lis))


def get_folders_with_target_files(directories, istarget):
    """Return folders that havy any file of interest."""
    return list(
        map(  # Folders with at least one target file
            lambda x: Folder(*x),
            filter(  # entries that have target files
                lambda x: any(istarget(y) for y in x[1]),
                filter(  # entries that have >0 files
                    lambda x: x[1],
                    map(  # only (path, files)
                        lambda x: (x[0], x[2]),
                        reduce(  # (path, subpath, files) for all dirs
                            itertools.chain,
                            map(  # (path, subpath, files) for each dir
                                os.walk, directories
                            ),
                            (),
                        ),
                    ),
                ),
            ),
        )
    )


def read_exp_folder_data(exp_folders, isprogress, isconfig, verbose=False):
    exps_data = []
    for path, files in exp_folders:
        try:
            progress_path = os.path.join(path, first_that(isprogress, files))
            progress = load_progress(progress_path, verbose=verbose)
            params_file = first_that(isconfig, files)
            params = (
                load_params(os.path.join(path, params_file))
                if params_file is not None
                else dict(exp_name="experiment")
            )
            if "trial_id" in progress:
                params["id"] = progress["trial_id"][0]
            exps_data.append(
                ExperimentData(
                    progress=progress, params=params, flat_params=flatten_dict(params)
                )
            )
        except (IOError, pd.errors.EmptyDataError) as error:
            if verbose:
                print(error)

    return exps_data


def load_exps_data(
    directories,
    progress_prefix="progress",
    config_prefix="params",
    error_prefix="error",
    include_errors=False,
    verbose=False,
):
    # pylint:disable=too-many-arguments
    if isinstance(directories, str):
        directories = [directories]

    def isprogress(file):
        return file.startswith(progress_prefix) and file.endswith(".csv")

    def isconfig(file):
        return file.startswith(config_prefix) and file.endswith(".json")

    exp_folders = get_folders_with_target_files(directories, isprogress)
    if not include_errors:
        exp_folders = [
            d
            for d in exp_folders
            if not any(f.startswith(error_prefix) for f in d.files)
        ]
    if verbose:
        print("finished walking exp folders")

    exps_data = read_exp_folder_data(exp_folders, isprogress, isconfig, verbose=verbose)
    return exps_data


def extract_distinct_params(exps_data, excluded_params=("seed", "log_dir")):
    repr_config_pairs = [repr(kv) for d in exps_data for kv in d.flat_params.items()]
    uniq_pairs = list(set(repr_config_pairs))
    evald_pairs = map(literal_eval, uniq_pairs)
    stringified_pairs = sorted(
        evald_pairs, key=lambda x: tuple("" if it is None else str(it) for it in x)
    )

    proposals = [
        (k, [x[1] for x in v])
        for k, v in itertools.groupby(stringified_pairs, lambda x: x[0])
    ]

    filtered = [
        (k, v)
        for (k, v) in proposals
        if v
        and all(k != excluded_param for excluded_param in excluded_params)
        and len(v) > 1
    ]
    return filtered


class Selector:
    def __init__(self, exps_data, filters=None):
        self._exps_data = exps_data
        self._filters = tuple() if filters is None else tuple(filters)

    def where(self, key, val):
        return Selector(
            self._exps_data,
            self._filters
            + (lambda exp: str(exp.flat_params.get(key, None)) == str(val),),
        )

    def where_not(self, key, val):
        return Selector(
            self._exps_data,
            self._filters
            + (lambda exp: str(exp.flat_params.get(key, None)) != str(val),),
        )

    def _check_exp(self, exp):
        return all(condition(exp) for condition in self._filters)

    def extract(self):
        return list(filter(self._check_exp, self._exps_data))


def filter_and_split_experiments(exps_data, split=None, include=(), exclude=()):
    selector = Selector(exps_data)
    for key, val in include:
        selector = selector.where(key, val)
    for key, val in exclude:
        selector = selector.where_not(key, val)

    if split is not None:
        exps_data = selector.extract()
        values = dict(sorted(extract_distinct_params(exps_data))).get(split, [])
        titles = list(map(str, values))
        selectors = [selector.where(split, t) for t in titles]
    else:
        selectors = [selector]
        titles = ["Experiment"]

    return selectors, titles
