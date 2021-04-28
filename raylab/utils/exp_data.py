"""Utilities for loading and processing experiment results."""
# pylint:disable=missing-docstring,unsubscriptable-object
from __future__ import annotations

import itertools
import json
import logging
import os
from ast import literal_eval
from collections import namedtuple
from functools import reduce
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NUMERIC_KINDS = set("uifc")
ExperimentData = namedtuple("ExperimentData", ["progress", "params", "flat_params"])
Folder = namedtuple("Folder", ["path", "files"])

###############################################################################
# Experiment data loading from disk
###############################################################################


def load_exps_data(
    directories: Union[list[str], str],
    progress_prefix: str = "progress",
    config_prefix: str = "params",
    error_prefix: str = "error",
    include_errors: bool = False,
    verbose: bool = False,
) -> list[ExperimentData]:
    """Extract experiment output from directories.

    Recursively searchs directories for experiment folders and loads data
    (csv logs, hyperparameters). Returns each experiment output as an
    ExperimentData object.

    Args:
        directories: directories to recursively search for experiment data
        progress_prefix: prefix for progress files containing logs for each run
        config_prefix: prefix for config files containing hyperparameters for
            each run
        error_prefix: prefix for files containing console logs for crashed runs
        include_errors: whether to include experiments that have at least one
            error file, identified by the `error_prefix` parameter, indicating
            that the run crashed
        verbose: whether to log intermediary steps from this function to stdout

    Returns:
        A list of ExperimentData containing each experiment's output
    """
    # pylint:disable=too-many-arguments
    if isinstance(directories, str):
        directories = [directories]

    def isprogress(file: str) -> bool:
        return file.startswith(progress_prefix) and file.endswith(".csv")

    def isconfig(file: str) -> bool:
        return file.startswith(config_prefix) and file.endswith(".json")

    exp_folders = get_folders_with_target_files(directories, isprogress)
    if not include_errors:
        exp_folders = [
            d
            for d in exp_folders
            if not any(f.startswith(error_prefix) for f in d.files)
        ]
    if verbose:
        logger.info("finished walking exp folders")

    exps_data = read_exp_folder_data(exp_folders, isprogress, isconfig, verbose=verbose)
    return exps_data


# noinspection PyTypeChecker,PyUnresolvedReferences
def get_folders_with_target_files(
    directories: list[str], istarget: callable[[str], bool]
):
    """Return folders that havy any file of interest."""
    # (path, subpath, files) for each dir
    paths_per_dir = map(os.walk, directories)
    # (path, subpath, files) for all dirs
    all_paths = reduce(itertools.chain, paths_per_dir, ())
    # only (path, files)
    paths_and_files = map(lambda x: (x[0], x[2]), all_paths)
    # entries that have >0 files (may be unnecessary)
    paths_with_files = filter(lambda x: x[1], paths_and_files)
    # entries that have target files
    paths_with_target_files = filter(
        lambda x: any(istarget(y) for y in x[1]), paths_with_files
    )
    # Folders with at least one target file
    folders = map(lambda x: Folder(*x), paths_with_target_files)
    return list(folders)


# noinspection PyUnresolvedReferences
def read_exp_folder_data(
    exp_folders: list[Folder],
    isprogress: callable[[str], bool],
    isconfig: callable[[str], bool],
    verbose: bool = False,
) -> list[ExperimentData]:
    exps_data = []
    for folder in exp_folders:
        try:
            progress = load_progress(
                os.path.join(folder.path, first_that(isprogress, folder.files)),
                verbose=verbose,
            )
            params = load_params(
                os.path.join(folder.path, first_that(isconfig, folder.files))
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
                logger.error(error)

    return exps_data


# noinspection PyUnresolvedReferences
def first_that(criterion: callable[[Any], bool], lis: list[Any]) -> Any:
    return next((x for x in lis if criterion(x)), None)


def load_progress(progress_path: str, verbose: bool = False) -> pd.DataFrame:
    if verbose:
        # pylint:disable=logging-format-interpolation
        logger.info("Reading {}".format(progress_path))

    if progress_path.endswith(".csv"):
        return pd.read_csv(progress_path, index_col=False, comment="#")

    # Assume progress is in JSON format
    dicts = []
    with open(progress_path, "rt") as file:
        for line in file:
            dicts.append(json.loads(line))
    return pd.DataFrame(dicts)


def flatten_dict(dic: dict) -> dict:
    flat_params = dict()
    for key, val in dic.items():
        if isinstance(val, dict):
            # pylint:disable=fixme
            # FIXME: calling flatten_dict twice seems to be a mistake
            val = flatten_dict(val)
            for subk, subv in flatten_dict(val).items():
                flat_params[key + "/" + subk] = subv
        else:
            flat_params[key] = val
    return flat_params


def load_params(params_json_path: Optional[str]) -> dict:
    if params_json_path is None:
        return dict(exp_name="experiment")

    with open(params_json_path, "r") as file:
        data = json.load(file)
        if "args_data" in data:
            del data["args_data"]
        if "exp_name" not in data:
            data["exp_name"] = params_json_path.split("/")[-3]
    return data


###############################################################################
# Experiment data filtering in memory
###############################################################################


class Selector:
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        exps_data: list[ExperimentData],
        filters: Optional[tuple[callable[[ExperimentData], bool]]] = None,
    ):
        self._exps_data = exps_data
        self._filters = tuple() if filters is None else tuple(filters)

    def where(self, key: str, val: str) -> Selector:
        return Selector(
            self._exps_data,
            self._filters
            + (lambda exp: str(exp.flat_params.get(key, None)) == str(val),),
        )

    def where_not(self, key: str, val: str) -> Selector:
        return Selector(
            self._exps_data,
            self._filters
            + (lambda exp: str(exp.flat_params.get(key, None)) != str(val),),
        )

    def _check_exp(self, exp: ExperimentData) -> bool:
        return all(condition(exp) for condition in self._filters)

    def extract(self) -> list[ExperimentData]:
        return list(filter(self._check_exp, self._exps_data))


def filter_and_split_experiments(
    exps_data: list[ExperimentData],
    split: Optional[str] = None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> tuple[list[Selector], list[str]]:
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


def extract_distinct_params(
    exps_data: list[ExperimentData],
    excluded_params: tuple[str, ...] = ("seed", "log_dir"),
) -> list[tuple[str, Any]]:
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


def is_numeric(array) -> bool:
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


def is_increasing_key(key: str, exps_data: list[ExperimentData]) -> bool:
    for exp in exps_data:
        if key in exp.progress and not is_increasing(exp.progress[key]):
            return False
    return True


def is_increasing(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    if not is_numeric(arr):
        return False

    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return False

    return np.all(np.less_equal(arr[:-1], arr[1:])) and np.max(arr) >= np.min(arr)


def get_plottable_keys(exps_data: list[ExperimentData]) -> list[str]:
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


def get_x_plottable_keys(
    plottable_keys: list[str], exps_data: list[ExperimentData]
) -> list[str]:
    return [key for key in plottable_keys if is_increasing_key(key, exps_data)]
