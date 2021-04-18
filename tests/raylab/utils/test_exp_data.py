from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pandas.util.testing as pdtest
import pytest

from raylab.utils.exp_data import ExperimentData
from raylab.utils.exp_data import load_exps_data


@pytest.fixture
def directories(tmpdir) -> list[str]:
    return [str(tmpdir)]


@pytest.fixture
def progress() -> pd.DataFrame:
    return pdtest.makeDataFrame().reset_index()


@pytest.fixture
def params() -> dict:
    return {"exp_name": "experiment"}


@pytest.fixture
def error_log() -> str:
    return "error"


@pytest.fixture
def progress_prefix() -> str:
    return "progress"


@pytest.fixture
def config_prefix() -> str:
    return "params"


@pytest.fixture(autouse=True)
def create_experiment_output(
    directories: list[str],
    progress: pd.DataFrame,
    params: dict,
    progress_prefix: str,
    config_prefix: str,
):
    for directory in directories:
        progress.to_csv(os.path.join(directory, progress_prefix + ".csv"), index=False)
        with open(os.path.join(directory, config_prefix + ".json"), "w") as file:
            json.dump(params, file)


def test_load_exps_data(
    directories: list[str],
    progress_prefix: str,
    config_prefix: str,
    progress: pd.DataFrame,
    params: dict,
):
    assert len(directories) == 1
    exps_data = load_exps_data(
        directories, progress_prefix=progress_prefix, config_prefix=config_prefix
    )
    assert all(list(isinstance(e, ExperimentData) for e in exps_data))

    assert len(exps_data) == 1
    exp_data = exps_data[0]
    assert np.all(progress.columns == exp_data.progress.columns)
    pd.testing.assert_frame_equal(progress, exp_data.progress)
    assert params == exp_data.params
