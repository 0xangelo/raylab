from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.util.testing as pdtest
import pytest

from raylab.utils.exp_data import ExperimentData, load_exps_data


@pytest.fixture
def progress() -> pd.DataFrame:
    # https://kanoki.org/2019/11/18/how-to-create-dataframe-for-testing/
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
    tmpdir,
    progress: pd.DataFrame,
    params: dict,
    progress_prefix: str,
    config_prefix: str,
):
    progress.to_csv(tmpdir.join(progress_prefix + ".csv"), index=False)
    tmpdir.join(config_prefix + ".json").write(json.dumps(params))


@pytest.fixture
def directories(tmpdir) -> list[str]:
    return [str(tmpdir)]


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
