import copy

import pytest

from raylab.agents.config import recursive_check_info


@pytest.fixture()
def common_config():
    from ray.rllib.agents.trainer import with_common_config

    return with_common_config({})


@pytest.fixture()
def common_info():
    from raylab.agents.config import with_rllib_info

    return with_rllib_info({})


@pytest.fixture()
def allow_new_subkey_list():
    from ray.rllib.agents.trainer import Trainer

    return copy.deepcopy(Trainer._allow_unknown_subkeys)


def test_rllib_info(common_config, common_info, allow_new_subkey_list):
    recursive_check_info(common_config, common_info, allow_new_subkey_list)
