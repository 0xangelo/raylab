from __future__ import annotations

from typing import Callable

import pytest
from ray.tune.logger import Logger


@pytest.fixture
def trainer_cls():
    from raylab.agents.mage import MAGETrainer

    return MAGETrainer


@pytest.fixture
def trainer(trainer_cls, logger_creator: Callable[[dict], Logger]):
    return trainer_cls(env="CartPoleSwingUp-v1", logger_creator=logger_creator)


def test_default_config(trainer_cls):
    defaults = trainer_cls.options.defaults
    assert "virtual_buffer_size" not in defaults
    assert "model_rollouts" not in defaults
    assert "real_data_ratio" not in defaults

    assert trainer_cls._name == "MAGE"


def test_init(trainer):
    assert hasattr(trainer, "workers")
    assert not hasattr(trainer, "virtual_replay")
