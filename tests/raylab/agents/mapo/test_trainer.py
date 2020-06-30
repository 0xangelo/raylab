# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.agents.model_based import ModelBasedTrainer


@pytest.fixture
def trainer_cls():
    from raylab.agents.mapo import MAPOTrainer

    return MAPOTrainer


@pytest.fixture
def trainer(trainer_cls):
    return trainer_cls(env="CartPoleSwingUp-v1")


def test_default_config(trainer_cls):
    assert trainer_cls._default_config is not None
    config = trainer_cls._default_config
    assert "polyak" in config
    assert "model_training" in config
    assert "model_sampling" not in config
    assert "virtual_buffer_size" in config
    assert config["virtual_buffer_size"] == 0
    assert "model_rollouts" in config
    assert config["model_rollouts"] == 0
    assert "real_data_ratio" in config
    assert config["real_data_ratio"] >= 1

    assert trainer_cls._name == "MAPO"


def test_init(trainer):
    assert isinstance(trainer, ModelBasedTrainer)
    assert hasattr(trainer, "workers")
    assert hasattr(trainer, "virtual_replay")
    assert len(trainer.virtual_replay) == 0
