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
    assert "virtual_buffer_size" not in config
    assert "model_rollouts" not in config
    assert "real_data_ratio" not in config

    assert trainer_cls._name == "MAPO"


def test_init(trainer):
    assert isinstance(trainer, ModelBasedTrainer)
    assert hasattr(trainer, "workers")
    assert not hasattr(trainer, "virtual_replay")
