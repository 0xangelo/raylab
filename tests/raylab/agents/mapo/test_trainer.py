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
    defaults = trainer_cls.options.defaults
    assert "polyak" in defaults
    assert "model_training" in defaults
    assert "model_sampling" not in defaults
    assert "virtual_buffer_size" not in defaults
    assert "model_rollouts" not in defaults
    assert "real_data_ratio" not in defaults

    assert trainer_cls._name == "MAPO"


def test_init(trainer):
    assert isinstance(trainer, ModelBasedTrainer)
    assert hasattr(trainer, "workers")
    assert not hasattr(trainer, "virtual_replay")
