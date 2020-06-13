# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import ray


@pytest.fixture
def trainer_cls():
    from raylab.agents.mage import MAGETrainer

    return MAGETrainer


@pytest.fixture
def trainer(trainer_cls, envs):  # pylint:disable=unused-argument
    ray.init()
    yield trainer_cls(env="CartPoleSwingUp-v1")
    ray.shutdown()


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

    assert trainer_cls._name == "MAGE"


def test_init(trainer):
    assert hasattr(trainer, "workers")
    assert hasattr(trainer, "virtual_replay")
    assert len(trainer.virtual_replay) == 0
