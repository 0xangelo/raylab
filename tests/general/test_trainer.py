# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import ray

from ..mock_env import MockEnv


def setup_module():
    ray.init()


def teardown_module():
    ray.shutdown()


@pytest.fixture(scope="module")
def trainer(trainer_cls):
    return trainer_cls(
        env=MockEnv,
        config={
            "num_workers": 0,
            "evaluation_config": {"explore": False},
            "evaluation_interval": 1,
        },
    )


@pytest.mark.slow
def test_trainer_step(trainer):
    trainer.train()


def test_trainer_eval(trainer):
    trainer._evaluate()


def test_trainer_restore(trainer):
    obj = trainer.save_to_object()
    trainer.restore_from_object(obj)
